"""Monotonic decoding and calibration-only gate-cutoff selection.

The projection is ordinary equal-weight least-squares isotonic regression with
the order constrained to be non-increasing.  It is implemented directly with
the pool-adjacent-violators algorithm so the decoder has no fitted state and is
identical on calibration, ProcessBench test, and native traces.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


def project_nonincreasing(
    values: Iterable[float], weights: Iterable[float] | None = None
) -> np.ndarray:
    """Return the closest weighted-L2 non-increasing sequence using PAVA."""

    source = np.asarray(list(values), dtype=float)
    if source.ndim != 1 or len(source) == 0 or not np.isfinite(source).all():
        raise ValueError("isotonic input must be a nonempty finite vector")
    resolved_weights = (
        np.ones(len(source), dtype=float)
        if weights is None
        else np.asarray(list(weights), dtype=float)
    )
    if resolved_weights.shape != source.shape or not np.isfinite(resolved_weights).all():
        raise ValueError("isotonic weights must be finite and aligned")
    if bool((resolved_weights <= 0).any()):
        raise ValueError("isotonic weights must be strictly positive")

    # PAVA for non-increasing values: adjacent blocks violate the order when
    # the left block is smaller than the right block.
    levels: list[float] = []
    masses: list[float] = []
    starts: list[int] = []
    ends: list[int] = []
    for index, (value, weight) in enumerate(zip(source, resolved_weights, strict=True)):
        levels.append(float(value))
        masses.append(float(weight))
        starts.append(index)
        ends.append(index + 1)
        while len(levels) >= 2 and levels[-2] < levels[-1]:
            mass = masses[-2] + masses[-1]
            level = (levels[-2] * masses[-2] + levels[-1] * masses[-1]) / mass
            levels[-2:] = [level]
            masses[-2:] = [mass]
            starts[-2:] = [starts[-2]]
            ends[-2:] = [ends[-1]]
    output = np.empty_like(source)
    for level, start, end in zip(levels, starts, ends, strict=True):
        output[start:end] = level
    if bool((np.diff(output) > 1e-12).any()):  # pragma: no cover - defensive
        raise RuntimeError("PAVA failed to produce a non-increasing sequence")
    return output


def apply_monotonic_projection(
    frame: pd.DataFrame,
    *,
    probability_column: str = "calibrated_probability",
    output_column: str = "monotonic_probability",
    model_column: str = "base_model",
    trace_column: str = "trace_id",
    order_column: str = "checkpoint_ordinal",
) -> pd.DataFrame:
    """Project each model-trace independently without changing row identity."""

    required = {model_column, trace_column, order_column, probability_column}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"monotonic projection lacks columns: {sorted(missing)}")
    output = frame.copy()
    output[output_column] = np.nan
    for _, trace in output.groupby([model_column, trace_column], sort=True):
        ordered = trace.sort_values(order_column, kind="stable")
        ordinals = ordered[order_column].to_numpy(int)
        if len(np.unique(ordinals)) != len(ordinals) or bool((np.diff(ordinals) <= 0).any()):
            raise RuntimeError("checkpoint order must be unique and strictly increasing")
        projected = project_nonincreasing(ordered[probability_column].to_numpy(float))
        output.loc[ordered.index, output_column] = projected
    if output[output_column].isna().any():  # pragma: no cover - defensive
        raise RuntimeError("monotonic projection left unassigned rows")
    return output


@dataclass(frozen=True)
class GateCutoff:
    gamma: float
    late_boundary_rate: float
    mean_retained_valid_prefix_fraction: float
    non_root_coverage: float
    calibration_traces: int
    validated_gate_exists: bool = True
    constraint: float = 0.05
    fit_split: str = "calibration_oof"
    decoder: str = "equal_weight_l2_nonincreasing_pava"

    def __post_init__(self) -> None:
        if not np.isfinite(self.gamma):
            raise ValueError("gate cutoff must be finite")
        if self.late_boundary_rate > self.constraint + 1e-12:
            raise ValueError("frozen gate violates its late-boundary constraint")
        if self.validated_gate_exists != (self.non_root_coverage > 0):
            raise ValueError("validated gate status must match non-root coverage")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _boundary_table(
    frame: pd.DataFrame,
    gamma: float,
    *,
    probability_column: str,
    trace_column: str,
    order_column: str,
    true_boundary_column: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for trace_id, trace in frame.groupby(trace_column, sort=True):
        true_values = trace[true_boundary_column].astype(int).unique()
        if len(true_values) != 1 or true_values[0] < 0:
            raise RuntimeError(f"trace {trace_id}: true boundary must be one nonnegative integer")
        true_boundary = int(true_values[0])
        eligible = trace.loc[trace[probability_column].ge(float(gamma)), order_column]
        predicted = int(eligible.max()) if len(eligible) else 0
        retained = (
            float(min(predicted, true_boundary) / true_boundary)
            if true_boundary > 0
            else float(predicted == 0)
        )
        records.append(
            {
                trace_column: str(trace_id),
                "predicted_boundary": predicted,
                "true_boundary": true_boundary,
                "late": predicted > true_boundary,
                "retained_valid_prefix_fraction": retained,
                "non_root": predicted > 0,
            }
        )
    return pd.DataFrame(records)


def select_gate_cutoff(
    oof_frame: pd.DataFrame,
    *,
    probability_column: str = "monotonic_probability",
    trace_column: str = "trace_id",
    order_column: str = "checkpoint_ordinal",
    true_boundary_column: str = "true_last_valid_checkpoint",
    max_late_rate: float = 0.05,
) -> tuple[GateCutoff, pd.DataFrame]:
    """Select a model-specific OOF cutoff under the frozen safety constraint.

    The objective is checkpoint-count retention: predicted valid checkpoints
    divided by truly valid checkpoints, with a root-only true trace scoring one
    only when the decoder also returns root.  The returned audit contains every
    candidate, including the all-root cutoff just above the maximum score.
    """

    if not 0 <= max_late_rate <= 1:
        raise ValueError("late-boundary constraint must lie in [0, 1]")
    required = {trace_column, order_column, true_boundary_column, probability_column}
    missing = required - set(oof_frame.columns)
    if missing:
        raise KeyError(f"gate selection lacks columns: {sorted(missing)}")
    if "split" in oof_frame and set(oof_frame["split"].astype(str)) != {"calibration"}:
        raise RuntimeError("gate selection may use calibration rows only")
    probabilities = oof_frame[probability_column].to_numpy(float)
    if not np.isfinite(probabilities).all() or bool(((probabilities < 0) | (probabilities > 1)).any()):
        raise ValueError("gate probabilities must be finite in [0, 1]")
    unique = np.unique(probabilities)
    candidates = np.concatenate([unique, [np.nextafter(float(unique.max()), np.inf)]])
    rows: list[dict[str, Any]] = []
    for gamma in candidates:
        boundaries = _boundary_table(
            oof_frame,
            float(gamma),
            probability_column=probability_column,
            trace_column=trace_column,
            order_column=order_column,
            true_boundary_column=true_boundary_column,
        )
        rows.append(
            {
                "gamma": float(gamma),
                "late_boundary_rate": float(boundaries["late"].mean()),
                "mean_retained_valid_prefix_fraction": float(
                    boundaries["retained_valid_prefix_fraction"].mean()
                ),
                "non_root_coverage": float(boundaries["non_root"].mean()),
                "traces": int(len(boundaries)),
            }
        )
    audit = pd.DataFrame(rows).sort_values("gamma", kind="stable").reset_index(drop=True)
    feasible = audit.loc[audit["late_boundary_rate"].le(max_late_rate + 1e-12)].copy()
    if feasible.empty:  # mathematically unreachable because all-root is safe
        raise RuntimeError("no gate cutoff satisfies the late-boundary constraint")
    feasible.sort_values(
        [
            "mean_retained_valid_prefix_fraction",
            "late_boundary_rate",
            "non_root_coverage",
            "gamma",
        ],
        ascending=[False, True, False, True],
        kind="stable",
        inplace=True,
    )
    best = feasible.iloc[0]
    cutoff = GateCutoff(
        gamma=float(best["gamma"]),
        late_boundary_rate=float(best["late_boundary_rate"]),
        mean_retained_valid_prefix_fraction=float(best["mean_retained_valid_prefix_fraction"]),
        non_root_coverage=float(best["non_root_coverage"]),
        calibration_traces=int(best["traces"]),
        validated_gate_exists=bool(best["non_root_coverage"] > 0),
        constraint=float(max_late_rate),
    )
    # A gate that returns root on every trace is valid but not useful.  Make
    # this explicit for the caller; the protocol says not to relax safety.
    audit["selected"] = np.isclose(audit["gamma"], cutoff.gamma, rtol=0, atol=0)
    audit["nontrivial_validated_gate"] = cutoff.validated_gate_exists
    return cutoff, audit


def decode_boundaries(
    frame: pd.DataFrame,
    cutoff: GateCutoff | float,
    *,
    probability_column: str = "monotonic_probability",
    trace_column: str = "trace_id",
    order_column: str = "checkpoint_ordinal",
    true_boundary_column: str = "true_last_valid_checkpoint",
) -> pd.DataFrame:
    """Materialize one predicted/true boundary row per trace."""

    gamma = cutoff.gamma if isinstance(cutoff, GateCutoff) else float(cutoff)
    return _boundary_table(
        frame,
        gamma,
        probability_column=probability_column,
        trace_column=trace_column,
        order_column=order_column,
        true_boundary_column=true_boundary_column,
    )
