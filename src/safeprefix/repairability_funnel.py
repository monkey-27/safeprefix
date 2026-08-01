"""Post-hoc repairability-funnel analysis over frozen SafePrefix H4 artifacts.

This module is deliberately analysis-only.  It accepts persisted teacher-forced
branch states and, optionally, hidden states recovered by deterministic replay
of the already stored failed token sequence.  It contains no sampling or native
repair-policy surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from safeprefix.config import load_config
from safeprefix.recoverability_geometry.runner import load_frozen_axes


EPS = 1e-12
MODEL_ORDER = (
    "family_a_small",
    "family_a_large",
    "family_b_small",
    "family_b_large",
)
HORIZON_ORDER = (32, 64, 128)
FORBIDDEN_PATH_TERMS = ("native", "native_eval", "native-eval", "final_test")


def _guard_analysis_path(path: str | Path) -> None:
    text = Path(path).as_posix().casefold()
    if any(term in text for term in FORBIDDEN_PATH_TERMS):
        raise RuntimeError(f"native/final-test path is forbidden: {path}")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _json_safe(value: Any) -> Any:
    """Convert numpy/path values and non-finite floats to strict JSON values."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            _json_safe(payload),
            indent=2,
            sort_keys=True,
            default=_json_default,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(base: int, *parts: Any) -> int:
    digest = hashlib.sha256(
        "\x1f".join([str(base), *(str(part) for part in parts)]).encode()
    ).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def _unit(vector: np.ndarray) -> np.ndarray | None:
    values = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if not math.isfinite(norm) or norm <= EPS:
        return None
    return values / norm


def _cosine(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return float("nan")
    return float(np.clip(np.dot(left, right), -1.0, 1.0))


def _safe_spearman(left: Sequence[float], right: Sequence[float]) -> float:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3 or np.ptp(x[finite]) <= 0 or np.ptp(y[finite]) <= 0:
        return float("nan")
    return float(spearmanr(x[finite], y[finite]).statistic)


def _one_sided_bootstrap_p(samples: np.ndarray, *, positive: bool) -> float:
    finite = np.asarray(samples, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return 1.0
    unfavorable = np.sum(finite <= 0) if positive else np.sum(finite >= 0)
    return float((unfavorable + 1) / (len(finite) + 1))


def _holm(p_values: Mapping[str, float]) -> dict[str, dict[str, float | bool]]:
    names = list(p_values)
    ordered = sorted(names, key=lambda name: (float(p_values[name]), name))
    running = 0.0
    adjusted: dict[str, float] = {}
    count = len(ordered)
    for rank, name in enumerate(ordered):
        candidate = min(1.0, float(p_values[name]) * (count - rank))
        running = max(running, candidate)
        adjusted[name] = running
    return {
        name: {
            "raw_p": float(p_values[name]),
            "holm_adjusted_p": float(adjusted[name]),
            "reject_at_0_05": bool(adjusted[name] < 0.05),
        }
        for name in names
    }


def _read_jsonl(path: Path) -> pd.DataFrame:
    return pd.DataFrame(
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    )


@dataclass
class FoldTransform:
    mean: np.ndarray
    components: np.ndarray
    variance: np.ndarray
    ridge: float
    retained_variance_fraction: float

    def apply(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        return ((array - self.mean) @ self.components.T) / np.sqrt(
            self.variance + self.ridge
        )

    @property
    def dimensions(self) -> int:
        return int(len(self.variance))


def _fold_assignment(trace_ids: Sequence[str], folds: int, seed: int) -> dict[str, int]:
    unique = sorted(set(map(str, trace_ids)), key=lambda value: (_stable_seed(seed, value), value))
    if len(unique) < 2:
        raise RuntimeError("cross-fitting requires at least two failed traces")
    count = min(int(folds), len(unique))
    return {trace_id: index % count for index, trace_id in enumerate(unique)}


def _fit_transform(
    training: np.ndarray,
    *,
    variance_target: float,
    maximum_dimensions: int,
    ridge_fraction: float,
    seed: int,
) -> FoldTransform:
    values = np.asarray(training, dtype=np.float64)
    maximum = min(int(maximum_dimensions), values.shape[0] - 1, values.shape[1])
    if maximum < 1:
        raise RuntimeError("insufficient fitting-fold rows for PCA")
    if values.shape[0] <= 128 and values.shape[0] < values.shape[1]:
        # Exact dual PCA is much faster for the 32-row parent-state fitting
        # folds than repeatedly running randomized SVD in 4K dimensions.
        mean = values.mean(axis=0)
        centered = values - mean
        gram = (centered @ centered.T) / max(values.shape[0] - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.clip(eigenvalues[order], 0, None)[:maximum]
        eigenvectors = eigenvectors[:, order][:, :maximum]
        singular = np.sqrt(eigenvalues * max(values.shape[0] - 1, 1))
        valid = singular > EPS
        eigenvalues = eigenvalues[valid]
        eigenvectors = eigenvectors[:, valid]
        singular = singular[valid]
        components_all = (eigenvectors.T @ centered) / singular[:, None]
        total_variance = float(np.var(values, axis=0, ddof=1).sum())
        ratios = eigenvalues / max(total_variance, EPS)
        fitted_mean = mean
        fitted_components = components_all
        fitted_variance = eigenvalues
    else:
        fit = PCA(
            n_components=maximum,
            svd_solver="randomized",
            iterated_power=2,
            n_oversamples=10,
            random_state=int(seed),
        )
        fit.fit(values)
        ratios = np.asarray(fit.explained_variance_ratio_, dtype=float)
        fitted_mean = np.asarray(fit.mean_, dtype=float)
        fitted_components = np.asarray(fit.components_, dtype=float)
        fitted_variance = np.asarray(fit.explained_variance_, dtype=float)
    cumulative = np.cumsum(ratios)
    retained = int(np.searchsorted(cumulative, float(variance_target), side="left") + 1)
    retained = min(retained, len(fitted_variance), maximum)
    variance = np.maximum(fitted_variance[:retained], EPS)
    ridge = max(float(ridge_fraction) * float(np.mean(variance)), EPS)
    return FoldTransform(
        mean=fitted_mean,
        components=fitted_components[:retained],
        variance=variance,
        ridge=ridge,
        retained_variance_fraction=float(cumulative[retained - 1]),
    )


def _cross_fit(
    values: Mapping[int, np.ndarray],
    traces: Mapping[int, str],
    *,
    folds: int,
    variance_target: float,
    maximum_dimensions: int,
    ridge_fraction: float,
    seed: int,
) -> tuple[dict[int, np.ndarray], dict[int, FoldTransform], dict[int, int], list[dict[str, Any]]]:
    indices = sorted(values)
    assignments_by_trace = _fold_assignment([traces[index] for index in indices], folds, seed)
    assignments = {index: assignments_by_trace[str(traces[index])] for index in indices}
    output: dict[int, np.ndarray] = {}
    transforms: dict[int, FoldTransform] = {}
    records: list[dict[str, Any]] = []
    for fold in sorted(set(assignments.values())):
        train_indices = [index for index in indices if assignments[index] != fold]
        test_indices = [index for index in indices if assignments[index] == fold]
        training = np.stack([values[index] for index in train_indices])
        transform = _fit_transform(
            training,
            variance_target=variance_target,
            maximum_dimensions=maximum_dimensions,
            ridge_fraction=ridge_fraction,
            seed=_stable_seed(seed, fold),
        )
        transformed = transform.apply(np.stack([values[index] for index in test_indices]))
        for index, vector in zip(test_indices, transformed, strict=True):
            output[index] = vector
        transforms[fold] = transform
        records.append(
            {
                "fold": int(fold),
                "fit_trace_count": len({str(traces[index]) for index in train_indices}),
                "heldout_trace_count": len({str(traces[index]) for index in test_indices}),
                "fit_row_count": len(train_indices),
                "heldout_row_count": len(test_indices),
                "retained_dimensions": transform.dimensions,
                "retained_variance_fraction": transform.retained_variance_fraction,
                "whitening_ridge": transform.ridge,
            }
        )
    return output, transforms, assignments, records


def _weighted_geometry(directions: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    u = np.asarray(directions, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if len(u) != len(w) or len(u) < 2 or np.any(w < 0) or float(w.sum()) <= EPS:
        return {key: float("nan") for key in ("effective_rank", "angular_dispersion", "leading_spectral_mass")}
    normalized = w / w.sum()
    mean = np.sum(normalized[:, None] * u, axis=0)
    centered = u - mean
    weighted = centered * np.sqrt(normalized[:, None])
    gram = weighted @ weighted.T
    eigenvalues = np.linalg.eigvalsh(gram)
    eigenvalues = np.clip(eigenvalues, 0, None)
    trace = float(eigenvalues.sum())
    square_trace = float(np.square(eigenvalues).sum())
    return {
        "effective_rank": float(trace * trace / square_trace) if square_trace > EPS else float("nan"),
        "angular_dispersion": float(1.0 - np.linalg.norm(np.sum(normalized[:, None] * u, axis=0))),
        "leading_spectral_mass": float(eigenvalues[-1] / trace) if trace > EPS else float("nan"),
    }


def _funnel_metrics(directions: np.ndarray, recoverability: np.ndarray) -> dict[str, float | np.ndarray | None]:
    u = np.asarray(directions, dtype=np.float64)
    r = np.asarray(recoverability, dtype=np.float64)
    contrast = r - r.mean()
    resultant = np.sum(contrast[:, None] * u, axis=0)
    denominator = float(np.abs(contrast).sum())
    concentration = float(np.linalg.norm(resultant) / denominator) if denominator > EPS else float("nan")
    direction = _unit(resultant)
    weighted = _weighted_geometry(u, r)
    unweighted = _weighted_geometry(u, np.ones(len(u)))
    return {
        "orientation": direction,
        "directional_concentration": concentration,
        **weighted,
        "unweighted_effective_rank": unweighted["effective_rank"],
        "unweighted_angular_dispersion": unweighted["angular_dispersion"],
        "unweighted_leading_spectral_mass": unweighted["leading_spectral_mass"],
    }


def _adherence(query: np.ndarray, directions: np.ndarray, recoverability: np.ndarray) -> float:
    cosine = np.asarray(directions, dtype=float) @ np.asarray(query, dtype=float)
    success = np.asarray(recoverability, dtype=float)
    failure = 1.0 - success
    if success.sum() <= EPS or failure.sum() <= EPS:
        return float("nan")
    return float(np.average(cosine, weights=success) - np.average(cosine, weights=failure))


def _orientation(directions: np.ndarray, recoverability: np.ndarray) -> np.ndarray | None:
    contrast = np.asarray(recoverability, dtype=float) - float(np.mean(recoverability))
    return _unit(np.sum(contrast[:, None] * np.asarray(directions, dtype=float), axis=0))


def _bootstrap_statistic(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    *,
    replicates: int,
    seed: int,
    trace_column: str = "trace_id",
) -> dict[str, Any]:
    point = float(statistic(frame))
    traces = sorted(frame[trace_column].astype(str).unique())
    rng = np.random.default_rng(int(seed))
    values: list[float] = []
    parts = {trace: frame.loc[frame[trace_column].astype(str) == trace] for trace in traces}
    for _ in range(int(replicates)):
        sampled = rng.choice(traces, size=len(traces), replace=True)
        replicate = pd.concat(
            [parts[str(trace)].assign(_bootstrap_trace=index) for index, trace in enumerate(sampled)],
            ignore_index=True,
        )
        value = float(statistic(replicate))
        if math.isfinite(value):
            values.append(value)
    array = np.asarray(values, dtype=float)
    return {
        "estimate": point,
        "ci_lower": float(np.quantile(array, 0.025)) if len(array) else float("nan"),
        "ci_upper": float(np.quantile(array, 0.975)) if len(array) else float("nan"),
        "bootstrap_replicates_requested": int(replicates),
        "bootstrap_replicates_finite": int(len(array)),
        "bootstrap_values": array,
        "trace_count": int(len(traces)),
    }


def _macro_model_mean(frame: pd.DataFrame, column: str) -> float:
    values = frame.groupby("base_model", sort=True)[column].mean()
    return float(values.mean()) if len(values) else float("nan")


def _relative_error_bin(relative: float) -> str:
    if not math.isfinite(float(relative)):
        return "missing_error_annotation"
    if relative <= -2:
        return "two_or_more_before"
    if relative == -1:
        return "immediately_before"
    if relative == 0:
        return "at_error"
    return "after_error"


def _prepare_corpus(
    children: pd.DataFrame,
    parents: pd.DataFrame,
    *,
    boundary_root: Path,
    config: Mapping[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[int, np.ndarray],
    dict[tuple[str, int, int], FoldTransform],
    dict[int, int],
    dict[tuple[str, int, str], np.ndarray],
    dict[tuple[str, int, str], np.ndarray],
    dict[tuple[str, int, str], np.ndarray],
    pd.DataFrame,
]:
    """Inventory, exact probe transform, drift removal, and cross-fitting."""

    required_children = {
        "base_model",
        "trace_id",
        "checkpoint_index",
        "parent_id",
        "branch_id",
        "branch_index",
        "horizon",
        "raw_hidden",
        "parent_raw_hidden",
        "child_success_count",
        "child_num_rollouts",
        "horizon_available",
        "parent_recoverability",
        "infrastructure_status",
        "domain",
    }
    if missing := required_children - set(children):
        raise KeyError(f"child corpus missing {sorted(missing)}")
    if children.duplicated(["base_model", "parent_id", "branch_id", "horizon"]).any():
        raise RuntimeError("duplicate branch-horizon identity")
    if len(parents) != parents[["base_model", "trace_id", "checkpoint_index"]].drop_duplicates().shape[0]:
        raise RuntimeError("parent manifest identity is not unique")

    metadata_columns = [
        "base_model",
        "trace_id",
        "checkpoint_index",
        "checkpoint_ordinal",
        "checkpoint_token_offset",
        "total_trace_token_count",
        "total_checkpoint_count",
        "first_error_zero_based_analysis_only",
        "common_trace_id",
        "split",
        "frozen_pipeline_split",
        "canonical_frozen_calibrated_probability",
        "dense_recoverability",
    ]
    metadata_columns = [column for column in metadata_columns if column in parents]
    frame = children.merge(
        parents[metadata_columns],
        on=["base_model", "trace_id", "checkpoint_index"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_manifest"),
    )
    if frame["checkpoint_token_offset"].isna().any():
        raise RuntimeError("child corpus contains a parent absent from the selection manifest")
    frame["normalized_checkpoint_position"] = (
        frame["checkpoint_token_offset"].astype(float)
        / frame["total_trace_token_count"].astype(float).clip(lower=1)
    ).clip(0, 1)
    frame["relative_to_first_visible_error"] = (
        frame["checkpoint_ordinal"].astype(float)
        - pd.to_numeric(frame["first_error_zero_based_analysis_only"], errors="coerce")
    )

    eligibility = config["eligibility"]
    required_rollouts = int(eligibility["required_child_rollouts"])
    frame["valid_branch"] = (
        frame["horizon_available"].astype(bool)
        & frame["raw_hidden"].notna()
        & frame["parent_raw_hidden"].notna()
        & (frame["child_num_rollouts"].astype(int) == required_rollouts)
        & (frame["infrastructure_status"].astype(str) == "complete")
        & frame["child_success_count"].between(0, required_rollouts)
    )
    inventory_records: list[dict[str, Any]] = []
    for keys, part in frame.groupby(
        ["base_model", "trace_id", "parent_id", "checkpoint_index", "domain", "horizon"],
        sort=True,
        dropna=False,
    ):
        model, trace, parent_id, checkpoint_index, domain, horizon = keys
        valid = part.loc[part["valid_branch"]]
        unique_valid = int(valid["branch_id"].nunique())
        distinct_counts = int(valid["child_success_count"].nunique())
        attrition_reasons: list[str] = []
        if int((~part["horizon_available"].astype(bool)).sum()):
            attrition_reasons.append("horizon_unavailable_by_protocol")
        if int((part["raw_hidden"].isna() | part["parent_raw_hidden"].isna()).sum()):
            attrition_reasons.append("missing_hidden_representation")
        if int((part["child_num_rollouts"].astype(int) != required_rollouts).sum()):
            attrition_reasons.append("invalid_continuation_count")
        if int((part["infrastructure_status"].astype(str) != "complete").sum()):
            attrition_reasons.append("infrastructure_incomplete")
        reasons: list[str] = []
        if unique_valid < int(eligibility["minimum_unique_valid_branches"]):
            reasons.append("fewer_than_eight_unique_valid_branches")
        if distinct_counts < int(eligibility["minimum_distinct_success_counts"]):
            reasons.append("fewer_than_two_distinct_child_success_counts")
        inventory_records.append(
            {
                "base_model": str(model),
                "trace_id": str(trace),
                "parent_id": str(parent_id),
                "checkpoint_index": int(checkpoint_index),
                "domain": str(domain),
                "horizon": int(horizon),
                "planned_branch_rows": int(len(part)),
                "unique_valid_branches": unique_valid,
                "distinct_child_success_counts": distinct_counts,
                "success_conditioned_geometry_eligible": not reasons,
                "exclusion_reason": "eligible" if not reasons else ";".join(sorted(set(reasons))),
                "branch_attrition_reason": (
                    "none" if not attrition_reasons else ";".join(sorted(set(attrition_reasons)))
                ),
                "f4_exact_twelve_eligible": bool(not reasons and unique_valid == 12),
            }
        )
    inventory = pd.DataFrame(inventory_records)
    eligible_keys = set(
        map(
            tuple,
            inventory.loc[
                inventory["success_conditioned_geometry_eligible"],
                ["base_model", "parent_id", "horizon"],
            ].to_numpy(),
        )
    )
    frame["eligible"] = [
        (str(row.base_model), str(row.parent_id), int(row.horizon)) in eligible_keys
        for row in frame.itertuples()
    ]

    # A persisted hidden vector can be finite yet provide no angular object:
    # several realized H4 parent-horizons contain identical FP16 child states
    # across all siblings.  Mean-drift subtraction then leaves exact numerical
    # zero (up to roundoff).  Inventory this as representation collapse rather
    # than normalizing floating-point noise into an arbitrary direction.
    frame["angular_valid"] = False
    angular_counts: dict[tuple[str, str, int], tuple[int, int, int]] = {}
    for model, model_frame in frame.loc[frame["eligible"] & frame["valid_branch"]].groupby(
        "base_model", sort=True
    ):
        axes, canonical_seed, _ = load_frozen_axes(boundary_root, str(model))
        axis = axes[canonical_seed]
        indices = list(map(int, model_frame.index))
        child = axis.transform_raw(
            np.stack(model_frame["raw_hidden"].map(lambda value: np.asarray(value, dtype=np.float32)))
        ).astype(np.float64)
        parent = axis.transform_raw(
            np.stack(model_frame["parent_raw_hidden"].map(lambda value: np.asarray(value, dtype=np.float32)))
        ).astype(np.float64)
        displacement = {index: child[offset] - parent[offset] for offset, index in enumerate(indices)}
        for (parent_id, horizon), part in model_frame.groupby(["parent_id", "horizon"], sort=True):
            part_indices = list(map(int, part.index))
            values = np.stack([displacement[index] for index in part_indices])
            centered = values - values.mean(axis=0)
            nonzero = np.linalg.norm(centered, axis=1) > 1e-8
            for index, valid in zip(part_indices, nonzero, strict=True):
                frame.at[index, "angular_valid"] = bool(valid)
            retained = part.loc[nonzero]
            angular_counts[(str(model), str(parent_id), int(horizon))] = (
                int(nonzero.sum()),
                int(retained["child_success_count"].nunique()),
                int((~nonzero).sum()),
            )
    inventory["unique_angular_valid_branches"] = 0
    inventory["zero_centered_displacement_branches"] = 0
    for index, row in inventory.iterrows():
        key = (str(row["base_model"]), str(row["parent_id"]), int(row["horizon"]))
        angular_count, distinct_count, zero_count = angular_counts.get(key, (0, 0, 0))
        inventory.at[index, "unique_angular_valid_branches"] = angular_count
        inventory.at[index, "zero_centered_displacement_branches"] = zero_count
        if bool(row["success_conditioned_geometry_eligible"]):
            extra: list[str] = []
            if angular_count < int(eligibility["minimum_unique_valid_branches"]):
                extra.append("fewer_than_eight_nonzero_centered_displacements")
            if distinct_count < int(eligibility["minimum_distinct_success_counts"]):
                extra.append("nonzero_displacements_have_fewer_than_two_success_counts")
            if extra:
                inventory.at[index, "success_conditioned_geometry_eligible"] = False
                prior = str(inventory.at[index, "exclusion_reason"])
                reasons = ([] if prior == "eligible" else prior.split(";")) + extra
                inventory.at[index, "exclusion_reason"] = ";".join(sorted(set(reasons)))
        inventory.at[index, "f4_exact_twelve_eligible"] = bool(
            inventory.at[index, "success_conditioned_geometry_eligible"]
            and angular_count == 12
        )
    inventory["unique_angular_valid_branches"] = inventory["unique_angular_valid_branches"].astype(int)
    inventory["zero_centered_displacement_branches"] = inventory["zero_centered_displacement_branches"].astype(int)
    eligible_keys = set(
        map(
            tuple,
            inventory.loc[
                inventory["success_conditioned_geometry_eligible"],
                ["base_model", "parent_id", "horizon"],
            ].to_numpy(),
        )
    )
    frame["eligible"] = [
        (str(row.base_model), str(row.parent_id), int(row.horizon)) in eligible_keys
        for row in frame.itertuples()
    ]
    frame["analysis_branch"] = frame["valid_branch"] & frame["angular_valid"]

    representation = config["representation"]
    direction_by_row: dict[int, np.ndarray] = {}
    transform_by_slice_fold: dict[tuple[str, int, int], FoldTransform] = {}
    fold_by_row: dict[int, int] = {}
    sibling_mean_by_parent: dict[tuple[str, int, str], np.ndarray] = {}
    parent_probe_feature: dict[tuple[str, int, str], np.ndarray] = {}
    parent_state_by_parent: dict[tuple[str, int, str], np.ndarray] = {}
    crossfit_records: list[dict[str, Any]] = []

    frame["r_tilde"] = (frame["child_success_count"].astype(float) + 0.5) / 5.0
    frame["r_unsmoothed"] = frame["child_success_count"].astype(float) / required_rollouts
    frame["crossfit_fold"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")

    for (model, horizon), slice_frame in frame.loc[frame["eligible"] & frame["analysis_branch"]].groupby(
        ["base_model", "horizon"], sort=True
    ):
        axes, canonical_seed, _ = load_frozen_axes(boundary_root, str(model))
        axis = axes[canonical_seed]
        indices = list(map(int, slice_frame.index))
        child_raw = np.stack(slice_frame["raw_hidden"].map(lambda value: np.asarray(value, dtype=np.float32)))
        parent_raw = np.stack(slice_frame["parent_raw_hidden"].map(lambda value: np.asarray(value, dtype=np.float32)))
        child_feature = axis.transform_raw(child_raw).astype(np.float64)
        parent_feature = axis.transform_raw(parent_raw).astype(np.float64)
        raw_displacement = {
            index: child_feature[offset] - parent_feature[offset]
            for offset, index in enumerate(indices)
        }
        traces = {index: str(frame.at[index, "trace_id"]) for index in indices}
        centered: dict[int, np.ndarray] = {}
        for parent_id, part in slice_frame.groupby("parent_id", sort=True):
            part_indices = list(map(int, part.index))
            mean = np.mean(np.stack([raw_displacement[index] for index in part_indices]), axis=0)
            parent_key = (str(model), int(horizon), str(parent_id))
            sibling_mean_by_parent[parent_key] = mean
            representative = part_indices[0]
            parent_probe_feature[parent_key] = parent_feature[indices.index(representative)]
            for index in part_indices:
                centered[index] = raw_displacement[index] - mean
        transformed, transforms, assignments, records = _cross_fit(
            centered,
            traces,
            folds=int(representation["cross_fit_folds"]),
            variance_target=float(representation["variance_retained"]),
            maximum_dimensions=int(representation["maximum_dimensions"]),
            ridge_fraction=float(representation["whitening_ridge_fraction_of_mean_retained_variance"]),
            seed=_stable_seed(int(config["experiment"]["seed"]), model, horizon, "displacement"),
        )
        for index, vector in transformed.items():
            unit = _unit(vector)
            if unit is None:
                raise RuntimeError(f"zero cross-fitted displacement: row {index}")
            direction_by_row[index] = unit
            fold_by_row[index] = assignments[index]
            frame.at[index, "crossfit_fold"] = assignments[index]
        for fold, transform in transforms.items():
            transform_by_slice_fold[(str(model), int(horizon), int(fold))] = transform
        for record in records:
            crossfit_records.append(
                {"space": "branch_displacement", "base_model": str(model), "horizon": int(horizon), **record}
            )

        unique_parent_rows = slice_frame.sort_index().groupby("parent_id", sort=True).head(1)
        parent_values = {
            int(index): parent_probe_feature[(str(model), int(horizon), str(row.parent_id))]
            for index, row in unique_parent_rows.iterrows()
        }
        parent_traces = {int(index): str(row.trace_id) for index, row in unique_parent_rows.iterrows()}
        parent_transformed, _, parent_assignments, parent_records = _cross_fit(
            parent_values,
            parent_traces,
            folds=int(representation["cross_fit_folds"]),
            variance_target=float(representation["variance_retained"]),
            maximum_dimensions=int(representation["maximum_dimensions"]),
            ridge_fraction=float(representation["whitening_ridge_fraction_of_mean_retained_variance"]),
            seed=_stable_seed(int(config["experiment"]["seed"]), model, horizon, "displacement"),
        )
        for index, vector in parent_transformed.items():
            row = frame.loc[index]
            key = (str(model), int(horizon), str(row["parent_id"]))
            parent_state_by_parent[key] = vector
            if int(parent_assignments[index]) != int(fold_by_row[index]):
                raise AssertionError("parent and displacement fold assignments differ")
        for record in parent_records:
            crossfit_records.append(
                {"space": "parent_state", "base_model": str(model), "horizon": int(horizon), **record}
            )

    if set(frame.index[frame["eligible"] & frame["analysis_branch"]].map(int)) != set(direction_by_row):
        raise RuntimeError("not every eligible valid branch received a cross-fitted representation")
    return (
        frame,
        inventory,
        direction_by_row,
        transform_by_slice_fold,
        fold_by_row,
        sibling_mean_by_parent,
        parent_probe_feature,
        parent_state_by_parent,
        pd.DataFrame(crossfit_records),
    )


def _run_f1(
    frame: pd.DataFrame,
    directions: Mapping[int, np.ndarray],
    *,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_parent: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    replicates = int(config["f1"]["permutation_replicates"])
    criteria = config["f1"]["descriptive_criteria"]
    grouping = ["base_model", "trace_id", "parent_id", "checkpoint_index", "domain", "horizon"]
    for keys, part in frame.loc[frame["eligible"] & frame["analysis_branch"]].groupby(grouping, sort=True):
        model, trace, parent_id, checkpoint_index, domain, horizon = keys
        part = part.sort_values("branch_index")
        u = np.stack([directions[int(index)] for index in part.index])
        smooth = part["r_tilde"].to_numpy(float)
        raw = part["r_unsmoothed"].to_numpy(float)
        observed = _funnel_metrics(u, smooth)
        observed_raw = _funnel_metrics(u, raw)
        rng = np.random.default_rng(
            _stable_seed(int(config["experiment"]["seed"]), "f1", model, parent_id, horizon)
        )
        metrics = [
            "directional_concentration",
            "effective_rank",
            "angular_dispersion",
            "leading_spectral_mass",
        ]
        null: dict[str, list[float]] = {metric: [] for metric in metrics}
        null_raw: dict[str, list[float]] = {metric: [] for metric in metrics}
        for replicate in range(replicates):
            permutation = rng.permutation(len(part))
            permuted = _funnel_metrics(u, smooth[permutation])
            permuted_raw = _funnel_metrics(u, raw[permutation])
            record: dict[str, Any] = {
                "base_model": str(model),
                "trace_id": str(trace),
                "parent_id": str(parent_id),
                "horizon": int(horizon),
                "permutation_index": int(replicate),
            }
            for metric in metrics:
                value = float(permuted[metric])
                raw_value = float(permuted_raw[metric])
                null[metric].append(value)
                null_raw[metric].append(raw_value)
                record[f"null_{metric}"] = value
                record[f"null_unsmoothed_{metric}"] = raw_value
            null_rows.append(record)
        result: dict[str, Any] = {
            "base_model": str(model),
            "trace_id": str(trace),
            "parent_id": str(parent_id),
            "checkpoint_index": int(checkpoint_index),
            "domain": str(domain),
            "horizon": int(horizon),
            "branch_count": int(len(part)),
            "mean_r_tilde": float(smooth.mean()),
            "crossfit_fold": int(part["crossfit_fold"].iloc[0]),
            "orientation": observed["orientation"].tolist() if observed["orientation"] is not None else None,
        }
        for metric in metrics:
            observed_value = float(observed[metric])
            samples = np.asarray(null[metric], dtype=float)
            mean, std = float(np.nanmean(samples)), float(np.nanstd(samples, ddof=1))
            result[metric] = observed_value
            result[f"null_mean_{metric}"] = mean
            result[f"null_sd_{metric}"] = std
            result[f"null_difference_{metric}"] = observed_value - mean
            result[f"z_{metric}"] = (observed_value - mean) / std if std > EPS else float("nan")
            result[f"permutation_two_sided_p_{metric}"] = float(
                (np.sum(np.abs(samples - mean) >= abs(observed_value - mean)) + 1) / (len(samples) + 1)
            )
            raw_observed = float(observed_raw[metric])
            raw_samples = np.asarray(null_raw[metric], dtype=float)
            finite_raw = raw_samples[np.isfinite(raw_samples)]
            raw_mean = float(finite_raw.mean()) if len(finite_raw) else float("nan")
            raw_std = float(finite_raw.std(ddof=1)) if len(finite_raw) > 1 else float("nan")
            result[f"unsmoothed_{metric}"] = raw_observed
            result[f"unsmoothed_z_{metric}"] = (
                (raw_observed - raw_mean) / raw_std if raw_std > EPS else float("nan")
            )
        for metric in (
            "unweighted_effective_rank",
            "unweighted_angular_dispersion",
            "unweighted_leading_spectral_mass",
        ):
            result[metric] = float(observed[metric])
        zc = float(result["z_directional_concentration"])
        zr = float(result["z_effective_rank"])
        zd = float(result["z_angular_dispersion"])
        zl = float(result["z_leading_spectral_mass"])
        narrow = criteria["narrow"]
        broad = criteria["broad"]
        multimodal = criteria["multimodal"]
        if (
            zc >= float(narrow["minimum_concentration_z"])
            and zr <= float(narrow["maximum_effective_rank_z"])
            and zl >= float(narrow["minimum_leading_mass_z"])
        ):
            category = "narrow"
        elif (
            abs(zc) <= float(broad["maximum_absolute_concentration_z"])
            and zr >= float(broad["minimum_effective_rank_z"])
            and zd >= float(broad["minimum_dispersion_z"])
        ):
            category = "broad"
        elif (
            zc <= float(multimodal["maximum_concentration_z"])
            and zl >= float(multimodal["minimum_leading_mass_z"])
        ):
            category = "multimodal"
        else:
            category = str(criteria["fallback"])
        result["descriptive_geometry"] = category
        per_parent.append(result)
    return pd.DataFrame(per_parent), pd.DataFrame(null_rows)


def _run_f2(
    frame: pd.DataFrame,
    directions: Mapping[int, np.ndarray],
    parent_states: Mapping[tuple[str, int, str], np.ndarray],
    *,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    f2 = config["f2"]
    repetitions = int(f2["split_half_replicates"])
    parent_records: list[dict[str, Any]] = []
    orientation_halves: dict[tuple[str, int, str], tuple[np.ndarray | None, np.ndarray | None]] = {}
    metadata: dict[tuple[str, int, str], dict[str, Any]] = {}
    grouping = ["base_model", "trace_id", "parent_id", "checkpoint_index", "domain", "horizon"]
    for keys, part in frame.loc[frame["eligible"] & frame["analysis_branch"]].groupby(grouping, sort=True):
        model, trace, parent_id, checkpoint_index, domain, horizon = keys
        part = part.sort_values("branch_index")
        u = np.stack([directions[int(index)] for index in part.index])
        r = part["r_tilde"].to_numpy(float)
        size = len(part) // 2
        rng = np.random.default_rng(
            _stable_seed(int(config["experiment"]["seed"]), "f2", model, parent_id, horizon)
        )
        cosines: list[float] = []
        first_halves: tuple[np.ndarray | None, np.ndarray | None] | None = None
        for repetition in range(repetitions):
            order = rng.permutation(len(part))
            left, right = order[:size], order[size : 2 * size]
            left_orientation = _orientation(u[left], r[left])
            right_orientation = _orientation(u[right], r[right])
            if first_halves is None and left_orientation is not None and right_orientation is not None:
                first_halves = (left_orientation, right_orientation)
            cosine = _cosine(left_orientation, right_orientation)
            if math.isfinite(cosine):
                cosines.append(cosine)
        key = (str(model), int(horizon), str(parent_id))
        if first_halves is None:
            first_halves = (None, None)
        orientation_halves[key] = first_halves
        row = part.iloc[0]
        position_bins = np.asarray(f2["normalized_position_bins"], dtype=float)
        recovery_bins = np.asarray(f2["parent_recoverability_bins"], dtype=float)
        metadata[key] = {
            "base_model": str(model),
            "trace_id": str(trace),
            "parent_id": str(parent_id),
            "checkpoint_index": int(checkpoint_index),
            "domain": str(domain),
            "horizon": int(horizon),
            "crossfit_fold": int(row["crossfit_fold"]),
            "normalized_checkpoint_position": float(row["normalized_checkpoint_position"]),
            "parent_recoverability": float(row["parent_recoverability"]),
            "position_bin": int(np.clip(np.digitize(float(row["normalized_checkpoint_position"]), position_bins) - 1, 0, len(position_bins) - 2)),
            "recoverability_bin": int(np.clip(np.digitize(float(row["parent_recoverability"]), recovery_bins) - 1, 0, len(recovery_bins) - 2)),
        }
        parent_records.append(
            {
                **metadata[key],
                "split_half_orientation_cosine_mean": float(np.mean(cosines)) if cosines else float("nan"),
                "split_half_orientation_cosine_median": float(np.median(cosines)) if cosines else float("nan"),
                "valid_split_half_repetitions": len(cosines),
            }
        )

    neighbor_records: list[dict[str, Any]] = []
    k_neighbors = int(f2["nearest_neighbors"])
    for model in MODEL_ORDER:
        for horizon in HORIZON_ORDER:
            keys = sorted(
                [key for key in metadata if key[0] == model and key[1] == horizon],
                key=lambda key: key[2],
            )
            for anchor in keys:
                anchor_meta = metadata[anchor]
                # Fold restriction ensures both states and orientations share the
                # same transform fitted without either evaluated failed trace.
                candidates = [
                    key
                    for key in keys
                    if key != anchor
                    and metadata[key]["trace_id"] != anchor_meta["trace_id"]
                    and metadata[key]["crossfit_fold"] == anchor_meta["crossfit_fold"]
                ]
                if not candidates:
                    continue
                anchor_state = parent_states[anchor]
                distances = sorted(
                    (
                        float(np.linalg.norm(anchor_state - parent_states[candidate])),
                        candidate,
                    )
                    for candidate in candidates
                )
                nearest = distances[: min(k_neighbors, len(distances))]
                excluded_nearest = {candidate for _, candidate in nearest}
                for rank, (distance, neighbor) in enumerate(distances, start=1):
                    neighbor_meta = metadata[neighbor]
                    left_a, right_a = orientation_halves[anchor]
                    left_b, right_b = orientation_halves[neighbor]
                    similarity_values = [_cosine(left_a, right_b), _cosine(right_a, left_b)]
                    finite_similarity = [value for value in similarity_values if math.isfinite(value)]
                    similarity = float(np.mean(finite_similarity)) if finite_similarity else float("nan")
                    controls = [
                        candidate
                        for candidate in candidates
                        if candidate != neighbor
                        and candidate not in excluded_nearest
                        and metadata[candidate]["domain"] == neighbor_meta["domain"]
                        and metadata[candidate]["position_bin"] == neighbor_meta["position_bin"]
                        and metadata[candidate]["recoverability_bin"] == neighbor_meta["recoverability_bin"]
                    ]
                    control = None
                    if controls:
                        control = min(
                            controls,
                            key=lambda candidate: (
                                _stable_seed(
                                    int(config["experiment"]["seed"]),
                                    "f2-control",
                                    anchor[2],
                                    neighbor[2],
                                    candidate[2],
                                ),
                                candidate[2],
                            ),
                        )
                    control_similarity = float("nan")
                    control_distance = float("nan")
                    control_id = None
                    control_trace_id = None
                    if control is not None:
                        control_id = control[2]
                        control_trace_id = metadata[control]["trace_id"]
                        left_c, right_c = orientation_halves[control]
                        values = [_cosine(left_a, right_c), _cosine(right_a, left_c)]
                        finite = [value for value in values if math.isfinite(value)]
                        control_similarity = float(np.mean(finite)) if finite else float("nan")
                        control_distance = float(
                            np.linalg.norm(anchor_state - parent_states[control])
                        )
                    neighbor_records.append(
                        {
                            "base_model": model,
                            "horizon": int(horizon),
                            "trace_id": anchor_meta["trace_id"],
                            "parent_id": anchor[2],
                            "neighbor_trace_id": neighbor_meta["trace_id"],
                            "neighbor_parent_id": neighbor[2],
                            "neighbor_rank": int(rank),
                            "is_nearest_neighbor": bool(rank <= k_neighbors),
                            "parent_state_distance": float(distance),
                            "orientation_similarity": similarity,
                            "control_parent_id": control_id,
                            "control_trace_id": control_trace_id,
                            "matched_control_similarity": control_similarity,
                            "matched_control_parent_state_distance": control_distance,
                            "excess_local_coherence": similarity - control_similarity,
                            "anchor_domain": anchor_meta["domain"],
                            "domain": anchor_meta["domain"],
                            "neighbor_domain": neighbor_meta["domain"],
                            "within_domain": bool(anchor_meta["domain"] == neighbor_meta["domain"]),
                            "control_match_available": bool(control is not None),
                            "crossfit_fold": int(anchor_meta["crossfit_fold"]),
                        }
                    )
    neighbors = pd.DataFrame(neighbor_records)
    if len(neighbors):
        quantiles = int(f2["distance_quantiles"])
        neighbors["distance_quantile"] = -1
        neighbors["control_distance_quantile"] = -1
        for (model, horizon), indices in neighbors.groupby(["base_model", "horizon"]).groups.items():
            ranks = neighbors.loc[indices, "parent_state_distance"].rank(method="first", pct=True)
            neighbors.loc[indices, "distance_quantile"] = np.minimum(
                np.floor((ranks - EPS) * quantiles).astype(int), quantiles - 1
            )
            finite_control = neighbors.loc[indices, "matched_control_parent_state_distance"].notna()
            control_indices = neighbors.loc[indices].index[finite_control]
            if len(control_indices):
                control_ranks = neighbors.loc[
                    control_indices, "matched_control_parent_state_distance"
                ].rank(method="first", pct=True)
                neighbors.loc[control_indices, "control_distance_quantile"] = np.minimum(
                    np.floor((control_ranks - EPS) * quantiles).astype(int), quantiles - 1
                )
        neighbors["distance_quantile"] = neighbors["distance_quantile"].astype(int)
        neighbors["control_distance_quantile"] = neighbors["control_distance_quantile"].astype(int)
    return pd.DataFrame(parent_records), neighbors


def _run_f3(
    frame: pd.DataFrame,
    directions: Mapping[int, np.ndarray],
    original_states: pd.DataFrame,
    transforms: Mapping[tuple[str, int, int], FoldTransform],
    sibling_means: Mapping[tuple[str, int, str], np.ndarray],
    parent_features: Mapping[tuple[str, int, str], np.ndarray],
    *,
    boundary_root: Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    loo_records: list[dict[str, Any]] = []
    grouping = ["base_model", "trace_id", "parent_id", "checkpoint_index", "domain", "horizon"]
    for keys, part in frame.loc[frame["eligible"] & frame["analysis_branch"]].groupby(grouping, sort=True):
        model, trace, parent_id, checkpoint_index, domain, horizon = keys
        part = part.sort_values("branch_index")
        u = np.stack([directions[int(index)] for index in part.index])
        r = part["r_tilde"].to_numpy(float)
        r_raw = part["r_unsmoothed"].to_numpy(float)
        for position, (_, branch) in enumerate(part.iterrows()):
            mask = np.arange(len(part)) != position
            score = _adherence(u[position], u[mask], r[mask])
            score_raw = _adherence(u[position], u[mask], r_raw[mask])
            loo_records.append(
                {
                    "base_model": str(model),
                    "trace_id": str(trace),
                    "parent_id": str(parent_id),
                    "checkpoint_index": int(checkpoint_index),
                    "domain": str(domain),
                    "horizon": int(horizon),
                    "branch_id": str(branch["branch_id"]),
                    "branch_recoverability": float(branch["r_tilde"]),
                    "branch_unsmoothed_recoverability": float(branch["r_unsmoothed"]),
                    "loo_adherence": score,
                    "loo_adherence_unsmoothed": score_raw,
                }
            )
    loo = pd.DataFrame(loo_records)
    validation_records: list[dict[str, Any]] = []
    bootstrap_replicates = int(config["statistics"]["bootstrap_replicates"])
    for (model, horizon), part in loo.groupby(["base_model", "horizon"], sort=True):
        trace_correlations = pd.DataFrame(
            [
                {
                    "trace_id": str(trace_id),
                    "loo_spearman": _safe_spearman(
                        trace_part["loo_adherence"], trace_part["branch_recoverability"]
                    ),
                }
                for trace_id, trace_part in part.groupby("trace_id", sort=True)
            ]
        ).dropna()
        result = _mean_trace_bootstrap(
            trace_correlations,
            "loo_spearman",
            replicates=bootstrap_replicates,
            seed=_stable_seed(int(config["experiment"]["seed"]), "f3-validation", model, horizon),
        )
        raw_trace_correlations = pd.DataFrame(
            [
                {
                    "trace_id": str(trace_id),
                    "loo_spearman_unsmoothed": _safe_spearman(
                        trace_part["loo_adherence_unsmoothed"],
                        trace_part["branch_unsmoothed_recoverability"],
                    ),
                }
                for trace_id, trace_part in part.groupby("trace_id", sort=True)
            ]
        ).dropna()
        raw_result = _mean_trace_bootstrap(
            raw_trace_correlations,
            "loo_spearman_unsmoothed",
            replicates=bootstrap_replicates,
            seed=_stable_seed(
                int(config["experiment"]["seed"]), "f3-validation-unsmoothed", model, horizon
            ),
        )
        validation_records.append(
            {
                "base_model": str(model),
                "horizon": int(horizon),
                "loo_spearman": result["estimate"],
                "ci_lower": result["ci_lower"],
                "ci_upper": result["ci_upper"],
                "trace_count": result["trace_count"],
                "expected_direction_validated": bool(result["ci_lower"] > 0),
                "unsmoothed_loo_spearman": raw_result["estimate"],
                "unsmoothed_ci_lower": raw_result["ci_lower"],
                "unsmoothed_ci_upper": raw_result["ci_upper"],
                "unsmoothed_expected_direction_validated": bool(raw_result["ci_lower"] > 0),
            }
        )
    validation = pd.DataFrame(validation_records)
    validation_lookup = {
        (str(row.base_model), int(row.horizon)): bool(row.expected_direction_validated)
        for row in validation.itertuples()
    }

    original = original_states.copy()
    if original.empty:
        columns = [
            "base_model", "trace_id", "parent_id", "checkpoint_index", "domain", "horizon",
            "original_path_adherence", "interpretation_eligible", "relative_to_first_visible_error",
            "relative_error_bin", "replay_status",
        ]
        return loo, validation, pd.DataFrame(columns=columns), pd.DataFrame()
    required = {"base_model", "trace_id", "checkpoint_index", "horizon", "raw_hidden", "replay_status"}
    if missing := required - set(original):
        raise KeyError(f"original-path state table missing {sorted(missing)}")
    parent_meta = frame.assign(_has_crossfit_fold=frame["crossfit_fold"].notna()).sort_values(
        ["base_model", "trace_id", "checkpoint_index", "horizon", "_has_crossfit_fold"],
        ascending=[True, True, True, True, False],
    ).groupby(
        ["base_model", "trace_id", "checkpoint_index", "horizon"], sort=True
    ).head(1)
    merge_columns = [
        "base_model", "trace_id", "checkpoint_index", "horizon", "parent_id", "domain",
        "crossfit_fold", "normalized_checkpoint_position", "relative_to_first_visible_error",
        "parent_recoverability", "eligible",
    ]
    original = original.merge(
        parent_meta[merge_columns],
        on=["base_model", "trace_id", "checkpoint_index", "horizon"],
        how="left",
        validate="one_to_one",
    )
    adherence_records: list[dict[str, Any]] = []
    axes_by_model: dict[str, Any] = {}
    for row in original.itertuples():
        model, horizon, parent_id = str(row.base_model), int(row.horizon), str(row.parent_id)
        key = (model, horizon, parent_id)
        record = {
            "base_model": model,
            "trace_id": str(row.trace_id),
            "parent_id": parent_id,
            "checkpoint_index": int(row.checkpoint_index),
            "domain": str(row.domain),
            "horizon": horizon,
            "replay_status": str(row.replay_status),
            "normalized_checkpoint_position": float(row.normalized_checkpoint_position),
            "relative_to_first_visible_error": float(row.relative_to_first_visible_error),
            "relative_error_bin": _relative_error_bin(float(row.relative_to_first_visible_error)),
            "parent_recoverability": float(row.parent_recoverability),
            "original_path_adherence": float("nan"),
            "original_path_adherence_unsmoothed": float("nan"),
            "interpretation_eligible": False,
        }
        if not bool(row.eligible) or str(row.replay_status) != "complete" or row.raw_hidden is None:
            adherence_records.append(record)
            continue
        if model not in axes_by_model:
            axes, canonical_seed, _ = load_frozen_axes(boundary_root, model)
            axes_by_model[model] = axes[canonical_seed]
        axis = axes_by_model[model]
        original_feature = axis.transform_raw(np.asarray(row.raw_hidden, dtype=np.float32)[None, :])[0]
        if hasattr(row, "replayed_parent_raw_hidden") and row.replayed_parent_raw_hidden is not None:
            replayed_parent_feature = axis.transform_raw(
                np.asarray(row.replayed_parent_raw_hidden, dtype=np.float32)[None, :]
            )[0]
        else:
            replayed_parent_feature = parent_features[key]
        centered_delta = original_feature - replayed_parent_feature - sibling_means[key]
        fold = int(row.crossfit_fold)
        transformed = transforms[(model, horizon, fold)].apply(centered_delta[None, :])[0]
        original_direction = _unit(transformed)
        if original_direction is None:
            record["replay_status"] = "zero_transformed_original_displacement"
            adherence_records.append(record)
            continue
        part = frame.loc[
            frame["eligible"]
            & frame["analysis_branch"]
            & (frame["base_model"].astype(str) == model)
            & (frame["parent_id"].astype(str) == parent_id)
            & (frame["horizon"].astype(int) == horizon)
        ]
        u = np.stack([directions[int(index)] for index in part.index])
        r = part["r_tilde"].to_numpy(float)
        r_raw = part["r_unsmoothed"].to_numpy(float)
        record["original_path_adherence"] = _adherence(original_direction, u, r)
        record["original_path_adherence_unsmoothed"] = _adherence(
            original_direction, u, r_raw
        )
        record["interpretation_eligible"] = validation_lookup.get((model, horizon), False)
        adherence_records.append(record)
    adherence = pd.DataFrame(adherence_records)

    alignment_records: list[dict[str, Any]] = []
    interpreted = adherence.loc[
        adherence["interpretation_eligible"] & adherence["original_path_adherence"].notna()
    ]
    for keys, part in interpreted.groupby(
        ["base_model", "horizon", "relative_error_bin"], sort=True
    ):
        model, horizon, error_bin = keys
        base_record = {
                "base_model": str(model),
                "horizon": int(horizon),
                "relative_error_bin": str(error_bin),
                "parent_count": int(len(part)),
                "trace_count": int(part["trace_id"].nunique()),
                "mean_adherence": float(part["original_path_adherence"].mean()),
                "median_adherence": float(part["original_path_adherence"].median()),
                "fraction_negative": float((part["original_path_adherence"] < 0).mean()),
                "unsmoothed_mean_adherence": float(
                    part["original_path_adherence_unsmoothed"].mean()
                ),
                "unsmoothed_median_adherence": float(
                    part["original_path_adherence_unsmoothed"].median()
                ),
                "unsmoothed_fraction_negative": float(
                    (part["original_path_adherence_unsmoothed"] < 0).mean()
                ),
            }
        trace_values = part.groupby("trace_id", sort=True).agg(
            adherence=("original_path_adherence", "mean"),
            adherence_unsmoothed=("original_path_adherence_unsmoothed", "mean"),
        )
        rng = np.random.default_rng(
            _stable_seed(
                int(config["experiment"]["seed"]), "f3-error-bin", model, horizon, error_bin
            )
        )
        values = trace_values["adherence"].to_numpy(float)
        raw_values = trace_values["adherence_unsmoothed"].to_numpy(float)
        if len(values):
            indices = rng.integers(
                0,
                len(values),
                size=(int(config["statistics"]["bootstrap_replicates"]), len(values)),
            )
            samples = values[indices]
            raw_samples = raw_values[indices]
            for prefix, sample in (("mean", samples.mean(axis=1)), ("median", np.median(samples, axis=1)), ("fraction_negative", (samples < 0).mean(axis=1))):
                base_record[f"{prefix}_ci_lower"] = float(np.quantile(sample, 0.025))
                base_record[f"{prefix}_ci_upper"] = float(np.quantile(sample, 0.975))
            for prefix, sample in (("unsmoothed_mean", raw_samples.mean(axis=1)), ("unsmoothed_median", np.median(raw_samples, axis=1)), ("unsmoothed_fraction_negative", (raw_samples < 0).mean(axis=1))):
                base_record[f"{prefix}_ci_lower"] = float(np.quantile(sample, 0.025))
                base_record[f"{prefix}_ci_upper"] = float(np.quantile(sample, 0.975))
        alignment_records.append(base_record)
    alignment_columns = [
        "base_model",
        "horizon",
        "relative_error_bin",
        "parent_count",
        "trace_count",
        "mean_adherence",
        "median_adherence",
        "fraction_negative",
        "unsmoothed_mean_adherence",
        "unsmoothed_median_adherence",
        "unsmoothed_fraction_negative",
        "mean_ci_lower",
        "mean_ci_upper",
        "median_ci_lower",
        "median_ci_upper",
        "fraction_negative_ci_lower",
        "fraction_negative_ci_upper",
        "unsmoothed_mean_ci_lower",
        "unsmoothed_mean_ci_upper",
        "unsmoothed_median_ci_lower",
        "unsmoothed_median_ci_upper",
        "unsmoothed_fraction_negative_ci_lower",
        "unsmoothed_fraction_negative_ci_upper",
    ]
    if alignment_records:
        alignment = pd.DataFrame.from_records(alignment_records, columns=alignment_columns)
    else:
        alignment = pd.DataFrame(
            {
                column: pd.Series(
                    dtype=(
                        "string"
                        if column in {"base_model", "relative_error_bin"}
                        else "int64"
                        if column in {"horizon", "parent_count", "trace_count"}
                        else "float64"
                    )
                )
                for column in alignment_columns
            }
        )
    return loo, validation, adherence, alignment


def _continuous_concordance(frame: pd.DataFrame, score: str, target: str) -> float:
    concordant = 0.0
    comparable = 0
    for _, part in frame.groupby(["base_model", "parent_id", "horizon"], sort=True):
        values = part[[score, target]].dropna().to_numpy(float)
        for left in range(len(values)):
            for right in range(left + 1, len(values)):
                target_difference = values[left, 1] - values[right, 1]
                if abs(target_difference) <= EPS:
                    continue
                score_difference = values[left, 0] - values[right, 0]
                comparable += 1
                product = target_difference * score_difference
                concordant += 1.0 if product > 0 else (0.5 if abs(product) <= EPS else 0.0)
    return float(concordant / comparable) if comparable else float("nan")


def _grouped_ridge_predictions(
    frame: pd.DataFrame,
    *,
    target: str,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    folds: int,
    alpha: float,
    seed: int,
) -> np.ndarray:
    data = frame.reset_index(drop=True)
    categorical = pd.get_dummies(
        data[list(categorical_features)].astype(str),
        columns=list(categorical_features),
        dtype=float,
    )
    numeric = data[list(numeric_features)].astype(float).reset_index(drop=True)
    missing = numeric.isna().astype(float).add_suffix("__missing")
    design = pd.concat(
        [numeric, missing, categorical.reset_index(drop=True)], axis=1
    ).to_numpy(float)
    target_values = data[target].to_numpy(float)
    trace_assignment = _fold_assignment(data["trace_id"].astype(str), folds, seed)
    assignments = data["trace_id"].astype(str).map(trace_assignment).to_numpy(int)
    predictions = np.full(len(data), np.nan, dtype=float)
    for fold in sorted(set(assignments)):
        train = assignments != fold
        test = assignments == fold
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(design[train], axis=0)
        mean[~np.isfinite(mean)] = 0.0
        train_design = np.where(np.isnan(design[train]), mean, design[train])
        test_design = np.where(np.isnan(design[test]), mean, design[test])
        scale = train_design.std(axis=0)
        scale[scale <= EPS] = 1.0
        model = Ridge(alpha=float(alpha))
        model.fit((train_design - mean) / scale, target_values[train])
        predictions[test] = model.predict((test_design - mean) / scale)
    if not np.isfinite(predictions).all():
        raise RuntimeError("grouped ridge cross-validation produced missing predictions")
    return predictions


def _run_f4(
    frame: pd.DataFrame,
    inventory: pd.DataFrame,
    directions: Mapping[int, np.ndarray],
    *,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    f4 = config["f4"]
    split_records: list[dict[str, Any]] = []
    breadth_records: list[dict[str, Any]] = []
    exact_keys = set(
        map(
            tuple,
            inventory.loc[
                inventory["f4_exact_twelve_eligible"], ["base_model", "parent_id", "horizon"]
            ].to_numpy(),
        )
    )
    grouping = ["base_model", "trace_id", "parent_id", "checkpoint_index", "domain", "horizon"]
    support_size = int(f4["support_branches"])
    query_size = int(f4["query_branches"])
    splits = int(f4["deterministic_splits_per_parent"])
    for keys, part in frame.loc[frame["eligible"] & frame["analysis_branch"]].groupby(grouping, sort=True):
        model, trace, parent_id, checkpoint_index, domain, horizon = keys
        if (str(model), str(parent_id), int(horizon)) not in exact_keys:
            continue
        part = part.sort_values("branch_index").reset_index().rename(columns={"index": "source_row_index"})
        if len(part) != support_size + query_size:
            continue
        u = np.stack([directions[int(index)] for index in part["source_row_index"]])
        r = part["r_tilde"].to_numpy(float)
        combinations_all = list(combinations(range(len(part)), support_size))
        rng = np.random.default_rng(
            _stable_seed(int(config["experiment"]["seed"]), "f4", model, parent_id, horizon)
        )
        selected = rng.permutation(len(combinations_all))[:splits]
        for split_index, combination_index in enumerate(selected):
            support = np.asarray(combinations_all[int(combination_index)], dtype=int)
            support_set = set(map(int, support))
            query = np.asarray([index for index in range(len(part)) if index not in support_set], dtype=int)
            if len(query) != query_size:
                raise AssertionError("support/query complement differs")
            support_r = r[support]
            support_r_raw = part["r_unsmoothed"].to_numpy(float)[support]
            support_u = u[support]
            contrast = support_r - support_r.mean()
            contrast_raw = support_r_raw - support_r_raw.mean()
            breadth = _funnel_metrics(support_u, support_r)
            breadth_raw = _funnel_metrics(support_u, support_r_raw)
            query_values = r[query]
            query_values_raw = part["r_unsmoothed"].to_numpy(float)[query]
            breadth_records.append(
                {
                    "base_model": str(model),
                    "trace_id": str(trace),
                    "parent_id": str(parent_id),
                    "checkpoint_index": int(checkpoint_index),
                    "domain": str(domain),
                    "horizon": int(horizon),
                    "split_index": int(split_index),
                    "support_directional_concentration": float(breadth["directional_concentration"]),
                    "support_effective_rank": float(breadth["effective_rank"]),
                    "support_angular_dispersion": float(breadth["angular_dispersion"]),
                    "support_leading_spectral_mass": float(breadth["leading_spectral_mass"]),
                    "support_unsmoothed_directional_concentration": float(
                        breadth_raw["directional_concentration"]
                    ),
                    "support_unsmoothed_effective_rank": float(breadth_raw["effective_rank"]),
                    "support_unsmoothed_angular_dispersion": float(
                        breadth_raw["angular_dispersion"]
                    ),
                    "support_unsmoothed_leading_spectral_mass": float(
                        breadth_raw["leading_spectral_mass"]
                    ),
                    "query_lower_quartile_recoverability": float(np.quantile(query_values, 0.25)),
                    "query_fraction_at_least_half": float(np.mean(query_values >= 0.5)),
                    "query_recoverability_variance": float(np.var(query_values, ddof=0)),
                    "query_unsmoothed_lower_quartile_recoverability": float(
                        np.quantile(query_values_raw, 0.25)
                    ),
                    "query_unsmoothed_fraction_at_least_half": float(
                        np.mean(query_values_raw >= 0.5)
                    ),
                    "query_unsmoothed_recoverability_variance": float(
                        np.var(query_values_raw, ddof=0)
                    ),
                    "parent_recoverability": float(part["parent_recoverability"].iloc[0]),
                    "normalized_checkpoint_position": float(part["normalized_checkpoint_position"].iloc[0]),
                }
            )
            cosine = u[query] @ support_u.T
            scores = cosine @ contrast
            scores_raw = cosine @ contrast_raw
            for query_offset, branch_position in enumerate(query):
                branch = part.iloc[int(branch_position)]
                split_records.append(
                    {
                        "base_model": str(model),
                        "trace_id": str(trace),
                        "parent_id": str(parent_id),
                        "checkpoint_index": int(checkpoint_index),
                        "domain": str(domain),
                        "horizon": int(horizon),
                        "split_index": int(split_index),
                        "query_branch_id": str(branch["branch_id"]),
                        "query_branch_index": int(branch["branch_index"]),
                        "query_recoverability": float(branch["r_tilde"]),
                        "query_unsmoothed_recoverability": float(branch["r_unsmoothed"]),
                        "support_funnel_score": float(scores[query_offset]),
                        "support_funnel_score_unsmoothed": float(scores_raw[query_offset]),
                        "parent_recoverability": float(branch["parent_recoverability"]),
                        "normalized_checkpoint_position": float(branch["normalized_checkpoint_position"]),
                    }
                )
    predictions = pd.DataFrame(split_records)
    breadth = pd.DataFrame(breadth_records)
    aggregate_columns = [
        "base_model", "trace_id", "parent_id", "checkpoint_index", "domain", "horizon",
        "query_branch_id", "query_branch_index", "query_recoverability",
        "query_unsmoothed_recoverability", "parent_recoverability", "normalized_checkpoint_position",
    ]
    aggregated = (
        predictions.groupby(aggregate_columns, sort=True, as_index=False)
        .agg(
            support_funnel_score=("support_funnel_score", "mean"),
            support_funnel_score_sd=("support_funnel_score", "std"),
            support_funnel_score_unsmoothed=("support_funnel_score_unsmoothed", "mean"),
            support_funnel_score_unsmoothed_sd=("support_funnel_score_unsmoothed", "std"),
            query_split_count=("split_index", "size"),
        )
        if len(predictions)
        else pd.DataFrame()
    )
    metrics_records: list[dict[str, Any]] = []
    if len(aggregated):
        base_numeric = ["parent_recoverability", "normalized_checkpoint_position"]
        categorical = ["base_model", "domain", "horizon"]
        baseline_prediction = _grouped_ridge_predictions(
            aggregated,
            target="query_recoverability",
            numeric_features=base_numeric,
            categorical_features=categorical,
            folds=int(f4["grouped_cv_folds"]),
            alpha=float(f4["ridge_alpha"]),
            seed=_stable_seed(int(config["experiment"]["seed"]), "f4-cv", "baseline"),
        )
        funnel_prediction = _grouped_ridge_predictions(
            aggregated,
            target="query_recoverability",
            numeric_features=[*base_numeric, "support_funnel_score"],
            categorical_features=categorical,
            folds=int(f4["grouped_cv_folds"]),
            alpha=float(f4["ridge_alpha"]),
            seed=_stable_seed(int(config["experiment"]["seed"]), "f4-cv", "baseline"),
        )
        aggregated["baseline_cv_prediction"] = baseline_prediction
        aggregated["funnel_cv_prediction"] = funnel_prediction
        aggregated["baseline_squared_error"] = np.square(
            aggregated["query_recoverability"] - baseline_prediction
        )
        aggregated["funnel_squared_error"] = np.square(
            aggregated["query_recoverability"] - funnel_prediction
        )
        aggregated["squared_error_improvement"] = (
            aggregated["baseline_squared_error"] - aggregated["funnel_squared_error"]
        )
        baseline_prediction_raw = _grouped_ridge_predictions(
            aggregated,
            target="query_unsmoothed_recoverability",
            numeric_features=base_numeric,
            categorical_features=categorical,
            folds=int(f4["grouped_cv_folds"]),
            alpha=float(f4["ridge_alpha"]),
            seed=_stable_seed(int(config["experiment"]["seed"]), "f4-cv-unsmoothed", "baseline"),
        )
        funnel_prediction_raw = _grouped_ridge_predictions(
            aggregated,
            target="query_unsmoothed_recoverability",
            numeric_features=[*base_numeric, "support_funnel_score_unsmoothed"],
            categorical_features=categorical,
            folds=int(f4["grouped_cv_folds"]),
            alpha=float(f4["ridge_alpha"]),
            seed=_stable_seed(int(config["experiment"]["seed"]), "f4-cv-unsmoothed", "baseline"),
        )
        aggregated["unsmoothed_baseline_cv_prediction"] = baseline_prediction_raw
        aggregated["unsmoothed_funnel_cv_prediction"] = funnel_prediction_raw
        aggregated["unsmoothed_baseline_squared_error"] = np.square(
            aggregated["query_unsmoothed_recoverability"] - baseline_prediction_raw
        )
        aggregated["unsmoothed_funnel_squared_error"] = np.square(
            aggregated["query_unsmoothed_recoverability"] - funnel_prediction_raw
        )
        aggregated["unsmoothed_squared_error_improvement"] = (
            aggregated["unsmoothed_baseline_squared_error"]
            - aggregated["unsmoothed_funnel_squared_error"]
        )
        for keys, part in aggregated.groupby(["base_model", "horizon"], sort=True):
            model, horizon = keys
            parent_correlations = [
                _safe_spearman(parent["support_funnel_score"], parent["query_recoverability"])
                for _, parent in part.groupby("parent_id", sort=True)
            ]
            parent_correlations = [value for value in parent_correlations if math.isfinite(value)]
            parent_stats = pd.DataFrame(
                [
                    {
                        "trace_id": str(parent_part["trace_id"].iloc[0]),
                        "rank_association": _safe_spearman(
                            parent_part["support_funnel_score"],
                            parent_part["query_recoverability"],
                        ),
                        "discrimination": _continuous_concordance(
                            parent_part,
                            "support_funnel_score",
                            "query_recoverability",
                        ),
                    }
                    for _, parent_part in part.groupby("parent_id", sort=True)
                ]
            ).dropna()
            rank_ci = _mean_trace_bootstrap(
                parent_stats,
                "rank_association",
                replicates=int(config["statistics"]["bootstrap_replicates"]),
                seed=_stable_seed(int(config["experiment"]["seed"]), "f4-rank-ci", model, horizon),
            )
            discrimination_ci = _mean_trace_bootstrap(
                parent_stats,
                "discrimination",
                replicates=int(config["statistics"]["bootstrap_replicates"]),
                seed=_stable_seed(
                    int(config["experiment"]["seed"]), "f4-discrimination-ci", model, horizon
                ),
            )
            metrics_records.append(
                {
                    "analysis": "query_prediction",
                    "base_model": str(model),
                    "horizon": int(horizon),
                    "query_branch_count": int(len(part)),
                    "trace_count": int(part["trace_id"].nunique()),
                    "within_parent_rank_association": float(np.mean(parent_correlations)) if parent_correlations else float("nan"),
                    "within_parent_rank_association_ci_lower": rank_ci["ci_lower"],
                    "within_parent_rank_association_ci_upper": rank_ci["ci_upper"],
                    "continuous_query_discrimination": discrimination_ci["estimate"],
                    "continuous_query_discrimination_ci_lower": discrimination_ci["ci_lower"],
                    "continuous_query_discrimination_ci_upper": discrimination_ci["ci_upper"],
                    "baseline_mse": float(part["baseline_squared_error"].mean()),
                    "funnel_mse": float(part["funnel_squared_error"].mean()),
                    "squared_error_improvement": float(part["squared_error_improvement"].mean()),
                    "unsmoothed_baseline_mse": float(
                        part["unsmoothed_baseline_squared_error"].mean()
                    ),
                    "unsmoothed_funnel_mse": float(
                        part["unsmoothed_funnel_squared_error"].mean()
                    ),
                    "unsmoothed_squared_error_improvement": float(
                        part["unsmoothed_squared_error_improvement"].mean()
                    ),
                }
            )

    if len(breadth):
        breadth_features = [
            "support_directional_concentration",
            "support_effective_rank",
            "support_angular_dispersion",
            "support_leading_spectral_mass",
        ]
        targets = [
            "query_lower_quartile_recoverability",
            "query_fraction_at_least_half",
            "query_recoverability_variance",
            "query_unsmoothed_lower_quartile_recoverability",
            "query_unsmoothed_fraction_at_least_half",
            "query_unsmoothed_recoverability_variance",
        ]
        categorical = ["base_model", "domain", "horizon"]
        base_numeric = ["parent_recoverability", "normalized_checkpoint_position"]
        for target in targets:
            baseline = _grouped_ridge_predictions(
                breadth,
                target=target,
                numeric_features=base_numeric,
                categorical_features=categorical,
                folds=int(f4["grouped_cv_folds"]),
                alpha=float(f4["ridge_alpha"]),
                seed=_stable_seed(int(config["experiment"]["seed"]), "f4-breadth", target),
            )
            target_breadth_features = (
                [f"support_unsmoothed_{column.removeprefix('support_')}" for column in breadth_features]
                if target.startswith("query_unsmoothed_")
                else breadth_features
            )
            geometry = _grouped_ridge_predictions(
                breadth,
                target=target,
                numeric_features=[*base_numeric, *target_breadth_features],
                categorical_features=categorical,
                folds=int(f4["grouped_cv_folds"]),
                alpha=float(f4["ridge_alpha"]),
                seed=_stable_seed(int(config["experiment"]["seed"]), "f4-breadth", target),
            )
            breadth[f"{target}_baseline_prediction"] = baseline
            breadth[f"{target}_geometry_prediction"] = geometry
            breadth[f"{target}_squared_error_improvement"] = np.square(
                breadth[target] - baseline
            ) - np.square(breadth[target] - geometry)
            for (model, horizon), part in breadth.groupby(["base_model", "horizon"], sort=True):
                metrics_records.append(
                    {
                        "analysis": f"breadth_prediction:{target}",
                        "base_model": str(model),
                        "horizon": int(horizon),
                        "query_branch_count": int(len(part)),
                        "trace_count": int(part["trace_id"].nunique()),
                        "within_parent_rank_association": float("nan"),
                        "continuous_query_discrimination": float("nan"),
                        "baseline_mse": float(np.square(part[target] - part[f"{target}_baseline_prediction"]).mean()),
                        "funnel_mse": float(np.square(part[target] - part[f"{target}_geometry_prediction"]).mean()),
                        "squared_error_improvement": float(part[f"{target}_squared_error_improvement"].mean()),
                    }
                )
    return predictions, aggregated, breadth, pd.DataFrame(metrics_records)


def _mean_trace_bootstrap(
    frame: pd.DataFrame,
    column: str,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    values = (
        frame[["trace_id", column]]
        .dropna()
        .groupby("trace_id", sort=True)[column]
        .mean()
        .to_numpy(float)
    )
    if not len(values):
        return {
            "estimate": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"),
            "trace_count": 0, "bootstrap_values": np.array([], dtype=float),
        }
    rng = np.random.default_rng(int(seed))
    samples = values[rng.integers(0, len(values), size=(int(replicates), len(values)))].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci_lower": float(np.quantile(samples, 0.025)),
        "ci_upper": float(np.quantile(samples, 0.975)),
        "trace_count": int(len(values)),
        "bootstrap_values": samples,
    }


def _macro_trace_bootstrap(
    frame: pd.DataFrame,
    column: str,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    by_model: dict[str, np.ndarray] = {}
    for model, part in frame.groupby("base_model", sort=True):
        values = (
            part[["trace_id", column]]
            .dropna()
            .groupby("trace_id", sort=True)[column]
            .mean()
            .to_numpy(float)
        )
        if len(values):
            by_model[str(model)] = values
    if not by_model:
        return {
            "estimate": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"),
            "model_count": 0, "trace_count": 0, "bootstrap_values": np.array([], dtype=float),
        }
    rng = np.random.default_rng(int(seed))
    samples = np.zeros(int(replicates), dtype=float)
    for values in by_model.values():
        indices = rng.integers(0, len(values), size=(int(replicates), len(values)))
        samples += values[indices].mean(axis=1)
    samples /= len(by_model)
    return {
        "estimate": float(np.mean([values.mean() for values in by_model.values()])),
        "ci_lower": float(np.quantile(samples, 0.025)),
        "ci_upper": float(np.quantile(samples, 0.975)),
        "model_count": int(len(by_model)),
        "trace_count": int(sum(len(values) for values in by_model.values())),
        "bootstrap_values": samples,
    }


def _macro_trace_sign_flip_p(
    frame: pd.DataFrame,
    column: str,
    *,
    positive: bool,
    replicates: int,
    seed: int,
) -> float:
    contributions: list[float] = []
    model_parts = list(frame.groupby("base_model", sort=True))
    model_parts = [
        (model, part)
        for model, part in model_parts
        if part[["trace_id", column]].dropna()["trace_id"].nunique() > 0
    ]
    if not model_parts:
        return 1.0
    for _, part in model_parts:
        trace_values = (
            part[["trace_id", column]]
            .dropna()
            .groupby("trace_id", sort=True)[column]
            .mean()
            .to_numpy(float)
        )
        sign = 1.0 if positive else -1.0
        contributions.extend(
            (sign * trace_values / (len(model_parts) * len(trace_values))).tolist()
        )
    weighted = np.asarray(contributions, dtype=float)
    observed = float(weighted.sum())
    if not len(weighted) or not math.isfinite(observed):
        return 1.0
    rng = np.random.default_rng(int(seed))
    exceed = 0
    completed = 0
    chunk_size = 10_000
    while completed < int(replicates):
        count = min(chunk_size, int(replicates) - completed)
        signs = rng.integers(0, 2, size=(count, len(weighted)), dtype=np.int8) * 2 - 1
        null = signs @ weighted
        exceed += int(np.sum(null >= observed - EPS))
        completed += count
    return float((exceed + 1) / (int(replicates) + 1))


def _f1_macro_permutation_p(
    f1: pd.DataFrame,
    permutation_nulls: pd.DataFrame,
) -> float:
    if f1.empty or permutation_nulls.empty:
        return 1.0
    observed = (
        f1.groupby(["base_model", "trace_id"], sort=True)["directional_concentration"]
        .mean()
        .groupby("base_model", sort=True)
        .mean()
        .mean()
    )
    trace_null = (
        permutation_nulls.groupby(
            ["permutation_index", "base_model", "trace_id"], sort=True
        )["null_directional_concentration"]
        .mean()
        .reset_index()
    )
    model_null = (
        trace_null.groupby(["permutation_index", "base_model"], sort=True)[
            "null_directional_concentration"
        ]
        .mean()
        .reset_index()
    )
    macro_null = model_null.groupby("permutation_index", sort=True)[
        "null_directional_concentration"
    ].mean().to_numpy(float)
    return float((np.sum(macro_null >= float(observed) - EPS) + 1) / (len(macro_null) + 1))


def _macro_dyadic_bootstrap(
    frame: pd.DataFrame,
    column: str,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    usable = frame.dropna(subset=[column, "trace_id", "neighbor_trace_id"])
    model_parts = [(str(model), part) for model, part in usable.groupby("base_model", sort=True)]
    if not model_parts:
        return {
            "estimate": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"),
            "model_count": 0, "trace_count": 0, "bootstrap_values": np.array([], dtype=float),
        }
    rng = np.random.default_rng(int(seed))
    samples = np.zeros(int(replicates), dtype=float)
    total_traces = 0
    for _, part in model_parts:
        traces = sorted(
            set(part["trace_id"].astype(str))
            | set(part["neighbor_trace_id"].astype(str))
            | set(part.get("control_trace_id", pd.Series(dtype=str)).dropna().astype(str))
        )
        total_traces += len(traces)
        index = {trace: position for position, trace in enumerate(traces)}
        anchor = part["trace_id"].astype(str).map(index).to_numpy(int)
        neighbor = part["neighbor_trace_id"].astype(str).map(index).to_numpy(int)
        control = np.asarray(
            [index.get(str(value), -1) if pd.notna(value) else -1 for value in part.get("control_trace_id", pd.Series([None] * len(part)))],
            dtype=int,
        )
        values = part[column].to_numpy(float)
        counts = rng.multinomial(
            len(traces), np.full(len(traces), 1.0 / len(traces)), size=int(replicates)
        )
        weights = counts[:, anchor] * counts[:, neighbor]
        has_control = control >= 0
        if has_control.any():
            weights[:, has_control] *= counts[:, control[has_control]]
        denominator = weights.sum(axis=1)
        model_sample = np.divide(
            weights @ values,
            denominator,
            out=np.full(int(replicates), np.nan),
            where=denominator > 0,
        )
        fallback = float(values.mean())
        model_sample[~np.isfinite(model_sample)] = fallback
        samples += model_sample / len(model_parts)
    return {
        "estimate": float(
            np.mean([part[column].mean() for _, part in model_parts])
        ),
        "ci_lower": float(np.quantile(samples, 0.025)),
        "ci_upper": float(np.quantile(samples, 0.975)),
        "model_count": len(model_parts),
        "trace_count": total_traces,
        "bootstrap_values": samples,
    }


def _macro_dyadic_sign_flip_p(
    frame: pd.DataFrame,
    column: str,
    *,
    replicates: int,
    seed: int,
) -> float:
    usable = frame.dropna(subset=[column, "trace_id", "neighbor_trace_id"])
    model_parts = [(str(model), part) for model, part in usable.groupby("base_model", sort=True)]
    if not model_parts:
        return 1.0
    contributions: list[float] = []
    for _, part in model_parts:
        trace_effect: dict[str, float] = {}
        for row in part.itertuples():
            participants = {
                str(row.trace_id),
                str(row.neighbor_trace_id),
            }
            if getattr(row, "control_trace_id", None) is not None and pd.notna(row.control_trace_id):
                participants.add(str(row.control_trace_id))
            share = float(getattr(row, column)) / (len(part) * len(participants) * len(model_parts))
            for trace in participants:
                trace_effect[trace] = trace_effect.get(trace, 0.0) + share
        contributions.extend(trace_effect.values())
    weighted = np.asarray(contributions, dtype=float)
    observed = float(weighted.sum())
    rng = np.random.default_rng(int(seed))
    exceed = 0
    completed = 0
    while completed < int(replicates):
        count = min(10_000, int(replicates) - completed)
        signs = rng.integers(0, 2, size=(count, len(weighted)), dtype=np.int8) * 2 - 1
        exceed += int(np.sum(signs @ weighted >= observed - EPS))
        completed += count
    return float((exceed + 1) / (int(replicates) + 1))


def _add_slice_summaries(
    records: list[dict[str, Any]],
    frame: pd.DataFrame,
    *,
    family: str,
    metrics: Sequence[str],
    config: Mapping[str, Any],
) -> None:
    if frame.empty:
        return
    replicates = int(config["statistics"]["bootstrap_replicates"])
    slice_columns: list[tuple[str, list[str]]] = [
        ("overall", []),
        ("model", ["base_model"]),
        ("horizon", ["horizon"]),
        ("domain", ["domain"]),
        ("model_horizon", ["base_model", "horizon"]),
        ("model_domain", ["base_model", "domain"]),
        ("horizon_domain", ["horizon", "domain"]),
        ("model_horizon_domain", ["base_model", "horizon", "domain"]),
    ]
    for slice_name, columns in slice_columns:
        groups: Iterable[tuple[Any, pd.DataFrame]]
        if columns:
            groups = frame.groupby(columns, sort=True, dropna=False)
        else:
            groups = [((), frame)]
        for keys, part in groups:
            if not isinstance(keys, tuple):
                keys = (keys,)
            labels = dict(zip(columns, keys, strict=True))
            for metric in metrics:
                if metric not in part:
                    continue
                if family == "F2" and "neighbor_trace_id" in part:
                    result = _macro_dyadic_bootstrap(
                        part,
                        metric,
                        replicates=replicates,
                        seed=_stable_seed(
                            int(config["experiment"]["seed"]),
                            "summary-dyadic",
                            family,
                            slice_name,
                            labels,
                            metric,
                        ),
                    )
                else:
                    result = _mean_trace_bootstrap(
                        part,
                        metric,
                        replicates=replicates,
                        seed=_stable_seed(int(config["experiment"]["seed"]), "summary", family, slice_name, labels, metric),
                    )
                records.append(
                    {
                        "family": family,
                        "metric": metric,
                        "slice_type": slice_name,
                        "base_model": labels.get("base_model"),
                        "horizon": labels.get("horizon"),
                        "domain": labels.get("domain"),
                        "estimate": result["estimate"],
                        "ci_lower": result["ci_lower"],
                        "ci_upper": result["ci_upper"],
                        "trace_count": result["trace_count"],
                        "row_count": int(len(part)),
                    }
                )
    for metric in metrics:
        if metric not in frame:
            continue
        result = _macro_trace_bootstrap(
            frame,
            metric,
            replicates=replicates,
            seed=_stable_seed(int(config["experiment"]["seed"]), "summary", family, "equal_model_macro", metric),
        )
        records.append(
            {
                "family": family,
                "metric": metric,
                "slice_type": "equal_model_macro",
                "base_model": None,
                "horizon": None,
                "domain": None,
                "estimate": result["estimate"],
                "ci_lower": result["ci_lower"],
                "ci_upper": result["ci_upper"],
                "trace_count": result["trace_count"],
                "row_count": int(len(frame)),
            }
        )


def _summaries_and_tests(
    f1: pd.DataFrame,
    permutation_nulls: pd.DataFrame,
    reliability: pd.DataFrame,
    neighbors: pd.DataFrame,
    validation: pd.DataFrame,
    adherence: pd.DataFrame,
    query_aggregated: pd.DataFrame,
    breadth: pd.DataFrame,
    *,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    summary_records: list[dict[str, Any]] = []
    _add_slice_summaries(
        summary_records,
        f1,
        family="F1",
        metrics=[
            "directional_concentration", "effective_rank", "angular_dispersion",
            "leading_spectral_mass", "z_directional_concentration", "z_effective_rank",
            "z_angular_dispersion", "z_leading_spectral_mass", "unweighted_effective_rank",
            "unweighted_leading_spectral_mass", "unsmoothed_z_directional_concentration",
        ],
        config=config,
    )
    _add_slice_summaries(
        summary_records,
        reliability,
        family="F2",
        metrics=["split_half_orientation_cosine_mean", "split_half_orientation_cosine_median"],
        config=config,
    )
    matched_neighbors = neighbors.loc[
        neighbors.get("control_match_available", False).astype(bool)
        & neighbors.get("is_nearest_neighbor", False).astype(bool)
    ].copy() if len(neighbors) else neighbors
    _add_slice_summaries(
        summary_records,
        matched_neighbors,
        family="F2",
        metrics=["orientation_similarity", "matched_control_similarity", "excess_local_coherence"],
        config=config,
    )
    if len(neighbors):
        replicates = int(config["statistics"]["bootstrap_replicates"])
        for keys, part in neighbors.groupby(
            ["base_model", "horizon", "distance_quantile", "within_domain"],
            sort=True,
        ):
            model, horizon, distance_quantile, within_domain = keys
            result = _macro_dyadic_bootstrap(
                part,
                "orientation_similarity",
                replicates=replicates,
                seed=_stable_seed(
                    int(config["experiment"]["seed"]),
                    "f2-distance-curve",
                    model,
                    horizon,
                    distance_quantile,
                    within_domain,
                ),
            )
            summary_records.append(
                {
                    "family": "F2",
                    "metric": "orientation_similarity_by_parent_distance",
                    "slice_type": "model_horizon_distance_domain_relation",
                    "base_model": str(model),
                    "horizon": int(horizon),
                    "domain": "within_domain" if within_domain else "cross_domain",
                    "distance_quantile": int(distance_quantile),
                    "estimate": result["estimate"],
                    "ci_lower": result["ci_lower"],
                    "ci_upper": result["ci_upper"],
                    "trace_count": result["trace_count"],
                    "row_count": int(len(part)),
                }
            )
    interpretable_adherence = adherence.loc[
        adherence.get("interpretation_eligible", False).astype(bool)
        & adherence.get("original_path_adherence", pd.Series(dtype=float)).notna()
    ].copy() if len(adherence) else adherence
    _add_slice_summaries(
        summary_records,
        interpretable_adherence,
        family="F3",
        metrics=["original_path_adherence", "original_path_adherence_unsmoothed"],
        config=config,
    )
    f4_parent_records: list[dict[str, Any]] = []
    if len(query_aggregated):
        for keys, part in query_aggregated.groupby(
            ["base_model", "trace_id", "parent_id", "horizon", "domain"], sort=True
        ):
            model, trace_id, parent_id, horizon, domain = keys
            f4_parent_records.append(
                {
                    "base_model": str(model),
                    "trace_id": str(trace_id),
                    "parent_id": str(parent_id),
                    "horizon": int(horizon),
                    "domain": str(domain),
                    "within_parent_rank_association": _safe_spearman(
                        part["support_funnel_score"], part["query_recoverability"]
                    ),
                    "continuous_query_discrimination": _continuous_concordance(
                        part, "support_funnel_score", "query_recoverability"
                    ),
                }
            )
    f4_parent_metrics = pd.DataFrame(f4_parent_records)
    _add_slice_summaries(
        summary_records,
        f4_parent_metrics,
        family="F4",
        metrics=["within_parent_rank_association", "continuous_query_discrimination"],
        config=config,
    )
    _add_slice_summaries(
        summary_records,
        query_aggregated,
        family="F4",
        metrics=[
            "support_funnel_score", "squared_error_improvement", "baseline_squared_error",
            "funnel_squared_error",
            "unsmoothed_squared_error_improvement",
            "unsmoothed_baseline_squared_error",
            "unsmoothed_funnel_squared_error",
        ],
        config=config,
    )
    breadth_improvement = [
        column for column in breadth if column.endswith("_squared_error_improvement")
    ] if len(breadth) else []
    _add_slice_summaries(
        summary_records,
        breadth,
        family="F4_breadth",
        metrics=breadth_improvement,
        config=config,
    )
    summaries = pd.DataFrame(summary_records)
    bootstraps = summaries.copy()
    replicates = int(config["statistics"]["bootstrap_replicates"])

    primary: dict[str, Any] = {}
    f1_test = _macro_trace_bootstrap(
        f1,
        "null_difference_directional_concentration",
        replicates=replicates,
        seed=_stable_seed(int(config["experiment"]["seed"]), "primary", "F1"),
    )
    primary["F1"] = {
        "estimand": "equal-model macro mean parent concentration minus within-parent permutation-null mean",
        **{key: value for key, value in f1_test.items() if key != "bootstrap_values"},
        "expected_direction": "positive",
        "raw_p": _f1_macro_permutation_p(f1, permutation_nulls),
        "null_test": "within-parent permutation aggregated by failed trace then equal-model macro",
    }

    reliable_slices: set[tuple[str, int]] = set()
    for (model, horizon), part in reliability.groupby(["base_model", "horizon"], sort=True):
        result = _mean_trace_bootstrap(
            part,
            "split_half_orientation_cosine_mean",
            replicates=replicates,
            seed=_stable_seed(int(config["experiment"]["seed"]), "f2-reliability", model, horizon),
        )
        if result["ci_lower"] > float(config["f2"]["reliability_claim_minimum"]):
            reliable_slices.add((str(model), int(horizon)))
    coherent = matched_neighbors.loc[
        [
            (str(row.base_model), int(row.horizon)) in reliable_slices
            for row in matched_neighbors.itertuples()
        ]
    ] if len(matched_neighbors) else matched_neighbors
    f2_test = _macro_dyadic_bootstrap(
        coherent,
        "excess_local_coherence",
        replicates=replicates,
        seed=_stable_seed(int(config["experiment"]["seed"]), "primary", "F2"),
    )
    primary["F2"] = {
        "estimand": "equal-model macro matched-control excess orientation cosine in reliability-qualified slices",
        **{key: value for key, value in f2_test.items() if key != "bootstrap_values"},
        "expected_direction": "positive",
        "reliability_qualified_model_horizon_slices": sorted([list(value) for value in reliable_slices]),
        "raw_p": _macro_dyadic_sign_flip_p(
            coherent,
            "excess_local_coherence",
            replicates=int(config["statistics"]["sign_flip_replicates"]),
            seed=_stable_seed(int(config["experiment"]["seed"]), "sign-flip", "F2"),
        ),
        "null_test": "trace-level sign flip with equal model weighting",
    }

    pre_error = interpretable_adherence.loc[
        interpretable_adherence["relative_error_bin"].isin(
            ["two_or_more_before", "immediately_before"]
        )
    ] if len(interpretable_adherence) else interpretable_adherence
    f3_test = _macro_trace_bootstrap(
        pre_error,
        "original_path_adherence",
        replicates=replicates,
        seed=_stable_seed(int(config["experiment"]["seed"]), "primary", "F3"),
    )
    primary["F3"] = {
        "estimand": "equal-model macro mean validated original-path adherence before first visible error",
        **{key: value for key, value in f3_test.items() if key != "bootstrap_values"},
        "estimability_status": (
            "estimable" if len(pre_error) else "not_estimable_no_validated_model_horizon_slice"
        ),
        "expected_direction": "negative",
        "validated_model_horizon_slices": validation.loc[
            validation["expected_direction_validated"], ["base_model", "horizon"]
        ].to_dict("records"),
        "raw_p": _macro_trace_sign_flip_p(
            pre_error,
            "original_path_adherence",
            positive=False,
            replicates=int(config["statistics"]["sign_flip_replicates"]),
            seed=_stable_seed(int(config["experiment"]["seed"]), "sign-flip", "F3"),
        ),
        "null_test": "trace-level sign flip with equal model weighting",
    }

    f4_test = _macro_trace_bootstrap(
        query_aggregated,
        "squared_error_improvement",
        replicates=replicates,
        seed=_stable_seed(int(config["experiment"]["seed"]), "primary", "F4"),
    )
    primary["F4"] = {
        "estimand": "equal-model macro grouped-CV squared-error improvement from support-derived funnel score",
        **{key: value for key, value in f4_test.items() if key != "bootstrap_values"},
        "expected_direction": "positive",
        "raw_p": _macro_trace_sign_flip_p(
            query_aggregated,
            "squared_error_improvement",
            positive=True,
            replicates=int(config["statistics"]["sign_flip_replicates"]),
            seed=_stable_seed(int(config["experiment"]["seed"]), "sign-flip", "F4"),
        ),
        "null_test": "trace-level sign flip with equal model weighting",
    }
    holm = _holm({family: float(result["raw_p"]) for family, result in primary.items()})
    for family in primary:
        primary[family].update(holm[family])
        primary[family]["p_value_status"] = (
            "holm_bookkeeping_sentinel_not_estimable"
            if family == "F3" and not len(pre_error)
            else "estimated"
        )
    return summaries, bootstraps, {
        "status": "post_hoc_exploratory",
        "confirmatory_language_permitted": False,
        "familywise_correction": "Holm across F1-F4",
        "primary_tests": primary,
    }


def _save_figure(fig: plt.Figure, root: Path, stem: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    fig.savefig(root / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(root / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _make_figures(
    f1: pd.DataFrame,
    neighbors: pd.DataFrame,
    adherence: pd.DataFrame,
    query: pd.DataFrame,
    *,
    root: Path,
) -> None:
    colors = dict(zip(MODEL_ORDER, ["#0072B2", "#56B4E9", "#D55E00", "#CC79A7"], strict=True))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, metric, label in [
        (axes[0], "directional_concentration", "Directional concentration"),
        (axes[1], "effective_rank", "Effective rank"),
    ]:
        summary = f1.groupby(["base_model", "horizon"], sort=True).agg(
            observed=(metric, "mean"), null=(f"null_mean_{metric}", "mean")
        ).reset_index()
        x = np.arange(len(summary))
        for index, row in summary.iterrows():
            axis.plot([index, index], [row["null"], row["observed"]], color="#999999", linewidth=1)
            axis.scatter(index, row["null"], facecolors="white", edgecolors=colors[row["base_model"]], marker="o", s=45)
            axis.scatter(index, row["observed"], color=colors[row["base_model"]], marker="o", s=45)
        axis.set_xticks(x, [f"{row.base_model.replace('family_', '')}\n{int(row.horizon)}" for row in summary.itertuples()], rotation=45, ha="right")
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_title("Observed (filled) versus permutation null (open)")
    axes[1].set_title("Observed (filled) versus permutation null (open)")
    fig.tight_layout()
    _save_figure(fig, root, "figure1_concentration_rank_vs_null")

    fig, axis = plt.subplots(figsize=(7, 4.8))
    if len(neighbors):
        curve = neighbors.groupby(["base_model", "distance_quantile"], sort=True).agg(
            similarity=("orientation_similarity", "mean"),
        ).reset_index()
        controls = (
            neighbors.loc[neighbors["control_distance_quantile"] >= 0]
            .groupby(["base_model", "control_distance_quantile"], sort=True)
            .agg(control=("matched_control_similarity", "mean"))
            .reset_index()
        )
        for model, part in curve.groupby("base_model", sort=True):
            axis.plot(part["distance_quantile"] + 1, part["similarity"], marker="o", color=colors[model], label=model)
            control_part = controls.loc[controls["base_model"] == model]
            axis.plot(control_part["control_distance_quantile"] + 1, control_part["control"], linestyle="--", color=colors[model], alpha=0.55)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Parent-state distance quantile (near to far)")
    axis.set_ylabel("Cross-half orientation cosine")
    axis.set_title("Local coherence (solid) and matched controls (dashed)")
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.25)
    _save_figure(fig, root, "figure2_coherence_vs_parent_distance")

    fig, axis = plt.subplots(figsize=(8, 4.8))
    interpreted = adherence.loc[
        adherence.get("interpretation_eligible", False).astype(bool)
        & adherence.get("original_path_adherence", pd.Series(dtype=float)).notna()
    ] if len(adherence) else adherence
    order = ["two_or_more_before", "immediately_before", "at_error", "after_error"]
    if len(interpreted):
        summary = interpreted.groupby(["base_model", "relative_error_bin"], sort=True)["original_path_adherence"].mean().reset_index()
        for model, part in summary.groupby("base_model", sort=True):
            lookup = dict(zip(part["relative_error_bin"], part["original_path_adherence"], strict=True))
            axis.plot(range(len(order)), [lookup.get(value, np.nan) for value in order], marker="o", color=colors[model], label=model)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(range(len(order)), ["≥2 before", "immediately\nbefore", "at error", "after"])
    axis.set_ylabel("Original-path adherence")
    axis.set_title("Failed trajectory relative to first visible error")
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(frameon=False, fontsize=8)
    else:
        axis.text(
            0.5,
            0.5,
            "No model–horizon slice passed\nleave-one-branch-out validation",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.grid(axis="y", alpha=0.25)
    _save_figure(fig, root, "figure3_failed_path_adherence")

    fig, axis = plt.subplots(figsize=(7, 4.8))
    if len(query):
        ranked = query.copy()
        ranked["score_bin"] = pd.qcut(ranked["support_funnel_score"].rank(method="first"), 10, labels=False, duplicates="drop")
        curve = ranked.groupby(["base_model", "score_bin"], sort=True).agg(
            score=("support_funnel_score", "mean"), recoverability=("query_recoverability", "mean")
        ).reset_index()
        for model, part in curve.groupby("base_model", sort=True):
            axis.plot(part["score"], part["recoverability"], marker="o", color=colors[model], label=model)
    axis.set_xlabel("Support-derived funnel score (within-model deciles)")
    axis.set_ylabel("Held-out smoothed recoverability")
    axis.set_title("Held-out branch recoverability")
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.25)
    _save_figure(fig, root, "figure4_heldout_recoverability_vs_funnel_score")


def _format_effect(result: Mapping[str, Any], *, digits: int = 4) -> str:
    try:
        estimate = float(result.get("estimate", float("nan")))
        lower = float(result.get("ci_lower", float("nan")))
        upper = float(result.get("ci_upper", float("nan")))
    except (TypeError, ValueError):
        return "unavailable"
    if not all(math.isfinite(value) for value in (estimate, lower, upper)):
        return "unavailable"
    return (
        f"{estimate:.{digits}f} "
        f"(95% CI {lower:.{digits}f} to {upper:.{digits}f})"
    )


def _write_report(
    path: Path,
    *,
    inventory: pd.DataFrame,
    inventory_counts: pd.DataFrame,
    f1: pd.DataFrame,
    reliability: pd.DataFrame,
    neighbors: pd.DataFrame,
    validation: pd.DataFrame,
    adherence: pd.DataFrame,
    query: pd.DataFrame,
    f4_metrics: pd.DataFrame,
    summaries: pd.DataFrame,
    tests: Mapping[str, Any],
    integrity: Mapping[str, Any],
) -> None:
    primary = tests["primary_tests"]
    eligible = int(inventory["success_conditioned_geometry_eligible"].sum())
    total = int(len(inventory))
    categories = f1["descriptive_geometry"].value_counts(normalize=True).to_dict() if len(f1) else {}
    reliability_effect = float(reliability["split_half_orientation_cosine_mean"].mean()) if len(reliability) else float("nan")
    matched = neighbors.loc[
        neighbors["control_match_available"] & neighbors["is_nearest_neighbor"]
    ] if len(neighbors) else neighbors
    coherence_effect = float(matched["excess_local_coherence"].mean()) if len(matched) else float("nan")
    reliable_slices = {
        (str(model), int(horizon))
        for model, horizon in primary["F2"].get(
            "reliability_qualified_model_horizon_slices", []
        )
    }
    matched_reliable = matched.loc[
        [
            (str(model), int(horizon)) in reliable_slices
            for model, horizon in zip(matched["base_model"], matched["horizon"])
        ]
    ] if len(matched) else matched
    relation_effects = (
        matched_reliable.groupby("within_domain", sort=True)["excess_local_coherence"]
        .agg(["mean", "size"])
        .to_dict("index")
        if len(matched_reliable)
        else {}
    )
    validated = validation.loc[validation["expected_direction_validated"]] if len(validation) else validation
    interpreted = adherence.loc[
        adherence["interpretation_eligible"] & adherence["original_path_adherence"].notna()
    ] if len(adherence) else adherence
    pre_error = interpreted.loc[
        interpreted["relative_error_bin"].isin(["two_or_more_before", "immediately_before"])
    ] if len(interpreted) else interpreted
    pre_error_description = (
        f"n={len(pre_error)}, mean={pre_error['original_path_adherence'].mean():.4f}, "
        f"median={pre_error['original_path_adherence'].median():.4f}, "
        f"fraction negative={(pre_error['original_path_adherence'] < 0).mean():.4f}"
        if len(pre_error)
        else "n=0; mean, median, and fraction negative unavailable"
    )
    f4_main = f4_metrics.loc[f4_metrics["analysis"] == "query_prediction"] if len(f4_metrics) else f4_metrics
    f4_positive_slices = int((f4_main["squared_error_improvement"] > 0).sum()) if len(f4_main) else 0
    def macro_summary(family: str, metric: str) -> Mapping[str, Any]:
        selected = summaries.loc[
            (summaries["family"] == family)
            & (summaries["metric"] == metric)
            & (summaries["slice_type"] == "equal_model_macro")
        ]
        return selected.iloc[0].to_dict() if len(selected) else {}

    f4_rank = macro_summary("F4", "within_parent_rank_association")
    f4_discrimination = macro_summary("F4", "continuous_query_discrimination")
    breadth_lower_quartile = macro_summary(
        "F4_breadth", "query_lower_quartile_recoverability_squared_error_improvement"
    )
    breadth_fraction = macro_summary(
        "F4_breadth", "query_fraction_at_least_half_squared_error_improvement"
    )
    breadth_variance = macro_summary(
        "F4_breadth", "query_recoverability_variance_squared_error_improvement"
    )
    replay_complete = int(adherence["replay_status"].eq("complete").sum()) if len(adherence) else 0
    replay_too_short = int(
        adherence["replay_status"].eq("stored_failed_sequence_too_short").sum()
    ) if len(adherence) else 0
    finite_adherence = int(adherence["original_path_adherence"].notna().sum()) if len(adherence) else 0
    f4_strongest = (
        f4_main.sort_values("squared_error_improvement", ascending=False).iloc[0]
        if len(f4_main)
        else None
    )

    f1_supported = bool(
        primary["F1"]["holm_adjusted_p"] < 0.05
        and primary["F1"]["estimate"] > 0
        and primary["F1"]["ci_lower"] > 0
    )
    f2_supported = bool(
        primary["F2"]["holm_adjusted_p"] < 0.05
        and primary["F2"]["estimate"] > 0
        and primary["F2"]["ci_lower"] > 0
    )
    f3_estimate = primary["F3"].get("estimate")
    f3_ci_upper = primary["F3"].get("ci_upper")
    f3_supported = bool(
        f3_estimate is not None
        and f3_ci_upper is not None
        and math.isfinite(float(f3_estimate))
        and math.isfinite(float(f3_ci_upper))
        and primary["F3"]["holm_adjusted_p"] < 0.05
        and float(f3_estimate) < 0
        and float(f3_ci_upper) < 0
    )
    f4_supported = bool(
        primary["F4"]["holm_adjusted_p"] < 0.05
        and primary["F4"]["estimate"] > 0
        and primary["F4"]["ci_lower"] > 0
    )

    if f1_supported:
        dominant = max(categories, key=categories.get) if categories else "undetectable"
        geometry_conclusion = {
            "narrow": "Success-conditioned geometry is detectably concentrated, with narrow local funnels the most common predefined description.",
            "broad": "Success conditioning is detectable, but broad or volumetric geometry is the most common predefined description.",
            "multimodal": "Success conditioning is detectable, but the parentwise spectra more often support multimodal than single-axis geometry.",
            "undetectable": "The pooled concentration departure is detectable, but most individual parents do not meet a narrow, broad, or multimodal descriptive criterion.",
        }[dominant]
    else:
        geometry_conclusion = "There is no familywise-corrected evidence for a concentrated repairability funnel; successful and failed futures remain locally interwoven at these horizons, or the available branch count is insufficient to resolve structure."
    coherence_conclusion = (
        "Reliable parentwise orientations show excess local coherence beyond matched controls; the result supports a locally varying field, not a universal axis."
        if f2_supported
        else "No local field claim is supported: the corrected sign-flip test is positive, but the dyadic trace-bootstrap interval crosses zero after accounting for anchor, neighbor, and control traces."
    )
    trajectory_conclusion = (
        "The stored failed trajectory shows a pre-error departure from validated success-conditioned geometry."
        if f3_supported
        else "No pre-error timing conclusion is permitted because original-path adherence failed the required held-out branch validation in every model–horizon slice."
    )
    trajectory_test_description = (
        "No corrected test is estimable; p=1 is retained in the machine output only as a Holm bookkeeping sentinel."
        if primary["F3"].get("p_value_status")
        == "holm_bookkeeping_sentinel_not_estimable"
        else f"Primary effect {_format_effect(primary['F3'])}; Holm-adjusted p={primary['F3']['holm_adjusted_p']:.4g}."
    )
    prediction_conclusion = (
        "Support-derived geometry improves grouped held-out branch prediction beyond scalar parent recoverability and structural controls."
        if f4_supported
        else "No detectable incremental held-out prediction value is established for funnel geometry beyond scalar recoverability and structural controls; this non-rejection does not prove scalar sufficiency."
    )

    lines = [
        "# SafePrefix repairability-funnel analysis",
        "",
        "## Status and evidence boundary",
        "",
        "This is a **post-hoc/exploratory, analysis-only** study over the already completed H4 local-branch corpus. The parent subset was originally drawn from the teacher-forced test split, but it is not genuinely untouched for these newly posed F1–F4 geometry hypotheses. No confirmatory language is used. The analysis did not modify SafePrefix, retrain the frozen probe, reselect a cutoff, inspect native repair outcomes, sample a branch, or generate a suffix rollout. Original-path states were recovered only by deterministic teacher forcing of stored failed token IDs.",
        "",
        "## Inventory",
        "",
        f"The corpus contains {total} planned parent–horizon pairs; {eligible} satisfy the success-conditioned eligibility rule and {total - eligible} remain in the inventory with explicit exclusion reasons. All four base models and horizons 32, 64, and 128 are represented. The full pair-level inventory and model/domain/horizon/reason census are machine-readable.",
        "",
        "## Primary results",
        "",
        f"- **F1 local funnel existence:** {_format_effect(primary['F1'])}; Holm-adjusted p={primary['F1']['holm_adjusted_p']:.4g}. {geometry_conclusion}",
        f"- **F2 local field coherence:** mean split-half orientation cosine={reliability_effect:.4f}; raw matched-control excess={coherence_effect:.4f}; primary effect {_format_effect(primary['F2'])}; Holm-adjusted p={primary['F2']['holm_adjusted_p']:.4g}. {coherence_conclusion}",
        f"- **F3 original-trajectory adherence:** {len(validated)} of {len(validation)} model–horizon slices pass leave-one-branch-out expected-direction validation. Among interpretable pre-error parents, {pre_error_description}. {trajectory_test_description} {trajectory_conclusion}",
        f"- **F4 held-out branch prediction:** {len(query)} unique query branches receive support-only scores. Mean slice-level baseline MSE={f4_main['baseline_mse'].mean() if len(f4_main) else float('nan'):.6f}, with-geometry MSE={f4_main['funnel_mse'].mean() if len(f4_main) else float('nan'):.6f}. Primary effect {_format_effect(primary['F4'])}; Holm-adjusted p={primary['F4']['holm_adjusted_p']:.4g}. {prediction_conclusion}",
        f"  The F4 effect is heterogeneous: {f4_positive_slices} of {len(f4_main)} model–horizon slices have positive squared-error improvement; the strongest slice is {(str(f4_strongest['base_model']) + ' at horizon ' + str(int(f4_strongest['horizon']))) if f4_strongest is not None else 'unavailable'}. The equal-model within-parent rank association is {_format_effect(f4_rank)}, and continuous query discrimination is {_format_effect(f4_discrimination)}; these near-null marginal summaries qualify the small conditional MSE gain.",
        "  Support-only breadth features give equal-model held-out squared-error improvements of "
        f"{_format_effect(breadth_lower_quartile, digits=6)} for the query lower quartile, "
        f"{_format_effect(breadth_fraction, digits=6)} for the fraction with r_tilde >= 0.5, and "
        f"{_format_effect(breadth_variance, digits=6)} for query recoverability variance. These are feasibility estimates, not additional corrected hypothesis families.",
        "",
        "## Geometric description",
        "",
        "The predefined parentwise descriptive proportions are "
        + ", ".join(f"{key}={value:.1%}" for key, value in sorted(categories.items()))
        + ". These labels are descriptive combinations of predefined z-score criteria, not discovered clusters. Unweighted anisotropy is reported alongside every success-conditioned spectral statistic, so ordinary branch-cloud anisotropy is not counted as a funnel.",
        "",
        "Nearest-parent orientation comparisons are restricted to parents in the same held-out cross-fitting fold, ensuring that each compared pair shares a transformation fitted without either evaluated trace. Controls must match domain, normalized checkpoint-position bin, and parent-recoverability bin; unmatched comparisons remain missing rather than being relaxed. Within-domain and cross-domain rows are preserved in the neighbor table.",
        "",
        "Because F2 does not pass the reliability-aware interval gate, neither a global orientation field nor domain-specific orientation transfer is supported. In the reliability-qualified slices, raw excess similarity is "
        f"{relation_effects.get(True, {}).get('mean', float('nan')):.4f} across {int(relation_effects.get(True, {}).get('size', 0))} within-domain comparisons and "
        f"{relation_effects.get(False, {}).get('mean', float('nan')):.4f} across {int(relation_effects.get(False, {}).get('size', 0))} cross-domain comparisons; these small realized cells are exploratory descriptions.",
        "",
        "## Failed-path timing limits",
        "",
        f"Deterministic replay recovered {replay_complete} original-path states; {replay_too_short} stored failed sequences ended before the requested horizon. Of the recovered rows, {finite_adherence} have a finite geometric adherence score, and {len(interpreted)} pass the model–horizon interpretation gate. Availability counts by model, domain, horizon, and replay status are saved separately.",
        "",
        "The original H4 design selected at most one parent per model–trace. Therefore zero traces have two sequential eligible parent checkpoints: no individual trace funnel-exit checkpoint and no within-trace paired checkpoint comparison is estimable. This is a design limitation, not evidence that no exit occurs.",
        "",
        "## Robustness and sensitivity",
        "",
        "Primary analyses use r_tilde=(s+0.5)/5 with K=4. F1, F3, and F4 include machine-readable unsmoothed s/4 sensitivity results. F2 success-contrast orientations are analytically identical under smoothing because r_tilde is a positive affine transform of s/4; covariance/breadth sensitivity remains in F1/F4. F4 keeps all support/query label use separated; query labels never enter support-derived geometry. Failed trace is the bootstrap and grouped-CV unit, dyadic F2 intervals resample every participating trace, models receive equal weight in macro summaries, and the four primary hypothesis families receive Holm correction over null-calibrated permutation/sign-flip tests.",
        "",
        "The optional parent-state actionability test was not run. F1–F4 exhaust the requested scientific questions, and this analysis alone does not justify a new repair or rewind policy.",
        "",
        "## Integrity",
        "",
        f"Final integrity status: **{'PASS' if integrity.get('passed') else 'FAIL'}**. "
        + ("All required machine-readable tables, statistics, reports, and figures were written." if integrity.get("passed") else "; ".join(integrity.get("failures", []))),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def run_analysis(
    *,
    config_path: Path,
    boundary_root: Path,
    child_states_path: Path,
    parent_manifest_path: Path,
    original_states_path: Path | None,
    output_root: Path,
) -> dict[str, Any]:
    """Execute F1–F4 and atomically materialize the complete analysis."""

    for path in (config_path, boundary_root, child_states_path, parent_manifest_path, output_root):
        _guard_analysis_path(path)
    if original_states_path is not None:
        _guard_analysis_path(original_states_path)
    config = load_config(config_path).data
    children = pd.read_parquet(child_states_path)
    parents = _read_jsonl(parent_manifest_path)
    original = (
        pd.read_parquet(original_states_path)
        if original_states_path is not None and original_states_path.is_file()
        else pd.DataFrame()
    )
    (
        frame,
        inventory,
        directions,
        transforms,
        folds,
        sibling_means,
        parent_features,
        parent_states,
        crossfit_manifest,
    ) = _prepare_corpus(children, parents, boundary_root=boundary_root, config=config)
    f1, permutation_nulls = _run_f1(frame, directions, config=config)
    reliability, neighbors = _run_f2(frame, directions, parent_states, config=config)
    loo, validation, adherence, error_alignment = _run_f3(
        frame,
        directions,
        original,
        transforms,
        sibling_means,
        parent_features,
        boundary_root=boundary_root,
        config=config,
    )
    query_predictions, query_aggregated, breadth, f4_metrics = _run_f4(
        frame, inventory, directions, config=config
    )
    summaries, bootstrap_intervals, tests = _summaries_and_tests(
        f1,
        permutation_nulls,
        reliability,
        neighbors,
        validation,
        adherence,
        query_aggregated,
        breadth,
        config=config,
    )

    inventory_exploded = inventory.assign(
        exclusion_reason=inventory["exclusion_reason"].str.split(";")
    ).explode("exclusion_reason")
    inventory_counts = (
        inventory_exploded.groupby(
            ["base_model", "domain", "horizon", "exclusion_reason"],
            sort=True,
            dropna=False,
        )
        .size()
        .rename("parent_horizon_pairs")
        .reset_index()
    )
    original_state_inventory = (
        adherence.assign(
            finite_original_path_adherence=adherence["original_path_adherence"].notna(),
            interpretation_eligible=adherence["interpretation_eligible"].astype(bool),
        )
        .groupby(
            ["base_model", "domain", "horizon", "replay_status"],
            sort=True,
            dropna=False,
        )
        .agg(
            parent_horizon_pairs=("parent_id", "size"),
            finite_original_path_adherence=("finite_original_path_adherence", "sum"),
            interpretation_eligible_pairs=("interpretation_eligible", "sum"),
        )
        .reset_index()
    )
    geometry_proportions = (
        f1.groupby(["base_model", "horizon", "domain", "descriptive_geometry"], sort=True)
        .size()
        .rename("parent_count")
        .reset_index()
    )
    geometry_proportions["proportion"] = geometry_proportions["parent_count"] / geometry_proportions.groupby(
        ["base_model", "horizon", "domain"], sort=True
    )["parent_count"].transform("sum")

    exit_eligibility = (
        inventory.loc[inventory["success_conditioned_geometry_eligible"]]
        .groupby(["base_model", "trace_id"], sort=True)
        .agg(sequential_eligible_checkpoint_count=("checkpoint_index", "nunique"))
        .reset_index()
    )
    exit_eligibility["individual_exit_time_estimable"] = (
        exit_eligibility["sequential_eligible_checkpoint_count"]
        >= int(config["f3"]["minimum_sequential_parents_for_exit"])
    )
    paired = pd.DataFrame(
        columns=[
            "base_model", "trace_id", "horizon", "earlier_checkpoint_index",
            "later_checkpoint_index", "adherence_difference",
        ]
    )

    tables = output_root / "tables"
    _atomic_parquet(tables / "inventory_parent_horizon.parquet", inventory)
    _atomic_csv(tables / "inventory_counts.csv", inventory_counts)
    _atomic_csv(tables / "original_path_state_inventory.csv", original_state_inventory)
    _atomic_parquet(tables / "crossfit_transform_manifest.parquet", crossfit_manifest)
    _atomic_parquet(tables / "per_parent_funnel_statistics.parquet", f1)
    _atomic_parquet(tables / "permutation_nulls.parquet", permutation_nulls)
    _atomic_parquet(tables / "geometry_descriptive_proportions.parquet", geometry_proportions)
    _atomic_parquet(tables / "split_half_orientation_reliability.parquet", reliability)
    _atomic_parquet(tables / "parent_neighbor_coherence.parquet", neighbors)
    _atomic_parquet(tables / "leave_one_branch_out_validation.parquet", loo)
    _atomic_parquet(tables / "loo_validation_by_model_horizon.parquet", validation)
    _atomic_parquet(tables / "original_path_adherence.parquet", adherence)
    _atomic_parquet(tables / "first_error_alignment.parquet", error_alignment)
    _atomic_parquet(tables / "trace_exit_eligibility.parquet", exit_eligibility)
    _atomic_parquet(tables / "paired_checkpoint_comparisons.parquet", paired)
    _atomic_parquet(tables / "support_query_predictions.parquet", query_predictions)
    _atomic_parquet(tables / "support_query_branch_aggregated.parquet", query_aggregated)
    _atomic_parquet(tables / "support_breadth_predictions.parquet", breadth)
    _atomic_parquet(tables / "f4_model_horizon_metrics.parquet", f4_metrics)
    _atomic_parquet(tables / "model_horizon_domain_summaries.parquet", summaries)
    _atomic_parquet(tables / "bootstrap_intervals.parquet", bootstrap_intervals)
    _atomic_json(output_root / "statistics/corrected_tests.json", tests)
    _make_figures(f1, neighbors, adherence, query_aggregated, root=output_root / "figures")

    required_relative = [
        "tables/inventory_parent_horizon.parquet",
        "tables/original_path_state_inventory.csv",
        "tables/per_parent_funnel_statistics.parquet",
        "tables/permutation_nulls.parquet",
        "tables/split_half_orientation_reliability.parquet",
        "tables/parent_neighbor_coherence.parquet",
        "tables/original_path_adherence.parquet",
        "tables/first_error_alignment.parquet",
        "tables/support_query_predictions.parquet",
        "tables/model_horizon_domain_summaries.parquet",
        "tables/bootstrap_intervals.parquet",
        "statistics/corrected_tests.json",
        "figures/figure1_concentration_rank_vs_null.png",
        "figures/figure2_coherence_vs_parent_distance.png",
        "figures/figure3_failed_path_adherence.png",
        "figures/figure4_heldout_recoverability_vs_funnel_score.png",
    ]
    failures: list[str] = []
    if set(inventory["base_model"].astype(str)) != set(MODEL_ORDER):
        failures.append("inventory does not contain all four base models")
    if set(inventory["horizon"].astype(int)) != set(HORIZON_ORDER):
        failures.append("inventory does not contain all three predefined horizons")
    eligible_count = int(inventory["success_conditioned_geometry_eligible"].sum())
    if len(f1) != eligible_count:
        failures.append("F1 parent count differs from eligible inventory")
    expected_null_rows = eligible_count * int(config["f1"]["permutation_replicates"])
    if len(permutation_nulls) != expected_null_rows:
        failures.append("permutation null row count differs from eligible parents times replicates")
    f4_parent_count = int(inventory["f4_exact_twelve_eligible"].sum())
    expected_query_rows = (
        f4_parent_count
        * int(config["f4"]["deterministic_splits_per_parent"])
        * int(config["f4"]["query_branches"])
    )
    if len(query_predictions) != expected_query_rows:
        failures.append("support/query prediction row count differs from exact protocol")
    if original_states_path is None or not original_states_path.is_file():
        failures.append("deterministic original-path states are absent")
    if int(exit_eligibility["individual_exit_time_estimable"].sum()) != 0:
        # This is not forbidden generally, but the realized H4 one-parent design
        # should make any contrary result a join/inventory defect.
        failures.append("unexpected sequential parents in one-parent-per-trace H4 corpus")
    missing_outputs = [relative for relative in required_relative if not (output_root / relative).is_file()]
    if missing_outputs:
        failures.append(f"missing required outputs: {missing_outputs}")
    integrity = {
        "passed": not failures,
        "failures": failures,
        "source_hashes": {
            "config": _hash_file(config_path),
            "child_states": _hash_file(child_states_path),
            "parent_manifest": _hash_file(parent_manifest_path),
            "original_states": _hash_file(original_states_path) if original_states_path and original_states_path.is_file() else None,
        },
        "native_artifacts_accessed": False,
        "new_branches_generated": 0,
        "new_suffix_rollouts_generated": 0,
        "probe_retrained": False,
        "cutoff_reselected": False,
        "inventory_parent_horizon_pairs": int(len(inventory)),
        "eligible_parent_horizon_pairs": eligible_count,
        "permutation_rows": int(len(permutation_nulls)),
        "support_query_rows": int(len(query_predictions)),
        "original_state_rows": int(len(original)),
        "required_outputs": required_relative,
    }
    _write_report(
        output_root / "SCIENTIFIC_REPORT.md",
        inventory=inventory,
        inventory_counts=inventory_counts,
        f1=f1,
        reliability=reliability,
        neighbors=neighbors,
        validation=validation,
        adherence=adherence,
        query=query_aggregated,
        f4_metrics=f4_metrics,
        summaries=summaries,
        tests=tests,
        integrity=integrity,
    )
    required_relative.append("SCIENTIFIC_REPORT.md")
    _atomic_json(output_root / "integrity.json", integrity)
    summary = {
        "status": "COMPLETE" if integrity["passed"] else "INCOMPLETE_INTEGRITY_FAILURE",
        "analysis_design": "post_hoc_exploratory",
        "inventory": {
            "parent_horizon_pairs": int(len(inventory)),
            "eligible_parent_horizon_pairs": eligible_count,
            "excluded_parent_horizon_pairs": int(len(inventory) - eligible_count),
            "f4_exact_twelve_parent_horizon_pairs": f4_parent_count,
        },
        "primary_tests": tests["primary_tests"],
        "claim_status": {
            "F1": (
                "supported_concentration_but_not_predominantly_narrow"
                if tests["primary_tests"]["F1"]["holm_adjusted_p"] < 0.05
                and tests["primary_tests"]["F1"]["ci_lower"] > 0
                else "not_supported"
            ),
            "F2": (
                "supported_local_coherence"
                if tests["primary_tests"]["F2"]["holm_adjusted_p"] < 0.05
                and tests["primary_tests"]["F2"]["ci_lower"] > 0
                else "inconclusive_signflip_positive_dyadic_ci_crosses_zero"
            ),
            "F3": (
                "not_interpretable_no_validated_model_horizon_slice"
                if not bool(validation["expected_direction_validated"].any())
                else "interpretable"
            ),
            "F4": (
                "supported_modest_heterogeneous_incremental_value"
                if tests["primary_tests"]["F4"]["holm_adjusted_p"] < 0.05
                and tests["primary_tests"]["F4"]["ci_lower"] > 0
                else "no_detectable_incremental_value"
            ),
        },
        "descriptive_geometry_proportions": f1["descriptive_geometry"].value_counts(normalize=True).to_dict(),
        "f3_validated_model_horizon_slices": validation.loc[
            validation["expected_direction_validated"], ["base_model", "horizon"]
        ].to_dict("records"),
        "individual_trace_exit_time_estimable": int(exit_eligibility["individual_exit_time_estimable"].sum()),
        "integrity": integrity,
    }
    _atomic_json(output_root / "summary.json", summary)
    return summary
