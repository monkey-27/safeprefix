"""Positive-slope calibration and trace-level five-fold OOF predictions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn


def _trace_weights(frame: pd.DataFrame) -> np.ndarray:
    if frame.empty or "trace_id" not in frame:
        raise ValueError("calibration rows must be nonempty and trace-indexed")
    count = frame.groupby("trace_id", sort=False)["trace_id"].transform("size").to_numpy(float)
    return 1.0 / count / float(frame["trace_id"].astype(str).nunique())


@dataclass(frozen=True)
class PositiveAffineCalibrator:
    a: float
    b: float
    initial_nll: float
    fitted_nll: float
    fit_split: str = "calibration"
    objective: str = "trace_weighted_binary_cross_entropy"
    positive_slope: bool = True
    test_outcomes_used: bool = False
    native_outcomes_used: bool = False

    def __post_init__(self) -> None:
        if not all(map(math.isfinite, (self.a, self.b, self.initial_nll, self.fitted_nll))):
            raise ValueError("calibration parameters and losses must be finite")
        if self.a <= 0 or not self.positive_slope:
            raise ValueError("calibration slope must be strictly positive")
        if self.fit_split != "calibration":
            raise ValueError("prefix-validity calibration must use calibration only")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PositiveAffineCalibrator":
        return cls(**{key: payload[key] for key in cls.__dataclass_fields__})


def fit_positive_affine_calibrator(
    frame: pd.DataFrame,
    *,
    logit_column: str = "raw_logit",
    target_column: str = "prefix_valid",
    max_iterations: int = 500,
) -> PositiveAffineCalibrator:
    """Fit ``sigmoid(a*logit+b)`` with ``a>0`` on calibration rows only."""

    required = {"trace_id", "split", logit_column, target_column}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"calibration frame lacks columns: {sorted(missing)}")
    if set(frame["split"].astype(str)) != {"calibration"}:
        raise RuntimeError("prefix-validity calibration may use calibration split only")
    logits_np = frame[logit_column].to_numpy(float)
    target_np = frame[target_column].to_numpy(float)
    if not np.isfinite(logits_np).all() or not np.isfinite(target_np).all():
        raise ValueError("calibration logits and targets must be finite")
    if set(np.unique(target_np)) != {0.0, 1.0}:
        raise RuntimeError("calibration requires both prefix-validity classes")
    logits = torch.tensor(logits_np, dtype=torch.float64)
    target = torch.tensor(target_np, dtype=torch.float64)
    weights = torch.tensor(_trace_weights(frame), dtype=torch.float64)
    theta = torch.nn.Parameter(torch.tensor(math.log(math.e - 1.0), dtype=torch.float64))
    intercept = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [theta, intercept],
        lr=0.5,
        max_iter=int(max_iterations),
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        slope = nn.functional.softplus(theta) + 1e-8
        loss = (
            nn.functional.binary_cross_entropy_with_logits(
                slope * logits + intercept, target, reduction="none"
            )
            * weights
        ).sum()
        loss.backward()
        return loss

    initial = float(closure().detach())
    optimizer.step(closure)
    with torch.no_grad():
        slope = float(nn.functional.softplus(theta) + 1e-8)
        bias = float(intercept)
        fitted = float(
            (
                nn.functional.binary_cross_entropy_with_logits(
                    slope * logits + bias, target, reduction="none"
                )
                * weights
            ).sum()
        )
    return PositiveAffineCalibrator(
        a=slope,
        b=bias,
        initial_nll=initial,
        fitted_nll=fitted,
    )


def apply_calibrator(
    frame: pd.DataFrame,
    calibrator: PositiveAffineCalibrator | Mapping[str, Any],
    *,
    logit_column: str = "raw_logit",
    output_column: str = "calibrated_probability",
) -> pd.DataFrame:
    resolved = (
        calibrator
        if isinstance(calibrator, PositiveAffineCalibrator)
        else PositiveAffineCalibrator.from_dict(calibrator)
    )
    output = frame.copy()
    value = np.clip(
        resolved.a * output[logit_column].to_numpy(float) + resolved.b,
        -60.0,
        60.0,
    )
    output[output_column] = 1.0 / (1.0 + np.exp(-value))
    return output


def assign_trace_folds(
    frame: pd.DataFrame, *, folds: int = 5, seed: int = 20260729
) -> pd.Series:
    """Assign whole traces to deterministic folds, never checkpoint rows."""

    if folds < 2:
        raise ValueError("OOF calibration requires at least two folds")
    if "trace_id" not in frame:
        raise KeyError("OOF calibration requires trace_id")
    traces = sorted(
        frame["trace_id"].astype(str).unique(),
        key=lambda value: hashlib.sha256(
            f"prefix-validity-oof-v1\0{seed}\0{value}".encode()
        ).digest(),
    )
    if len(traces) < folds:
        raise RuntimeError(f"five-fold calibration requires at least {folds} traces")
    # Hash-sort then round-robin gives deterministic pseudorandom membership
    # while guaranteeing that every fold is populated.
    mapping = {trace_id: index % folds for index, trace_id in enumerate(traces)}
    result = frame["trace_id"].astype(str).map(mapping).astype(int)
    if result.nunique() != folds:
        raise RuntimeError(f"calibration traces do not populate all {folds} folds")
    return result


@dataclass(frozen=True)
class OOFCalibrationResult:
    final_calibrator: PositiveAffineCalibrator
    fold_calibrators: tuple[PositiveAffineCalibrator, ...]
    predictions: pd.DataFrame
    folds: int
    fold_seed: int


def fit_fivefold_oof_calibration(
    calibration_predictions: pd.DataFrame,
    *,
    logit_column: str = "raw_logit",
    target_column: str = "prefix_valid",
    fold_seed: int = 20260729,
    max_iterations: int = 500,
) -> OOFCalibrationResult:
    """Produce leakage-free OOF probabilities and a final all-calibration map.

    The OOF probabilities are exclusively for selecting ``gamma_m``.  The
    returned final calibrator, fit after OOF prediction is complete, is the map
    frozen for ProcessBench test and native application.
    """

    if set(calibration_predictions["split"].astype(str)) != {"calibration"}:
        raise RuntimeError("OOF input must be the ProcessBench calibration split")
    frame = calibration_predictions.copy()
    frame["oof_fold"] = assign_trace_folds(frame, folds=5, seed=fold_seed)
    frame["calibrated_probability"] = np.nan
    fitted: list[PositiveAffineCalibrator] = []
    for fold in range(5):
        train = frame.loc[frame["oof_fold"].ne(fold)]
        holdout = frame.loc[frame["oof_fold"].eq(fold)]
        if holdout.empty:
            raise RuntimeError(f"OOF fold {fold} is empty")
        if set(train["trace_id"].astype(str)) & set(holdout["trace_id"].astype(str)):
            raise RuntimeError("trace leakage across OOF calibration fold")
        calibrator = fit_positive_affine_calibrator(
            train,
            logit_column=logit_column,
            target_column=target_column,
            max_iterations=max_iterations,
        )
        fitted.append(calibrator)
        scored = apply_calibrator(
            holdout,
            calibrator,
            logit_column=logit_column,
            output_column="calibrated_probability",
        )
        frame.loc[holdout.index, "calibrated_probability"] = scored[
            "calibrated_probability"
        ].to_numpy(float)
    if frame["calibrated_probability"].isna().any():
        raise RuntimeError("OOF calibration failed to score every row exactly once")
    final = fit_positive_affine_calibrator(
        frame,
        logit_column=logit_column,
        target_column=target_column,
        max_iterations=max_iterations,
    )
    return OOFCalibrationResult(
        final_calibrator=final,
        fold_calibrators=tuple(fitted),
        predictions=frame,
        folds=5,
        fold_seed=int(fold_seed),
    )
