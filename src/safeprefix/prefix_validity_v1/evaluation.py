"""Checkpoint, boundary, and paired trace-bootstrap evaluation primitives."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)


EPSILON = 1e-12


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise KeyError(f"matched rows are missing: {sorted(missing)}")


def _trace_weights(frame: pd.DataFrame) -> np.ndarray:
    _require_columns(frame, ["trace_id"])
    if frame.empty:
        raise ValueError("metric frame must be nonempty")
    counts = frame.groupby("trace_id", sort=False)["trace_id"].transform("size")
    weights = 1.0 / counts.to_numpy(float)
    return weights / weights.sum()


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(np.asarray(values, dtype=float) * weights) / np.sum(weights))


def _safe_binary_metric(function: Any, target: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(target)) < 2:
        return float("nan")
    return float(function(target, score))


def _weighted_ece(
    target: np.ndarray, probability: np.ndarray, weights: np.ndarray, *, bins: int
) -> float:
    if bins <= 0:
        raise ValueError("ECE bins must be positive")
    order = np.argsort(probability, kind="stable")
    total = 0.0
    for chunk in np.array_split(order, min(bins, len(order))):
        if not len(chunk):
            continue
        mass = float(weights[chunk].sum())
        predicted = _weighted_mean(probability[chunk], weights[chunk])
        observed = _weighted_mean(target[chunk], weights[chunk])
        total += mass * abs(predicted - observed)
    return float(total / weights.sum())


def _correctness_metrics(
    frame: pd.DataFrame,
    *,
    logit_column: str,
    probability_column: str,
    ece_bins: int = 10,
) -> dict[str, float | int]:
    _require_columns(frame, ["correctness_label", logit_column, probability_column])
    target = frame["correctness_label"].to_numpy(int)
    logits = frame[logit_column].to_numpy(float)
    probability = frame[probability_column].to_numpy(float)
    if not set(np.unique(target)).issubset({0, 1}):
        raise ValueError("correctness_label must be binary")
    if not np.isfinite(logits).all() or not np.isfinite(probability).all():
        raise ValueError("probe predictions must be finite")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("calibrated probabilities must lie in [0, 1]")
    predicted = probability >= 0.5
    weights = _trace_weights(frame)
    clipped = np.clip(probability, EPSILON, 1.0 - EPSILON)
    nll = -(target * np.log(clipped) + (1 - target) * np.log1p(-clipped))
    return {
        "checkpoints": int(len(frame)),
        "traces": int(frame["trace_id"].nunique()),
        "prevalence": _weighted_mean(target, weights),
        "roc_auc": _safe_binary_metric(roc_auc_score, target, logits),
        "average_precision": _safe_binary_metric(average_precision_score, target, logits),
        "accuracy": float(accuracy_score(target, predicted, sample_weight=weights)),
        "balanced_accuracy": _safe_binary_metric(
            lambda y, p: balanced_accuracy_score(y, p), target, predicted
        ),
        "trace_weighted_nll": _weighted_mean(nll, weights),
        "trace_weighted_brier": _weighted_mean((probability - target) ** 2, weights),
        "trace_weighted_ece": _weighted_ece(target, probability, weights, bins=ece_bins),
    }


def checkpoint_metrics(
    frame: pd.DataFrame,
    *,
    logit_column: str = "raw_logit",
    probability_column: str = "monotonic_probability",
    ece_bins: int = 10,
) -> dict[str, float | int]:
    """Trace-weighted checkpoint classification and calibration metrics."""

    if "prefix_valid" not in frame:
        raise KeyError("checkpoint evaluation requires prefix_valid")
    renamed = frame.rename(columns={"prefix_valid": "correctness_label"})
    return _correctness_metrics(
        renamed,
        logit_column=logit_column,
        probability_column=probability_column,
        ece_bins=ece_bins,
    )


def _trace_boundary_record(
    trace: pd.DataFrame,
    *,
    gamma: float,
    probability_column: str,
    order_column: str,
    true_boundary_column: str,
    token_column: str,
    total_token_column: str,
) -> dict[str, Any]:
    true_values = trace[true_boundary_column].astype(int).unique()
    if len(true_values) != 1 or int(true_values[0]) < 0:
        raise RuntimeError("each trace requires one nonnegative true last-valid boundary")
    true_boundary = int(true_values[0])
    eligible = trace.loc[trace[probability_column].ge(float(gamma)), order_column]
    predicted = int(eligible.max()) if len(eligible) else 0
    ordinal_to_token = {
        int(row[order_column]): int(row[token_column]) for _, row in trace.iterrows()
    }
    if predicted > 0 and predicted not in ordinal_to_token:
        raise RuntimeError("predicted checkpoint is absent from the trace")
    if true_boundary > 0 and true_boundary not in ordinal_to_token:
        raise RuntimeError("true last-valid checkpoint is absent from the trace")
    predicted_token = ordinal_to_token.get(predicted, 0)
    true_token = ordinal_to_token.get(true_boundary, 0)
    totals = trace[total_token_column].astype(int).unique()
    if len(totals) != 1 or int(totals[0]) <= 0:
        raise RuntimeError("trace token length must be one positive integer")
    absolute = abs(predicted - true_boundary)
    retained = (
        float(min(predicted, true_boundary) / true_boundary)
        if true_boundary > 0
        else float(predicted == 0)
    )
    return {
        "predicted_boundary": predicted,
        "true_boundary": true_boundary,
        "checkpoint_error": predicted - true_boundary,
        "absolute_checkpoint_error": absolute,
        "normalized_token_position_error": abs(predicted_token - true_token) / int(totals[0]),
        "exact": predicted == true_boundary,
        "within_one": absolute <= 1,
        "late": predicted > true_boundary,
        "early": predicted < true_boundary,
        "retained_valid_prefix_fraction": retained,
        "non_root": predicted > 0,
    }


def boundary_records(
    frame: pd.DataFrame,
    *,
    gamma: float,
    probability_column: str = "monotonic_probability",
    order_column: str = "checkpoint_ordinal",
    true_boundary_column: str = "true_last_valid_checkpoint",
    token_column: str = "checkpoint_token_offset",
    total_token_column: str = "total_trace_token_count",
) -> pd.DataFrame:
    required = {
        "trace_id",
        probability_column,
        order_column,
        true_boundary_column,
        token_column,
        total_token_column,
    }
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"boundary evaluation lacks columns: {sorted(missing)}")
    records: list[dict[str, Any]] = []
    for trace_id, trace in frame.groupby("trace_id", sort=True):
        metadata = {
            key: trace.iloc[0][key]
            for key in ("base_model", "domain", "split")
            if key in trace
        }
        records.append(
            {
                **metadata,
                "trace_id": str(trace_id),
                **_trace_boundary_record(
                    trace,
                    gamma=gamma,
                    probability_column=probability_column,
                    order_column=order_column,
                    true_boundary_column=true_boundary_column,
                    token_column=token_column,
                    total_token_column=total_token_column,
                ),
            }
        )
    return pd.DataFrame(records)


def boundary_metrics(
    frame: pd.DataFrame,
    *,
    gamma: float,
    probability_column: str = "monotonic_probability",
) -> dict[str, float | int]:
    records = boundary_records(frame, gamma=gamma, probability_column=probability_column)
    return {
        "traces": int(len(records)),
        "exact_last_valid_checkpoint_accuracy": float(records["exact"].mean()),
        "within_one_checkpoint_accuracy": float(records["within_one"].mean()),
        "mean_absolute_checkpoint_error": float(records["absolute_checkpoint_error"].mean()),
        "median_absolute_checkpoint_error": float(records["absolute_checkpoint_error"].median()),
        "normalized_token_position_error": float(records["normalized_token_position_error"].mean()),
        "late_boundary_rate": float(records["late"].mean()),
        "early_boundary_rate": float(records["early"].mean()),
        "mean_retained_valid_prefix_fraction": float(
            records["retained_valid_prefix_fraction"].mean()
        ),
        "non_root_predicted_boundary_coverage": float(records["non_root"].mean()),
    }


def evaluate_by_domain(
    frame: pd.DataFrame,
    *,
    gamma: float,
    logit_column: str = "raw_logit",
    probability_column: str = "monotonic_probability",
) -> dict[str, Any]:
    """Return overall and per-ProcessBench-domain checkpoint/boundary metrics."""

    if "domain" not in frame:
        raise KeyError("domain-stratified evaluation requires domain")
    return {
        "overall": {
            "checkpoint": checkpoint_metrics(
                frame, logit_column=logit_column, probability_column=probability_column
            ),
            "boundary": boundary_metrics(
                frame, gamma=gamma, probability_column=probability_column
            ),
        },
        "by_domain": {
            str(domain): {
                "checkpoint": checkpoint_metrics(
                    part, logit_column=logit_column, probability_column=probability_column
                ),
                "boundary": boundary_metrics(
                    part, gamma=gamma, probability_column=probability_column
                ),
            }
            for domain, part in frame.groupby("domain", sort=True)
        },
    }


def align_matched_probe_predictions(
    hidden: pd.DataFrame,
    position: pd.DataFrame,
) -> pd.DataFrame:
    """Join matched held-out predictions and reject any row-set drift.

    Both probes are required to consume the identical ProcessBench checkpoint
    rows.  The join keeps one copy of all scientific metadata and two frozen
    score/probability pairs for paired trace-level inference.
    """

    identity = [
        column
        for column in ("model_key", "base_model", "trace_id", "checkpoint_id")
        if column in hidden.columns and column in position.columns
    ]
    if "trace_id" not in identity or "checkpoint_id" not in identity:
        raise KeyError("matched probe predictions require trace/checkpoint identities")
    if hidden.duplicated(identity).any() or position.duplicated(identity).any():
        raise RuntimeError("matched probe predictions contain duplicate checkpoint rows")
    if len(hidden) != len(position):
        raise RuntimeError("hidden and position probes used different checkpoint counts")
    shared_metadata = [
        column
        for column in (
            "domain",
            "split",
            "prefix_valid",
            "true_last_valid_checkpoint",
            "checkpoint_ordinal",
            "checkpoint_token_offset",
            "total_trace_token_count",
        )
        if column in hidden.columns and column in position.columns
    ]
    left = hidden[identity + shared_metadata + ["raw_logit", "monotonic_probability"]].copy()
    right = position[identity + shared_metadata + ["raw_logit", "monotonic_probability"]].copy()
    joined = left.merge(
        right,
        on=identity,
        how="outer",
        suffixes=("_hidden", "_position"),
        indicator=True,
        validate="one_to_one",
    )
    if not joined["_merge"].eq("both").all():
        raise RuntimeError("hidden and position probes used different checkpoint identities")
    joined.drop(columns="_merge", inplace=True)
    for column in shared_metadata:
        lhs = joined.pop(f"{column}_hidden")
        rhs = joined.pop(f"{column}_position")
        if not lhs.equals(rhs):
            raise RuntimeError(f"matched probe metadata differs in {column}")
        joined[column] = lhs
    joined.rename(
        columns={
            "raw_logit_hidden": "hidden_raw_logit",
            "monotonic_probability_hidden": "hidden_monotonic_probability",
            "raw_logit_position": "position_raw_logit",
            "monotonic_probability_position": "position_monotonic_probability",
        },
        inplace=True,
    )
    return joined


def paired_probe_metric_vector(
    matched: pd.DataFrame,
    *,
    hidden_gamma: float | Mapping[str, float],
    position_gamma: float | Mapping[str, float],
) -> dict[str, float]:
    """Return paired point metrics and hidden-minus-position contrasts."""

    def resolve_gamma(value: float | Mapping[str, float]) -> float:
        if not isinstance(value, Mapping):
            return float(value)
        model_column = next(
            (name for name in ("model_key", "base_model", "model_id") if name in matched),
            None,
        )
        if model_column is None or matched[model_column].astype(str).nunique() != 1:
            raise RuntimeError("model-specific gamma requires exactly one model per statistic")
        model = str(matched[model_column].astype(str).iloc[0])
        if model not in value:
            raise KeyError(f"gate cutoff is absent for model {model}")
        return float(value[model])

    values: dict[str, float] = {}
    by_probe: dict[str, dict[str, float]] = {}
    for probe, gamma in (
        ("hidden", resolve_gamma(hidden_gamma)),
        ("position", resolve_gamma(position_gamma)),
    ):
        scored = matched.rename(
            columns={
                f"{probe}_raw_logit": "raw_logit",
                f"{probe}_monotonic_probability": "monotonic_probability",
            }
        )
        checkpoint = checkpoint_metrics(scored)
        boundary = boundary_metrics(scored, gamma=float(gamma))
        metrics = {
            **{
                f"checkpoint_{name}": float(value)
                for name, value in checkpoint.items()
                if name not in {"checkpoints", "traces"}
            },
            **{
                f"boundary_{name}": float(value)
                for name, value in boundary.items()
                if name != "traces"
            },
        }
        by_probe[probe] = metrics
        values.update({f"{probe}_{name}": value for name, value in metrics.items()})
    if set(by_probe["hidden"]) != set(by_probe["position"]):  # pragma: no cover
        raise RuntimeError("paired probes expose different metric sets")
    values.update(
        {
            f"hidden_minus_position_{name}": (
                by_probe["hidden"][name] - by_probe["position"][name]
            )
            for name in by_probe["hidden"]
        }
    )
    return values


def paired_probe_test_bootstrap(
    hidden: pd.DataFrame,
    position: pd.DataFrame,
    *,
    hidden_gamma: float | Mapping[str, float],
    position_gamma: float | Mapping[str, float],
    replicates: int = 10_000,
    seed: int = 20260729,
    expected_models: int | None = None,
) -> dict[str, Any]:
    """Fast paired complete-trace CIs for held-out safety metrics.

    The resampling law is identical to :func:`paired_trace_bootstrap`, but the
    per-trace sufficient statistics are computed once.  This makes 10,000
    replicates practical while retaining every checkpoint from each sampled
    trace.  ROC AUC, average precision, balanced accuracy, and ECE remain point
    metrics because they do not decompose into trace-level sufficient
    statistics; all boundary metrics plus trace-weighted NLL/Brier/accuracy
    receive paired intervals.
    """

    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    matched = align_matched_probe_predictions(hidden, position)
    test_only = set(matched["split"].astype(str)) == {"teacher_forced_test"}
    if not test_only:
        raise RuntimeError("ProcessBench held-out bootstrap contains non-test rows")
    model_column = next(
        (name for name in ("base_model", "model_key", "model_id") if name in matched),
        None,
    )
    if model_column is None:
        raise KeyError("paired test bootstrap requires a model identity")
    models = sorted(matched[model_column].astype(str).unique())
    if expected_models is not None and len(models) != int(expected_models):
        raise RuntimeError(f"expected {expected_models} models, found {len(models)}")

    def gamma_for(value: float | Mapping[str, float], model: str) -> float:
        return float(value[model]) if isinstance(value, Mapping) else float(value)

    trace_tables: dict[str, pd.DataFrame] = {}
    for probe, gamma_source in (("hidden", hidden_gamma), ("position", position_gamma)):
        pieces = []
        for model in models:
            part = matched.loc[matched[model_column].astype(str).eq(model)].copy()
            scored = part.rename(
                columns={
                    f"{probe}_raw_logit": "raw_logit",
                    f"{probe}_monotonic_probability": "monotonic_probability",
                }
            )
            boundaries = boundary_records(
                scored,
                gamma=gamma_for(gamma_source, model),
            )
            probability = np.clip(
                scored["monotonic_probability"].to_numpy(float), 1e-12, 1.0 - 1e-12
            )
            target = scored["prefix_valid"].to_numpy(float)
            checkpoint = scored[["trace_id"]].copy()
            checkpoint["checkpoint_trace_weighted_nll"] = -(
                target * np.log(probability) + (1.0 - target) * np.log1p(-probability)
            )
            checkpoint["checkpoint_trace_weighted_brier"] = (probability - target) ** 2
            checkpoint["checkpoint_trace_weighted_accuracy"] = (
                (probability >= 0.5) == target.astype(bool)
            ).astype(float)
            checkpoint = checkpoint.groupby("trace_id", as_index=False).mean(numeric_only=True)
            joined = boundaries.merge(checkpoint, on="trace_id", validate="one_to_one")
            joined[model_column] = model
            pieces.append(joined)
        trace_tables[probe] = pd.concat(pieces, ignore_index=True)

    identity = [model_column, "trace_id"]
    hidden_trace = trace_tables["hidden"].sort_values(identity, kind="mergesort")
    position_trace = trace_tables["position"].sort_values(identity, kind="mergesort")
    if hidden_trace[identity].reset_index(drop=True).equals(
        position_trace[identity].reset_index(drop=True)
    ) is False:
        raise RuntimeError("paired boundary bootstrap trace identities differ")

    mean_columns = {
        "boundary_exact_last_valid_checkpoint_accuracy": "exact",
        "boundary_within_one_checkpoint_accuracy": "within_one",
        "boundary_mean_absolute_checkpoint_error": "absolute_checkpoint_error",
        "boundary_normalized_token_position_error": "normalized_token_position_error",
        "boundary_late_boundary_rate": "late",
        "boundary_early_boundary_rate": "early",
        "boundary_mean_retained_valid_prefix_fraction": "retained_valid_prefix_fraction",
        "boundary_non_root_predicted_boundary_coverage": "non_root",
        "checkpoint_trace_weighted_nll": "checkpoint_trace_weighted_nll",
        "checkpoint_trace_weighted_brier": "checkpoint_trace_weighted_brier",
        "checkpoint_trace_weighted_accuracy": "checkpoint_trace_weighted_accuracy",
    }
    median_columns = {
        "boundary_median_absolute_checkpoint_error": "absolute_checkpoint_error",
    }
    generator = np.random.default_rng(int(seed))
    per_model_points: dict[str, dict[str, float]] = {}
    per_model_samples: dict[str, dict[str, np.ndarray]] = {}
    for model in models:
        per_model_points[model] = {}
        per_model_samples[model] = {}
        probe_parts = {
            probe: table.loc[table[model_column].astype(str).eq(model)]
            .sort_values("trace_id", kind="mergesort")
            .reset_index(drop=True)
            for probe, table in trace_tables.items()
        }
        count = len(probe_parts["hidden"])
        if count == 0 or len(probe_parts["position"]) != count:
            raise RuntimeError(f"{model}: paired bootstrap trace population is empty or unequal")
        draw = generator.integers(0, count, size=(int(replicates), count))
        for metric, column in {**mean_columns, **median_columns}.items():
            for probe in ("hidden", "position"):
                values = probe_parts[probe][column].to_numpy(float)
                reducer = np.median if metric in median_columns else np.mean
                point = float(reducer(values))
                samples = reducer(values[draw], axis=1)
                per_model_points[model][f"{probe}_{metric}"] = point
                per_model_samples[model][f"{probe}_{metric}"] = samples
            contrast = f"hidden_minus_position_{metric}"
            per_model_points[model][contrast] = (
                per_model_points[model][f"hidden_{metric}"]
                - per_model_points[model][f"position_{metric}"]
            )
            per_model_samples[model][contrast] = (
                per_model_samples[model][f"hidden_{metric}"]
                - per_model_samples[model][f"position_{metric}"]
            )

    metric_names = sorted(next(iter(per_model_points.values())))
    intervals: dict[str, Any] = {}
    for metric in metric_names:
        point = float(np.mean([per_model_points[model][metric] for model in models]))
        samples = np.mean(
            np.stack([per_model_samples[model][metric] for model in models], axis=0),
            axis=0,
        )
        intervals[metric] = {
            "estimate": point,
            "ci95": [
                float(np.quantile(samples, 0.025)),
                float(np.quantile(samples, 0.975)),
            ],
            "finite_replicates": int(np.isfinite(samples).sum()),
        }
    return {
        "unit": "complete failed trace",
        "paired_predictions_within_sample": True,
        "models_resampled_independently": True,
        "macro_weighting": "equal model",
        "replicates": int(replicates),
        "seed": int(seed),
        "per_model_point_estimates": per_model_points,
        "macro": intervals,
        "row_identity_matched": True,
        "test_only": True,
        "interval_metric_scope": {
            "included": sorted([*mean_columns, *median_columns]),
            "point_only": [
                "checkpoint_roc_auc",
                "checkpoint_average_precision",
                "checkpoint_balanced_accuracy",
                "checkpoint_trace_weighted_ece",
            ],
            "reason": "nondecomposable ranking/calibration metrics remain held-out point estimates",
        },
    }


Statistic = Callable[[pd.DataFrame], Mapping[str, float] | float]


def _normalize_statistic(value: Mapping[str, float] | float) -> dict[str, float]:
    return (
        {str(key): float(item) for key, item in value.items()}
        if isinstance(value, Mapping)
        else {"value": float(value)}
    )


def paired_trace_bootstrap(
    frame: pd.DataFrame,
    statistic: Statistic,
    *,
    replicates: int = 10_000,
    seed: int = 20260729,
    expected_models: int | None = None,
) -> dict[str, Any]:
    """Resample complete traces per model and macro-average models equally.

    All compared probe columns remain in the same resampled frame, preserving
    paired contrasts.  Model problem sets need not be cross-model paired.
    """

    if frame.empty or "trace_id" not in frame:
        raise ValueError("bootstrap requires nonempty trace-indexed rows")
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    model_column = next(
        (name for name in ("base_model", "model_key", "model_id") if name in frame), None
    )
    if model_column is None:
        working = frame.copy()
        working["_single_model"] = "model"
        model_column = "_single_model"
    else:
        working = frame
    models = sorted(working[model_column].astype(str).unique())
    if expected_models is not None and len(models) != int(expected_models):
        raise RuntimeError(f"expected {expected_models} models, found {len(models)}")
    clusters: dict[str, list[pd.DataFrame]] = {
        model: [part.copy() for _, part in working.loc[working[model_column].astype(str).eq(model)].groupby("trace_id", sort=True)]
        for model in models
    }
    point_by_model = {
        model: _normalize_statistic(statistic(pd.concat(parts, ignore_index=True)))
        for model, parts in clusters.items()
    }
    names = set(next(iter(point_by_model.values())))
    if any(set(value) != names for value in point_by_model.values()):
        raise ValueError("bootstrap statistic keys differ across models")
    point_macro = {
        name: float(np.mean([point_by_model[model][name] for model in models])) for name in names
    }
    generator = np.random.default_rng(seed)
    samples = {name: np.empty(replicates, dtype=float) for name in names}
    for replicate in range(replicates):
        model_values: dict[str, dict[str, float]] = {}
        for model in models:
            source = clusters[model]
            draw = generator.integers(0, len(source), size=len(source))
            pieces: list[pd.DataFrame] = []
            for draw_index, source_index in enumerate(draw):
                piece = source[int(source_index)].copy()
                piece["trace_id"] = f"bootstrap_{model}_{draw_index}"
                pieces.append(piece)
            model_values[model] = _normalize_statistic(
                statistic(pd.concat(pieces, ignore_index=True))
            )
        for name in names:
            samples[name][replicate] = np.mean(
                [model_values[model][name] for model in models]
            )
    intervals: dict[str, Any] = {}
    for name, values in samples.items():
        finite = values[np.isfinite(values)]
        intervals[name] = {
            "estimate": point_macro[name],
            "ci95": (
                [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]
                if len(finite) == replicates
                else [float("nan"), float("nan")]
            ),
            "finite_replicates": int(len(finite)),
        }
    return {
        "unit": "complete failed trace",
        "paired_predictions_within_sample": True,
        "models_resampled_independently": True,
        "macro_weighting": "equal model",
        "replicates": int(replicates),
        "seed": int(seed),
        "per_model_point_estimates": point_by_model,
        "macro": intervals,
    }
