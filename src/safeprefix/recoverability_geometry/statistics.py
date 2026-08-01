"""Cluster-respecting inference for the recoverability-geometry study."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

import numpy as np


ArrayStatistic = Callable[[np.ndarray], float]


def _cluster_values(
    cluster_ids: Iterable[Any], values: Iterable[float]
) -> tuple[list[str], dict[str, np.ndarray]]:
    clusters = [str(value) for value in cluster_ids]
    observations = np.asarray(list(values), dtype=float)
    if observations.ndim != 1 or not len(observations) or len(clusters) != len(observations):
        raise ValueError("cluster ids and values must be aligned nonempty vectors")
    if not np.isfinite(observations).all():
        raise ValueError("bootstrap values must be finite")
    grouped: dict[str, list[float]] = {}
    for cluster, value in zip(clusters, observations, strict=True):
        grouped.setdefault(cluster, []).append(float(value))
    ordered = sorted(grouped)
    return ordered, {key: np.asarray(grouped[key], dtype=float) for key in ordered}


def _array_clustered_bootstrap(
    cluster_ids: Iterable[Any],
    values: Iterable[float],
    *,
    replicates: int = 10_000,
    seed: int = 20260728,
    within_cluster: ArrayStatistic = np.mean,
    across_clusters: ArrayStatistic = np.mean,
) -> dict[str, float | int]:
    """Bootstrap whole traces/parents, never their checkpoint/child rows."""
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    ordered, grouped = _cluster_values(cluster_ids, values)
    collapsed = np.asarray([within_cluster(grouped[key]) for key in ordered], dtype=float)
    if not np.isfinite(collapsed).all():
        raise ValueError("cluster statistic returned a non-finite value")
    generator = np.random.default_rng(seed)
    sampled = np.empty(replicates, dtype=float)
    for index in range(replicates):
        selection = generator.integers(0, len(collapsed), size=len(collapsed))
        sampled[index] = float(across_clusters(collapsed[selection]))
    estimate = float(across_clusters(collapsed))
    # Two-sided bootstrap sign test with a one-replicate finite-sample
    # correction.  This is descriptive inference for pre-registered paired
    # contrasts; it never treats checkpoint rows as independent.
    tail_low = (float(np.count_nonzero(sampled <= 0.0)) + 1.0) / (replicates + 1.0)
    tail_high = (float(np.count_nonzero(sampled >= 0.0)) + 1.0) / (replicates + 1.0)
    return {
        "estimate": estimate,
        "ci_low": float(np.quantile(sampled, 0.025)),
        "ci_high": float(np.quantile(sampled, 0.975)),
        "replicates": int(replicates),
        "clusters": int(len(collapsed)),
        "rows": int(sum(len(grouped[key]) for key in ordered)),
        "seed": int(seed),
        "two_sided_sign_p": float(min(1.0, 2.0 * min(tail_low, tail_high))),
    }


@dataclass(frozen=True)
class BootstrapResult:
    cluster_column: str
    cluster_count: int
    estimates: dict[str, float]
    intervals: dict[str, tuple[float, float]]
    replicate_values: dict[str, list[float]]
    replicates: int
    seed: int

    def to_dict(self, *, include_replicates: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "cluster_column": self.cluster_column,
            "cluster_count": self.cluster_count,
            "estimates": self.estimates,
            "intervals": {key: list(value) for key, value in self.intervals.items()},
            "replicates": self.replicates,
            "seed": self.seed,
        }
        if include_replicates:
            result["replicate_values"] = self.replicate_values
        return result


def _normalize_statistic(value: Any) -> dict[str, float]:
    if isinstance(value, Mapping):
        output = {str(key): float(item) for key, item in value.items()}
    else:
        output = {"value": float(value)}
    if not output or not all(np.isfinite(item) for item in output.values()):
        raise ValueError("bootstrap statistic must return finite scalar values")
    return output


def _dataframe_clustered_bootstrap(
    frame: pd.DataFrame,
    *,
    cluster_col: str,
    statistic: Callable[[pd.DataFrame], Any],
    replicates: int,
    seed: int,
    stratify_col: str | None,
) -> BootstrapResult:
    if frame.empty or cluster_col not in frame:
        raise ValueError("bootstrap frame must be nonempty and contain its cluster column")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    source = frame.copy()
    source[cluster_col] = source[cluster_col].astype(str)
    clusters = sorted(source[cluster_col].unique())
    if stratify_col is not None:
        if stratify_col not in source:
            raise KeyError(stratify_col)
        stratum_counts = source.groupby(cluster_col)[stratify_col].nunique()
        if bool((stratum_counts != 1).any()):
            raise ValueError("each cluster must belong to exactly one stratum")
        cluster_strata = source.groupby(cluster_col)[stratify_col].first().astype(str).to_dict()
        strata = {
            value: sorted([cluster for cluster in clusters if cluster_strata[cluster] == value])
            for value in sorted(set(cluster_strata.values()))
        }
    else:
        strata = {"__all__": clusters}
    estimate = _normalize_statistic(statistic(source))
    values = {key: [] for key in estimate}
    generator = np.random.default_rng(seed)
    for _ in range(replicates):
        pieces: list[pd.DataFrame] = []
        sample_counter = 0
        for stratum_clusters in strata.values():
            sampled = generator.choice(stratum_clusters, size=len(stratum_clusters), replace=True)
            for cluster in sampled:
                piece = source.loc[source[cluster_col] == cluster].copy()
                # Give repeated draws unique bootstrap cluster IDs so a statistic
                # that re-aggregates by the cluster column keeps multiplicity.
                piece[cluster_col] = f"bootstrap_{sample_counter}"
                sample_counter += 1
                pieces.append(piece)
        observed = _normalize_statistic(statistic(pd.concat(pieces, ignore_index=True)))
        if set(observed) != set(estimate):
            raise ValueError("bootstrap statistic keys changed across replicates")
        for key, value in observed.items():
            values[key].append(value)
    intervals = {
        key: (float(np.quantile(items, 0.025)), float(np.quantile(items, 0.975)))
        for key, items in values.items()
    }
    return BootstrapResult(
        cluster_column=cluster_col,
        cluster_count=len(clusters),
        estimates=estimate,
        intervals=intervals,
        replicate_values=values,
        replicates=replicates,
        seed=seed,
    )


def clustered_bootstrap(
    cluster_ids_or_frame: Iterable[Any] | pd.DataFrame,
    values: Iterable[float] | None = None,
    *,
    cluster_col: str | None = None,
    statistic: Callable[..., Any] = np.mean,
    replicates: int = 10_000,
    seed: int = 20260728,
    within_cluster: ArrayStatistic = np.mean,
    across_clusters: ArrayStatistic = np.mean,
    stratify_col: str | None = None,
) -> dict[str, float | int] | BootstrapResult:
    """Dispatch to array or DataFrame cluster-resampling interfaces."""
    if isinstance(cluster_ids_or_frame, pd.DataFrame):
        if cluster_col is None:
            raise ValueError("DataFrame bootstrap requires cluster_col")
        return _dataframe_clustered_bootstrap(
            cluster_ids_or_frame,
            cluster_col=cluster_col,
            statistic=statistic,
            replicates=replicates,
            seed=seed,
            stratify_col=stratify_col,
        )
    if values is None:
        raise ValueError("array bootstrap requires values")
    return _array_clustered_bootstrap(
        cluster_ids_or_frame,
        values,
        replicates=replicates,
        seed=seed,
        within_cluster=within_cluster,
        across_clusters=across_clusters,
    )


def clustered_paired_bootstrap(
    cluster_ids: Iterable[Any],
    left: Iterable[float],
    right: Iterable[float],
    *,
    replicates: int = 10_000,
    seed: int = 20260728,
    within_cluster: ArrayStatistic = np.mean,
) -> dict[str, float | int]:
    """Paired difference with cluster-level resampling."""
    clusters = list(cluster_ids)
    lhs = np.asarray(list(left), dtype=float)
    rhs = np.asarray(list(right), dtype=float)
    if lhs.shape != rhs.shape or lhs.ndim != 1 or len(lhs) != len(clusters):
        raise ValueError("paired inputs must be aligned nonempty vectors")
    result = _array_clustered_bootstrap(
        clusters,
        lhs - rhs,
        replicates=replicates,
        seed=seed,
        within_cluster=within_cluster,
    )
    result["comparison"] = "left_minus_right"
    return result


def shared_trace_model_bootstrap(
    shared_trace_ids: Iterable[Any],
    model_keys: Iterable[Any],
    values: Iterable[float],
    *,
    replicates: int = 10_000,
    seed: int = 20260728,
) -> dict[str, float | int]:
    """Aggregate four-model inference while resampling the shared trace block."""
    traces, models, observations = list(shared_trace_ids), list(model_keys), list(values)
    if not traces or len(traces) != len(models) or len(traces) != len(observations):
        raise ValueError("shared-trace inputs must be aligned and nonempty")
    # A duplicated model row for one shared trace would overweight that model.
    seen: set[tuple[str, str]] = set()
    for trace, model in zip(traces, models, strict=True):
        identity = (str(trace), str(model))
        if identity in seen:
            raise ValueError(f"duplicate shared trace/model row: {identity}")
        seen.add(identity)
    return _array_clustered_bootstrap(
        traces, observations, replicates=replicates, seed=seed
    )


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    """Holm step-down adjusted p-values in original order."""
    values = np.asarray(list(p_values), dtype=float)
    if values.ndim != 1 or not len(values) or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be a nonempty vector in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def holm_primary_families(p_values: Mapping[str, float]) -> dict[str, float]:
    """Apply the pre-registered correction across H1--H4."""
    required = ("H1", "H2", "H3", "H4")
    if set(p_values) != set(required):
        raise ValueError("Holm family input must contain exactly H1, H2, H3, and H4")
    adjusted = holm_adjust([p_values[key] for key in required])
    return dict(zip(required, adjusted, strict=True))


def trace_clustered_bootstrap(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], Any],
    *,
    replicates: int = 10_000,
    seed: int = 20260728,
    stratify_col: str | None = None,
) -> BootstrapResult:
    result = clustered_bootstrap(
        frame,
        cluster_col="trace_id",
        statistic=statistic,
        replicates=replicates,
        seed=seed,
        stratify_col=stratify_col,
    )
    assert isinstance(result, BootstrapResult)
    return result


def parent_clustered_bootstrap(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], Any],
    *,
    replicates: int = 10_000,
    seed: int = 20260728,
    stratify_col: str | None = None,
) -> BootstrapResult:
    result = clustered_bootstrap(
        frame,
        cluster_col="parent_id",
        statistic=statistic,
        replicates=replicates,
        seed=seed,
        stratify_col=stratify_col,
    )
    assert isinstance(result, BootstrapResult)
    return result


def holm_correction(
    p_values: Mapping[str, float] | Iterable[float], *, alpha: float = 0.05
) -> dict[str, dict[str, float | bool]] | list[dict[str, float | bool]]:
    if isinstance(p_values, Mapping):
        keys = list(p_values)
        original = [float(p_values[key]) for key in keys]
        adjusted = holm_adjust(original)
        return {
            key: {"raw_p": raw, "holm_p": corrected, "reject": bool(corrected <= alpha)}
            for key, raw, corrected in zip(keys, original, adjusted, strict=True)
        }
    original = list(map(float, p_values))
    adjusted = holm_adjust(original)
    return [
        {"raw_p": raw, "holm_p": corrected, "reject": bool(corrected <= alpha)}
        for raw, corrected in zip(original, adjusted, strict=True)
    ]
