"""Paired problem-level inference; checkpoints and branches are never units."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Iterable

import numpy as np
from scipy.stats import binomtest


def paired_bootstrap(
    left: Iterable[float],
    right: Iterable[float],
    *,
    replicates: int = 10_000,
    seed: int = 0,
    statistic: Callable[[np.ndarray], float] = np.mean,
) -> dict[str, float]:
    a, b = np.asarray(list(left), dtype=float), np.asarray(list(right), dtype=float)
    if a.shape != b.shape or a.ndim != 1 or len(a) == 0:
        raise ValueError("paired bootstrap inputs must be aligned nonempty vectors")
    differences = a - b
    generator = np.random.default_rng(seed)
    sampled = np.empty(replicates)
    for index in range(replicates):
        positions = generator.integers(0, len(differences), len(differences))
        sampled[index] = statistic(differences[positions])
    return {
        "estimate": float(statistic(differences)),
        "ci_low": float(np.quantile(sampled, 0.025)),
        "ci_high": float(np.quantile(sampled, 0.975)),
        "replicates": int(replicates),
        "problems": int(len(differences)),
    }


def clustered_paired_bootstrap(
    cluster_ids: Iterable[Any],
    left: Iterable[float],
    right: Iterable[float],
    *,
    replicates: int = 10_000,
    seed: int = 0,
) -> dict[str, float | int]:
    """Bootstrap paired differences after collapsing rows to problems."""
    clusters, a, b = list(cluster_ids), list(left), list(right)
    if not clusters or len(clusters) != len(a) or len(a) != len(b):
        raise ValueError("clustered paired bootstrap inputs must be aligned and nonempty")
    values: dict[str, list[float]] = defaultdict(list)
    for cluster, lhs, rhs in zip(clusters, a, b):
        values[str(cluster)].append(float(lhs) - float(rhs))
    problem_differences = np.asarray(
        [np.mean(values[key]) for key in sorted(values)], dtype=float
    )
    generator = np.random.default_rng(seed)
    sampled = np.empty(int(replicates))
    for index in range(int(replicates)):
        positions = generator.integers(0, len(problem_differences), len(problem_differences))
        sampled[index] = float(np.mean(problem_differences[positions]))
    return {
        "estimate": float(np.mean(problem_differences)),
        "ci_low": float(np.quantile(sampled, 0.025)),
        "ci_high": float(np.quantile(sampled, 0.975)),
        "replicates": int(replicates),
        "problems": int(len(problem_differences)),
        "model_problem_rows": int(len(clusters)),
    }


def mcnemar_exact(left: Iterable[bool], right: Iterable[bool]) -> dict[str, float | int]:
    a, b = list(map(bool, left)), list(map(bool, right))
    if len(a) != len(b) or not a:
        raise ValueError("McNemar inputs must be aligned and nonempty")
    left_only = sum(x and not y for x, y in zip(a, b))
    right_only = sum(y and not x for x, y in zip(a, b))
    discordant = left_only + right_only
    p = 1.0 if discordant == 0 else float(binomtest(min(left_only, right_only), discordant, 0.5, alternative="two-sided").pvalue)
    return {"left_only": left_only, "right_only": right_only, "discordant": discordant, "p_value": p}


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = np.asarray(list(p_values), dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def paired_equivalence(left: Iterable[float], right: Iterable[float], margin: float, **kwargs: object) -> dict[str, float | bool]:
    if margin <= 0:
        raise ValueError("equivalence margin must be positive")
    result = paired_bootstrap(left, right, **kwargs)
    result["margin"] = float(margin)
    result["equivalent"] = bool(result["ci_low"] > -margin and result["ci_high"] < margin)
    return result
