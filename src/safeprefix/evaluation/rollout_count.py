"""Select the smallest rollout count satisfying preregistered non-inferiority."""

from __future__ import annotations

from typing import Any


def choose_rollout_count(
    metrics_by_k: dict[int, dict[str, float]],
    *,
    accuracy_gap: float = 0.01,
    utility_regret_gap: float = 0.025,
    token_cost_gap: float = 0.03,
    further_gain: float = 0.005,
) -> dict[str, Any]:
    if len(metrics_by_k) < 2:
        raise ValueError("at least two nested-k conditions are required")
    ordered = sorted(metrics_by_k)
    reference_k = ordered[-1]
    reference = metrics_by_k[reference_k]
    checks = []
    selected = None
    for position, k in enumerate(ordered):
        current = metrics_by_k[k]
        next_k = ordered[min(position + 1, len(ordered) - 1)]
        later = metrics_by_k[next_k]
        passed = (
            reference["repair_accuracy"] - current["repair_accuracy"] <= accuracy_gap
            and current["utility_regret"] - reference["utility_regret"] <= utility_regret_gap
            and current["repair_token_cost"] <= reference["repair_token_cost"] * (1 + token_cost_gap)
            and later["repair_accuracy"] - current["repair_accuracy"] < further_gain
        )
        checks.append({"k": k, "passed": passed})
        if passed and selected is None:
            selected = k
    return {
        "selected_k": selected,
        "reference_k": reference_k,
        "checks": checks,
        "passed": selected is not None,
    }
