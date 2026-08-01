"""One-shot and budget-matched repair accounting."""

from __future__ import annotations

from typing import Any

import numpy as np


def summarize_repairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"problems": 0}
    initial = [bool(row["initial_verifier_pass"]) for row in rows]
    final = [bool(row["final_verifier_pass"]) for row in rows]
    failures = [index for index, value in enumerate(initial) if not value]
    return {
        "problems": len(rows),
        "initial_pass_at_1": float(np.mean(initial)),
        "final_all_problem_verified_accuracy": float(np.mean(final)),
        "initial_failures": len(failures),
        "repair_success_among_failures": float(np.mean([final[index] for index in failures])) if failures else None,
        "mean_generated_repair_tokens": float(np.mean([row.get("generated_repair_tokens", 0) for row in rows])),
        "mean_prefix_tokens_recomputed": float(np.mean([row.get("prefix_tokens_recomputed", 0) for row in rows])),
        "mean_latency_seconds": float(np.mean([row.get("latency_seconds", 0.0) for row in rows])),
        "mean_verifier_calls": float(np.mean([row.get("verifier_calls", 0) for row in rows])),
        "mean_checkpoint_memory_bytes": float(np.mean([row.get("checkpoint_memory_bytes", 0) for row in rows])),
    }


def selected_checkpoint_regret(rows: list[dict[str, Any]]) -> dict[str, float]:
    regrets, token_regrets, agreements, distances = [], [], [], []
    for row in rows:
        selected = int(row["selected_checkpoint_index"])
        candidates = list(map(int, row["checkpoint_indices"]))
        position = candidates.index(selected)
        values = list(map(float, row["dense_repairability"]))
        best = max(values)
        best_positions = [index for index, value in zip(candidates, values) if value == best]
        regrets.append(best - values[position])
        agreements.append(selected in best_positions)
        distances.append(min(abs(selected - value) for value in best_positions))
        costs = list(map(float, row.get("repair_token_costs", [0] * len(candidates))))
        token_regrets.append(costs[position] - min(costs[index] for index, value in enumerate(values) if value == best))
    return {
        "selected_checkpoint_regret": float(np.mean(regrets)),
        "candidate_set_agreement": float(np.mean(agreements)),
        "token_cost_regret": float(np.mean(token_regrets)),
        "checkpoint_distance_error": float(np.mean(distances)),
    }
