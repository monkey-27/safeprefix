"""Prompt-pilot metrics and paired condition comparisons."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .statistics import paired_bootstrap, paired_equivalence


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()), "minimum": float(array.min()),
        "q25": float(np.quantile(array, 0.25)), "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)), "maximum": float(array.max()),
    }


def _summarize_group(rows: list[dict[str, Any]], *, equivalence_margin: float, seed: int) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_condition.setdefault(str(row["prompt_condition"]), []).append(row)
    summaries = {}
    for condition, values in sorted(by_condition.items()):
        summaries[condition] = {
            "examples": len(values),
            "verified_accuracy": float(np.mean([bool(row["verifier_pass"]) for row in values])),
            "answer_parser_success": float(np.mean([bool(row["answer_parser_success"]) for row in values])),
            "mean_reasoning_tokens": float(np.mean([int(row["reasoning_token_count"]) for row in values])),
            "mean_completion_tokens": float(np.mean([int(row["completion_token_count"]) for row in values])),
            "reasoning_token_distribution": _distribution([int(row["reasoning_token_count"]) for row in values]),
            "completion_token_distribution": _distribution([int(row["completion_token_count"]) for row in values]),
            "span_count_distribution": _distribution([int(row["span_count"]) for row in values]),
            "mean_spans": float(np.mean([int(row["span_count"]) for row in values])),
            "at_least_two_checkpoints": float(np.mean([int(row["span_count"]) >= 2 for row in values])),
            "malformed_or_truncated": float(np.mean([bool(row.get("malformed", False) or row.get("truncated", False)) for row in values])),
            "formatting_adherence": float(np.mean([bool(row.get("formatting_adherent", False)) for row in values])),
        }
    paired = {}
    conditions = sorted(by_condition)
    maps = {}
    for condition, values in by_condition.items():
        by_problem: dict[str, list[float]] = {}
        for row in values:
            by_problem.setdefault(str(row["problem_id"]), []).append(float(row["verifier_pass"]))
        maps[condition] = {problem_id: float(np.mean(outcomes)) for problem_id, outcomes in by_problem.items()}
    for i, left in enumerate(conditions):
        for right in conditions[i + 1 :]:
            common = sorted(set(maps[left]) & set(maps[right]))
            a = [maps[left][key] for key in common]
            b = [maps[right][key] for key in common]
            paired[f"{left}_vs_{right}"] = {
                "accuracy_difference": paired_bootstrap(a, b, seed=seed),
                "equivalence": paired_equivalence(a, b, equivalence_margin, seed=seed),
            }
    return {"conditions": summaries, "paired": paired}


def summarize_prompt_rows(rows: list[dict[str, Any]], *, equivalence_margin: float, seed: int) -> dict[str, Any]:
    aggregate = _summarize_group(rows, equivalence_margin=equivalence_margin, seed=seed)
    aggregate["statistical_unit"] = "underlying_problem"
    aggregate["per_model"] = {
        model_name: _summarize_group(
            [row for row in rows if str(row["model_name"]) == model_name],
            equivalence_margin=equivalence_margin,
            seed=seed,
        )
        for model_name in sorted({str(row["model_name"]) for row in rows})
    }
    return aggregate
