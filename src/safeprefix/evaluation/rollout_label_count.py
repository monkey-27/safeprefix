"""Metrics and immutable cohorts for the 4/6/8 rollout-label pilot."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

import numpy as np

from safeprefix.reproducibility import stable_hash
from safeprefix.scalable import source_bucket


def select_label_count_cohort(
    rows: Iterable[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select a deterministic, source-stratified, problem-disjoint cohort."""
    spec = config["rollout_label_count_pilot"]["composition"]
    seed = int(config["seed"])
    selected: list[dict[str, Any]] = []
    shortfalls: list[dict[str, Any]] = []
    used: set[str] = set()
    values = [dict(row) for row in rows]
    for split, composition in spec.items():
        for bucket, requested in composition.items():
            candidates = []
            for row in values:
                identity = str(row.get("problem_group_hash") or row["problem_id"])
                if identity in used or str(row.get("split")) != str(split):
                    continue
                observed_bucket = source_bucket(row)
                if observed_bucket is None and str(row.get("source_dataset", "")).startswith("mock/"):
                    observed_bucket = "mock"
                if observed_bucket != str(bucket):
                    continue
                final_correct = row.get("final_answer_correct")
                if final_correct is None or bool(final_correct):
                    continue
                error = row.get("first_error_index")
                if error is None or error != error:
                    continue
                candidates.append(row)
            candidates.sort(
                key=lambda row: stable_hash(
                    [
                        "rollout-label-count-v1",
                        seed,
                        split,
                        bucket,
                        row.get("source_trace_id") or row["problem_id"],
                    ]
                )
            )
            retained = candidates[: int(requested)]
            if len(retained) != int(requested):
                shortfalls.append(
                    {
                        "split": str(split),
                        "source_bucket": str(bucket),
                        "requested": int(requested),
                        "available": len(candidates),
                    }
                )
            for row in retained:
                identity = str(row.get("problem_group_hash") or row["problem_id"])
                used.add(identity)
                row.update(
                    pipeline_split=str(split),
                    rollout_eligible=str(split) in {"train", "dev"},
                    semantic_safety_only=False,
                    source_bucket=str(bucket),
                )
                selected.append(row)
    return selected, shortfalls


def unsafe_span_records(
    rows: Iterable[Mapping[str, Any]], tau: float
) -> list[dict[str, Any]]:
    """Turn per-checkpoint safety probabilities into first-unsafe predictions."""
    output = []
    for row in rows:
        indices = list(map(int, row["checkpoint_indices"]))
        labels = list(map(float, row["safety_labels"]))
        probabilities = list(map(float, row["safety_probabilities"]))
        if not indices or not (len(indices) == len(labels) == len(probabilities)):
            raise ValueError("unsafe-span rows must contain aligned nonempty sequences")
        unsafe_positions = [index for index, label in zip(indices, labels) if label < 0.5]
        if not unsafe_positions:
            continue
        truth = unsafe_positions[0]
        predicted_positions = [
            index for index, probability in zip(indices, probabilities) if probability < float(tau)
        ]
        predicted = predicted_positions[0] if predicted_positions else max(indices) + 1
        error = predicted - truth
        output.append(
            {
                "problem_id": str(row["problem_id"]),
                "trace_id": str(row["trace_id"]),
                "true_first_unsafe_checkpoint": int(truth),
                "predicted_first_unsafe_checkpoint": int(predicted),
                "signed_checkpoint_error": int(error),
                "absolute_checkpoint_error": int(abs(error)),
                "exact": bool(error == 0),
                "within_one": bool(abs(error) <= 1),
                "early": bool(error < 0),
                "late": bool(error > 0),
                "no_crossing": bool(not predicted_positions),
            }
        )
    return output


def unsafe_span_metrics(rows: Iterable[Mapping[str, Any]], tau: float) -> dict[str, Any]:
    records = unsafe_span_records(rows, tau)
    if not records:
        return {
            "traces": 0,
            "exact_first_unsafe_accuracy": None,
            "within_one_accuracy": None,
            "mean_absolute_checkpoint_error": None,
            "early_rate": None,
            "late_rate": None,
            "no_crossing_rate": None,
        }
    return {
        "traces": len(records),
        "exact_first_unsafe_accuracy": float(np.mean([row["exact"] for row in records])),
        "within_one_accuracy": float(np.mean([row["within_one"] for row in records])),
        "mean_absolute_checkpoint_error": float(
            np.mean([row["absolute_checkpoint_error"] for row in records])
        ),
        "early_rate": float(np.mean([row["early"] for row in records])),
        "late_rate": float(np.mean([row["late"] for row in records])),
        "no_crossing_rate": float(np.mean([row["no_crossing"] for row in records])),
    }


def tune_unsafe_tau(
    rows: Iterable[Mapping[str, Any]], taus: Iterable[float]
) -> dict[str, Any]:
    values = list(rows)
    candidates = []
    for tau in map(float, taus):
        metrics = unsafe_span_metrics(values, tau)
        if not metrics["traces"]:
            continue
        score = (
            float(metrics["exact_first_unsafe_accuracy"]),
            float(metrics["within_one_accuracy"]),
            -float(metrics["mean_absolute_checkpoint_error"]),
            -float(metrics["late_rate"]),
            -tau,
        )
        candidates.append((score, tau, metrics))
    if not candidates:
        raise ValueError("no development traces contain an unsafe checkpoint")
    _score, tau, metrics = max(candidates, key=lambda value: value[0])
    return {"tau": tau, "development_metrics": metrics}


def nested_repairability_stability(
    rollouts: Iterable[Mapping[str, Any]],
    *,
    counts: Iterable[int] = (4, 6, 8),
    alpha: float = 0.5,
    beta: float = 0.5,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compare nested repairability estimates without calling them safety labels."""
    ordered_counts = sorted(set(map(int, counts)))
    reference = ordered_counts[-1]
    grouped: dict[tuple[str, int], list[bool]] = defaultdict(list)
    problems: dict[str, str] = {}
    for row in sorted(
        rollouts,
        key=lambda value: (
            str(value["trace_id"]),
            int(value["checkpoint_index"]),
            int(value["rollout_index"]),
        ),
    ):
        trace = str(row["trace_id"])
        grouped[(trace, int(row["checkpoint_index"]))].append(bool(row["verifier_pass"]))
        problems[trace] = str(row.get("problem_id", trace))
    eligible = {key: values for key, values in grouped.items() if len(values) >= reference}
    if not eligible:
        return {"status": "UNAVAILABLE", "reason": f"no checkpoint has {reference} outcomes"}
    means: dict[int, dict[tuple[str, int], float]] = {}
    for count in ordered_counts:
        means[count] = {
            key: (sum(values[:count]) + alpha) / (count + alpha + beta)
            for key, values in eligible.items()
        }
    per_k: dict[str, Any] = {}
    reference_means = means[reference]
    for count in ordered_counts:
        differences = [abs(means[count][key] - reference_means[key]) for key in eligible]
        agreements = [
            (means[count][key] >= threshold) == (reference_means[key] >= threshold)
            for key in eligible
        ]
        per_k[str(count)] = {
            "checkpoint_count": len(eligible),
            "posterior_mean_absolute_difference_from_k8": float(np.mean(differences)),
            "repairability_threshold_agreement_with_k8": float(np.mean(agreements)),
        }
    trace_keys: dict[str, list[int]] = defaultdict(list)
    for trace, checkpoint in eligible:
        trace_keys[trace].append(checkpoint)
    reference_boundaries = {}
    for trace, indices in trace_keys.items():
        ordered = sorted(indices)
        below = [index for index in ordered if reference_means[(trace, index)] < threshold]
        reference_boundaries[trace] = below[0] if below else max(ordered) + 1
    for count in ordered_counts:
        exact = []
        distances = []
        for trace, indices in trace_keys.items():
            ordered = sorted(indices)
            below = [index for index in ordered if means[count][(trace, index)] < threshold]
            boundary = below[0] if below else max(ordered) + 1
            exact.append(boundary == reference_boundaries[trace])
            distances.append(abs(boundary - reference_boundaries[trace]))
        per_k[str(count)].update(
            {
                "trace_count": len(exact),
                "operational_repairability_boundary_agreement_with_k8": float(np.mean(exact)),
                "operational_repairability_boundary_mae_from_k8": float(np.mean(distances)),
            }
        )
    return {
        "status": "COMPLETE",
        "counts": ordered_counts,
        "reference_k": reference,
        "threshold": float(threshold),
        "prior": {"alpha": float(alpha), "beta": float(beta)},
        "semantic_warning": (
            "This operational repairability boundary is not the annotated first visible error. "
            "Rollouts supervise repairability; dataset annotations supervise semantic safety."
        ),
        "per_k": per_k,
    }
