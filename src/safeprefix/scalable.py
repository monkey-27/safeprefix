"""Shared cohort, allocation, runtime, and artifact rules for scalable runs."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping

from safeprefix.reproducibility import stable_hash


SOURCE_BUCKETS = (
    "crv_arithmetic",
    "processbench_math",
    "processbench_olympiadbench",
    "processbench_omnimath",
)

NATIVE_BUCKETS = (
    "gsm8k",
    "math_level",
    "olympiad_or_omnimath",
    "hard_synthetic_arithmetic",
)


def source_bucket(row: Mapping[str, Any]) -> str | None:
    dataset = str(row.get("source_dataset", "")).casefold()
    subset = str(row.get("source_subset", "")).casefold().replace("-", "_")
    if "facebook/crv" in dataset and ("arithmetic" in subset or subset.startswith("arith.")):
        return "crv_arithmetic"
    if "processbench" not in dataset:
        return None
    if "olympiad" in subset:
        return "processbench_olympiadbench"
    if "omnimath" in subset or "omni_math" in subset:
        return "processbench_omnimath"
    if subset == "math" or subset.endswith("/math"):
        return "processbench_math"
    return None


def native_bucket(row: Mapping[str, Any]) -> str | None:
    dataset = str(row.get("source_dataset", "")).casefold()
    subset = str(row.get("source_subset", "")).casefold().replace("-", "_")
    if "facebook/crv" in dataset and ("arithmetic" in subset or subset.startswith("arith.")):
        return "hard_synthetic_arithmetic"
    if "processbench" not in dataset:
        return None
    if "gsm8k" in subset:
        return "gsm8k"
    if "olympiad" in subset or "omnimath" in subset or "omni_math" in subset:
        return "olympiad_or_omnimath"
    if subset == "math" or subset.endswith("/math"):
        return "math_level"
    return None


def select_native_problem_pool(
    rows: Iterable[Mapping[str, Any]], config: Mapping[str, Any], *, mock: bool = False
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return the preregistered initial pool followed by deterministic hard extensions."""
    values = _one_trace_per_problem(rows, int(config["seed"]), "native-pool")
    if mock:
        return values, []
    initial_counts = config["native"]["initial_pool"]
    initial: list[dict[str, Any]] = []
    shortfalls = []
    used: set[str] = set()
    for bucket in NATIVE_BUCKETS:
        candidates = [row for row in values if native_bucket(row) == bucket]
        candidates.sort(key=lambda row: _rank(row, int(config["seed"]), f"native:{bucket}"))
        retained = candidates[: int(initial_counts[bucket])]
        if len(retained) < int(initial_counts[bucket]):
            shortfalls.append(
                {"source_bucket": bucket, "requested": int(initial_counts[bucket]), "available": len(candidates)}
            )
        for row in retained:
            row["native_bucket"] = bucket
            row["native_pool_phase"] = "initial"
            used.add(str(row.get("problem_group_hash") or row.get("problem_id")))
        initial.extend(retained)
    if shortfalls:
        raise RuntimeError(f"native evaluation initial-pool shortfall: {shortfalls}")
    remaining = [
        row for row in values
        if str(row.get("problem_group_hash") or row.get("problem_id")) not in used
    ]
    remaining.sort(
        key=lambda row: (
            0 if native_bucket(row) in {"olympiad_or_omnimath", "hard_synthetic_arithmetic"} else 1,
            -len(row.get("reasoning_steps") or []),
            _rank(row, int(config["seed"]), "native-extension"),
        )
    )
    for row in remaining:
        row["native_bucket"] = native_bucket(row)
        row["native_pool_phase"] = "hard_extension"
    maximum = int(config["native"]["maximum_problems_per_model"])
    return (initial + remaining)[:maximum], shortfalls


def _rank(row: Mapping[str, Any], seed: int, purpose: str) -> str:
    identity = row.get("source_trace_id") or row.get("trace_id") or row.get("problem_id")
    return stable_hash(["safeprefix-scalable", int(seed), purpose, identity])


def _boolean_value(value: Any, expected: bool) -> bool:
    """Compare Python and parquet/numpy scalar booleans without identity checks."""
    try:
        return bool(value == expected)
    except (TypeError, ValueError):
        return False


def _present_scalar(value: Any) -> bool:
    """Return false for None, NaN, and pandas missing scalar values."""
    if value is None:
        return False
    try:
        return bool(value == value)
    except (TypeError, ValueError):
        return False


def _one_trace_per_problem(rows: Iterable[Mapping[str, Any]], seed: int, purpose: str) -> list[dict[str, Any]]:
    ordered = sorted((dict(row) for row in rows), key=lambda row: _rank(row, seed, purpose))
    seen: set[str] = set()
    output = []
    for row in ordered:
        group = str(row.get("problem_group_hash") or row.get("problem_id"))
        if group in seen:
            continue
        seen.add(group)
        output.append(row)
    return output


def _select_composition(
    rows: Iterable[Mapping[str, Any]],
    composition: Mapping[str, int],
    *,
    seed: int,
    purpose: str,
    strict: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    values = list(rows)
    selected: list[dict[str, Any]] = []
    shortfalls: list[dict[str, Any]] = []
    for bucket, requested in composition.items():
        candidates = _one_trace_per_problem(
            (row for row in values if source_bucket(row) == bucket), seed, f"{purpose}:{bucket}"
        )
        retained = candidates[: int(requested)]
        for row in retained:
            row["source_bucket"] = bucket
        selected.extend(retained)
        if len(retained) < int(requested):
            shortfalls.append(
                {"purpose": purpose, "source_bucket": bucket, "requested": int(requested), "available": len(candidates)}
            )
    if strict and shortfalls:
        raise RuntimeError(f"teacher-forced cohort composition shortfall: {shortfalls}")
    return selected, shortfalls


def proportional_counts(total: int, buckets: Iterable[str]) -> dict[str, int]:
    names = list(buckets)
    if total < 0 or not names:
        raise ValueError("proportional allocation requires a nonnegative total and buckets")
    base, remainder = divmod(int(total), len(names))
    return {name: base + int(index < remainder) for index, name in enumerate(names)}


def select_teacher_forced_cohort(
    rows: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    mock: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select one disjoint, deterministic cohort for every target model."""
    values = [dict(row) for row in rows]
    seed = int(config["seed"])
    if mock:
        failed = [row for row in values if _boolean_value(row.get("final_answer_correct"), False) and _present_scalar(row.get("first_error_index"))]
        successful = [row for row in values if _boolean_value(row.get("final_answer_correct"), True)]
        train = _one_trace_per_problem((row for row in failed if row.get("split") == "train"), seed, "mock-train")[:4]
        dev = _one_trace_per_problem((row for row in failed if row.get("split") == "dev"), seed, "mock-dev")[:2]
        dense = _one_trace_per_problem((row for row in failed if row.get("split") == "dense_audit"), seed, "mock-dense")[:2]
        safety = _one_trace_per_problem((row for row in successful if row.get("split") == "train"), seed, "mock-safety")[:2]
        groups = [(train, "train", True), (dev, "dev", True), (dense, "dense_teacher_forced", True), (safety, "safety_train", False)]
        output = []
        for group, pipeline_split, rollout_eligible in groups:
            for row in group:
                row.update(pipeline_split=pipeline_split, rollout_eligible=rollout_eligible, semantic_safety_only=not rollout_eligible, source_bucket=source_bucket(row) or "mock")
                output.append(row)
        return output, []

    failed = [
        row for row in values
        if _boolean_value(row.get("final_answer_correct"), False) and _present_scalar(row.get("first_error_index"))
    ]
    composition = config["teacher_forced"]["failed_composition"]
    train, train_shortfalls = _select_composition(
        (row for row in failed if row.get("split") == "train"),
        composition["train"], seed=seed, purpose="train", strict=True,
    )
    dev, dev_shortfalls = _select_composition(
        (row for row in failed if row.get("split") == "dev"),
        composition["dev"], seed=seed, purpose="dev", strict=True,
    )
    dense_counts = proportional_counts(
        int(config["sample_counts"]["dense_teacher_forced_failures_per_model"]),
        config["teacher_forced"]["dense_source_proportions"],
    )
    dense, dense_shortfalls = _select_composition(
        (row for row in failed if row.get("split") == "dense_audit"),
        dense_counts, seed=seed, purpose="dense_teacher_forced", strict=True,
    )
    used = {str(row.get("problem_group_hash")) for row in (*train, *dev, *dense)}
    successful_pool = [
        row for row in values
        if row.get("split") == "train"
        and _boolean_value(row.get("final_answer_correct"), True)
        and str(row.get("problem_group_hash")) not in used
    ]
    successful = _one_trace_per_problem(successful_pool, seed, "successful-safety")
    successful = successful[: int(config["sample_counts"]["successful_safety_traces_per_model"])]
    if len(successful) < int(config["sample_counts"]["successful_safety_traces_per_model"]):
        raise RuntimeError(
            "successful semantic-safety cohort shortfall: "
            f"requested {config['sample_counts']['successful_safety_traces_per_model']}, available {len(successful)}"
        )
    groups = [(train, "train", True), (dev, "dev", True), (dense, "dense_teacher_forced", True), (successful, "safety_train", False)]
    output = []
    for group, pipeline_split, rollout_eligible in groups:
        for row in group:
            row.update(
                pipeline_split=pipeline_split,
                rollout_eligible=rollout_eligible,
                semantic_safety_only=not rollout_eligible,
                source_bucket=source_bucket(row),
            )
            output.append(row)
    return output, [*train_shortfalls, *dev_shortfalls, *dense_shortfalls]


def select_stage_a_trace_ids(index_rows: Iterable[Mapping[str, Any]], count: int, seed: int) -> set[str]:
    eligible = [row for row in index_rows if row.get("pipeline_split") == "train" and row.get("rollout_eligible", True)]
    ordered = sorted(eligible, key=lambda row: _rank(row, seed, "stage-a"))
    if len(ordered) < int(count):
        raise RuntimeError(f"Stage A requires {count} training failures, found {len(ordered)}")
    return {str(row["trace_id"]) for row in ordered[: int(count)]}


def select_k6_trace_ids(
    index_rows: Iterable[Mapping[str, Any]],
    rollout_rows: Iterable[Mapping[str, Any]],
    *,
    count: int,
    seed: int,
) -> set[str]:
    """Stratify by source/checkpoint count, then prioritize uncertainty."""
    outcomes: dict[tuple[str, int], list[bool]] = defaultdict(list)
    for row in sorted(rollout_rows, key=lambda item: (str(item["trace_id"]), int(item["checkpoint_index"]), int(item["rollout_index"]))):
        if int(row["rollout_index"]) < 4:
            outcomes[(str(row["trace_id"]), int(row["checkpoint_index"]))].append(bool(row["verifier_pass"]))
    strata: dict[tuple[str, str], list[tuple[float, str, str]]] = defaultdict(list)
    for row in index_rows:
        if row.get("pipeline_split") != "train" or not row.get("rollout_eligible", True):
            continue
        trace_id = str(row["trace_id"])
        checkpoint_values = [values for (candidate_trace, _checkpoint), values in outcomes.items() if candidate_trace == trace_id]
        if not checkpoint_values or any(len(values) < 4 for values in checkpoint_values):
            continue
        probabilities = [(sum(values) + 0.5) / 5.0 for values in checkpoint_values]
        uncertainty = sum(1.0 - 2.0 * abs(value - 0.5) for value in probabilities) / len(probabilities)
        candidate_count = len(checkpoint_values)
        count_bin = "1_2" if candidate_count <= 2 else "3_4" if candidate_count <= 4 else "5_6"
        rank = stable_hash(["safeprefix-k6", int(seed), trace_id])
        strata[(str(row.get("source_bucket")), count_bin)].append((-uncertainty, rank, trace_id))
    for values in strata.values():
        values.sort()
    selected: list[str] = []
    keys = sorted(strata)
    while len(selected) < int(count):
        progressed = False
        for key in keys:
            if strata[key] and len(selected) < int(count):
                selected.append(strata[key].pop(0)[2])
                progressed = True
        if not progressed:
            break
    if len(selected) < int(count):
        raise RuntimeError(f"k=6 selection requires {count} eligible traces, found {len(selected)}")
    return set(selected)


def runtime_gate(seconds: float, thresholds_minutes: Mapping[str, float]) -> str:
    minutes = float(seconds) / 60.0
    if minutes <= float(thresholds_minutes["excellent"]):
        return "EXCELLENT"
    if minutes <= float(thresholds_minutes["pass"]):
        return "PASS"
    if minutes <= float(thresholds_minutes["warning"]):
        return "WARNING"
    if minutes > float(thresholds_minutes["fail"]):
        return "FAIL"
    return "STOP_OPTIMIZE"


def percentile(values: Iterable[int | float], quantile: float) -> float | None:
    sequence = sorted(map(float, values))
    if not sequence:
        return None
    position = (len(sequence) - 1) * float(quantile)
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return sequence[low]
    weight = position - low
    return sequence[low] * (1 - weight) + sequence[high] * weight
