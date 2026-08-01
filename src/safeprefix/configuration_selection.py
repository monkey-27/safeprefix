"""Immutable cohorts and preregistered decisions for configuration selection."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from safeprefix.reproducibility import stable_hash
from safeprefix.scalable import native_bucket, source_bucket
from safeprefix.manifests import decode_reference_answer


class FinalTestAccessError(RuntimeError):
    """Raised before any final-test prompt or output can be read or written."""


@dataclass(frozen=True)
class ModelAccessRecord:
    model_key: str
    model_id: str
    requested_revision: str
    resolved_revision: str | None
    tokenizer_id: str
    requested_tokenizer_revision: str
    resolved_tokenizer_revision: str | None
    accessible: bool
    error_type: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return vars(self).copy()


def assert_final_test_locked(config: Mapping[str, Any], *, operation: str, path: str | Path | None = None) -> None:
    guard = config.get("final_test_guard", {})
    lowered = f"{operation} {path or ''}".casefold()
    forbidden = [str(value).casefold() for value in guard.get("forbidden_artifact_terms", [])]
    if any(value in lowered for value in forbidden):
        raise FinalTestAccessError(f"final-test artifact access is locked: {operation}")
    if "final" in lowered and "test" in lowered:
        if not all(bool(guard.get(key, False)) for key in ("allow_prompt_loading", "allow_generation", "allow_verifier_outputs")):
            raise FinalTestAccessError(f"final-test access is locked: {operation}")


def _identity(row: Mapping[str, Any]) -> str:
    return str(row.get("problem_group_hash") or row.get("problem_id"))


def _trace_identity(row: Mapping[str, Any]) -> str:
    return str(row.get("source_trace_id") or row.get("trace_id") or row.get("problem_id"))


def _rank(row: Mapping[str, Any], seed: int, role: str) -> str:
    return stable_hash(["safeprefix-configuration-selection-v1", int(seed), role, _trace_identity(row)])


def _is_failed_annotated(row: Mapping[str, Any]) -> bool:
    correct = row.get("final_answer_correct")
    error = row.get("first_error_index")
    try:
        present = error is not None and bool(error == error)
    except (TypeError, ValueError):
        present = False
    return bool(correct == False) and present  # noqa: E712


def _has_reference_answer(row: Mapping[str, Any]) -> bool:
    value = decode_reference_answer(row.get("reference_answer"))
    return value is not None and value != "" and value != []


def _unique_rows(rows: Iterable[Mapping[str, Any]], seed: int, role: str) -> list[dict[str, Any]]:
    ordered = sorted((dict(row) for row in rows), key=lambda row: _rank(row, seed, role))
    seen: set[str] = set()
    result = []
    for row in ordered:
        identity = _identity(row)
        if identity in seen:
            continue
        seen.add(identity)
        result.append(row)
    return result


def _select_composition(
    rows: Iterable[Mapping[str, Any]],
    composition: Mapping[str, int],
    *,
    seed: int,
    role: str,
    bucket_function: Any,
    excluded: set[str],
) -> list[dict[str, Any]]:
    values = [dict(row) for row in rows if _identity(row) not in excluded]
    selected: list[dict[str, Any]] = []
    shortfalls = []
    for bucket, requested in composition.items():
        candidates = _unique_rows((row for row in values if bucket_function(row) == bucket), seed, f"{role}:{bucket}")
        retained = candidates[: int(requested)]
        if len(retained) != int(requested):
            shortfalls.append({"role": role, "bucket": bucket, "requested": int(requested), "available": len(candidates)})
        for row in retained:
            row.update(manifest_role=role, manifest_bucket=bucket)
        selected.extend(retained)
        excluded.update(_identity(row) for row in retained)
    if shortfalls:
        raise RuntimeError(f"immutable cohort shortfall: {shortfalls}")
    return selected


def _select_mixed_teacher_forced(
    rows: Iterable[Mapping[str, Any]],
    entry: Mapping[str, Any],
    *,
    seed: int,
    role: str,
    excluded: set[str],
) -> list[dict[str, Any]]:
    """Select a mixed teacher-forced cohort with a guaranteed failed floor.

    Safety supervision is defined on both successful and failed traces. Only
    the separately configured failed floor is reserved for later repairability
    rollout selection. Treating every teacher-forced row as failed would exceed
    the number of annotated failures in the pinned corpora and is not part of
    the configuration-pilot protocol.
    """
    values = [dict(row) for row in rows if source_bucket(row) is not None]
    minimum_failed = {str(key): int(value) for key, value in entry.get("minimum_failed_composition", {}).items()}
    composition = {str(key): int(value) for key, value in entry["composition"].items()}
    unknown = set(minimum_failed) - set(composition)
    if unknown:
        raise ValueError(f"{role} failed floors have unknown buckets: {sorted(unknown)}")
    if any(minimum_failed.get(key, 0) > count for key, count in composition.items()):
        raise ValueError(f"{role} failed floor exceeds total bucket composition")
    failed = [row for row in values if _is_failed_annotated(row) and _has_reference_answer(row)]
    selected = _select_composition(
        failed,
        minimum_failed,
        seed=seed,
        role=f"{role}_failed_floor",
        bucket_function=source_bucket,
        excluded=excluded,
    )
    selected_ids = {_identity(row) for row in selected}
    fill_counts = {key: count - minimum_failed.get(key, 0) for key, count in composition.items()}
    selected.extend(_select_composition(
        values,
        fill_counts,
        seed=seed,
        role=role,
        bucket_function=source_bucket,
        excluded=excluded,
    ))
    if len({_identity(row) for row in selected}) != len(selected):
        raise AssertionError(f"{role} contains duplicate problems")
    for row in selected:
        row["manifest_role"] = role
        row["minimum_failed_floor_member"] = _identity(row) in selected_ids
    return selected


def build_immutable_manifests(rows: Iterable[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Create all problem-level partitions before scientific generation.

    Final-test reservations contain identity and provenance only. Prompt text,
    references, reasoning traces, and outputs are deliberately omitted.
    """
    values = [dict(row) for row in rows]
    seed = int(config["seed"])
    excluded: set[str] = set()
    reservation_fields = list(config["final_test_guard"]["reservation_fields"])
    reservation_count = int(config["final_test_guard"]["reservation_count"])
    final_candidates = _unique_rows(values, seed, "final-test-reservation")
    if len(final_candidates) < reservation_count:
        raise RuntimeError(f"final-test reservation requires {reservation_count} unique problems")
    reservations = []
    for row in final_candidates[:reservation_count]:
        reservation = {field: row.get(field) for field in reservation_fields}
        reservation["manifest_role"] = "final_test_reservation"
        reservations.append(reservation)
        excluded.add(_identity(row))

    referenced_values = [row for row in values if _has_reference_answer(row)]
    prompt = _select_composition(
        referenced_values,
        config["phase_a"]["composition"],
        seed=seed,
        role="prompt_segmentation",
        bucket_function=native_bucket,
        excluded=excluded,
    )
    native_dev = _select_composition(
        referenced_values,
        config["immutable_splits"]["native_configuration_development"]["initial_pool"],
        seed=seed,
        role="native_configuration_development",
        bucket_function=native_bucket,
        excluded=excluded,
    )
    train = _select_mixed_teacher_forced(
        values,
        config["immutable_splits"]["teacher_forced_train"],
        seed=seed,
        role="teacher_forced_train",
        excluded=excluded,
    )
    dev = _select_mixed_teacher_forced(
        values,
        config["immutable_splits"]["teacher_forced_dev"],
        seed=seed,
        role="teacher_forced_dev",
        excluded=excluded,
    )
    manifests = {
        "final_test_reservations": reservations,
        "prompt_segmentation": prompt,
        "native_configuration_development": native_dev,
        "teacher_forced_train": train,
        "teacher_forced_dev": dev,
    }
    assert_manifest_disjoint(manifests)
    return manifests


def assert_manifest_disjoint(manifests: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    roles: dict[str, set[str]] = defaultdict(set)
    for role, rows in manifests.items():
        for row in rows:
            roles[_identity(row)].add(str(role))
    overlap = {identity: sorted(values) for identity, values in roles.items() if len(values) > 1}
    if overlap:
        sample = dict(list(overlap.items())[:10])
        raise ValueError(f"problem-level configuration-pilot leakage: {sample}")


def select_phase_c_traces(rows: Iterable[Mapping[str, Any]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return _select_composition(
        (row for row in rows if _is_failed_annotated(row) and _has_reference_answer(row)),
        config["phase_c"]["composition"],
        seed=int(config["seed"]),
        role="phase_c_rollout_count",
        bucket_function=source_bucket,
        excluded=set(),
    )


def select_configuration_teacher_forced_cohort(
    rows: Iterable[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return the frozen 5,100/800 cohort with a disjoint dense-dev subset."""
    manifests = build_immutable_manifests(rows, config)
    train = [dict(row) for row in manifests["teacher_forced_train"]]
    dev = [dict(row) for row in manifests["teacher_forced_dev"]]
    dense_total = int(config["phase_c"]["dense_failures_per_model"])
    buckets = list(config["phase_c"]["composition"])
    base, remainder = divmod(dense_total, len(buckets))
    dense_composition = {bucket: base + int(index < remainder) for index, bucket in enumerate(buckets)}
    failed_dev = [row for row in dev if _is_failed_annotated(row)]
    dense = _select_composition(
        failed_dev,
        dense_composition,
        seed=int(config["seed"]),
        role="dense_teacher_forced",
        bucket_function=source_bucket,
        excluded=set(),
    )
    dense_ids = {_identity(row) for row in dense}
    development = [row for row in dev if _identity(row) not in dense_ids]
    output = []
    for group, split in ((train, "train"), (development, "dev"), (dense, "dense_teacher_forced")):
        for row in group:
            rollout_eligible = _is_failed_annotated(row)
            row.update(
                pipeline_split=split,
                rollout_eligible=rollout_eligible,
                semantic_safety_only=not rollout_eligible,
                source_bucket=source_bucket(row),
            )
            output.append(row)
    return output, []


def _metric(row: Mapping[str, Any], name: str) -> float:
    value = row.get(name)
    if value is None:
        raise ValueError(f"missing selection metric {name!r}")
    return float(value)


def select_prompt_and_segmenter(rows: Sequence[Mapping[str, Any]], rules: Mapping[str, Any]) -> dict[str, Any]:
    if not rows:
        raise ValueError("prompt selection requires measured rows")
    best_accuracy = max(_metric(row, "verified_accuracy") for row in rows)
    best_segmentation = max(_metric(row, "valid_segmentation") for row in rows)
    prompt_order = {"P0": 0, "P1": 1, "P2": 2}
    segmenter_order = {"natural_paragraph": 0, "hybrid": 1, "fixed_token": 2}
    qualified = []
    for row in rows:
        segmentation_gain = float(row.get("segmentation_gain_over_least_intrusive", 0.0))
        inflation_ok = _metric(row, "reasoning_token_inflation") <= float(rules["maximum_reasoning_token_inflation"])
        if not inflation_ok:
            inflation_ok = segmentation_gain >= float(rules["material_segmentation_gain"])
        passes = (
            _metric(row, "answer_parser_success") >= float(rules["answer_parser_success"])
            and _metric(row, "valid_segmentation") >= float(rules["valid_segmentation"])
            and best_accuracy - _metric(row, "verified_accuracy") <= float(rules["accuracy_gap_from_best"])
            and inflation_ok
            and bool(row.get("normalized_first_error_position_stable", False))
        )
        if passes:
            qualified.append(dict(row))
    if not qualified:
        raise RuntimeError("no prompt and segmenter passed the frozen Phase A criteria")
    qualified.sort(key=lambda row: (prompt_order[str(row["prompt_condition"])], segmenter_order[str(row["segmenter"])], -_metric(row, "verified_accuracy")))
    return qualified[0]


def select_within_best(
    rows: Sequence[Mapping[str, Any]],
    *,
    size_order: Sequence[str],
    repair_gap: float,
    regret_gap: float,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("configuration selection requires measured rows")
    best_repair = max(_metric(row, "native_repair_accuracy") for row in rows)
    best_regret = min(_metric(row, "dense_audit_regret") for row in rows)
    eligible = [
        dict(row)
        for row in rows
        if best_repair - _metric(row, "native_repair_accuracy") <= float(repair_gap)
        and _metric(row, "dense_audit_regret") - best_regret <= float(regret_gap)
    ]
    if not eligible:
        raise RuntimeError("no configuration satisfies frozen non-inferiority rules")
    order = {name: index for index, name in enumerate(size_order)}
    eligible.sort(key=lambda row: (order.get(str(row["configuration"]), len(order)), -_metric(row, "native_repair_accuracy"), _metric(row, "dense_audit_regret")))
    return eligible[0]


def decide_post_error(*, optima_rate: float, pooled_gain: float, config: Mapping[str, Any], native_prefers_post_error: bool | None = None) -> str:
    settings = config["phase_e"]
    if optima_rate <= float(settings["retain_pre_error_maximum_optima_rate"]):
        return "strict_pre_error"
    if optima_rate > float(settings["include_post_error_minimum_optima_rate"]) or pooled_gain > float(settings["include_post_error_minimum_pooled_gain"]):
        return "include_post_error"
    if native_prefers_post_error is None:
        return "retain_both_pending_native_development"
    return "include_post_error" if native_prefers_post_error else "strict_pre_error"


def model_matrix_complete(config: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> bool:
    selected = list(map(str, config.get("selected_models", [])))
    indexed = {str(row["model_key"]): bool(row.get("accessible")) for row in records}
    return len(selected) == 4 and set(indexed) == set(selected) and all(indexed.values())
