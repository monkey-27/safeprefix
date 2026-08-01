"""Read-only forensic reconciliation for SafePrefix trace populations.

This module deliberately does not import model code or mutate source datasets.
It reconstructs the configuration-pilot partitions from persisted rows, then
applies the frozen rollout eligibility predicate and records each transition.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import yaml

from safeprefix.configuration_selection import (
    _has_reference_answer,
    _identity,
    build_immutable_manifests,
)
from safeprefix.full_teacher_forced import (
    _valid_failure,
    eligible_source_bucket,
    reasoning_steps_list,
    select_frozen_manifest_failures,
)


DESIGNATED_BUCKETS = (
    "crv_arithmetic",
    "processbench_math",
    "processbench_olympiadbench",
    "processbench_omnimath",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_true(value: Any) -> bool:
    return value is not None and not _is_nan(value) and bool(value)


def _is_false(value: Any) -> bool:
    return value is not None and not _is_nan(value) and not bool(value)


def _is_nan(value: Any) -> bool:
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def normalized_error_index(row: Mapping[str, Any]) -> int | None:
    if not _valid_failure(row):
        return None
    try:
        return int(row["first_error_index"]) - int(row.get("index_base") == "one")
    except (TypeError, ValueError, OverflowError):
        return None


def reasoning_step_count(row: Mapping[str, Any]) -> int | None:
    try:
        return len(reasoning_steps_list(row.get("reasoning_steps")))
    except (TypeError, ValueError):
        return None


def has_usable_first_error(row: Mapping[str, Any]) -> bool:
    error = normalized_error_index(row)
    steps = reasoning_step_count(row)
    return error is not None and steps is not None and 0 <= error < steps


def pre_error_checkpoint_count(row: Mapping[str, Any]) -> int:
    error = normalized_error_index(row)
    return error + 1 if error is not None and has_usable_first_error(row) else 0


def all_dataset_checkpoint_count(row: Mapping[str, Any]) -> int:
    steps = reasoning_step_count(row)
    return steps + 1 if steps is not None else 0


def _frozen_category(row: Mapping[str, Any], *, production_eligible: bool) -> str:
    reference = _has_reference_answer(row)
    if production_eligible:
        return "rollout_eligible_in_completed_suite"
    if _is_true(row.get("final_answer_correct")):
        return "verified_correct_with_reference" if reference else "verified_correct_missing_reference"
    if _is_false(row.get("final_answer_correct")):
        if not has_usable_first_error(row):
            return "verified_incorrect_without_usable_first_error"
        if not reference:
            return "verified_incorrect_usable_error_missing_reference"
        return "verified_incorrect_other_rejection"
    return "unknown_terminal_correctness"


def _raw_3180_fate(
    row: Mapping[str, Any],
    *,
    role_by_group: Mapping[str, str],
    production_ids: set[str],
) -> str | None:
    if eligible_source_bucket(row) is None or not has_usable_first_error(row):
        return None
    source_trace_id = str(row.get("source_trace_id") or "")
    if source_trace_id in production_ids:
        return "production_eligible"
    if not _has_reference_answer(row):
        return "missing_reference"
    role = role_by_group.get(_identity(row), "unassigned")
    return {
        "teacher_forced_train": "teacher_group_duplicate_or_unselected_variant",
        "teacher_forced_dev": "teacher_group_duplicate_or_unselected_variant",
        "native_configuration_development": "reserved_native_development_group",
        "prompt_segmentation": "reserved_prompt_group",
        "final_test_reservations": "reserved_final_test_group",
        "unassigned": "not_selected_into_any_frozen_role",
    }.get(role, f"other_role:{role}")


def _summary_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    population: str,
    split: str,
    bucket: str,
    production_ids: set[str],
) -> dict[str, Any]:
    usable = [row for row in rows if has_usable_first_error(row)]
    exact_reference_usable = [row for row in usable if _has_reference_answer(row)]
    selected = [row for row in rows if str(row.get("source_trace_id") or "") in production_ids]
    groups = [_identity(row) for row in rows]
    problem_ids = [str(row.get("problem_id")) for row in rows]
    return {
        "population": population,
        "pipeline_split": split,
        "source_bucket": bucket,
        "total_source_rows": len(rows),
        "unique_problem_ids": len(set(problem_ids)),
        "unique_problem_groups": len(set(groups)),
        "duplicate_problem_group_rows": len(rows) - len(set(groups)),
        "verified_correct_traces": sum(_is_true(row.get("final_answer_correct")) for row in rows),
        "verified_incorrect_traces": sum(_is_false(row.get("final_answer_correct")) for row in rows),
        "incorrect_with_usable_first_error": len(usable),
        "incorrect_without_usable_first_error": sum(
            _is_false(row.get("final_answer_correct")) and not has_usable_first_error(row)
            for row in rows
        ),
        "traces_lacking_exact_terminal_reference": sum(not _has_reference_answer(row) for row in rows),
        "other_rejected_traces": sum(
            _is_false(row.get("final_answer_correct"))
            and has_usable_first_error(row)
            and _has_reference_answer(row)
            and str(row.get("source_trace_id") or "") not in production_ids
            for row in rows
        ),
        "final_selected_rollout_eligible_traces": len(selected),
        "all_dataset_defined_checkpoints": sum(all_dataset_checkpoint_count(row) for row in rows),
        "pre_error_checkpoints_before_reference_filter": sum(pre_error_checkpoint_count(row) for row in usable),
        "pre_error_checkpoints_after_reference_filter": sum(
            pre_error_checkpoint_count(row) for row in exact_reference_usable
        ),
        "final_selected_rollout_checkpoints": sum(pre_error_checkpoint_count(row) for row in selected),
    }


def reconcile_counts(
    *,
    enriched_path: Path,
    frozen_root: Path,
    production_root: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Reconstruct all populations and return summary, lineage, and count table."""

    train_path = frozen_root / "teacher_forced_train.jsonl"
    dev_path = frozen_root / "teacher_forced_dev.jsonl"
    config_path = frozen_root / "resolved_config.yaml"
    summary_path = frozen_root / "summary.json"
    common_path = production_root / "immutable_manifests/common_trace_manifest.jsonl"
    protocol_path = production_root / "immutable_manifests/immutable_protocol_manifest.json"
    integrity_path = production_root / "integrity_validation_report.json"

    required = (
        enriched_path,
        train_path,
        dev_path,
        config_path,
        summary_path,
        common_path,
        protocol_path,
        integrity_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"reconciliation inputs are missing: {missing}")

    enriched_rows = pd.read_parquet(enriched_path).to_dict("records")
    train_rows = read_jsonl(train_path)
    dev_rows = read_jsonl(dev_path)
    frozen_rows = train_rows + dev_rows
    common_rows = read_jsonl(common_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    frozen_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))

    rebuilt = build_immutable_manifests(enriched_rows, config)
    for role, frozen in (
        ("teacher_forced_train", train_rows),
        ("teacher_forced_dev", dev_rows),
    ):
        rebuilt_ids = {str(row.get("source_trace_id")) for row in rebuilt[role]}
        frozen_ids = {str(row.get("source_trace_id")) for row in frozen}
        if rebuilt_ids != frozen_ids:
            raise AssertionError(f"reconstructed {role} does not match frozen manifest")

    selected, selection_summary, exclusions = select_frozen_manifest_failures(train_rows, dev_rows)
    selected_ids = {str(row["source_trace_id"]) for row in selected}
    common_ids = {str(row["source_trace_id"]) for row in common_rows}
    if selected_ids != common_ids:
        raise AssertionError("recomputed production selection differs from common trace manifest")

    role_by_group = {
        _identity(row): role
        for role, rows in rebuilt.items()
        for row in rows
    }
    selected_trace_role = {
        str(row.get("source_trace_id")): role
        for role, rows in rebuilt.items()
        for row in rows
    }
    frozen_split_by_id = {
        str(row.get("source_trace_id")): split
        for split, rows in (("train", train_rows), ("dev", dev_rows))
        for row in rows
    }
    frozen_row_by_id = {
        str(row.get("source_trace_id")): row
        for row in frozen_rows
    }
    duplicate_sizes = Counter(_identity(row) for row in enriched_rows)
    exclusion_reason = {str(row["source_trace_id"]): str(row["reason"]) for row in exclusions}

    lineage_records: list[dict[str, Any]] = []
    for row_index, row in enumerate(enriched_rows):
        source_trace_id = str(row.get("source_trace_id") or "")
        group = _identity(row)
        split = frozen_split_by_id.get(source_trace_id)
        production_eligible = source_trace_id in selected_ids
        usable = has_usable_first_error(row)
        raw_member = eligible_source_bucket(row) is not None and usable
        lineage_records.append(
            {
                "enriched_row_index": row_index,
                "source_trace_id": source_trace_id,
                "problem_id": str(row.get("problem_id") or ""),
                "problem_group_hash": group,
                "source_dataset": str(row.get("source_dataset") or ""),
                "source_subset": str(row.get("source_subset") or ""),
                "source_bucket": eligible_source_bucket(row),
                "source_split": str(row.get("split") or ""),
                "problem_group_row_count": duplicate_sizes[group],
                "is_problem_group_duplicate_row": duplicate_sizes[group] > 1,
                "terminal_correct": _is_true(row.get("final_answer_correct")),
                "terminal_incorrect": _is_false(row.get("final_answer_correct")),
                "first_error_annotation_present": normalized_error_index(row) is not None,
                "usable_first_error": usable,
                "exact_terminal_reference": _has_reference_answer(row),
                "reasoning_step_count": reasoning_step_count(row),
                "all_dataset_checkpoint_count": all_dataset_checkpoint_count(row),
                "pre_error_checkpoint_count": pre_error_checkpoint_count(row),
                "assigned_problem_role": role_by_group.get(group, "unassigned"),
                "selected_row_role": selected_trace_role.get(source_trace_id),
                "in_frozen_5900": split is not None,
                "frozen_pipeline_split": split,
                "minimum_failed_floor_member": bool(
                    frozen_row_by_id.get(source_trace_id, {}).get(
                        "minimum_failed_floor_member", False
                    )
                ),
                "production_eligible": production_eligible,
                "production_exclusion_reason": exclusion_reason.get(source_trace_id),
                "raw_3180_member": raw_member,
                "raw_3180_fate": _raw_3180_fate(
                    row, role_by_group=role_by_group, production_ids=selected_ids
                ),
                "frozen_5900_category": (
                    _frozen_category(row, production_eligible=production_eligible)
                    if split is not None
                    else None
                ),
            }
        )
    lineage = pd.DataFrame(lineage_records)

    count_rows: list[dict[str, Any]] = []
    designated = [row for row in enriched_rows if eligible_source_bucket(row) is not None]
    for bucket in (*DESIGNATED_BUCKETS, "ALL"):
        values = designated if bucket == "ALL" else [
            row for row in designated if eligible_source_bucket(row) == bucket
        ]
        count_rows.append(
            _summary_row(
                values,
                population="enriched_designated_source_rows",
                split="all",
                bucket=bucket,
                production_ids=selected_ids,
            )
        )
    for split, rows in (("train", train_rows), ("dev", dev_rows), ("train_plus_dev", frozen_rows)):
        for bucket in (*DESIGNATED_BUCKETS, "ALL"):
            values = rows if bucket == "ALL" else [
                row for row in rows if eligible_source_bucket(row) == bucket
            ]
            count_rows.append(
                _summary_row(
                    values,
                    population="frozen_teacher_forced_5900",
                    split=split,
                    bucket=bucket,
                    production_ids=selected_ids,
                )
            )
    count_table = pd.DataFrame(count_rows)

    raw_3180 = lineage[lineage["raw_3180_member"]]
    frozen_lineage = lineage[lineage["in_frozen_5900"]]
    raw_fates = {
        str(key): int(value)
        for key, value in raw_3180["raw_3180_fate"].value_counts().sort_index().items()
    }
    raw_fate_checkpoints = {
        str(key): int(group["pre_error_checkpoint_count"].sum())
        for key, group in raw_3180.groupby("raw_3180_fate", dropna=False)
    }
    frozen_categories = {
        str(key): int(value)
        for key, value in frozen_lineage["frozen_5900_category"].value_counts().sort_index().items()
    }

    allowed_fates = {
        "production_eligible",
        "teacher_group_duplicate_or_unselected_variant",
        "not_selected_into_any_frozen_role",
    }
    allowed_rows = raw_3180[
        raw_3180["exact_terminal_reference"]
        & raw_3180["raw_3180_fate"].isin(allowed_fates)
    ]
    additional = allowed_rows[~allowed_rows["production_eligible"]]

    filters = [
        {
            "stage": "enriched_all_rows",
            "rows": len(enriched_rows),
            "unique_problem_groups": int(lineage["problem_group_hash"].nunique()),
            "pre_error_checkpoints": int(lineage["pre_error_checkpoint_count"].sum()),
        },
        {
            "stage": "designated_sources",
            "rows": len(designated),
            "unique_problem_groups": len({_identity(row) for row in designated}),
            "pre_error_checkpoints": sum(pre_error_checkpoint_count(row) for row in designated),
        },
        {
            "stage": "terminally_incorrect",
            "rows": sum(_is_false(row.get("final_answer_correct")) for row in designated),
            "unique_problem_groups": len({
                _identity(row) for row in designated if _is_false(row.get("final_answer_correct"))
            }),
            "pre_error_checkpoints": sum(
                pre_error_checkpoint_count(row)
                for row in designated
                if _is_false(row.get("final_answer_correct"))
            ),
        },
        {
            "stage": "raw_usable_visible_error_population",
            "rows": len(raw_3180),
            "unique_problem_groups": int(raw_3180["problem_group_hash"].nunique()),
            "pre_error_checkpoints": int(raw_3180["pre_error_checkpoint_count"].sum()),
        },
        {
            "stage": "raw_usable_error_with_exact_reference",
            "rows": int(raw_3180["exact_terminal_reference"].sum()),
            "unique_problem_groups": int(
                raw_3180.loc[raw_3180["exact_terminal_reference"], "problem_group_hash"].nunique()
            ),
            "pre_error_checkpoints": int(
                raw_3180.loc[raw_3180["exact_terminal_reference"], "pre_error_checkpoint_count"].sum()
            ),
        },
        {
            "stage": "eligible_after_protected_role_exclusion",
            "rows": len(allowed_rows),
            "unique_problem_groups": int(allowed_rows["problem_group_hash"].nunique()),
            "pre_error_checkpoints": int(allowed_rows["pre_error_checkpoint_count"].sum()),
        },
        {
            "stage": "frozen_teacher_forced_mixed_rows",
            "rows": len(frozen_rows),
            "unique_problem_groups": int(frozen_lineage["problem_group_hash"].nunique()),
            "pre_error_checkpoints": int(frozen_lineage["pre_error_checkpoint_count"].sum()),
        },
        {
            "stage": "frozen_terminally_incorrect",
            "rows": int(frozen_lineage["terminal_incorrect"].sum()),
            "unique_problem_groups": int(
                frozen_lineage.loc[frozen_lineage["terminal_incorrect"], "problem_group_hash"].nunique()
            ),
            "pre_error_checkpoints": int(
                frozen_lineage.loc[frozen_lineage["terminal_incorrect"], "pre_error_checkpoint_count"].sum()
            ),
        },
        {
            "stage": "frozen_incorrect_with_usable_error",
            "rows": int(frozen_lineage["usable_first_error"].sum()),
            "unique_problem_groups": int(
                frozen_lineage.loc[frozen_lineage["usable_first_error"], "problem_group_hash"].nunique()
            ),
            "pre_error_checkpoints": int(
                frozen_lineage.loc[frozen_lineage["usable_first_error"], "pre_error_checkpoint_count"].sum()
            ),
        },
        {
            "stage": "completed_production_selection",
            "rows": len(selected),
            "unique_problem_groups": len({str(row["production_problem_group"]) for row in selected}),
            "pre_error_checkpoints": sum(pre_error_checkpoint_count(row) for row in selected),
        },
    ]

    summary = {
        "status": "PASS",
        "scope": "forensic_read_only_count_reconciliation",
        "input_files": {
            str(path): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in required
        },
        "source_counts": {
            "enriched_rows": len(enriched_rows),
            "enriched_unique_problem_ids": len({str(row.get("problem_id")) for row in enriched_rows}),
            "enriched_unique_problem_groups": len({_identity(row) for row in enriched_rows}),
            "enriched_problem_group_duplicate_rows": len(enriched_rows) - len({_identity(row) for row in enriched_rows}),
            "frozen_train_rows": len(train_rows),
            "frozen_dev_rows": len(dev_rows),
            "frozen_total_rows": len(frozen_rows),
        },
        "raw_claim_reproduction": {
            "traces": len(raw_3180),
            "checkpoints": int(raw_3180["pre_error_checkpoint_count"].sum()),
            "unique_problem_groups": int(raw_3180["problem_group_hash"].nunique()),
            "exact_reference_traces": int(raw_3180["exact_terminal_reference"].sum()),
            "exact_reference_checkpoints": int(
                raw_3180.loc[raw_3180["exact_terminal_reference"], "pre_error_checkpoint_count"].sum()
            ),
            "fates": raw_fates,
            "fate_checkpoints": raw_fate_checkpoints,
        },
        "frozen_corpus": {
            "terminally_correct": int(frozen_lineage["terminal_correct"].sum()),
            "terminally_incorrect": int(frozen_lineage["terminal_incorrect"].sum()),
            "incorrect_with_usable_first_error": int(frozen_lineage["usable_first_error"].sum()),
            "rollout_eligible": int(frozen_lineage["production_eligible"].sum()),
            "categories": frozen_categories,
            "minimum_failed_floor_members": int(frozen_lineage["minimum_failed_floor_member"].sum()),
        },
        "completed_run": {
            "selection_summary": selection_summary,
            "common_manifest_rows": len(common_rows),
            "protocol_eligible_traces": int(protocol["cohort"]["eligible_traces"]),
            "protocol_checkpoints_per_model": int(protocol["cohort"]["anticipated_checkpoints_per_model"]),
            "integrity_status": str(integrity["status"]),
            "models_complete": int(integrity["models_complete"]),
            "native_final_test_access_count": int(integrity["native_final_test_access_count"]),
        },
        "additional_generation": {
            "literal_all_eligible_trace_rows_after_protected_role_exclusion": len(allowed_rows),
            "unique_problem_groups": int(allowed_rows["problem_group_hash"].nunique()),
            "checkpoints_per_model": int(allowed_rows["pre_error_checkpoint_count"].sum()),
            "already_completed_trace_rows": len(selected),
            "already_completed_checkpoints_per_model": sum(pre_error_checkpoint_count(row) for row in selected),
            "additional_trace_rows": len(additional),
            "additional_unique_problem_groups_represented": int(additional["problem_group_hash"].nunique()),
            "additional_checkpoints_per_model": int(additional["pre_error_checkpoint_count"].sum()),
            "additional_rollouts_per_model_at_k4": int(additional["pre_error_checkpoint_count"].sum()) * 4,
            "additional_rollouts_four_models_at_k4": int(additional["pre_error_checkpoint_count"].sum()) * 16,
            "note": "Counts retain every eligible trace row while excluding problem groups reserved for final-test, prompt, or native-development roles.",
        },
        "filter_stages": filters,
        "assertions": {
            "rebuild_matches_frozen_manifests": True,
            "selection_matches_common_trace_manifest": True,
            "raw_3180_fates_sum": sum(raw_fates.values()) == 3180,
            "frozen_categories_sum": sum(frozen_categories.values()) == 5900,
            "completed_integrity_validated": integrity["status"] == "INTEGRITY_VALIDATED",
        },
        "frozen_manifest_summary": {
            "status": frozen_summary.get("status"),
            "configuration_hash": frozen_summary.get("configuration_hash"),
            "counts": frozen_summary.get("counts"),
        },
    }

    assert summary["source_counts"] == {
        "enriched_rows": 44517,
        "enriched_unique_problem_ids": 44517,
        "enriched_unique_problem_groups": 43123,
        "enriched_problem_group_duplicate_rows": 1394,
        "frozen_train_rows": 5100,
        "frozen_dev_rows": 800,
        "frozen_total_rows": 5900,
    }
    assert summary["raw_claim_reproduction"]["traces"] == 3180
    assert summary["raw_claim_reproduction"]["checkpoints"] == 13318
    assert summary["frozen_corpus"]["terminally_incorrect"] == 948
    assert summary["frozen_corpus"]["incorrect_with_usable_first_error"] == 945
    assert summary["frozen_corpus"]["rollout_eligible"] == 942
    assert selection_summary["split_counts"] == {"dev": 90, "train": 852}
    assert selection_summary["anticipated_checkpoints_per_model"] == 3895
    assert all(summary["assertions"].values())
    return summary, lineage, count_table
