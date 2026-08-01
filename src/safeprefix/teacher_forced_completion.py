"""Complete the reconciled repairability and safety-feature corpora.

This module deliberately contains no model training, boundary selection,
threshold tuning, or native-evaluation code.  It extends the frozen k=4 raw
rollout corpus and separately performs forward-only feature extraction over the
mixed 5,900-row teacher-forced corpus.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import torch

from safeprefix.config import dump_resolved
from safeprefix.full_teacher_forced import (
    aggregate_checkpoint_outcomes,
    assign_packs_to_workers,
    build_execution_packs,
    first_error_bin,
    reasoning_steps_list,
)
from safeprefix.models.hidden_features import build_checkpoint_features
from safeprefix.models.loader import load_tokenizer
from safeprefix.models.teacher_forcing import teacher_force_token_ids_chunked
from safeprefix.production_suite import (
    _model_trace_rows,
    canonical_completion,
    engine_revision,
    execute_pack,
    parser_verifier_hashes,
    read_jsonl,
    sha256_file,
    source_step_boundary_audits,
    valid_pack_artifact,
)
from safeprefix.prompting.chat_format import render_chat_prompt
from safeprefix.prompting.templates import problem_instruction
from safeprefix.parsing.token_alignment import tokenize_with_offsets
from safeprefix.reproducibility import (
    atomic_json,
    atomic_jsonl,
    atomic_parquet,
    atomic_text,
    now_iso,
    provenance,
    stable_hash,
)


ROOT = Path(__file__).resolve().parents[2]
REPAIRABILITY_FATES = {
    "production_eligible",
    "not_selected_into_any_frozen_role",
    "teacher_group_duplicate_or_unselected_variant",
}
PROTECTED_FATES = {
    "reserved_native_development_group",
    "reserved_final_test_group",
    "reserved_prompt_group",
    "missing_reference",
}
SCHEMA_VERSION = 1
SAFETY_REPRESENTATION = "three_layer_checkpoint_final_with_difference_and_trace_summary_v1"


def completion_worker_allocation(config: Mapping[str, Any]) -> dict[str, int]:
    """Return the frozen concurrent production allocation.

    Allocation changes only which immutable pack a model-resident worker claims;
    it does not change pack membership, decoding, seeds, or record identity.
    """

    selected = list(map(str, config["selected_models"]))
    raw = config.get("teacher_forced_completion", {}).get("gpu_workers_by_model", {})
    allocation = {model_key: int(raw.get(model_key, 0)) for model_key in selected}
    if set(raw) != set(selected):
        raise RuntimeError("GPU allocation must name every and only selected model")
    if any(count < 1 for count in allocation.values()):
        raise RuntimeError("every selected model requires at least one H100 worker")
    expected_total = int(
        config.get("teacher_forced_completion", {}).get("total_gpu_workers", 0)
    )
    if expected_total < len(selected) or sum(allocation.values()) != expected_total:
        raise RuntimeError(
            "completion production allocation must equal the configured positive worker total"
        )
    return allocation


def feature_reuse_equivalence(
    newly_extracted: torch.Tensor, existing: torch.Tensor
) -> dict[str, Any]:
    """Decide whether an existing feature tensor may be reused exactly.

    A numerically different tensor is not a failed extraction: it is retained as
    evidence that the old feature cannot be mixed into the newly frozen safety
    corpus. Shape equality remains mandatory because it verifies that the same
    layer/pooling representation is being extracted before all rows are
    recomputed under the new batch protocol.
    """

    shape_equal = list(newly_extracted.shape) == list(existing.shape)
    maximum = None
    if shape_equal and newly_extracted.numel():
        maximum = float(
            (newly_extracted.float() - existing.float()).abs().max().item()
        )
    exact_equal = bool(shape_equal and torch.equal(newly_extracted, existing))
    return {
        "shape_equal": shape_equal,
        "exact_equal": exact_equal,
        "max_abs_difference": maximum,
        "reuse_permitted": exact_equal,
        "action": "reuse" if exact_equal else "recompute_all_safety_features",
    }


def validate_completed_aggregate_counts(aggregate: pd.DataFrame) -> None:
    """Validate the frozen completed-run aggregate without schema aliases."""

    required = {"trace_id", "checkpoint_index", "trial_count"}
    missing = required - set(aggregate.columns)
    if missing:
        raise RuntimeError(f"completed aggregate schema is missing {sorted(missing)}")
    if (
        len(aggregate) != 3895
        or not aggregate["trial_count"].eq(4).all()
        or int(aggregate["trial_count"].sum()) != 15580
    ):
        raise RuntimeError("completed aggregate count differs")


def completion_root(config: Mapping[str, Any], run_id: str) -> Path:
    if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
        raise ValueError("run_id must be one safe path component")
    external = os.environ.get("SAFEPREFIX_RUNS_ROOT")
    if external:
        return Path(external) / run_id / "artifacts" / "teacher_forced_completion"
    return ROOT / str(config["artifacts_root"]) / run_id


def completion_engine_revision() -> str:
    files = [
        ROOT / "src/safeprefix/teacher_forced_completion.py",
        ROOT / "src/safeprefix/production_suite.py",
        ROOT / "src/safeprefix/full_teacher_forced.py",
        ROOT / "src/safeprefix/models/teacher_forcing.py",
        ROOT / "src/safeprefix/models/hidden_features.py",
        ROOT / "src/safeprefix/rollout/production_engine.py",
        ROOT / "src/safeprefix/parsing/answer_parsers.py",
        ROOT / "src/safeprefix/rollout/verifier.py",
    ]
    return stable_hash([(str(path.relative_to(ROOT)), sha256_file(path)) for path in files])


def assert_completion_scope(config: Mapping[str, Any]) -> None:
    scope = config.get("teacher_forced_completion", {})
    expected = scope.get("expected", {})
    if not scope.get("rollout_extension_only") or not scope.get("safety_forward_only"):
        raise RuntimeError("completion run requires rollout-extension and forward-only scope")
    required = {
        "eligible_failed_traces": 2614,
        "eligible_checkpoints_per_model": 10988,
        "reused_failed_traces": 942,
        "reused_checkpoints_per_model": 3895,
        "extension_failed_traces": 1672,
        "extension_checkpoints_per_model": 7093,
        "mixed_train_traces": 5100,
        "mixed_dev_traces": 800,
    }
    for key, value in required.items():
        if int(expected.get(key, -1)) != value:
            raise RuntimeError(f"frozen completion count differs for {key}")
    if int(config.get("rollout", {}).get("max_new_tokens", -1)) != 4096:
        raise RuntimeError("repairability extension requires max_new_tokens=4096")
    if int(config.get("full_teacher_forced_suite", {}).get("rollouts_per_checkpoint", -1)) != 4:
        raise RuntimeError("repairability extension requires exactly k=4")
    completion_worker_allocation(config)
    disabled = config.get("disabled_stages", {})
    prohibited = {
        "boundary_model_training",
        "safety_head_training",
        "repairability_head_training",
        "binary_first_unsafe_label_derivation",
        "operational_boundary_selection",
        "threshold_tuning",
        "boundary_localization_evaluation",
        "native_repair_evaluation",
        "native_final_test_access",
    }
    if any(not bool(disabled.get(key)) for key in prohibited):
        raise RuntimeError("all training, selection, and native stages must remain disabled")


def _atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _commit_modal_volume() -> None:
    if not os.environ.get("MODAL_TASK_ID"):
        return
    import modal

    name = os.environ.get(
        "SAFEPREFIX_RUN_VOLUME", "safeprefix-full-teacher-forced-runs-v2"
    )
    modal.Volume.from_name(name).commit()


def _lineage(config: Mapping[str, Any]) -> pd.DataFrame:
    path = ROOT / str(config["teacher_forced_completion"]["reconciliation_lineage"])
    frame = pd.read_parquet(path)
    if len(frame) != 44_517 or int(frame["raw_3180_member"].sum()) != 3_180:
        raise RuntimeError("reconciliation lineage does not reproduce the frozen audit")
    return frame


def _source_root(config: Mapping[str, Any]) -> Path:
    return Path(str(config["frozen_teacher_forced_manifests"]["root"]))


def _load_frozen_mixed(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    settings = config["frozen_teacher_forced_manifests"]
    root = _source_root(config)
    result: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    seen_groups: dict[str, str] = {}
    for split, key in (("train", "train_file"), ("dev", "dev_file")):
        path = root / str(settings[key])
        hashes[split] = sha256_file(path)
        rows = read_jsonl(path)
        expected = int(settings["expected_counts"][f"teacher_forced_{split}"])
        if len(rows) != expected:
            raise RuntimeError(f"frozen {split} manifest count differs")
        for raw in rows:
            row = dict(raw)
            group = str(row.get("problem_group_hash") or row["problem_id"])
            prior = seen_groups.setdefault(group, split)
            if prior != split:
                raise RuntimeError("frozen mixed corpus has problem-group leakage")
            row["reasoning_steps"] = reasoning_steps_list(row.get("reasoning_steps"))
            row["pipeline_split"] = split
            row["production_problem_group"] = group
            result.append(row)
    if len(result) != 5900:
        raise AssertionError("mixed corpus must contain 5,900 traces")
    return result, hashes


def _read_selected_enriched(path: Path, source_trace_ids: set[str]) -> list[dict[str, Any]]:
    """Read only selected records from the broader enriched parquet table."""

    import pyarrow.dataset as ds

    dataset = ds.dataset(path, format="parquet")
    table = dataset.to_table(
        filter=ds.field("source_trace_id").isin(sorted(source_trace_ids))
    )
    rows = table.to_pylist()
    observed = {str(row["source_trace_id"]) for row in rows}
    if observed != source_trace_ids or len(rows) != len(source_trace_ids):
        raise RuntimeError("selected enriched records are missing or duplicated")
    return rows


def _assign_splits(
    selected: pd.DataFrame,
    frozen_mixed: Sequence[Mapping[str, Any]],
    completed: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    target_dev: int,
) -> dict[str, str]:
    """Assign whole problem groups while preserving every frozen assignment."""

    frozen_group_split: dict[str, str] = {}
    for row in frozen_mixed:
        group = str(row["problem_group_hash"])
        split = str(row["pipeline_split"])
        prior = frozen_group_split.setdefault(group, split)
        if prior != split:
            raise RuntimeError("frozen group appears in both train and dev")
    completed_group_split: dict[str, str] = {}
    for row in completed:
        group = str(row["production_problem_group"])
        split = str(row["pipeline_split"])
        prior = completed_group_split.setdefault(group, split)
        if prior != split:
            raise RuntimeError("completed cohort contains split leakage")
        if frozen_group_split.get(group) != split:
            raise RuntimeError("completed split differs from frozen mixed split")

    groups: dict[str, dict[str, Any]] = {}
    for group, frame in selected.groupby("problem_group_hash", sort=True):
        group = str(group)
        strata = Counter(
            (
                str(row.source_bucket),
                first_error_bin(int(row.pre_error_checkpoint_count) - 1),
            )
            for row in frame.itertuples()
        )
        stratum = sorted(strata.items(), key=lambda item: (-item[1], item[0]))[0][0]
        groups[group] = {
            "size": int(len(frame)),
            "stratum": stratum,
            "forced": frozen_group_split.get(group),
            "tie": stable_hash(["safeprefix-extension-split-v1", seed, group]),
        }

    assignments: dict[str, str] = {
        group: str(info["forced"])
        for group, info in groups.items()
        if info["forced"] is not None
    }
    movable = [group for group in groups if group not in assignments]
    total_by_stratum = Counter()
    dev_by_stratum = Counter()
    for group, info in groups.items():
        total_by_stratum[info["stratum"]] += info["size"]
        if assignments.get(group) == "dev":
            dev_by_stratum[info["stratum"]] += info["size"]
    target_by_stratum = {
        key: int(round(value * target_dev / max(len(selected), 1)))
        for key, value in total_by_stratum.items()
    }
    for group in sorted(movable, key=lambda value: groups[value]["tie"]):
        info = groups[group]
        key = info["stratum"]
        before = abs(dev_by_stratum[key] - target_by_stratum[key])
        after = abs(dev_by_stratum[key] + info["size"] - target_by_stratum[key])
        split = "dev" if after < before else "train"
        assignments[group] = split
        if split == "dev":
            dev_by_stratum[key] += info["size"]

    def dev_rows() -> int:
        return sum(groups[group]["size"] for group, split in assignments.items() if split == "dev")

    # Improve the global target without changing any inherited assignment.
    while True:
        current = dev_rows()
        if current == target_dev:
            break
        direction = "train" if current < target_dev else "dev"
        candidates = [group for group in movable if assignments[group] == direction]
        if not candidates:
            break
        best: tuple[Any, ...] | None = None
        best_group: str | None = None
        for group in candidates:
            info = groups[group]
            delta = info["size"] if direction == "train" else -info["size"]
            new_total = current + delta
            if abs(new_total - target_dev) >= abs(current - target_dev):
                continue
            key = info["stratum"]
            new_stratum = dev_by_stratum[key] + delta
            score = (
                abs(new_total - target_dev),
                abs(new_stratum - target_by_stratum[key]),
                info["tie"],
            )
            if best is None or score < best:
                best, best_group = score, group
        if best_group is None:
            break
        info = groups[best_group]
        delta = info["size"] if direction == "train" else -info["size"]
        assignments[best_group] = "dev" if direction == "train" else "train"
        dev_by_stratum[info["stratum"]] += delta
    return assignments


def _normalize_extension_row(
    raw: Mapping[str, Any], lineage: Mapping[str, Any], split: str, error_bins: Iterable[int]
) -> dict[str, Any]:
    steps = reasoning_steps_list(raw.get("reasoning_steps"))
    error = int(raw["first_error_index"]) - int(str(raw.get("index_base")) == "one")
    if not 0 <= error < len(steps):
        raise RuntimeError("reconciled first error lies outside reasoning steps")
    source_trace_id = str(raw["source_trace_id"])
    group = str(raw["problem_group_hash"])
    return {
        **{str(key): value for key, value in raw.items()},
        "reasoning_steps": steps,
        "source_bucket": str(lineage["source_bucket"]),
        "first_error_zero_based": error,
        "first_error_bin": first_error_bin(error, error_bins),
        "rollout_eligible": True,
        "semantic_safety_only": False,
        "pipeline_split": split,
        "production_problem_group": group,
        "trace_id": stable_hash(["safeprefix-full-tf-trace-v2", source_trace_id])[:24],
        "reconciliation_fate": str(lineage["raw_3180_fate"]),
    }


def freeze_membership_artifacts(
    config: Mapping[str, Any],
    *,
    output_root: Path,
    frozen_source_root: Path,
    completed_root: Path,
) -> dict[str, Any]:
    """Freeze content-free membership and split decisions before GPU execution."""

    local = dict(config)
    local["frozen_teacher_forced_manifests"] = {
        **dict(config["frozen_teacher_forced_manifests"]),
        "root": str(frozen_source_root),
    }
    local["teacher_forced_completion"] = {
        **dict(config["teacher_forced_completion"]),
        "completed_rollout_root": str(completed_root),
    }
    lineage = _lineage(local)
    selected = lineage[lineage["raw_3180_fate"].isin(REPAIRABILITY_FATES)].copy()
    mixed, _ = _load_frozen_mixed(local)
    old_common = read_jsonl(completed_root / "immutable_manifests/common_trace_manifest.jsonl")
    split_map = _assign_splits(
        selected,
        mixed,
        old_common,
        seed=int(config["seed"]),
        target_dev=int(round(len(selected) * 0.15)),
    )
    old_ids = {str(row["source_trace_id"]) for row in old_common}
    repair = selected[
        [
            "source_trace_id",
            "problem_id",
            "problem_group_hash",
            "source_bucket",
            "pre_error_checkpoint_count",
            "raw_3180_fate",
        ]
    ].copy()
    repair["pipeline_split"] = repair["problem_group_hash"].astype(str).map(split_map)
    repair["reused_completed_942"] = repair["source_trace_id"].astype(str).isin(old_ids)
    extension_ids = set(repair["source_trace_id"].astype(str)) - old_ids
    extension_rows = _read_selected_enriched(
        frozen_source_root / "enriched_examples.parquet", extension_ids
    )
    repair = repair.sort_values(
        ["pipeline_split", "source_bucket", "source_trace_id"], kind="stable"
    ).reset_index(drop=True)
    safety = lineage[lineage["in_frozen_5900"]][
        [
            "source_trace_id",
            "problem_id",
            "problem_group_hash",
            "source_bucket",
            "frozen_pipeline_split",
            "terminal_correct",
            "terminal_incorrect",
            "usable_first_error",
            "all_dataset_checkpoint_count",
        ]
    ].copy()
    safety = safety.rename(columns={"frozen_pipeline_split": "pipeline_split"})
    safety = safety.sort_values(["pipeline_split", "source_trace_id"], kind="stable").reset_index(drop=True)
    protected = lineage[lineage["raw_3180_fate"].isin(PROTECTED_FATES)][
        ["source_trace_id", "problem_group_hash", "raw_3180_fate"]
    ].copy()
    output_root.mkdir(parents=True, exist_ok=True)
    repair_path = output_root / "repairability_membership.parquet"
    safety_path = output_root / "safety_membership.parquet"
    protected_path = output_root / "protected_exclusions.parquet"
    extension_path = output_root / "extension_examples.parquet"
    atomic_parquet(repair_path, repair)
    atomic_parquet(safety_path, safety)
    atomic_parquet(protected_path, protected)
    atomic_parquet(extension_path, pd.DataFrame(extension_rows))
    summary = {
        "status": "FROZEN",
        "created_at": now_iso(),
        "repairability": {
            "traces": len(repair),
            "problem_groups": int(repair["problem_group_hash"].nunique()),
            "checkpoints_per_model": int(repair["pre_error_checkpoint_count"].sum()),
            "reused_traces": int(repair["reused_completed_942"].sum()),
            "split_counts": repair["pipeline_split"].value_counts().sort_index().to_dict(),
        },
        "safety": {
            "traces": len(safety),
            "split_counts": safety["pipeline_split"].value_counts().sort_index().to_dict(),
        },
        "protected_exclusions": len(protected),
        "files": {
            "repairability_membership.parquet": sha256_file(repair_path),
            "safety_membership.parquet": sha256_file(safety_path),
            "protected_exclusions.parquet": sha256_file(protected_path),
            "extension_examples.parquet": sha256_file(extension_path),
        },
    }
    atomic_json(output_root / "membership_summary.json", summary)
    return summary


def _old_root(config: Mapping[str, Any]) -> Path:
    return Path(str(config["teacher_forced_completion"]["completed_rollout_root"]))


def completed_integrity_status_valid(
    final: Mapping[str, Any], integrity: Mapping[str, Any]
) -> bool:
    return (
        final.get("status") == "INTEGRITY_VALIDATED"
        and integrity.get("status") == "INTEGRITY_VALIDATED"
        and int(final.get("native_final_test_access_count", -1)) == 0
    )


def _validate_old_reuse(
    config: Mapping[str, Any],
    full_rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    root = _old_root(config)
    final = json.loads((root / "final_summary.json").read_text(encoding="utf-8"))
    integrity = json.loads(
        (root / "integrity_validation_report.json").read_text(encoding="utf-8")
    )
    protocol = json.loads(
        (root / "immutable_manifests/immutable_protocol_manifest.json").read_text(encoding="utf-8")
    )
    frozen = json.loads((root / "frozen_execution_manifest.json").read_text(encoding="utf-8"))
    if not completed_integrity_status_valid(final, integrity):
        raise RuntimeError("completed rollout suite lacks a passing final integrity state")
    if int(protocol["cohort"]["eligible_traces"]) != 942:
        raise RuntimeError("completed cohort is not the reconciled 942-trace subset")

    ledger: dict[str, Any] = {
        "status": "PASS",
        "source_root": str(root),
        "source_run_status": final["status"],
        "source_configuration_hash": protocol["configuration_hash"],
        "source_engine_revision": protocol["engine_revision"],
        "source_freeze_digest": frozen["freeze_digest"],
        "models": {},
    }
    for model_key in config["selected_models"]:
        model_key = str(model_key)
        old_rows = read_jsonl(
            root / "immutable_manifests/per_model" / model_key / "trace_manifest.jsonl"
        )
        old_by_source = {str(row["source_trace_id"]): row for row in old_rows}
        current_by_source = {
            str(row["source_trace_id"]): row
            for row in full_rows_by_model[model_key]
            if str(row.get("reconciliation_fate")) == "production_eligible"
        }
        if set(old_by_source) != set(current_by_source) or len(old_by_source) != 942:
            raise RuntimeError(f"{model_key}: completed trace membership differs")
        fields = (
            "trace_id",
            "prompt_token_ids",
            "completion_token_ids",
            "checkpoint_offsets",
            "eligible_checkpoint_offsets",
            "model_revision",
            "tokenizer_revision",
        )
        mismatches = [
            source
            for source, old in old_by_source.items()
            if any(old.get(field) != current_by_source[source].get(field) for field in fields)
        ]
        if mismatches:
            raise RuntimeError(f"{model_key}: reuse identity mismatch: {mismatches[:5]}")
        aggregate_path = root / "aggregated_checkpoint_outcomes" / f"{model_key}.parquet"
        aggregate = pd.read_parquet(aggregate_path)
        try:
            validate_completed_aggregate_counts(aggregate)
        except RuntimeError as exc:
            raise RuntimeError(f"{model_key}: {exc}") from exc
        complete_markers = sorted(
            (root / "raw_rollout_shards" / model_key).glob("*/complete.json")
        )
        if len(complete_markers) != 79:
            raise RuntimeError(f"{model_key}: expected 79 completed source packs")
        checksum_failures = []
        for marker_path in complete_markers:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            pack_root = marker_path.parent
            for filename, field in (
                ("rollouts.parquet", "rollouts_sha256"),
                ("checkpoint_features.pt", "features_sha256"),
            ):
                if sha256_file(pack_root / filename) != marker.get(field):
                    checksum_failures.append(str(pack_root))
                    break
        if checksum_failures:
            raise RuntimeError(f"{model_key}: reused pack checksum failure")
        ledger["models"][model_key] = {
            "traces": 942,
            "checkpoints": 3895,
            "rollouts": 15580,
            "packs": 79,
            "trace_manifest_sha256": sha256_file(
                root / "immutable_manifests/per_model" / model_key / "trace_manifest.jsonl"
            ),
            "aggregate_sha256": sha256_file(aggregate_path),
            "all_pack_checksums_valid": True,
        }
    return ledger


def _safety_model_rows(
    common: Sequence[Mapping[str, Any]], config: Mapping[str, Any], model_key: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    entry = config["models"][model_key]
    loaded = load_tokenizer(entry)
    tokenizer = loaded.tokenizer
    prompt_condition = str(config.get("prompting", {}).get("selected_condition", "P0"))
    rows: list[dict[str, Any]] = []
    checkpoints = 0
    masked_failures = 0
    for common_row in common:
        steps = reasoning_steps_list(common_row.get("reasoning_steps"))
        completion, ranges = canonical_completion(steps)
        prompt = render_chat_prompt(
            tokenizer,
            problem_instruction(str(common_row["problem_text"]), prompt_condition),
            override=entry.get("chat_template_override"),
            template_kwargs=entry.get("chat_template_kwargs"),
        )
        prompt_ids = list(map(int, tokenizer(prompt, add_special_tokens=False)["input_ids"]))
        alignment = tokenize_with_offsets(tokenizer, completion)
        completion_ids = list(map(int, alignment.input_ids))
        audits = source_step_boundary_audits(alignment, ranges)
        offsets = [len(prompt_ids)] + [
            len(prompt_ids) + int(item["token_offset"]) for item in audits
        ]
        correct = bool(common_row.get("final_answer_correct"))
        raw_error = common_row.get("first_error_index")
        usable = (
            not correct
            and raw_error is not None
            and not (isinstance(raw_error, float) and math.isnan(raw_error))
        )
        error = None
        if usable:
            error = int(raw_error) - int(str(common_row.get("index_base")) == "one")
            usable = 0 <= error < len(steps)
        labels, mask = safety_label_sequence(
            checkpoint_count=len(offsets),
            terminal_correct=correct,
            first_error_zero_based=error if usable else None,
        )
        if not correct and not usable:
            masked_failures += 1
        row = {
            **{str(key): value for key, value in common_row.items()},
            "trace_id": stable_hash(
                ["safeprefix-safety-feature-trace-v1", model_key, common_row["source_trace_id"]]
            )[:24],
            "common_trace_id": str(common_row.get("source_trace_id")),
            "model_key": model_key,
            "model_id": entry["hf_model_id"],
            "model_revision": entry.get("revision"),
            "tokenizer_revision": loaded.tokenizer_revision,
            "prompt_text": prompt,
            "recorded_completion": completion,
            "prompt_token_ids": prompt_ids,
            "completion_token_ids": completion_ids,
            "checkpoint_offsets": offsets,
            "checkpoint_indices": list(range(len(offsets))),
            "visible_safety_labels": labels,
            "safety_supervision_mask": mask,
            "first_error_zero_based": error,
            "terminal_correct": correct,
            "representation": SAFETY_REPRESENTATION,
            "full_token_count": len(prompt_ids) + len(completion_ids),
            "source_step_boundary_audits": audits,
        }
        rows.append(row)
        checkpoints += len(offsets)
    return rows, {
        "traces": len(rows),
        "checkpoints": checkpoints,
        "terminal_correct": sum(bool(row["terminal_correct"]) for row in rows),
        "terminal_failed": sum(not bool(row["terminal_correct"]) for row in rows),
        "failed_without_usable_first_error": masked_failures,
    }


def safety_label_sequence(
    *, checkpoint_count: int, terminal_correct: bool, first_error_zero_based: int | None
) -> tuple[list[bool | None], list[bool]]:
    """Visible-safety targets for root plus every source-step checkpoint.

    If source step ``e`` is the first visible error, checkpoint ``e`` is the
    state after the last clean step (root is checkpoint zero), while checkpoint
    ``e + 1`` is after the erroneous step.  Missing annotations are retained
    with a false supervision mask rather than guessed.
    """

    if checkpoint_count < 1:
        raise ValueError("a trace must contain at least the prompt-root checkpoint")
    if terminal_correct:
        return [True] * checkpoint_count, [True] * checkpoint_count
    if first_error_zero_based is None:
        return [None] * checkpoint_count, [False] * checkpoint_count
    error = int(first_error_zero_based)
    if not 0 <= error < checkpoint_count - 1:
        raise ValueError("first error lies outside source reasoning steps")
    return [index <= error for index in range(checkpoint_count)], [True] * checkpoint_count


def _simple_packs(
    rows: Sequence[Mapping[str, Any]], *, model_key: str, kind: str, size: int, digest: str
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (-int(row["full_token_count"]), str(row["trace_id"])))
    packs = []
    for index in range(0, len(ordered), size):
        members = ordered[index : index + size]
        identity = stable_hash([kind, digest, model_key, [row["trace_id"] for row in members]])
        packs.append(
            {
                "pack_id": f"{model_key}-{kind}-{index // size:05d}-{identity[:12]}",
                "pack_hash": identity,
                "model_key": model_key,
                "kind": kind,
                "trace_ids": [str(row["trace_id"]) for row in members],
                "trace_count": len(members),
                "checkpoint_count": sum(len(row["checkpoint_offsets"]) for row in members),
                "estimated_work": sum(int(row["full_token_count"]) for row in members),
            }
        )
    return packs


def prepare_completion(config: Mapping[str, Any], *, run_id: str, config_path: Path) -> dict[str, Any]:
    assert_completion_scope(config)
    root = completion_root(config, run_id)
    root.mkdir(parents=True, exist_ok=True)
    digest = stable_hash(config)
    lineage = _lineage(config)
    selected = lineage[lineage["raw_3180_fate"].isin(REPAIRABILITY_FATES)].copy()
    if len(selected) != 2614 or int(selected["pre_error_checkpoint_count"].sum()) != 10988:
        raise RuntimeError("reconciled rollout population does not match 2,614/10,988")
    protected_groups = set(
        lineage.loc[lineage["raw_3180_fate"].isin(PROTECTED_FATES), "problem_group_hash"].astype(str)
    )
    if set(selected["problem_group_hash"].astype(str)) & protected_groups:
        raise RuntimeError("protected problem group entered repairability manifest")

    mixed, mixed_hashes = _load_frozen_mixed(config)
    old_common = read_jsonl(_old_root(config) / "immutable_manifests/common_trace_manifest.jsonl")
    old_source_ids = {str(row["source_trace_id"]) for row in old_common}
    selected_source_ids = set(selected["source_trace_id"].astype(str))
    if not old_source_ids.issubset(selected_source_ids) or len(old_source_ids) != 942:
        raise RuntimeError("completed subset is not contained in reconciled population")

    computed_split_map = _assign_splits(
        selected,
        mixed,
        old_common,
        seed=int(config["seed"]),
        target_dev=int(round(len(selected) * 0.15)),
    )
    membership_root = ROOT / str(config["teacher_forced_completion"]["membership_manifest_root"])
    frozen_membership = pd.read_parquet(membership_root / "repairability_membership.parquet")
    if set(frozen_membership["source_trace_id"].astype(str)) != selected_source_ids:
        raise RuntimeError("precommitted repairability membership differs")
    split_map = dict(
        zip(
            frozen_membership["problem_group_hash"].astype(str),
            frozen_membership["pipeline_split"].astype(str),
        )
    )
    if any(split_map[group] != split for group, split in computed_split_map.items()):
        raise RuntimeError("precommitted group split differs from deterministic reconstruction")
    membership_summary = json.loads((membership_root / "membership_summary.json").read_text())
    for filename, expected_hash in membership_summary["files"].items():
        if sha256_file(membership_root / filename) != expected_hash:
            raise RuntimeError("precommitted membership checksum differs")
    run_membership_root = root / "immutable_manifests/membership"
    run_membership_root.mkdir(parents=True, exist_ok=True)
    for filename in (*membership_summary["files"], "membership_summary.json"):
        shutil.copy2(membership_root / filename, run_membership_root / filename)
    extra_source_ids = selected_source_ids - old_source_ids
    extension_examples_path = ROOT / str(
        config["teacher_forced_completion"]["extension_examples_file"]
    )
    enriched_extra = _read_selected_enriched(extension_examples_path, extra_source_ids)
    selected_by_source = {
        str(row.source_trace_id): row._asdict() for row in selected.itertuples(index=False)
    }
    errors = config["full_teacher_forced_suite"]["first_error_bins"]
    extras = [
        _normalize_extension_row(
            row,
            selected_by_source[str(row["source_trace_id"])],
            split_map[str(row["problem_group_hash"])],
            errors,
        )
        for row in enriched_extra
    ]
    completed = []
    for row in old_common:
        item = dict(row)
        item["reconciliation_fate"] = "production_eligible"
        item["pipeline_split"] = split_map[str(item["production_problem_group"])]
        completed.append(item)
    common = sorted(
        [*completed, *extras],
        key=lambda row: (str(row["pipeline_split"]), str(row["source_bucket"]), str(row["source_trace_id"])),
    )
    if len(common) != 2614 or len(extras) != 1672:
        raise AssertionError("repairability common manifest count differs")
    groups_by_split = defaultdict(set)
    for row in common:
        groups_by_split[str(row["pipeline_split"])].add(str(row["production_problem_group"]))
    if groups_by_split["train"] & groups_by_split["dev"]:
        raise RuntimeError("repairability manifest has problem-group leakage")

    manifest_root = root / "immutable_manifests"
    atomic_jsonl(manifest_root / "repairability/common_trace_manifest.jsonl", common)
    atomic_jsonl(manifest_root / "repairability/extension_trace_manifest.jsonl", extras)
    atomic_jsonl(manifest_root / "safety/common_trace_manifest.jsonl", mixed)
    model_summaries: dict[str, Any] = {}
    full_rows_by_model: dict[str, list[dict[str, Any]]] = {}
    gpu_allocation = completion_worker_allocation(config)
    safety_summaries: dict[str, Any] = {}
    identity_hash = str(config["teacher_forced_completion"]["completed_configuration_hash"])
    settings_source = json.loads((_old_root(config) / "frozen_execution_manifest.json").read_text())
    model_settings = settings_source["model_settings"]
    for model_key in map(str, config["selected_models"]):
        model_rows, model_summary = _model_trace_rows(
            common,
            config=config,
            configuration_hash=digest,
            trace_identity_hash=identity_hash,
            model_key=model_key,
        )
        full_rows_by_model[model_key] = model_rows
        extension_rows = [
            row for row in model_rows if str(row["source_trace_id"]) in extra_source_ids
        ]
        if len(extension_rows) != 1672 or sum(len(row["eligible_checkpoint_offsets"]) for row in extension_rows) != 7093:
            raise RuntimeError(f"{model_key}: extension count differs")
        repair_root = manifest_root / "repairability/per_model" / model_key
        atomic_jsonl(repair_root / "trace_manifest.jsonl", model_rows)
        atomic_jsonl(repair_root / "extension_trace_manifest.jsonl", extension_rows)
        packs = build_execution_packs(
            extension_rows,
            model_key=model_key,
            base_seed=int(config["rollout"]["base_seed"]),
            configuration_hash=digest,
            engine_revision=completion_engine_revision(),
            traces_per_pack=int(config["execution_packs"]["traces_per_pack"]),
        )
        assignments = assign_packs_to_workers(packs, gpu_allocation[model_key])
        atomic_jsonl(repair_root / "extension_execution_packs.jsonl", packs)
        atomic_json(repair_root / "worker_assignments.json", {str(k): v for k, v in assignments.items()})

        safety_rows, safety_summary = _safety_model_rows(mixed, config, model_key)
        safety_root = manifest_root / "safety/per_model" / model_key
        atomic_jsonl(safety_root / "trace_manifest.jsonl", safety_rows)
        safety_packs = _simple_packs(
            safety_rows,
            model_key=model_key,
            kind="safety",
            size=int(config["teacher_forced_completion"]["safety_traces_per_pack"]),
            digest=digest,
        )
        atomic_jsonl(safety_root / "execution_packs.jsonl", safety_packs)
        safety_assignments = assign_packs_to_workers(
            safety_packs, gpu_allocation[model_key]
        )
        atomic_json(
            safety_root / "worker_assignments.json",
            {str(k): v for k, v in safety_assignments.items()},
        )
        model_summaries[model_key] = {
            **model_summary,
            "extension_traces": len(extension_rows),
            "extension_checkpoints": sum(len(row["eligible_checkpoint_offsets"]) for row in extension_rows),
            "extension_rollouts": sum(int(pack["rollout_count"]) for pack in packs),
            "extension_packs": len(packs),
        }
        safety_summaries[model_key] = {**safety_summary, "packs": len(safety_packs)}

    reuse = _validate_old_reuse(config, full_rows_by_model)
    atomic_json(root / "reuse_ledger/completed_942.json", reuse)
    split_counts = Counter(row["pipeline_split"] for row in common)
    source_split = Counter((row["pipeline_split"], row["source_bucket"]) for row in common)
    frozen_execution = {
        "status": "FROZEN_FOR_PRODUCTION",
        "configuration_hash": digest,
        "engine_revision": completion_engine_revision(),
        "generation": dict(settings_source["generation"]),
        "model_settings": model_settings,
        "source_completed_freeze_digest": settings_source["freeze_digest"],
        "gpu_worker_allocation": gpu_allocation,
        "scheduler_policy": "concurrent_model_weighted_lpt_pack_assignment_v1",
    }
    frozen_execution["freeze_digest"] = stable_hash(frozen_execution)
    atomic_json(root / "frozen_execution_manifest.json", frozen_execution)
    payload = {
        **provenance(config),
        "status": "PREPARED",
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "configuration_hash": digest,
        "engine_revision": completion_engine_revision(),
        "source_commit": os.environ.get("SAFEPREFIX_SOURCE_COMMIT"),
        "scope": {
            "boundary_training": False,
            "unsafe_label_derivation": False,
            "native_evaluation": False,
            "final_test_access": False,
        },
        "repairability": {
            "traces": len(common),
            "problem_groups": len(set(row["production_problem_group"] for row in common)),
            "split_counts": dict(split_counts),
            "source_split_counts": {f"{a}:{b}": n for (a, b), n in sorted(source_split.items())},
            "checkpoints_per_model": 10988,
            "rollouts_per_model": 43952,
            "reused_traces": 942,
            "reused_checkpoints_per_model": 3895,
            "extension_traces": 1672,
            "extension_checkpoints_per_model": 7093,
            "extension_rollouts_per_model": 28372,
            "included_reconciliation_fates": sorted(REPAIRABILITY_FATES),
            "protected_fates": sorted(PROTECTED_FATES),
        },
        "safety": {
            "traces": 5900,
            "train": 5100,
            "dev": 800,
            "representation": SAFETY_REPRESENTATION,
            "model_summaries": safety_summaries,
        },
        "models": model_summaries,
        "mixed_manifest_hashes": mixed_hashes,
        "gpu_worker_allocation": gpu_allocation,
        "reconciliation_lineage_sha256": sha256_file(
            ROOT / str(config["teacher_forced_completion"]["reconciliation_lineage"])
        ),
        "reuse_ledger_sha256": sha256_file(root / "reuse_ledger/completed_942.json"),
        **parser_verifier_hashes(),
    }
    atomic_json(manifest_root / "immutable_protocol_manifest.json", payload)
    atomic_text(root / "resolved_config.yaml", dump_resolved(config))
    atomic_json(root / "pre_run_summary.json", payload)
    atomic_json(
        root / "source_access_ledger.json",
        {
            "status": "PASS",
            "opened": ["teacher_forced_train", "teacher_forced_dev", "selected_enriched_rows"],
            "selected_enriched_source_trace_ids": len(extra_source_ids),
            "native_development_outputs_opened": False,
            "prompt_pilot_outputs_opened": False,
            "final_test_outputs_opened": False,
            "protected_group_content_opened": False,
        },
    )
    return payload


def load_completion_context(
    config: Mapping[str, Any], *, run_id: str, model_key: str
) -> dict[str, Any]:
    root = completion_root(config, run_id)
    frozen = json.loads((root / "frozen_execution_manifest.json").read_text())
    if frozen.get("status") != "FROZEN_FOR_PRODUCTION":
        raise RuntimeError("completion execution is not frozen")
    if frozen.get("configuration_hash") != stable_hash(config) or frozen.get("engine_revision") != completion_engine_revision():
        raise RuntimeError("completion execution identity differs")
    repair_root = root / "immutable_manifests/repairability/per_model" / model_key
    safety_root = root / "immutable_manifests/safety/per_model" / model_key
    repair_assignments = json.loads((repair_root / "worker_assignments.json").read_text())
    safety_assignments = json.loads((safety_root / "worker_assignments.json").read_text())
    return {
        "root": root,
        "frozen": frozen,
        "repair_rows": {row["trace_id"]: row for row in read_jsonl(repair_root / "extension_trace_manifest.jsonl")},
        "repair_packs": {pack["pack_id"]: pack for pack in read_jsonl(repair_root / "extension_execution_packs.jsonl")},
        "repair_assignments": {
            int(worker): list(map(str, pack_ids))
            for worker, pack_ids in repair_assignments.items()
        },
        "safety_rows": {row["trace_id"]: row for row in read_jsonl(safety_root / "trace_manifest.jsonl")},
        "safety_packs": {pack["pack_id"]: pack for pack in read_jsonl(safety_root / "execution_packs.jsonl")},
        "safety_assignments": {
            int(worker): list(map(str, pack_ids))
            for worker, pack_ids in safety_assignments.items()
        },
        "settings": frozen["model_settings"][model_key],
        "generation": frozen["generation"],
    }


def execute_extension_pack(
    config: Mapping[str, Any], *, run_id: str, model_key: str, pack_id: str, loaded: Any,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    context = dict(context or load_completion_context(config, run_id=run_id, model_key=model_key))
    pack = context["repair_packs"][pack_id]
    settings = context["settings"]
    return execute_pack(
        config=config,
        model_key=model_key,
        loaded=loaded,
        pack=pack,
        trace_rows=context["repair_rows"],
        pack_root=context["root"] / "repairability/raw_rollout_shards" / model_key / pack_id,
        generation=context["generation"],
        batch_size=int(settings["branch_batch_size"]),
        compaction_quantum=int(settings["compaction_quantum"]),
        prefill_chunk_size=int(settings["prefill_chunk_size"]),
        traces_per_decode_group=int(settings["traces_per_decode_group"]),
        maximum_decode_kv_bytes=int(settings["maximum_decode_kv_bytes"]),
        mode="production",
        execution_freeze_digest=str(context["frozen"]["freeze_digest"]),
        model_execution_settings_hash=str(settings["settings_hash"]),
    )


def _safety_pack_root(root: Path, model_key: str, pack_id: str) -> Path:
    return root / "safety/hidden_state_feature_shards" / model_key / pack_id


def valid_safety_pack(
    pack: Mapping[str, Any], pack_root: Path, rows: Mapping[str, Mapping[str, Any]], *,
    model_revision: str | None, selected_layers: Sequence[int],
) -> bool:
    marker_path = pack_root / "complete.json"
    features_path = pack_root / "features.pt"
    metadata_path = pack_root / "checkpoint_metadata.parquet"
    if not all(path.is_file() for path in (marker_path, features_path, metadata_path)):
        return False
    try:
        marker = json.loads(marker_path.read_text())
        if marker.get("pack_hash") != pack["pack_hash"]:
            return False
        if marker.get("features_sha256") != sha256_file(features_path) or marker.get("metadata_sha256") != sha256_file(metadata_path):
            return False
        payload = torch.load(features_path, map_location="cpu", weights_only=False)
        if set(payload) != set(pack["trace_ids"]):
            return False
        metadata = pd.read_parquet(metadata_path)
        for trace_id in pack["trace_ids"]:
            row = rows[trace_id]
            value = payload[trace_id]
            if list(value["checkpoint_offsets"]) != list(row["checkpoint_offsets"]):
                return False
            if list(value["visible_safety_labels"]) != list(row["visible_safety_labels"]):
                return False
            if list(value["selected_layers"]) != list(selected_layers):
                return False
            if str(value["model_revision"]) != str(model_revision):
                return False
            if value["features"].shape[0] != len(row["checkpoint_offsets"]):
                return False
        return len(metadata) == int(pack["checkpoint_count"])
    except Exception:
        return False


def execute_safety_pack(
    config: Mapping[str, Any], *, run_id: str, model_key: str, pack_id: str, loaded: Any,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    context = dict(context or load_completion_context(config, run_id=run_id, model_key=model_key))
    pack = context["safety_packs"][pack_id]
    rows = context["safety_rows"]
    layers = list(map(int, config["models"][model_key]["selected_hidden_state_layers"]))
    pack_root = _safety_pack_root(context["root"], model_key, pack_id)
    if valid_safety_pack(pack, pack_root, rows, model_revision=loaded.model_revision, selected_layers=layers):
        return {"status": "SKIPPED_VALID", "pack_id": pack_id}
    started = time.perf_counter()
    feature_payload: dict[str, Any] = {}
    metadata: list[dict[str, Any]] = []
    for trace_id in pack["trace_ids"]:
        row = rows[trace_id]
        ids = list(map(int, row["prompt_token_ids"])) + list(map(int, row["completion_token_ids"]))
        offsets = list(map(int, row["checkpoint_offsets"]))
        token_offsets = [value - 1 for value in offsets]
        final_offset = len(ids) - 1
        forced = teacher_force_token_ids_chunked(
            loaded.model,
            ids,
            prompt_count=len(row["prompt_token_ids"]),
            chunk_size=int(context["settings"]["prefill_chunk_size"]),
            selected_layers=layers,
            selected_token_offsets=sorted(set([*token_offsets, final_offset])),
            selected_checkpoint_offsets=(),
        )
        nll = -forced.token_log_probabilities
        features = build_checkpoint_features(
            forced.selected_hidden_states,
            token_offsets,
            final_offset=final_offset,
            prefix_total_tokens=len(ids),
            nll_by_token=nll,
        ).to(torch.float16)
        feature_payload[trace_id] = {
            "features": features,
            "checkpoint_offsets": offsets,
            "checkpoint_indices": list(row["checkpoint_indices"]),
            "visible_safety_labels": list(row["visible_safety_labels"]),
            "safety_supervision_mask": list(row["safety_supervision_mask"]),
            "first_error_zero_based": row["first_error_zero_based"],
            "selected_layers": layers,
            "model_revision": loaded.model_revision,
            "tokenizer_revision": row["tokenizer_revision"],
            "representation": SAFETY_REPRESENTATION,
            "pooling_rule": "checkpoint_final_token",
            "mean_all_token_nll": float(nll.mean()),
        }
        for index, offset in enumerate(offsets):
            metadata.append(
                {
                    "model_key": model_key,
                    "trace_id": trace_id,
                    "source_trace_id": row["source_trace_id"],
                    "problem_id": row["problem_id"],
                    "problem_group_hash": row["problem_group_hash"],
                    "pipeline_split": row["pipeline_split"],
                    "checkpoint_index": index,
                    "checkpoint_token_offset": offset,
                    "visible_safety_label": row["visible_safety_labels"][index],
                    "safety_supervision_mask": row["safety_supervision_mask"][index],
                    "terminal_correct": row["terminal_correct"],
                    "first_error_zero_based": row["first_error_zero_based"],
                    "feature_row_index": index,
                }
            )
        del forced
    features_path = pack_root / "features.pt"
    metadata_path = pack_root / "checkpoint_metadata.parquet"
    _atomic_torch(features_path, feature_payload)
    atomic_parquet(metadata_path, pd.DataFrame(metadata))
    marker = {
        "status": "COMPLETE",
        "pack_id": pack_id,
        "pack_hash": pack["pack_hash"],
        "model_key": model_key,
        "trace_count": len(pack["trace_ids"]),
        "checkpoint_count": len(metadata),
        "features_sha256": sha256_file(features_path),
        "metadata_sha256": sha256_file(metadata_path),
        "elapsed_seconds": time.perf_counter() - started,
        "rollouts_generated": 0,
        "boundary_training_occurred": False,
    }
    atomic_json(pack_root / "complete.json", marker)
    _commit_modal_volume()
    return marker


def validate_prepared_completion(config: Mapping[str, Any], *, run_id: str) -> dict[str, Any]:
    root = completion_root(config, run_id)
    protocol = json.loads((root / "immutable_manifests/immutable_protocol_manifest.json").read_text())
    failures = []
    if protocol.get("configuration_hash") != stable_hash(config):
        failures.append("configuration_hash")
    if protocol.get("engine_revision") != completion_engine_revision():
        failures.append("engine_revision")
    if protocol["repairability"]["traces"] != 2614 or protocol["repairability"]["checkpoints_per_model"] != 10988:
        failures.append("repairability_counts")
    if protocol["safety"]["traces"] != 5900:
        failures.append("safety_counts")
    for model_key in map(str, config["selected_models"]):
        context = load_completion_context(config, run_id=run_id, model_key=model_key)
        if len(context["repair_rows"]) != 1672 or sum(len(row["eligible_checkpoint_offsets"]) for row in context["repair_rows"].values()) != 7093:
            failures.append(f"{model_key}:extension")
        if len(context["safety_rows"]) != 5900:
            failures.append(f"{model_key}:safety")
    access = json.loads((root / "source_access_ledger.json").read_text())
    if access.get("final_test_outputs_opened") or access.get("native_development_outputs_opened"):
        failures.append("prohibited_access")
    result = {"passed": not failures, "failures": failures}
    if failures:
        raise RuntimeError(f"prepared completion manifests invalid: {failures}")
    return result


def aggregate_completion(config: Mapping[str, Any], *, run_id: str) -> dict[str, Any]:
    root = completion_root(config, run_id)
    old = _old_root(config)
    per_model: dict[str, Any] = {}
    compute_by_model: dict[str, Any] = {}
    total_rollouts = 0
    for model_key in map(str, config["selected_models"]):
        context = load_completion_context(config, run_id=run_id, model_key=model_key)
        settings = context["settings"]
        new_rows: list[dict[str, Any]] = []
        repair_markers: list[dict[str, Any]] = []
        for pack_id, pack in context["repair_packs"].items():
            pack_root = root / "repairability/raw_rollout_shards" / model_key / pack_id
            subset = {trace_id: context["repair_rows"][trace_id] for trace_id in pack["trace_ids"]}
            if not valid_pack_artifact(
                pack,
                pack_root,
                trace_rows=subset,
                expected_freeze_digest=context["frozen"]["freeze_digest"],
                expected_settings_hash=settings["settings_hash"],
                expected_layers=config["models"][model_key]["selected_hidden_state_layers"],
                expected_model_revision=config["models"][model_key]["revision"],
            ):
                raise RuntimeError(f"{model_key}: invalid extension pack {pack_id}")
            new_rows.extend(pd.read_parquet(pack_root / "rollouts.parquet").to_dict("records"))
            repair_markers.append(json.loads((pack_root / "complete.json").read_text()))
        old_rows = []
        for path in sorted((old / "raw_rollout_shards" / model_key).glob("*/rollouts.parquet")):
            old_rows.extend(pd.read_parquet(path).to_dict("records"))
        raw = [*old_rows, *new_rows]
        logical = [
            (row["trace_id"], int(row["checkpoint_index"]), int(row["rollout_index"]))
            for row in raw
        ]
        if len(raw) != 43952 or len(logical) != len(set(logical)):
            raise RuntimeError(f"{model_key}: complete raw rollout coverage differs")
        counts = Counter((row["trace_id"], int(row["checkpoint_index"])) for row in raw)
        if len(counts) != 10988 or set(counts.values()) != {4}:
            raise RuntimeError(f"{model_key}: checkpoints do not have exactly four outcomes")
        aggregate = aggregate_checkpoint_outcomes(raw)
        atomic_parquet(root / "repairability/aggregated_checkpoint_outcomes" / f"{model_key}.parquet", pd.DataFrame(aggregate))

        safety_rows = 0
        safety_checkpoints = 0
        safety_rollouts = 0
        safety_markers: list[dict[str, Any]] = []
        safety_metadata_frames: list[pd.DataFrame] = []
        for pack_id, pack in context["safety_packs"].items():
            pack_root = _safety_pack_root(root, model_key, pack_id)
            if not valid_safety_pack(
                pack,
                pack_root,
                context["safety_rows"],
                model_revision=config["models"][model_key]["revision"],
                selected_layers=config["models"][model_key]["selected_hidden_state_layers"],
            ):
                raise RuntimeError(f"{model_key}: invalid safety pack {pack_id}")
            marker = json.loads((pack_root / "complete.json").read_text())
            safety_markers.append(marker)
            safety_metadata_frames.append(pd.read_parquet(pack_root / "checkpoint_metadata.parquet"))
            safety_rows += int(marker["trace_count"])
            safety_checkpoints += int(marker["checkpoint_count"])
            safety_rollouts += int(marker["rollouts_generated"])
        if safety_rows != 5900 or safety_rollouts != 0:
            raise RuntimeError(f"{model_key}: safety corpus coverage differs")
        safety_index = pd.concat(safety_metadata_frames, ignore_index=True)
        if len(safety_index) != safety_checkpoints:
            raise RuntimeError(f"{model_key}: safety checkpoint index count differs")
        if safety_index.duplicated(["trace_id", "checkpoint_index"]).any():
            raise RuntimeError(f"{model_key}: duplicate safety checkpoint metadata")
        atomic_parquet(
            root / "safety/aggregated_checkpoint_metadata" / f"{model_key}.parquet",
            safety_index,
        )
        per_model[model_key] = {
            "repairability_traces": 2614,
            "repairability_checkpoints": len(counts),
            "repairability_rollouts": len(raw),
            "reused_rollouts": len(old_rows),
            "new_rollouts": len(new_rows),
            "safety_traces": safety_rows,
            "safety_checkpoints": safety_checkpoints,
            "safety_rollouts": safety_rollouts,
        }
        compute_by_model[model_key] = {
            "extension_pack_count": len(repair_markers),
            "safety_pack_count": len(safety_markers),
            "extension_pack_wall_seconds_sum": sum(float(row.get("total_wall_seconds", 0.0)) for row in repair_markers),
            "extension_prefill_seconds_sum": sum(float(row.get("prefill_seconds", 0.0)) for row in repair_markers),
            "extension_decode_seconds_sum": sum(float(row.get("decode_metrics", {}).get("decode_wall_seconds", 0.0)) for row in repair_markers),
            "extension_verifier_seconds_sum": sum(float(row.get("verifier_seconds", 0.0)) for row in repair_markers),
            "extension_generated_tokens": sum(int(row.get("decode_metrics", {}).get("useful_output_tokens", 0)) for row in repair_markers),
            "safety_forward_wall_seconds_sum": sum(float(row.get("elapsed_seconds", 0.0)) for row in safety_markers),
            "safety_rollouts_generated": 0,
        }
        total_rollouts += len(raw)
    summary = {
        "status": "INTEGRITY_VALIDATED",
        "completed_at": now_iso(),
        "per_model": per_model,
        "compute_by_model": compute_by_model,
        "total_rollouts": total_rollouts,
        "boundary_training_occurred": False,
        "unsafe_binary_labels_derived": False,
        "thresholds_tuned": False,
        "native_evaluation_occurred": False,
        "native_final_test_access_count": 0,
    }
    atomic_json(root / "final_summary.json", summary)
    atomic_json(root / "integrity_report.json", {"status": "PASS", **summary})
    atomic_json(
        root / "compute_and_infrastructure_report.json",
        {
            "status": "COMPLETE",
            "hardware_pool": "8 x H100",
            "execution_policy": "model-major dynamic immutable-pack dispatch",
            "compute_by_model": compute_by_model,
            "training_compute": 0,
            "native_evaluation_compute": 0,
        },
    )
    report = [
        "# SafePrefix teacher-forced corpus completion",
        "",
        "Status: **COMPLETE — INTEGRITY VALIDATED**",
        "",
        "The run extended raw k=4 repairability outcomes and extracted forward-only safety features. It did not train a boundary model, derive binary unsafe labels, tune thresholds, run native evaluation, or access final-test data.",
        "",
        "## Counts",
        "",
        "| Model | Repair traces | Checkpoints | Reused rollouts | New rollouts | Safety traces | Safety checkpoints |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_key, values in per_model.items():
        report.append(
            f"| {model_key} | {values['repairability_traces']} | {values['repairability_checkpoints']} | {values['reused_rollouts']} | {values['new_rollouts']} | {values['safety_traces']} | {values['safety_checkpoints']} |"
        )
    report += [
        "",
        "The 942-trace completed corpus was checksum-validated and reused without regeneration. The extension contains 1,672 traces and 7,093 checkpoints per model. Safety-only checkpoints generated zero suffix rollouts.",
    ]
    atomic_text(root / "FINAL_REPORT.md", "\n".join(report) + "\n")
    return summary
