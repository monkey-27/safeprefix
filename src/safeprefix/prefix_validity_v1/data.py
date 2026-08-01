"""Frozen full-step ProcessBench data for the prefix-validity probe.

The production recoverability corpus contains only root and pre-error states,
so it cannot train a prefix-validity classifier.  The earlier teacher-forced
completion run separately saved *all* source-step states for a 5,900-trace
safety corpus.  This module reuses those shards and extracts the exact
final-transformer-layer/current-checkpoint slice used by the production
recoverability linear probe.  It never launches inference or reads suffix
outcomes.

Indexing is deliberately explicit.  ProcessBench's ``first_error_zero_based``
is a zero-based source-step index.  Safety checkpoint zero is the prompt root;
checkpoint ``c > 0`` is the state after source step ``c - 1``.  Consequently
the label for checkpoint ``c`` is ``(c - 1) < first_error_zero_based``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import pandas as pd
import torch
import yaml

from safeprefix.boundary_v1.data import _extract_final_layer, sha256_file, stable_hash


PROCESSBENCH_DATASET = "Qwen/ProcessBench"
PROCESSBENCH_DOMAINS = {
    "math": "processbench_math",
    "olympiadbench": "processbench_olympiadbench",
    "omnimath": "processbench_omnimath",
}
FROZEN_SPLITS = ("train", "architecture_dev", "calibration", "teacher_forced_test")
SOURCE_REPRESENTATION = "three_layer_checkpoint_final_with_difference_and_trace_summary_v1"
TARGET_REPRESENTATION = "final_transformer_layer_checkpoint_final_token_v1"


class PrefixValidityDataError(RuntimeError):
    """A frozen-data invariant required by the validity probe was violated."""


@dataclass(frozen=True)
class PrefixValidityData:
    """Compact model-specific features and their shared scientific manifest."""

    rows: pd.DataFrame
    features: dict[str, torch.Tensor]
    exclusions: pd.DataFrame
    inventory: pd.DataFrame
    pack_inventory: pd.DataFrame
    indexing_convention: dict[str, Any]
    integrity: dict[str, Any]


@dataclass(frozen=True)
class PrefixValidityExtractionPlan:
    """All eligible traces, separated into reusable and new forward passes."""

    rows: pd.DataFrame
    summary: dict[str, Any]
    traces_by_model: dict[str, list[dict[str, Any]]]
    exclusions: pd.DataFrame | None = None


def indexing_convention() -> dict[str, Any]:
    """Return the resolved ProcessBench/source-step checkpoint convention."""

    return {
        "status": "RESOLVED",
        "annotation_field": "first_error_zero_based",
        "annotation_index_base": "zero",
        "annotation_meaning": "first annotated visibly erroneous source reasoning step",
        "checkpoint_index_base": "zero",
        "checkpoint_zero": "prompt-only root; excluded from probe training",
        "completed_step_mapping": "completed_step_zero_based = checkpoint_index - 1",
        "prefix_valid_rule": "prefix_valid = int(completed_step_zero_based < first_error_zero_based)",
        "true_boundary_rule": "latest valid checkpoint is first_error_zero_based; zero means root fallback",
        "first_erroneous_checkpoint": "checkpoint_index = first_error_zero_based + 1",
        "final_answer_accuracy_used": False,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _integer(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        result = int(value)
        if float(value) != float(result):
            return None
        return result
    except (TypeError, ValueError, OverflowError):
        return None


def completed_step_and_label(
    checkpoint_index: Any, first_error_zero_based: Any
) -> tuple[int | None, int | None, str | None]:
    """Map a full-step checkpoint to its completed step and binary target."""

    checkpoint = _integer(checkpoint_index)
    error = _integer(first_error_zero_based)
    if error is None or error < 0:
        return None, None, "missing_or_ambiguous_first_error_annotation"
    if checkpoint is None or checkpoint < 0:
        return None, None, "invalid_checkpoint_index"
    if checkpoint == 0:
        return None, None, "prompt_root_is_not_training_checkpoint"
    step = checkpoint - 1
    return step, int(step < error), None


def _final_answer_only(step_text: Any, final_answer_text: Any) -> bool:
    """Conservatively identify states containing an answer and no reasoning.

    ProcessBench often ends with a reasoned conclusion that happens to contain a
    box; that is a valid annotated reasoning step.  We exclude only an exact
    answer value or a bare deterministic answer marker around that value.
    """

    step = str(step_text or "").strip()
    answer = str(final_answer_text or "").strip()
    if not step or not answer:
        return False

    def normalize(value: str) -> str:
        value = value.strip().rstrip(".")
        value = re.sub(r"\s+", "", value)
        return value.lower()

    candidates = {normalize(answer), normalize(f"\\boxed{{{answer}}}")}
    stripped = re.sub(
        r"^(?:final\s+answer|answer|the\s+final\s+answer\s+is)\s*[:=]?\s*",
        "",
        step,
        flags=re.IGNORECASE,
    )
    return normalize(step) in candidates or normalize(stripped) in candidates


def _resolve_safety_manifest_root(root: Path) -> Path:
    candidates = [
        root,
        root / "immutable_manifests",
        root / "artifacts/teacher_forced_completion/immutable_manifests",
    ]
    for candidate in candidates:
        if (candidate / "safety/per_model").is_dir():
            return candidate
    raise FileNotFoundError(f"cannot locate frozen safety manifests below {root}")


def _load_frozen_splits(boundary_root: Path) -> tuple[dict[str, str], dict[str, str]]:
    split_root = boundary_root / "data/splits"
    assignments: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for split in FROZEN_SPLITS:
        path = split_root / f"{split}_problems.jsonl"
        rows = _read_jsonl(path)
        hashes[split] = sha256_file(path)
        for row in rows:
            group = str(row["problem_group"])
            previous = assignments.setdefault(group, split)
            if previous != split:
                raise PrefixValidityDataError(
                    f"problem group {group} crosses frozen splits {previous}/{split}"
                )
    if not assignments:
        raise PrefixValidityDataError("frozen problem-level split manifests are empty")
    return assignments, hashes


def _load_model_specs(config_path: Path) -> dict[str, dict[str, Any]]:
    payload = yaml.safe_load(config_path.read_text())
    source = payload.get("source", {}).get("expected_models", {})
    if not isinstance(source, dict) or len(source) != 4:
        raise PrefixValidityDataError("boundary configuration does not define four models")
    result: dict[str, dict[str, Any]] = {}
    for model, raw in source.items():
        entry = dict(raw)
        if int(entry.get("hidden_size", 0)) <= 0 or -1 not in entry.get("selected_layers", []):
            raise PrefixValidityDataError(f"{model}: missing final-layer representation metadata")
        result[str(model)] = entry
    return result


def _candidate_model_pack_roots(root: Path, model_key: str) -> list[Path]:
    candidates = [
        root / model_key,
        root / "safety" / model_key,
        root / "safety/hidden_state_feature_shards" / model_key,
        root / "artifacts/teacher_forced_completion/safety/hidden_state_feature_shards" / model_key,
    ]
    # Distributed completion artifacts can be merged under one directory per
    # workspace.  This bounded glob avoids a recursive filesystem crawl.
    candidates.extend(root.glob(f"*/safety/{model_key}"))
    return sorted({path.resolve() for path in candidates if path.is_dir()})


def _discover_pack_dirs(feature_roots: Sequence[Path], model_key: str) -> dict[str, Path]:
    by_id: dict[str, tuple[Path, tuple[str, str]]] = {}
    for root in feature_roots:
        for model_root in _candidate_model_pack_roots(Path(root), model_key):
            for pack in sorted(path for path in model_root.iterdir() if path.is_dir()):
                marker_path = pack / "complete.json"
                feature_path = pack / "features.pt"
                metadata_path = pack / "checkpoint_metadata.parquet"
                if not all(path.is_file() for path in (marker_path, feature_path, metadata_path)):
                    continue
                marker = json.loads(marker_path.read_text())
                if marker.get("status") != "COMPLETE":
                    continue
                pack_id = str(marker.get("pack_id") or pack.name)
                if str(marker.get("model_key")) != model_key or pack_id != pack.name:
                    raise PrefixValidityDataError(f"pack identity mismatch: {pack}")
                checksums = (sha256_file(feature_path), sha256_file(metadata_path))
                if marker.get("features_sha256") != checksums[0] or marker.get("metadata_sha256") != checksums[1]:
                    raise PrefixValidityDataError(f"pack checksum mismatch: {pack}")
                if pack_id in by_id:
                    previous, previous_hashes = by_id[pack_id]
                    if previous_hashes != checksums:
                        raise PrefixValidityDataError(
                            f"conflicting duplicate safety pack {pack_id}: {previous} / {pack}"
                        )
                    continue
                by_id[pack_id] = (pack, checksums)
    return {pack_id: value[0] for pack_id, value in by_id.items()}


def _alignment_status(trace: Mapping[str, Any], checkpoint_index: int) -> tuple[bool, dict[str, Any]]:
    audits = list(trace.get("source_step_boundary_audits") or [])
    step_index = checkpoint_index - 1
    if not 0 <= step_index < len(audits):
        return False, {"reason": "missing_source_step_boundary_audit"}
    audit = dict(audits[step_index])
    valid = (
        bool(audit.get("complete_step_included"))
        and not bool(audit.get("next_step_content_included"))
        and str(audit.get("boundary_policy")) == "snap_after_complete_source_step_v1"
        and _integer(audit.get("step_index")) == step_index
        and _integer(audit.get("token_offset")) is not None
    )
    return valid, {
        "alignment_exact": bool(audit.get("exact")),
        "alignment_snap_policy": audit.get("snap_policy"),
        "alignment_character_displacement": _integer(audit.get("character_displacement")),
        "alignment_boundary_policy": audit.get("boundary_policy"),
    }


def _selected_trace_rows(
    traces: Sequence[Mapping[str, Any]], assignments: Mapping[str, str], *, model_key: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for source in traces:
        row = dict(source)
        reason: str | None = None
        if row.get("source_dataset") != PROCESSBENCH_DATASET:
            reason = "excluded_non_processbench_or_crv"
        elif str(row.get("source_subset")) not in PROCESSBENCH_DOMAINS:
            reason = "excluded_unknown_processbench_domain"
        elif bool(row.get("terminal_correct")):
            reason = "excluded_terminally_correct_trace"
        elif str(row.get("index_base")) != "zero" or _integer(row.get("first_error_zero_based")) is None:
            reason = "missing_or_ambiguous_first_error_annotation"
        else:
            steps = list(row.get("reasoning_steps") or [])
            error = _integer(row.get("first_error_zero_based"))
            indices = list(map(int, row.get("checkpoint_indices") or range(len(steps) + 1)))
            offsets = list(map(int, row.get("checkpoint_offsets") or []))
            if error is None or not 0 <= error < len(steps):
                reason = "first_error_outside_reasoning_steps"
            elif indices != list(range(len(steps) + 1)) or len(offsets) != len(indices):
                reason = "source_step_checkpoint_mapping_is_not_exact"
            elif str(row.get("production_problem_group")) not in assignments:
                reason = "no_existing_four_way_problem_split_assignment"
        if reason is not None:
            excluded.append(
                {
                    "model_key": model_key,
                    "trace_id": row.get("trace_id"),
                    "source_trace_id": row.get("source_trace_id"),
                    "problem_id": row.get("problem_id"),
                    "problem_group": row.get("production_problem_group"),
                    "domain": PROCESSBENCH_DOMAINS.get(str(row.get("source_subset"))),
                    "pipeline_split": row.get("pipeline_split"),
                    "split": assignments.get(
                        str(row.get("production_problem_group")), row.get("pipeline_split")
                    ),
                    "exclusion_reason": reason,
                }
            )
            continue
        row["checkpoint_indices"] = list(range(len(row["reasoning_steps"]) + 1))
        selected.append(row)
    return selected, excluded


def _attach_reuse_identity(
    selected: Sequence[Mapping[str, Any]], safety_traces: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Attach an existing safety-shard trace ID when exact source rows match."""

    by_source: dict[str, Mapping[str, Any]] = {}
    for row in safety_traces:
        source = str(row.get("source_trace_id"))
        if source in by_source:
            raise PrefixValidityDataError(f"duplicate safety source trace {source}")
        by_source[source] = row
    output: list[dict[str, Any]] = []
    for raw in selected:
        row = dict(raw)
        existing = by_source.get(str(row["source_trace_id"]))
        if existing is not None:
            fields = (
                "problem_id", "problem_text", "reasoning_steps", "first_error_zero_based",
                "prompt_token_ids", "completion_token_ids", "checkpoint_offsets",
                "model_revision", "tokenizer_revision",
            )
            differences = [field for field in fields if existing.get(field) != row.get(field)]
            if differences:
                raise PrefixValidityDataError(
                    f"reusable safety trace differs for {row['source_trace_id']}: {differences}"
                )
            row["feature_trace_id"] = str(existing["trace_id"])
            row["feature_origin"] = "reused_teacher_forced_completion_r5"
        else:
            row["feature_trace_id"] = str(row["trace_id"])
            row["feature_origin"] = "new_full_step_teacher_forcing_required"
        output.append(row)
    return output


def build_prefix_validity_extraction_plan(
    *,
    safety_manifest_root: Path,
    boundary_root: Path,
    boundary_config_path: Path,
    reuse_completed_safety_features: bool = True,
) -> PrefixValidityExtractionPlan:
    """Plan exact reuse/new teacher forcing over all 979 ProcessBench traces.

    The eligible universe is the production recoverability manifest because it
    carries the existing four-way split and all 979 annotated ProcessBench
    failures.  The 5,900-trace safety corpus is a feature cache, not the source
    population: it covers only 466 of those failures.
    """

    manifest_root = _resolve_safety_manifest_root(Path(safety_manifest_root))
    assignments, split_hashes = _load_frozen_splits(Path(boundary_root))
    specs = _load_model_specs(Path(boundary_config_path))
    plan_rows: list[dict[str, Any]] = []
    traces_by_model: dict[str, list[dict[str, Any]]] = {}
    exclusion_rows: list[dict[str, Any]] = []
    per_model: dict[str, Any] = {}
    for model_key, spec in specs.items():
        repair_path = manifest_root / f"repairability/per_model/{model_key}/trace_manifest.jsonl"
        safety_path = manifest_root / f"safety/per_model/{model_key}/trace_manifest.jsonl"
        repair = _read_jsonl(repair_path)
        safety = _read_jsonl(safety_path)
        selected, excluded = _selected_trace_rows(repair, assignments, model_key=model_key)
        exclusion_rows.extend(excluded)
        if reuse_completed_safety_features:
            selected = _attach_reuse_identity(selected, safety)
        else:
            # The completion publication retained the safety manifests but not
            # the corresponding full-step tensor shards.  A run may therefore
            # freeze one uniform, fresh teacher-forcing pass rather than claim
            # reuse of tensors that are not present.  Trace/problem/split
            # membership is unchanged; only the feature-cache provenance is.
            selected = [
                {
                    **dict(row),
                    "feature_trace_id": str(row["trace_id"]),
                    "feature_origin": "new_full_step_teacher_forcing_required",
                }
                for row in selected
            ]
        if len(selected) != 979:
            raise PrefixValidityDataError(
                f"{model_key}: eligible ProcessBench population is {len(selected)}, expected 979"
            )
        traces_by_model[model_key] = selected
        counts = Counter(str(row["feature_origin"]) for row in selected)
        per_model[model_key] = {
            "eligible_traces": len(selected),
            "reused_full_step_traces": counts["reused_teacher_forced_completion_r5"],
            "new_teacher_forcing_traces": counts["new_full_step_teacher_forcing_required"],
            "source_steps": sum(len(row["reasoning_steps"]) for row in selected),
            "model_revision": spec["model_revision"],
            "tokenizer_revision": spec["tokenizer_revision"],
            "repairability_manifest_sha256": sha256_file(repair_path),
            "safety_manifest_sha256": sha256_file(safety_path),
        }
        for row in selected:
            plan_rows.append(
                {
                    "model_key": model_key,
                    "model_id": spec["model_id"],
                    "model_revision": spec["model_revision"],
                    "tokenizer_revision": spec["tokenizer_revision"],
                    "trace_id": row["trace_id"],
                    "feature_trace_id": row["feature_trace_id"],
                    "source_trace_id": row["source_trace_id"],
                    "problem_id": row["problem_id"],
                    "problem_group": row["production_problem_group"],
                    "domain": row["source_bucket"],
                    "split": assignments[str(row["production_problem_group"])],
                    "first_error_zero_based": int(row["first_error_zero_based"]),
                    "source_step_count": len(row["reasoning_steps"]),
                    "full_token_count": int(row["full_token_count"]),
                    "feature_origin": row["feature_origin"],
                }
            )
    frame = pd.DataFrame(plan_rows).sort_values(
        ["model_key", "split", "domain", "trace_id"]
    ).reset_index(drop=True)
    summary = {
        "status": "FROZEN",
        "source_population": "production repairability ProcessBench traces with frozen split",
        "split_manifest_hashes": split_hashes,
        "per_model": per_model,
        "total_model_traces": len(frame),
        "total_reused_traces": int(frame["feature_origin"].eq("reused_teacher_forced_completion_r5").sum()),
        "total_new_teacher_forcing_traces": int(
            frame["feature_origin"].eq("new_full_step_teacher_forcing_required").sum()
        ),
        "reuse_completed_safety_features": bool(reuse_completed_safety_features),
        "new_suffix_rollouts": 0,
        "excluded_trace_rows": len(exclusion_rows),
        "excluded_reason_counts": dict(
            Counter(str(row["exclusion_reason"]) for row in exclusion_rows)
        ),
    }
    exclusions = pd.DataFrame(exclusion_rows)
    return PrefixValidityExtractionPlan(frame, summary, traces_by_model, exclusions)


def _pack_trace_index(pack_dirs: Mapping[str, Path]) -> tuple[dict[str, tuple[str, int]], list[dict[str, Any]]]:
    index: dict[str, tuple[str, int]] = {}
    inventory: list[dict[str, Any]] = []
    ordered_packs = sorted(pack_dirs.items())
    for pack_number, (pack_id, pack) in enumerate(ordered_packs, start=1):
        print(
            f"[prefix-validity:data] indexing pack {pack_number}/{len(ordered_packs)}: {pack_id}",
            flush=True,
        )
        metadata = pd.read_parquet(pack / "checkpoint_metadata.parquet")
        required = {
            "model_key", "trace_id", "checkpoint_index", "checkpoint_token_offset",
            "visible_safety_label", "safety_supervision_mask", "feature_row_index",
        }
        missing = required - set(metadata.columns)
        if missing:
            raise PrefixValidityDataError(f"{pack_id}: metadata missing {sorted(missing)}")
        for trace_id, part in metadata.groupby("trace_id", sort=False):
            trace_id = str(trace_id)
            if trace_id in index:
                raise PrefixValidityDataError(f"trace appears in multiple safety packs: {trace_id}")
            index[trace_id] = (pack_id, int(len(part)))
        inventory.append(
            {
                "pack_id": pack_id,
                "pack_path": str(pack),
                "trace_count": int(metadata["trace_id"].nunique()),
                "checkpoint_count": int(len(metadata)),
                "features_sha256": sha256_file(pack / "features.pt"),
                "metadata_sha256": sha256_file(pack / "checkpoint_metadata.parquet"),
            }
        )
    return index, inventory


def _load_model_features(
    *,
    model_key: str,
    selected: Sequence[Mapping[str, Any]],
    pack_dirs: Mapping[str, Path],
    spec: Mapping[str, Any],
    assignments: Mapping[str, str],
    checkpoint_exclusions: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], torch.Tensor, list[dict[str, Any]]]:
    trace_index, pack_inventory = _pack_trace_index(pack_dirs)
    # ``features.pt`` identity is a pack-level invariant.  Hashing the same
    # multi-hundred-MB file once per checkpoint row caused thousands of
    # redundant full-file reads; reuse the already-audited pack digest.
    feature_sha_by_pack = {
        str(row["pack_id"]): str(row["features_sha256"])
        for row in pack_inventory
    }
    selected_ids = {str(row["feature_trace_id"]) for row in selected}
    missing = selected_ids - set(trace_index)
    if missing:
        raise PrefixValidityDataError(
            f"{model_key}: {len(missing)} selected traces lack frozen safety features; "
            "reuse is incomplete and teacher forcing would be required"
        )
    needed_packs = sorted({trace_index[trace_id][0] for trace_id in selected_ids})
    selected_by_id = {str(row["feature_trace_id"]): row for row in selected}
    feature_rows: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []
    hidden_size = int(spec["hidden_size"])
    expected_revision = str(spec["model_revision"])
    expected_tokenizer = str(spec["tokenizer_revision"])

    for pack_number, pack_id in enumerate(needed_packs, start=1):
        print(
            f"[prefix-validity:data] {model_key} loading pack "
            f"{pack_number}/{len(needed_packs)}: {pack_id}",
            flush=True,
        )
        pack = pack_dirs[pack_id]
        payload = torch.load(pack / "features.pt", map_location="cpu", weights_only=False)
        metadata = pd.read_parquet(pack / "checkpoint_metadata.parquet")
        for feature_trace_id in sorted(set(map(str, metadata["trace_id"])) & selected_ids):
            trace = selected_by_id[feature_trace_id]
            trace_id = str(trace["trace_id"])
            if feature_trace_id not in payload:
                raise PrefixValidityDataError(f"{pack_id}: payload lacks trace {feature_trace_id}")
            item = payload[feature_trace_id]
            if str(item.get("model_revision")) != expected_revision:
                raise PrefixValidityDataError(f"{model_key}/{trace_id}: model revision mismatch")
            if str(item.get("tokenizer_revision")) != expected_tokenizer:
                raise PrefixValidityDataError(f"{model_key}/{trace_id}: tokenizer revision mismatch")
            if str(item.get("representation")) != SOURCE_REPRESENTATION:
                raise PrefixValidityDataError(f"{model_key}/{trace_id}: representation mismatch")
            if list(map(int, item.get("selected_layers", []))) != list(map(int, spec["selected_layers"])):
                raise PrefixValidityDataError(f"{model_key}/{trace_id}: selected hidden layers mismatch")
            raw = item["features"]
            current_final = _extract_final_layer(raw, hidden_size)
            offsets = list(map(int, item["checkpoint_offsets"]))
            indices = list(map(int, item.get("checkpoint_indices", range(len(offsets)))))
            expected_offsets = list(map(int, trace["checkpoint_offsets"]))
            if offsets != expected_offsets or indices != list(range(len(offsets))):
                raise PrefixValidityDataError(f"{model_key}/{trace_id}: checkpoint offsets differ")
            if current_final.shape != (len(offsets), hidden_size):
                raise PrefixValidityDataError(f"{model_key}/{trace_id}: compact feature shape differs")
            if not torch.isfinite(current_final).all():
                raise PrefixValidityDataError(f"{model_key}/{trace_id}: non-finite hidden representation")

            trace_meta = metadata[metadata["trace_id"].astype(str).eq(feature_trace_id)].sort_values(
                "checkpoint_index"
            )
            if list(trace_meta["checkpoint_index"].astype(int)) != indices:
                raise PrefixValidityDataError(f"{model_key}/{trace_id}: metadata indices differ")
            if list(trace_meta["checkpoint_token_offset"].astype(int)) != offsets:
                raise PrefixValidityDataError(f"{model_key}/{trace_id}: metadata offsets differ")

            steps = list(trace["reasoning_steps"])
            error = int(trace["first_error_zero_based"])
            first_feature_row = len(feature_rows)
            first_record_row = len(records)
            for checkpoint_index in indices:
                completed_step, label, reason = completed_step_and_label(checkpoint_index, error)
                if reason is not None:
                    continue  # prompt root is an inference fallback, never a training row
                valid_alignment, alignment = _alignment_status(trace, checkpoint_index)
                if not valid_alignment:
                    raise PrefixValidityDataError(
                        f"{model_key}/{trace_id}/{checkpoint_index}: ambiguous step alignment"
                    )
                step_text = steps[int(completed_step)]
                if _final_answer_only(step_text, trace.get("final_answer_text")):
                    # A bare final answer is not a reasoning checkpoint.
                    if checkpoint_exclusions is not None:
                        checkpoint_exclusions.append({
                            "model_key": model_key,
                            "trace_id": trace_id,
                            "source_trace_id": trace["source_trace_id"],
                            "problem_id": trace["problem_id"],
                            "problem_group": trace["production_problem_group"],
                            "domain": PROCESSBENCH_DOMAINS[str(trace["source_subset"])],
                            "split": assignments[str(trace["production_problem_group"])],
                            "checkpoint_index": checkpoint_index,
                            "completed_step_zero_based": int(completed_step),
                            "exclusion_reason": "final_answer_only_state",
                        })
                    continue
                row_index = len(feature_rows)
                feature_rows.append(current_final[checkpoint_index].contiguous())
                checkpoint_meta = trace_meta[trace_meta["checkpoint_index"].eq(checkpoint_index)].iloc[0]
                if bool(checkpoint_meta["visible_safety_label"]) != bool(label):
                    raise PrefixValidityDataError(
                        f"{model_key}/{trace_id}/{checkpoint_index}: frozen label/index convention differs"
                    )
                if not bool(checkpoint_meta["safety_supervision_mask"]):
                    raise PrefixValidityDataError(
                        f"{model_key}/{trace_id}/{checkpoint_index}: masked annotation entered dataset"
                    )
                prefix_tokens = int(offsets[checkpoint_index])
                full_tokens = int(trace["full_token_count"])
                records.append(
                    {
                        "model_key": model_key,
                        "model_id": spec["model_id"],
                        "model_revision": expected_revision,
                        "tokenizer_revision": expected_tokenizer,
                        "problem_id": trace["problem_id"],
                        "problem_group": trace["production_problem_group"],
                        "trace_id": trace_id,
                        "source_trace_id": trace["source_trace_id"],
                        "domain": PROCESSBENCH_DOMAINS[str(trace["source_subset"])],
                        "split": assignments[str(trace["production_problem_group"])],
                        "checkpoint_id": f"{trace_id}:{checkpoint_index}",
                        "checkpoint_index": checkpoint_index,
                        "completed_step_zero_based": int(completed_step),
                        "first_error_zero_based": error,
                        "prefix_valid": int(label),
                        # Replaced below with the latest *retained* valid
                        # checkpoint.  This differs from ``error`` only when a
                        # non-reasoning answer-only state was excluded.
                        "true_last_valid_checkpoint": None,
                        "checkpoint_token_offset": prefix_tokens,
                        "normalized_token_position": prefix_tokens / max(full_tokens, 1),
                        "full_trace_token_count": full_tokens,
                        "hidden_feature_row_index": row_index,
                        "hidden_state_dimension": hidden_size,
                        "hidden_state_representation": TARGET_REPRESENTATION,
                        "source_feature_pack_id": pack_id,
                        "source_feature_trace_id": feature_trace_id,
                        "source_feature_origin": trace["feature_origin"],
                        "source_feature_sha256": feature_sha_by_pack[pack_id],
                        "source_feature_local_row": checkpoint_index,
                        "row_id": stable_hash([model_key, trace_id, checkpoint_index]),
                        **alignment,
                    }
                )
            if len(feature_rows) == first_feature_row:
                raise PrefixValidityDataError(f"{model_key}/{trace_id}: no eligible reasoning checkpoint")
            trace_records = records[first_record_row:]
            valid_checkpoints = [
                int(row["checkpoint_index"]) for row in trace_records if int(row["prefix_valid"]) == 1
            ]
            true_boundary = max(valid_checkpoints, default=0)
            for row in trace_records:
                row["true_last_valid_checkpoint"] = true_boundary
        del payload
    compact = torch.stack(feature_rows).to(torch.float16).contiguous()
    return records, compact, pack_inventory


def _inventory(rows: pd.DataFrame, exclusions: pd.DataFrame) -> pd.DataFrame:
    group = ["model_key", "domain", "split"]
    retained = (
        rows.groupby(group, dropna=False)
        .agg(
            problems=("problem_group", "nunique"),
            traces=("trace_id", "nunique"),
            checkpoints=("checkpoint_id", "size"),
            prefix_valid_checkpoints=("prefix_valid", "sum"),
        )
        .reset_index()
    )
    retained["prefix_invalid_checkpoints"] = (
        retained["checkpoints"] - retained["prefix_valid_checkpoints"]
    )
    relevant = exclusions[exclusions["domain"].isin(PROCESSBENCH_DOMAINS.values())]
    if len(relevant):
        count = relevant.groupby(group, dropna=False).size().rename("excluded_traces").reset_index()
        retained = retained.merge(count, on=group, how="outer")
    else:
        retained["excluded_traces"] = 0
    for column in (
        "problems", "traces", "checkpoints", "prefix_valid_checkpoints",
        "prefix_invalid_checkpoints", "excluded_traces",
    ):
        retained[column] = retained[column].fillna(0).astype(int)
    return retained.sort_values(group).reset_index(drop=True)


def _assert_integrity(rows: pd.DataFrame, features: Mapping[str, torch.Tensor]) -> None:
    if rows.empty:
        raise PrefixValidityDataError("no eligible full-step ProcessBench rows")
    row_key = ["model_key", "trace_id", "checkpoint_index"]
    if rows.duplicated(row_key).any():
        raise PrefixValidityDataError("duplicate model/trace/checkpoint rows")
    if not rows["prefix_valid"].isin([0, 1]).all():
        raise PrefixValidityDataError("non-binary prefix-validity target")
    for model, part in rows.groupby("model_key"):
        if set(part["prefix_valid"].astype(int)) != {0, 1}:
            raise PrefixValidityDataError(f"{model}: prefix-validity target lacks both classes")
        indices = sorted(part["hidden_feature_row_index"].astype(int))
        if indices != list(range(len(part))) or len(features[model]) != len(part):
            raise PrefixValidityDataError(f"{model}: compact feature index is not one-to-one")
        for _, trace in part.sort_values("checkpoint_index").groupby("trace_id", sort=False):
            labels = trace["prefix_valid"].astype(int).tolist()
            if any(left < right for left, right in zip(labels, labels[1:])):
                raise PrefixValidityDataError("ground-truth prefix validity is not non-increasing")
            boundaries = trace["true_last_valid_checkpoint"].astype(int).unique()
            if len(boundaries) != 1:
                raise PrefixValidityDataError("true boundary changes within one trace")
    leakage = rows.groupby("problem_group")["split"].nunique()
    if (leakage != 1).any():
        raise PrefixValidityDataError("problem group crosses frozen train/dev/calibration/test splits")


def validate_recoverability_feature_equivalence(
    rows: pd.DataFrame,
    features: Mapping[str, torch.Tensor],
    boundary_root: Path,
    *,
    minimum_cosine_similarity: float = 0.99,
    maximum_relative_l2_difference: float = 0.10,
) -> dict[str, Any]:
    """Compare shared pre-error states with the production feature store.

    Shape, checkpoint identity, layer identity, revisions, and token offsets are
    exact invariants checked elsewhere.  Bitwise equality is recorded here but
    cannot be required across two independent H100 BF16 SDPA executions: that
    would conflate the representation convention with low-order kernel
    numerics.  Non-bitwise rows must instead remain inside conservative,
    fixed geometric bounds.  The full diagnostics are retained rather than
    silently labelling the tensors identical.
    """

    if not 0.0 < float(minimum_cosine_similarity) <= 1.0:
        raise ValueError("minimum cosine similarity must lie in (0, 1]")
    if not 0.0 <= float(maximum_relative_l2_difference) < 1.0:
        raise ValueError("maximum relative L2 difference must lie in [0, 1)")

    canonical_path = Path(boundary_root) / "data/canonical_checkpoint_manifest.parquet"
    if not canonical_path.is_file():
        raise FileNotFoundError(canonical_path)
    canonical = pd.read_parquet(canonical_path)
    required = {
        "base_model", "source_trace_id", "checkpoint_ordinal", "feature_row_index",
    }
    if required - set(canonical.columns):
        raise PrefixValidityDataError("production canonical manifest lacks feature identities")
    result: dict[str, Any] = {}
    for model, full_rows in rows.groupby("model_key", sort=True):
        production = canonical[canonical["base_model"].eq(model)]
        store_path = Path(boundary_root) / f"data/features/{model}.pt"
        payload = torch.load(store_path, map_location="cpu", weights_only=False)
        production_features = payload["features"] if isinstance(payload, dict) else payload
        production_index = {
            (str(row.source_trace_id), int(row.checkpoint_ordinal)): int(row.feature_row_index)
            for row in production.itertuples(index=False)
        }
        compared = 0
        exact = 0
        maximum = 0.0
        total_mean = 0.0
        minimum_cosine = 1.0
        maximum_relative_l2 = 0.0
        non_bitwise_rows: list[dict[str, Any]] = []
        for row in full_rows.itertuples(index=False):
            key = (str(row.source_trace_id), int(row.checkpoint_index))
            if key not in production_index:
                continue
            left = features[model][int(row.hidden_feature_row_index)]
            right = production_features[production_index[key]]
            if left.shape != right.shape:
                raise PrefixValidityDataError(f"{model}: shared feature shape differs at {key}")
            difference = (left.float() - right.float()).abs()
            compared += 1
            is_exact = torch.equal(left, right)
            exact += int(is_exact)
            maximum = max(maximum, float(difference.max()))
            total_mean += float(difference.mean())
            if not is_exact:
                left_float = left.float()
                right_float = right.float()
                cosine = float(torch.nn.functional.cosine_similarity(
                    left_float, right_float, dim=0, eps=1e-12
                ))
                relative_l2 = float(
                    torch.linalg.vector_norm(left_float - right_float)
                    / torch.linalg.vector_norm(right_float).clamp_min(1e-12)
                )
                minimum_cosine = min(minimum_cosine, cosine)
                maximum_relative_l2 = max(maximum_relative_l2, relative_l2)
                non_bitwise_rows.append({
                    "source_trace_id": str(row.source_trace_id),
                    "checkpoint_index": int(row.checkpoint_index),
                    "maximum_absolute_difference": float(difference.max()),
                    "mean_absolute_difference": float(difference.mean()),
                    "cosine_similarity": cosine,
                    "relative_l2_difference": relative_l2,
                })
        if compared == 0:
            raise PrefixValidityDataError(f"{model}: no shared state for feature-equivalence audit")
        result[model] = {
            "shared_checkpoints_compared": compared,
            "bitwise_equal_checkpoints": exact,
            "bitwise_equal_rate": exact / compared,
            "maximum_absolute_difference": maximum,
            "mean_absolute_difference": total_mean / compared,
            "minimum_cosine_similarity": minimum_cosine,
            "maximum_relative_l2_difference": maximum_relative_l2,
            "non_bitwise_rows": non_bitwise_rows,
            "production_feature_store_sha256": sha256_file(store_path),
        }
        if (
            minimum_cosine < float(minimum_cosine_similarity)
            or maximum_relative_l2 > float(maximum_relative_l2_difference)
        ):
            raise PrefixValidityDataError(
                f"{model}: full-step representation exceeds frozen numerical bounds; "
                f"minimum cosine={minimum_cosine:.6f}, "
                f"maximum relative L2={maximum_relative_l2:.6f}"
            )
    return {
        "status": "PASS",
        "criterion": (
            "exact representation identity plus bitwise audit; independent BF16 reruns "
            "must have cosine >= 0.99 and relative L2 <= 0.10 at every shared checkpoint"
        ),
        "minimum_cosine_similarity": float(minimum_cosine_similarity),
        "maximum_relative_l2_difference": float(maximum_relative_l2_difference),
        "canonical_manifest_sha256": sha256_file(canonical_path),
        "models": result,
    }


def build_prefix_validity_data(
    *,
    safety_manifest_root: Path,
    safety_feature_roots: Sequence[Path],
    boundary_root: Path,
    boundary_config_path: Path,
    reuse_completed_safety_features: bool = True,
) -> PrefixValidityData:
    """Load, validate, and compact the frozen full-step feature corpus.

    All selected feature shards must already exist.  Missing coverage raises an
    error identifying that teacher forcing would be required; this function
    never silently recomputes it.
    """

    manifest_root = _resolve_safety_manifest_root(Path(safety_manifest_root))
    assignments, split_hashes = _load_frozen_splits(Path(boundary_root))
    specs = _load_model_specs(Path(boundary_config_path))
    plan = build_prefix_validity_extraction_plan(
        safety_manifest_root=manifest_root,
        boundary_root=boundary_root,
        boundary_config_path=boundary_config_path,
        reuse_completed_safety_features=reuse_completed_safety_features,
    )
    all_rows: list[dict[str, Any]] = []
    all_exclusions: list[dict[str, Any]] = (
        [] if plan.exclusions is None else plan.exclusions.to_dict("records")
    )
    all_pack_inventory: list[dict[str, Any]] = []
    features: dict[str, torch.Tensor] = {}
    manifest_hashes: dict[str, str] = {}

    for model_key, spec in specs.items():
        print(f"[prefix-validity:data] {model_key} materialization started", flush=True)
        path = manifest_root / f"repairability/per_model/{model_key}/trace_manifest.jsonl"
        manifest_hashes[model_key] = sha256_file(path)
        selected = plan.traces_by_model[model_key]
        # The extraction plan has already applied all trace-level eligibility
        # checks.  Its universe deliberately includes 513 traces/model absent
        # from the earlier 5,900-row safety sample.
        pack_dirs = _discover_pack_dirs(safety_feature_roots, model_key)
        records, compact, packs = _load_model_features(
            model_key=model_key,
            selected=selected,
            pack_dirs=pack_dirs,
            spec=spec,
            assignments=assignments,
            checkpoint_exclusions=all_exclusions,
        )
        all_rows.extend(records)
        all_pack_inventory.extend({"model_key": model_key, **row} for row in packs)
        features[model_key] = compact
        print(
            f"[prefix-validity:data] {model_key} materialization complete: "
            f"{len(records)} checkpoints from {len(selected)} traces",
            flush=True,
        )

    rows = pd.DataFrame(all_rows).sort_values(
        ["model_key", "trace_id", "checkpoint_index"]
    ).reset_index(drop=True)
    # Compaction indices were created model-locally before the global sort.  A
    # deterministic trace order makes both orders equal; assert rather than
    # rewriting identities after the fact.
    exclusions = pd.DataFrame(all_exclusions)
    if exclusions.empty:
        exclusions = pd.DataFrame(
            columns=[
                "model_key", "trace_id", "source_trace_id", "problem_id", "domain",
                "pipeline_split", "exclusion_reason",
            ]
        )
    pack_inventory = pd.DataFrame(all_pack_inventory)
    _assert_integrity(rows, features)
    print("[prefix-validity:data] recoverability feature-equivalence audit started", flush=True)
    equivalence = validate_recoverability_feature_equivalence(rows, features, boundary_root)
    print("[prefix-validity:data] recoverability feature-equivalence audit complete", flush=True)
    inventory = _inventory(rows, exclusions)
    convention = indexing_convention()
    per_model = {}
    for model, part in rows.groupby("model_key"):
        per_model[model] = {
            "traces": int(part["trace_id"].nunique()),
            "checkpoints": int(len(part)),
            "prefix_valid": int(part["prefix_valid"].sum()),
            "prefix_invalid": int((part["prefix_valid"] == 0).sum()),
            "feature_shape": list(features[model].shape),
            "feature_dtype": str(features[model].dtype),
        }
    integrity = {
        "status": "PASS",
        "source_completion_run": "safeprefix_teacher_forced_completion_20260727_r5",
        "source_safety_traces_per_model": 5900,
        "eligible_processbench_traces_per_model": 979,
        "teacher_forcing_reused": bool(reuse_completed_safety_features),
        "new_teacher_forcing_required": True,
        "extraction_plan": plan.summary,
        "new_suffix_rollouts_generated": False,
        "native_data_accessed": False,
        "source_representation": SOURCE_REPRESENTATION,
        "target_representation": TARGET_REPRESENTATION,
        "split_manifest_hashes": split_hashes,
        "safety_trace_manifest_hashes": manifest_hashes,
        "recoverability_feature_equivalence": equivalence,
        "per_model": per_model,
        "excluded_reason_counts": dict(Counter(exclusions["exclusion_reason"])) if len(exclusions) else {},
    }
    return PrefixValidityData(
        rows=rows,
        features=features,
        exclusions=exclusions,
        inventory=inventory,
        pack_inventory=pack_inventory,
        indexing_convention=convention,
        integrity=integrity,
    )
