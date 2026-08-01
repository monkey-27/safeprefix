"""Rollout-only production manifests, execution, and exact-once artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import torch

from safeprefix.config import dump_resolved
from safeprefix.full_teacher_forced import (
    aggregate_checkpoint_outcomes,
    assign_packs_to_workers,
    build_execution_packs,
    reasoning_steps_list,
    select_frozen_manifest_failures,
    validate_pack_rows,
)
from safeprefix.manifests import decode_reference_answer
from safeprefix.models.cache_checkpoint import tokenizer_checkpoint_metadata
from safeprefix.models.hidden_features import build_checkpoint_features
from safeprefix.models.loader import load_model, load_tokenizer
from safeprefix.models.teacher_forcing import teacher_force_token_ids_chunked
from safeprefix.parsing.answer_parsers import parse_answer_region
from safeprefix.parsing.token_alignment import tokenize_with_offsets
from safeprefix.prompting.chat_format import render_chat_prompt
from safeprefix.prompting.templates import problem_instruction
from safeprefix.reproducibility import (
    atomic_json,
    atomic_jsonl,
    atomic_parquet,
    atomic_text,
    now_iso,
    package_versions,
    provenance,
    stable_hash,
)
from safeprefix.rollout.production_engine import (
    ProductionRolloutRequest,
    decode_execution_pack,
)
from safeprefix.rollout.verifier import resolve_verifier


ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_SCHEMA_VERSION = 1
SOURCE_STEP_BOUNDARY_POLICY = "snap_after_complete_source_step_v1"
ANSWER_REGION_AUDIT_POLICY = (
    "source_terminal_or_self_delimiting_high_confidence_v1"
)


def run_root(config: Mapping[str, Any], run_id: str) -> Path:
    if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
        raise ValueError("run_id must be one safe path component")
    external = os.environ.get("SAFEPREFIX_RUNS_ROOT")
    if external:
        return Path(external) / run_id / "artifacts" / "full_teacher_forced_suite"
    return ROOT / str(config["artifacts_root"]) / run_id


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def engine_revision() -> str:
    files = [
        ROOT / "src/safeprefix/full_teacher_forced.py",
        ROOT / "src/safeprefix/production_suite.py",
        ROOT / "src/safeprefix/models/cache_checkpoint.py",
        ROOT / "src/safeprefix/models/cache_utils.py",
        ROOT / "src/safeprefix/models/teacher_forcing.py",
        ROOT / "src/safeprefix/rollout/production_engine.py",
        ROOT / "src/safeprefix/parsing/answer_parsers.py",
        ROOT / "src/safeprefix/rollout/verifier.py",
        ROOT / "src/safeprefix/data/reference_join.py",
        ROOT / "scripts/16_audit_processbench_references.py",
    ]
    return stable_hash(
        [(str(path.relative_to(ROOT)), sha256_file(path)) for path in files]
    )


def parser_verifier_hashes() -> dict[str, str]:
    return {
        "parser_sha256": sha256_file(
            ROOT / "src/safeprefix/parsing/answer_parsers.py"
        ),
        "verifier_sha256": sha256_file(
            ROOT / "src/safeprefix/rollout/verifier.py"
        ),
    }


def assert_rollout_only(config: Mapping[str, Any]) -> None:
    suite = config.get("full_teacher_forced_suite", {})
    gates = config.get("phase_gates", {})
    disabled = config.get("disabled_stages", {})
    if not suite.get("rollout_only") or not suite.get("teacher_forced_only"):
        raise RuntimeError("production suite must be teacher-forced rollout-only")
    if int(suite.get("rollouts_per_checkpoint", -1)) != 4:
        raise RuntimeError("production suite requires exactly k=4")
    if (
        suite.get("source_step_token_boundary_policy")
        != SOURCE_STEP_BOUNDARY_POLICY
    ):
        raise RuntimeError("source-step token boundary policy must remain frozen")
    if (
        suite.get("answer_region_checkpoint_audit_policy")
        != ANSWER_REGION_AUDIT_POLICY
    ):
        raise RuntimeError("answer-region checkpoint audit policy must remain frozen")
    if int(config.get("rollout", {}).get("max_new_tokens", -1)) != 4096:
        raise RuntimeError("production suite requires max_new_tokens=4096")
    required_locked = {
        "repairability_label_processing",
        "teacher_forced_boundary_training",
        "native_free_form_evaluation",
        "native_final_test",
    }
    if any(gates.get(key) != "locked" for key in required_locked):
        raise RuntimeError("training, label processing, and native evaluation must remain locked")
    if not all(bool(value) for value in disabled.values()):
        raise RuntimeError("every prohibited downstream stage must remain disabled")
    if os.environ.get("SAFEPREFIX_ROLLOUT_ONLY", "1") != "1":
        raise RuntimeError("SAFEPREFIX_ROLLOUT_ONLY guard is not enabled")


def _json_value(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def canonical_completion(steps: Sequence[str]) -> tuple[str, list[tuple[int, int]]]:
    pieces = ["\n"]
    ranges: list[tuple[int, int]] = []
    cursor = 1
    for index, raw in enumerate(steps):
        if index:
            pieces.append("\n\n")
            cursor += 2
        step = str(raw)
        start = cursor
        pieces.append(step)
        cursor += len(step)
        ranges.append((start, cursor))
    return "".join(pieces), ranges


def source_step_boundary_audits(
    completion_alignment: Any,
    ranges: Sequence[tuple[int, int]],
) -> list[dict[str, Any]]:
    """Map source-step ends to token prefixes that contain the complete step.

    A subword can span the final characters of a step and the canonical ``\n\n``
    separator.  Snapping before that token silently drops part of the annotated
    step.  Production therefore snaps after the token, while asserting that the
    resulting prefix never reaches characters belonging to the next step.
    """

    audits: list[dict[str, Any]] = []
    for step_index, (_, requested_end) in enumerate(ranges):
        resolved = dict(
            completion_alignment.resolve_boundary(int(requested_end), policy="after")
        )
        next_step_start = (
            int(ranges[step_index + 1][0])
            if step_index + 1 < len(ranges)
            else int(completion_alignment.text_length)
        )
        resolved_end = int(resolved["resolved_char_offset"])
        if resolved_end < int(requested_end):
            raise RuntimeError("source-step token boundary omitted step characters")
        if resolved_end > next_step_start:
            raise RuntimeError(
                "source-step token boundary entered the next dataset step"
            )
        audits.append(
            {
                "step_index": int(step_index),
                "requested_char_offset": int(requested_end),
                "token_offset": int(resolved["token_offset"]),
                "resolved_char_offset": resolved_end,
                "exact": bool(resolved["exact"]),
                "snap_policy": "after",
                "boundary_policy": SOURCE_STEP_BOUNDARY_POLICY,
                "character_displacement": int(resolved["character_displacement"]),
                "next_step_char_start": next_step_start,
                "complete_step_included": True,
                "next_step_content_included": False,
            }
        )
    return audits


def answer_region_checkpoint_audit(
    completion: str,
    ranges: Sequence[tuple[int, int]],
    *,
    final_answer_text: Any,
    first_error_zero_based: int,
    eligible_step_boundaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Prove eligible prefixes do not enter an operational final-answer region.

    The primary boundary is either a source-declared answer occurring terminally
    in the recorded completion or a self-delimiting, high-confidence parser
    region.  We deliberately do not promote medium-confidence isolated-line or
    prose fallbacks: on recorded ProcessBench traces they can identify an
    intermediate equation rather than the final answer.  Those candidates are
    retained as diagnostics instead of silently becoming scientific boundaries.
    """

    import re

    error = int(first_error_zero_based)
    if not 0 <= error < len(ranges):
        raise ValueError("first error lies outside source-step ranges")
    declared = str(final_answer_text or "").strip()
    terminal_matches = []
    if declared:
        terminal_matches = [
            match
            for match in re.finditer(re.escape(declared), completion)
            if not completion[match.end() :].strip(" .!\n\t")
        ]
    parsed = parse_answer_region(completion)
    high_confidence_methods = {
        "last_boxed",
        "final_answer_marker",
        "dataset_specific",
    }
    candidates: list[tuple[int, int, str]] = []
    if terminal_matches:
        match = terminal_matches[-1]
        candidates.append(
            (int(match.start()), int(match.end()), "source_final_answer_terminal_match")
        )
    if (
        parsed.success
        and parsed.method in high_confidence_methods
        and parsed.char_start is not None
        and parsed.char_end is not None
    ):
        candidates.append(
            (int(parsed.char_start), int(parsed.char_end), str(parsed.method))
        )
    # Use the earliest accepted region so the exclusion check is conservative.
    answer_start: int | None = None
    answer_end: int | None = None
    answer_method = "conservative_final_source_step_container"
    used_conservative_container = False
    if candidates:
        answer_start, answer_end, answer_method = min(
            candidates, key=lambda value: (value[0], value[1], value[2])
        )
    else:
        # When neither the source declaration nor a self-delimiting parser rule
        # localizes a narrower region, conservatively treat the entire final
        # dataset-provided step as answer-containing. Eligible checkpoints end
        # before the first error, and every valid first error is at or before the
        # final step, so this is a schema-backed exclusion boundary rather than a
        # vacuous "no answer found" pass.
        answer_start, answer_end = map(int, ranges[-1])
        used_conservative_container = True
    latest_requested = 0 if error == 0 else int(ranges[error - 1][1])
    latest_resolved = (
        0
        if error == 0
        else int(eligible_step_boundaries[error - 1]["resolved_char_offset"])
    )
    before_answer = answer_start is None or latest_resolved <= answer_start
    audit = {
        "policy": ANSWER_REGION_AUDIT_POLICY,
        "status": "PASS" if before_answer else "FAIL",
        "answer_region_identified": not used_conservative_container,
        "answer_region_exclusion_boundary_identified": True,
        "answer_region_method": answer_method,
        "answer_char_start": answer_start,
        "answer_char_end": answer_end,
        "declared_final_answer_present": bool(declared),
        "declared_terminal_match": bool(terminal_matches),
        "answer_region_schema_proof": (
            "entire_final_dataset_step_treated_as_answer_container"
            if used_conservative_container
            else None
        ),
        "parser_success": bool(parsed.success),
        "parser_method": str(parsed.method),
        "parser_confidence": str(parsed.confidence),
        "parser_candidate_char_start": parsed.char_start,
        "parser_candidate_accepted": bool(parsed.method in high_confidence_methods),
        "latest_eligible_requested_char_offset": latest_requested,
        "latest_eligible_resolved_char_offset": latest_resolved,
        "eligible_checkpoint_count": error + 1,
        "all_eligible_checkpoints_before_answer_region": bool(before_answer),
    }
    if not before_answer:
        raise RuntimeError(
            "eligible source-step checkpoint enters the deterministic final-answer region"
        )
    return audit


def _frozen_manifest_root(config: Mapping[str, Any]) -> Path:
    settings = config["frozen_teacher_forced_manifests"]
    override = os.environ.get("SAFEPREFIX_FROZEN_MANIFEST_ROOT")
    return Path(override or str(settings["root"]))


def _load_authoritative_frozen_cohort(
    config: Mapping[str, Any], root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read only the pre-authorized train/dev roles and their non-content summary.

    Native-development, prompt-pilot, and final-test manifest files are never
    opened or scanned by this rollout-only pipeline.
    """

    settings = config["frozen_teacher_forced_manifests"]
    source_root = _frozen_manifest_root(config)
    allowed = {
        "summary": source_root / str(settings["summary_file"]),
        "teacher_forced_train": source_root
        / str(settings["train_file"]),
        "teacher_forced_dev": source_root / str(settings["dev_file"]),
    }
    missing = [name for name, path in allowed.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"authoritative frozen teacher-forced inputs are missing: {missing}"
        )
    summary = json.loads(allowed["summary"].read_text(encoding="utf-8"))
    if summary.get("status") != "COMPLETE":
        raise RuntimeError("frozen configuration manifest summary is not complete")
    if str(summary.get("configuration_hash")) != str(
        settings["source_configuration_hash"]
    ):
        raise RuntimeError("frozen source configuration hash differs")
    rows_by_role = {
        role: read_jsonl(path)
        for role, path in allowed.items()
        if role != "summary"
    }
    for role, rows in rows_by_role.items():
        expected_count = int(settings["expected_counts"][role])
        expected_hash = str(settings["manifest_hashes"][role])
        if len(rows) != expected_count:
            raise RuntimeError(
                f"frozen {role} count {len(rows)} != {expected_count}"
            )
        if stable_hash(rows) != expected_hash:
            raise RuntimeError(f"frozen {role} content hash differs")
        if str(summary["manifest_hashes"].get(role)) != expected_hash:
            raise RuntimeError(f"frozen {role} summary hash differs")
    common, cohort_summary, exclusions = select_frozen_manifest_failures(
        rows_by_role["teacher_forced_train"],
        rows_by_role["teacher_forced_dev"],
        error_bins=config["full_teacher_forced_suite"]["first_error_bins"],
    )
    access = {
        "status": "PASS",
        "source_root": str(source_root),
        "files_opened": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in allowed.items()
        },
        "allowed_roles": ["teacher_forced_train", "teacher_forced_dev"],
        "native_configuration_development_opened": False,
        "prompt_pilot_manifest_opened": False,
        "native_final_test_manifest_opened": False,
        "native_or_final_outputs_opened": False,
        "source_configuration_hash": summary["configuration_hash"],
        "source_manifest_hashes": {
            role: settings["manifest_hashes"][role]
            for role in ("teacher_forced_train", "teacher_forced_dev")
        },
        "selected_eligible_traces": len(common),
        "excluded_rows": len(exclusions),
    }
    atomic_json(root / "source_manifest_access_ledger.json", access)
    atomic_jsonl(root / "immutable_manifests/source_exclusions.jsonl", exclusions)
    atomic_json(root / "immutable_manifests/source_summary.json", summary)
    return common, {"cohort": cohort_summary, "access": access}


def _model_trace_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    configuration_hash: str,
    model_key: str,
    trace_identity_hash: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    entry = config["models"][model_key]
    loaded = load_tokenizer(entry)
    tokenizer = loaded.tokenizer
    prompt_condition = str(config.get("prompting", {}).get("selected_condition", "P0"))
    output: list[dict[str, Any]] = []
    length_counts: list[int] = []
    checkpoint_lengths: list[int] = []
    for common in rows:
        steps = list(map(str, common["reasoning_steps"]))
        completion, ranges = canonical_completion(steps)
        prompt = render_chat_prompt(
            tokenizer,
            problem_instruction(str(common["problem_text"]), prompt_condition),
            override=entry.get("chat_template_override"),
            template_kwargs=entry.get("chat_template_kwargs"),
        )
        prompt_ids = list(
            map(int, tokenizer(prompt, add_special_tokens=False)["input_ids"])
        )
        completion_alignment = tokenize_with_offsets(tokenizer, completion)
        completion_ids = list(completion_alignment.input_ids)
        boundary_audits = source_step_boundary_audits(completion_alignment, ranges)
        step_token_ends = [int(item["token_offset"]) for item in boundary_audits]
        if any(right <= left for left, right in zip([0, *step_token_ends[:-1]], step_token_ends)):
            raise RuntimeError(
                f"dataset step boundaries collapsed under {model_key}: {common['source_trace_id']}"
            )
        offsets = [len(prompt_ids)] + [
            len(prompt_ids) + value for value in step_token_ends
        ]
        error = int(common["first_error_zero_based"])
        eligible = offsets[: error + 1]
        if len(eligible) != error + 1 or not eligible:
            raise AssertionError("eligible source-step checkpoint enumeration drifted")
        answer_audit = answer_region_checkpoint_audit(
            completion,
            ranges,
            final_answer_text=common.get("final_answer_text"),
            first_error_zero_based=error,
            eligible_step_boundaries=boundary_audits,
        )
        checkpoint_boundary_audits = [
            {
                "checkpoint_index": 0,
                "checkpoint_kind": "prompt_root",
                "checkpoint_token_offset": len(prompt_ids),
                "source_step_index": None,
                "boundary_policy": "exact_prompt_token_boundary",
                "exact": True,
                "character_displacement": 0,
            }
        ] + [
            {
                **item,
                "checkpoint_index": int(item["step_index"]) + 1,
                "checkpoint_kind": "after_complete_source_step",
                "source_step_index": int(item["step_index"]),
                "checkpoint_token_offset": len(prompt_ids)
                + int(item["token_offset"]),
            }
            for item in boundary_audits
        ]
        configured_context = int(entry["max_context_length"])
        max_new = int(config["rollout"]["max_new_tokens"])
        full_token_count = len(prompt_ids) + len(completion_ids)
        if full_token_count > configured_context:
            raise RuntimeError(
                f"teacher-forced trace exceeds context for {model_key}: "
                f"{common['source_trace_id']} {full_token_count}>{configured_context}"
            )
        invalid_checkpoint = next(
            (
                offset
                for offset in eligible
                if int(offset) + max_new > configured_context
            ),
            None,
        )
        if invalid_checkpoint is not None:
            raise RuntimeError(
                f"checkpoint plus 4096-token continuation exceeds context for "
                f"{model_key}: {common['source_trace_id']} "
                f"{invalid_checkpoint}+{max_new}>{configured_context}"
            )
        model_trace_id = stable_hash(
            [
                "safeprefix-full-tf-model-trace-v1",
                trace_identity_hash or configuration_hash,
                model_key,
                common["source_trace_id"],
            ]
        )[:24]
        row = {
            **{str(key): _json_value(value) for key, value in common.items()},
            "common_trace_id": common["trace_id"],
            "trace_id": model_trace_id,
            "model_key": model_key,
            "model_id": entry["hf_model_id"],
            "model_revision": entry.get("revision"),
            "tokenizer_id": entry.get("tokenizer_id") or entry["hf_model_id"],
            "tokenizer_revision": loaded.tokenizer_revision,
            "prompt_condition": prompt_condition,
            "prompt_text": prompt,
            "recorded_completion": completion,
            "prompt_token_ids": prompt_ids,
            "completion_token_ids": completion_ids,
            "checkpoint_offsets": offsets,
            "eligible_checkpoint_offsets": eligible,
            "source_step_boundary_policy": SOURCE_STEP_BOUNDARY_POLICY,
            "source_step_boundary_audits": boundary_audits,
            "checkpoint_boundary_audits": checkpoint_boundary_audits,
            "eligible_checkpoint_boundary_audits": checkpoint_boundary_audits[
                : error + 1
            ],
            "answer_region_checkpoint_audit": answer_audit,
            "full_token_count": full_token_count,
            # Deterministic scheduling proxy only: one teacher-forced prefill
            # plus four suffix branches whose attention cost includes the
            # exact checkpoint prefix and the frozen continuation allowance.
            # Actual early stopping can only reduce the realized work.
            "estimated_rollout_work": (
                len(prompt_ids) + len(completion_ids)
                + 4 * sum(int(offset) + max_new for offset in eligible)
            ),
            "configuration_hash": configuration_hash,
        }
        output.append(row)
        length_counts.append(row["full_token_count"])
        checkpoint_lengths.extend(eligible)
    boundary_count = sum(
        len(row["source_step_boundary_audits"]) for row in output
    )
    exact_boundary_count = sum(
        int(bool(item["exact"]))
        for row in output
        for item in row["source_step_boundary_audits"]
    )
    eligible_boundary_count = sum(
        max(len(row["eligible_checkpoint_boundary_audits"]) - 1, 0)
        for row in output
    )
    eligible_exact_boundary_count = sum(
        int(bool(item["exact"]))
        for row in output
        for item in row["eligible_checkpoint_boundary_audits"][1:]
    )
    answer_methods = Counter(
        str(row["answer_region_checkpoint_audit"]["answer_region_method"])
        for row in output
    )
    boundary_by_source: dict[str, Any] = {}
    for source in sorted({str(row["source_bucket"]) for row in output}):
        source_items = [
            item
            for row in output
            if str(row["source_bucket"]) == source
            for item in row["source_step_boundary_audits"]
        ]
        source_answer_methods = Counter(
            str(row["answer_region_checkpoint_audit"]["answer_region_method"])
            for row in output
            if str(row["source_bucket"]) == source
        )
        boundary_by_source[source] = {
            "boundaries": len(source_items),
            "exact": sum(bool(item["exact"]) for item in source_items),
            "snapped": sum(not bool(item["exact"]) for item in source_items),
            "maximum_character_displacement": max(
                (int(item["character_displacement"]) for item in source_items),
                default=0,
            ),
            "answer_region_methods": dict(source_answer_methods),
        }
    return output, {
        "model_key": model_key,
        "traces": len(output),
        "checkpoints": len(checkpoint_lengths),
        "rollouts": len(checkpoint_lengths) * 4,
        "maximum_full_trace_tokens": max(length_counts),
        "maximum_checkpoint_tokens": max(checkpoint_lengths),
        "tokenizer_revision": loaded.tokenizer_revision,
        "source_step_boundary_policy": SOURCE_STEP_BOUNDARY_POLICY,
        "source_step_boundaries": boundary_count,
        "source_step_boundaries_exact": exact_boundary_count,
        "source_step_boundaries_snapped": boundary_count - exact_boundary_count,
        "eligible_nonroot_boundaries": eligible_boundary_count,
        "eligible_nonroot_boundaries_exact": eligible_exact_boundary_count,
        "eligible_nonroot_boundaries_snapped": (
            eligible_boundary_count - eligible_exact_boundary_count
        ),
        "maximum_boundary_character_displacement": max(
            (
                int(item["character_displacement"])
                for row in output
                for item in row["source_step_boundary_audits"]
            ),
            default=0,
        ),
        "answer_region_audit_policy": ANSWER_REGION_AUDIT_POLICY,
        "answer_region_methods": dict(answer_methods),
        "answer_regions_identified": sum(
            bool(row["answer_region_checkpoint_audit"]["answer_region_identified"])
            for row in output
        ),
        "answer_regions_not_identified": sum(
            not bool(row["answer_region_checkpoint_audit"]["answer_region_identified"])
            for row in output
        ),
        "answer_regions_conservative_final_step": sum(
            row["answer_region_checkpoint_audit"]["answer_region_method"]
            == "conservative_final_source_step_container"
            for row in output
        ),
        "answer_region_audit_failures": sum(
            row["answer_region_checkpoint_audit"]["status"] != "PASS"
            for row in output
        ),
        "boundary_audit_by_source": boundary_by_source,
    }


def prepare_manifests(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    run_id: str,
) -> dict[str, Any]:
    assert_rollout_only(config)
    root = run_root(config, run_id)
    root.mkdir(parents=True, exist_ok=True)
    configuration_hash = stable_hash(config)
    revision = engine_revision()
    common, source_details = _load_authoritative_frozen_cohort(config, root)
    cohort_summary = source_details["cohort"]
    answer_types = Counter(
        type(decode_reference_answer(row["reference_answer"])).__name__
        for row in common
    )
    unsupported_answer_types = sorted(
        answer_type
        for answer_type in answer_types
        if answer_type not in {"str", "int", "float"}
    )
    answer_schema = {
        "status": "PASS" if not unsupported_answer_types else "FAIL",
        "decoded_reference_types": dict(answer_types),
        "unsupported_types": unsupported_answer_types,
        "verifier": "ExactAnswerVerifier scalar exact/numeric tolerance",
        "multi_answer_reference_count": 0,
    }
    atomic_json(root / "reference_answer_schema_audit.json", answer_schema)
    if unsupported_answer_types:
        raise RuntimeError(
            f"exact verifier does not support frozen reference types: "
            f"{unsupported_answer_types}"
        )
    manifest_root = root / "immutable_manifests"
    atomic_jsonl(manifest_root / "common_trace_manifest.jsonl", common)
    model_summaries: dict[str, Any] = {}
    pack_summaries: dict[str, Any] = {}
    for model_key in config["selected_models"]:
        model_rows, model_summary = _model_trace_rows(
            common,
            config=config,
            configuration_hash=configuration_hash,
            model_key=str(model_key),
        )
        atomic_jsonl(
            manifest_root / "per_model" / str(model_key) / "trace_manifest.jsonl",
            model_rows,
        )
        atomic_json(
            manifest_root
            / "per_model"
            / str(model_key)
            / "checkpoint_boundary_manifest.json",
            {
                "status": "PASS",
                "model_key": str(model_key),
                "tokenizer_id": config["models"][model_key].get("tokenizer_id")
                or config["models"][model_key]["hf_model_id"],
                **model_summary,
            },
        )
        packs = build_execution_packs(
            model_rows,
            model_key=str(model_key),
            base_seed=int(config["rollout"]["base_seed"]),
            configuration_hash=configuration_hash,
            engine_revision=revision,
            traces_per_pack=int(config["execution_packs"]["traces_per_pack"]),
        )
        assignments = assign_packs_to_workers(
            packs, int(config["scheduler"]["workers_per_model_wave"])
        )
        atomic_jsonl(
            manifest_root / "per_model" / str(model_key) / "execution_packs.jsonl",
            packs,
        )
        atomic_json(
            manifest_root / "per_model" / str(model_key) / "worker_assignments.json",
            {str(key): value for key, value in assignments.items()},
        )
        model_summaries[str(model_key)] = model_summary
        pack_summaries[str(model_key)] = {
            "packs": len(packs),
            "workers": len(assignments),
            "worker_rollout_counts": {
                str(worker): sum(
                    int(pack["rollout_count"])
                    for pack in packs
                    if pack["pack_id"] in set(ids)
                )
                for worker, ids in assignments.items()
            },
        }
    payload = {
        **provenance(config),
        "status": "PREPARED",
        "run_id": run_id,
        "schema_version": PRODUCTION_SCHEMA_VERSION,
        "configuration_hash": configuration_hash,
        "engine_revision": revision,
        "source_commit": os.environ.get("SAFEPREFIX_SOURCE_COMMIT"),
        "source_bundle_sha256": os.environ.get("SAFEPREFIX_SOURCE_BUNDLE_SHA256"),
        "rollout_only": True,
        "boundary_training_enabled": False,
        "native_evaluation_enabled": False,
        "cohort": cohort_summary,
        "reference_join": {
            "source": "authoritative configuration-pilot teacher-forced manifests",
            "eligible_missing_references": 0,
            "excluded_missing_references": int(
                cohort_summary["exclusion_counts"].get(
                    "missing_exact_terminal_reference", 0
                )
            ),
            "answer_schema": answer_schema,
        },
        "source_manifest_access": source_details["access"],
        "models": model_summaries,
        "execution_packs": pack_summaries,
        "rollout": dict(config["rollout"]),
        **parser_verifier_hashes(),
    }
    atomic_json(manifest_root / "immutable_protocol_manifest.json", payload)
    atomic_text(root / "resolved_config.yaml", dump_resolved(config))
    atomic_json(root / "pre_run_summary.json", payload)
    return payload


def manifest_paths(config: Mapping[str, Any], run_id: str, model_key: str) -> tuple[Path, Path, Path]:
    base = run_root(config, run_id) / "immutable_manifests" / "per_model" / model_key
    return (
        base / "trace_manifest.jsonl",
        base / "execution_packs.jsonl",
        base / "worker_assignments.json",
    )


def validate_prepared_manifests(
    config: Mapping[str, Any], *, run_id: str
) -> dict[str, Any]:
    root = run_root(config, run_id)
    protocol_path = root / "immutable_manifests/immutable_protocol_manifest.json"
    if not protocol_path.is_file():
        raise FileNotFoundError("immutable production protocol manifest is missing")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if str(protocol.get("configuration_hash")) != stable_hash(config):
        failures.append("configuration_hash")
    if str(protocol.get("engine_revision")) != engine_revision():
        failures.append("engine_revision")
    source_access = root / "source_manifest_access_ledger.json"
    source_summary = root / "immutable_manifests/source_summary.json"
    source_exclusions = root / "immutable_manifests/source_exclusions.jsonl"
    if not all(path.is_file() for path in (source_access, source_summary, source_exclusions)):
        failures.append("source_manifest_artifacts")
    elif source_access.is_file():
        access = json.loads(source_access.read_text(encoding="utf-8"))
        forbidden_access = any(
            bool(access.get(field))
            for field in (
                "native_configuration_development_opened",
                "prompt_pilot_manifest_opened",
                "native_final_test_manifest_opened",
                "native_or_final_outputs_opened",
            )
        )
        opened_roles = set(map(str, access.get("files_opened", {})))
        allowed_opened_roles = {
            "summary",
            "teacher_forced_train",
            "teacher_forced_dev",
        }
        if (
            access.get("status") != "PASS"
            or forbidden_access
            or not opened_roles.issubset(allowed_opened_roles)
        ):
            failures.append("prohibited source access")
    model_checks: dict[str, Any] = {}
    for model_key in config["selected_models"]:
        trace_path, pack_path, assignment_path = manifest_paths(
            config, run_id, str(model_key)
        )
        boundary_manifest_path = (
            trace_path.parent / "checkpoint_boundary_manifest.json"
        )
        if not all(
            path.is_file()
            for path in (
                trace_path,
                pack_path,
                assignment_path,
                boundary_manifest_path,
            )
        ):
            failures.append(f"{model_key}:missing_files")
            continue
        rows = read_jsonl(trace_path)
        packs = read_jsonl(pack_path)
        assignments = json.loads(assignment_path.read_text(encoding="utf-8"))
        boundary_manifest = json.loads(
            boundary_manifest_path.read_text(encoding="utf-8")
        )
        pack_ids = [str(pack["pack_id"]) for pack in packs]
        assigned_ids = [
            str(pack_id)
            for values in assignments.values()
            for pack_id in values
        ]
        logical = [
            (
                str(key["model_key"]),
                str(key["trace_id"]),
                int(key["checkpoint_index"]),
                int(key["rollout_index"]),
                int(key["rollout_seed"]),
            )
            for pack in packs
            for key in pack["logical_rollout_keys"]
        ]
        boundary_failures: list[dict[str, Any]] = []
        for row in rows:
            steps = reasoning_steps_list(row.get("reasoning_steps"))
            audits = list(row.get("source_step_boundary_audits", []))
            error = int(row.get("first_error_zero_based", -1))
            eligible_audits = list(
                row.get("eligible_checkpoint_boundary_audits", [])
            )
            answer_audit = row.get("answer_region_checkpoint_audit", {})
            reasons: list[str] = []
            if row.get("source_step_boundary_policy") != SOURCE_STEP_BOUNDARY_POLICY:
                reasons.append("source_step_boundary_policy")
            if len(audits) != len(steps):
                reasons.append("source_step_boundary_count")
            checkpoint_audits = list(row.get("checkpoint_boundary_audits", []))
            if len(checkpoint_audits) != len(audits) + 1:
                reasons.append("checkpoint_boundary_audit_count")
            expected_eligible_audits = checkpoint_audits[: error + 1]
            if eligible_audits != expected_eligible_audits:
                reasons.append("eligible_checkpoint_boundary_audits")
            prompt_count = len(row.get("prompt_token_ids", []))
            observed_offsets = [prompt_count] + [
                prompt_count + int(item.get("token_offset", -1))
                for item in audits
            ]
            if observed_offsets != list(map(int, row.get("checkpoint_offsets", []))):
                reasons.append("checkpoint_offsets_from_boundary_audit")
            if list(map(int, row.get("eligible_checkpoint_offsets", []))) != observed_offsets[
                : error + 1
            ]:
                reasons.append("eligible_offsets_from_boundary_audit")
            if [
                int(item.get("checkpoint_token_offset", -1))
                for item in checkpoint_audits
            ] != observed_offsets:
                reasons.append("checkpoint_boundary_audit_offsets")
            if checkpoint_audits:
                root_audit = checkpoint_audits[0]
                if (
                    root_audit.get("checkpoint_kind") != "prompt_root"
                    or int(root_audit.get("checkpoint_index", -1)) != 0
                    or root_audit.get("boundary_policy")
                    != "exact_prompt_token_boundary"
                    or root_audit.get("exact") is not True
                ):
                    reasons.append("root_checkpoint_boundary_audit")
            for index, item in enumerate(audits):
                requested = int(item.get("requested_char_offset", -1))
                resolved = int(item.get("resolved_char_offset", -1))
                next_start = int(item.get("next_step_char_start", -1))
                if (
                    int(item.get("step_index", -1)) != index
                    or item.get("boundary_policy") != SOURCE_STEP_BOUNDARY_POLICY
                    or item.get("snap_policy") != "after"
                    or resolved < requested
                    or resolved > next_start
                    or int(item.get("character_displacement", -1))
                    != resolved - requested
                    or bool(item.get("exact")) != (resolved == requested)
                    or item.get("complete_step_included") is not True
                    or item.get("next_step_content_included") is not False
                ):
                    reasons.append("source_step_boundary_semantics")
                    break
            if (
                answer_audit.get("policy") != ANSWER_REGION_AUDIT_POLICY
                or answer_audit.get("status") != "PASS"
                or answer_audit.get(
                    "all_eligible_checkpoints_before_answer_region"
                )
                is not True
                or int(answer_audit.get("eligible_checkpoint_count", -1))
                != error + 1
            ):
                reasons.append("answer_region_checkpoint_audit")
            answer_start = answer_audit.get("answer_char_start")
            latest = answer_audit.get("latest_eligible_resolved_char_offset")
            if (
                answer_start is not None
                and latest is not None
                and int(latest) > int(answer_start)
            ):
                reasons.append("checkpoint_enters_answer_region")
            if reasons:
                boundary_failures.append(
                    {
                        "trace_id": str(row.get("trace_id")),
                        "reasons": sorted(set(reasons)),
                    }
                )
        boundary_manifest_valid = (
            boundary_manifest.get("status") == "PASS"
            and boundary_manifest.get("source_step_boundary_policy")
            == SOURCE_STEP_BOUNDARY_POLICY
            and boundary_manifest.get("answer_region_audit_policy")
            == ANSWER_REGION_AUDIT_POLICY
            and int(boundary_manifest.get("answer_region_audit_failures", -1)) == 0
            and int(boundary_manifest.get("traces", -1)) == len(rows)
        )
        passed = (
            len(rows) == int(protocol["models"][model_key]["traces"])
            and len(logical) == int(protocol["models"][model_key]["rollouts"])
            and len(logical) == len(set(logical))
            and sorted(pack_ids) == sorted(assigned_ids)
            and not boundary_failures
            and boundary_manifest_valid
            and all(
                str(pack["configuration_hash"])
                == str(protocol["configuration_hash"])
                and str(pack["engine_revision"])
                == str(protocol["engine_revision"])
                and stable_hash(
                    {
                        key: value
                        for key, value in pack.items()
                        if key
                        not in {
                            "pack_id",
                            "pack_hash",
                            "trace_count",
                            "checkpoint_count",
                            "rollout_count",
                            "estimated_work",
                        }
                    }
                )
                == str(pack["pack_hash"])
                for pack in packs
            )
        )
        model_checks[str(model_key)] = {
            "passed": passed,
            "traces": len(rows),
            "packs": len(packs),
            "rollouts": len(logical),
            "boundary_audit_failures": boundary_failures,
            "boundary_manifest_valid": boundary_manifest_valid,
        }
        if not passed:
            failures.append(f"{model_key}:content")
    result = {
        "passed": not failures,
        "failures": failures,
        "models": model_checks,
    }
    if failures:
        raise RuntimeError(f"prepared immutable manifests are invalid: {failures}")
    return result


def _pack_output_root(root: Path, mode: str, model_key: str, pack_id: str) -> Path:
    base = root / "raw_rollout_shards" if mode == "production" else root / "validation" / mode
    return base / model_key / pack_id


def load_frozen_production_context(
    config: Mapping[str, Any], *, run_id: str, model_key: str
) -> dict[str, Any]:
    root = run_root(config, run_id)
    freeze_path = root / "frozen_execution_manifest.json"
    if not freeze_path.is_file():
        raise RuntimeError("production requires a frozen execution manifest")
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN_FOR_PRODUCTION":
        raise RuntimeError("execution manifest is not frozen for production")
    if str(frozen.get("configuration_hash")) != stable_hash(config):
        raise RuntimeError("frozen production configuration differs")
    if str(frozen.get("engine_revision")) != engine_revision():
        raise RuntimeError("frozen production engine differs")
    expected_digest = stable_hash(
        {key: value for key, value in frozen.items() if key != "freeze_digest"}
    )
    if str(frozen.get("freeze_digest")) != expected_digest:
        raise RuntimeError("frozen execution manifest digest differs")
    trace_path, pack_path, _ = manifest_paths(config, run_id, model_key)
    rows = read_jsonl(trace_path)
    packs = read_jsonl(pack_path)
    settings = dict(frozen["model_settings"][model_key])
    expected_settings_hash = stable_hash(
        {key: value for key, value in settings.items() if key != "settings_hash"}
    )
    if str(settings.get("settings_hash")) != expected_settings_hash:
        raise RuntimeError(f"frozen settings hash differs for {model_key}")
    return {
        "root": root,
        "frozen": frozen,
        "freeze_digest": expected_digest,
        "trace_rows": {str(row["trace_id"]): row for row in rows},
        "packs": {str(pack["pack_id"]): pack for pack in packs},
        "settings": settings,
        "generation": dict(frozen["generation"]),
    }


def execute_frozen_production_pack(
    config: Mapping[str, Any],
    *,
    run_id: str,
    model_key: str,
    pack_id: str,
    loaded: Any,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    context = dict(context) if context is not None else load_frozen_production_context(
        config, run_id=run_id, model_key=model_key
    )
    if pack_id not in context["packs"]:
        raise KeyError(f"unknown immutable production pack: {pack_id}")
    settings = context["settings"]
    return execute_pack(
        config=config,
        model_key=model_key,
        loaded=loaded,
        pack=context["packs"][pack_id],
        trace_rows=context["trace_rows"],
        pack_root=_pack_output_root(
            context["root"], "production", model_key, pack_id
        ),
        generation=context["generation"],
        batch_size=int(settings["branch_batch_size"]),
        compaction_quantum=int(settings["compaction_quantum"]),
        prefill_chunk_size=int(settings["prefill_chunk_size"]),
        traces_per_decode_group=int(settings["traces_per_decode_group"]),
        maximum_decode_kv_bytes=int(settings["maximum_decode_kv_bytes"]),
        mode="production",
        execution_freeze_digest=context["freeze_digest"],
        model_execution_settings_hash=str(settings["settings_hash"]),
    )


def _marker_path(pack_root: Path) -> Path:
    return pack_root / "complete.json"


def _load_pack_rows(pack_root: Path) -> list[dict[str, Any]]:
    return pd.read_parquet(pack_root / "rollouts.parquet").to_dict("records")


def _validate_feature_payload(
    pack: Mapping[str, Any],
    payload: Mapping[str, Any],
    trace_rows: Mapping[str, Mapping[str, Any]],
    *,
    selected_layers: Sequence[int],
    model_revision: str | None,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    expected_ids = set(map(str, pack["trace_ids"]))
    if set(map(str, payload)) != expected_ids:
        failures.append({"reason": "trace_ids"})
    for trace_id in sorted(expected_ids & set(map(str, payload))):
        value = payload[trace_id]
        trace = trace_rows[trace_id]
        offsets = list(map(int, trace["eligible_checkpoint_offsets"]))
        reasons: list[str] = []
        if list(map(int, value.get("checkpoint_offsets", []))) != offsets:
            reasons.append("checkpoint_offsets")
        if list(map(int, value.get("selected_layers", []))) != list(
            map(int, selected_layers)
        ):
            reasons.append("selected_layers")
        if str(value.get("model_revision")) != str(model_revision):
            reasons.append("model_revision")
        features = value.get("features")
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            reasons.append("features_shape")
        elif int(features.shape[0]) != len(offsets):
            reasons.append("features_checkpoint_count")
        if reasons:
            failures.append({"trace_id": trace_id, "reasons": reasons})
    return {"passed": not failures, "failures": failures}


def valid_pack_artifact(
    pack: Mapping[str, Any],
    pack_root: Path,
    *,
    trace_rows: Mapping[str, Mapping[str, Any]] | None = None,
    expected_freeze_digest: str | None = None,
    expected_settings_hash: str | None = None,
    expected_layers: Sequence[int] | None = None,
    expected_model_revision: str | None = None,
) -> bool:
    marker = _marker_path(pack_root)
    data = pack_root / "rollouts.parquet"
    features = pack_root / "checkpoint_features.pt"
    if not (marker.is_file() and data.is_file() and features.is_file()):
        return False
    try:
        state = json.loads(marker.read_text(encoding="utf-8"))
        if state.get("pack_hash") != pack["pack_hash"]:
            return False
        if state.get("rollouts_sha256") != sha256_file(data):
            return False
        if state.get("features_sha256") != sha256_file(features):
            return False
        if expected_freeze_digest is not None and str(
            state.get("execution_freeze_digest")
        ) != str(expected_freeze_digest):
            return False
        if expected_settings_hash is not None and str(
            state.get("model_execution_settings_hash")
        ) != str(expected_settings_hash):
            return False
        hashes = parser_verifier_hashes()
        rows_valid = validate_pack_rows(
            pack,
            _load_pack_rows(pack_root),
            trace_rows=trace_rows,
            expected_freeze_digest=expected_freeze_digest,
            expected_settings_hash=expected_settings_hash,
            expected_parser_sha256=hashes["parser_sha256"],
            expected_verifier_sha256=hashes["verifier_sha256"],
        )["passed"]
        if not rows_valid:
            return False
        if trace_rows is not None and expected_layers is not None:
            feature_payload = torch.load(
                features, map_location="cpu", weights_only=False
            )
            if not _validate_feature_payload(
                pack,
                feature_payload,
                trace_rows,
                selected_layers=expected_layers,
                model_revision=expected_model_revision,
            )["passed"]:
                return False
        return True
    except Exception:
        return False


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
    name = os.environ.get(
        "SAFEPREFIX_RUN_VOLUME", "safeprefix-full-teacher-forced-runs-v2"
    )
    if not os.environ.get("MODAL_TASK_ID"):
        return
    try:
        import modal

        modal.Volume.from_name(name).commit()
    except Exception as exc:
        raise RuntimeError(f"failed to commit completed execution pack: {exc}") from exc


def _feature_payload(
    forced: Any,
    eligible_offsets: list[int],
    layers: list[int],
    model_revision: str | None,
    prompt_token_count: int,
) -> dict[str, Any]:
    token_offsets = [offset - 1 for offset in eligible_offsets]
    final_offset = int(forced.token_ids.shape[1]) - 1
    nll = -forced.token_log_probabilities
    features = build_checkpoint_features(
        forced.selected_hidden_states,
        token_offsets,
        final_offset=final_offset,
        prefix_total_tokens=int(forced.token_ids.shape[1]),
        nll_by_token=nll,
    )
    return {
        "features": features.to(torch.float16),
        "checkpoint_offsets": eligible_offsets,
        "selected_layers": layers,
        "model_revision": model_revision,
        "mean_assistant_token_nll": float(
            nll[max(int(prompt_token_count) - 1, 0) :].mean()
        ),
        "mean_all_token_nll": float(nll.mean()),
    }


def _merge_decode_metric_payloads(
    payloads: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not payloads:
        raise ValueError("at least one decode metric payload is required")
    summed = {
        "useful_output_tokens",
        "forwarded_row_steps",
        "wasted_forwarded_row_steps",
        "model_forward_calls",
        "compaction_refill_events",
        "admitted_branches",
        "completed_branches",
        "preallocated_cache_waves",
        "refilled_branches",
        "heterogeneous_prefix_waves",
        "completion_removal_events",
    }
    maximum = {
        "maximum_batch_size_observed",
        "maximum_physical_cache_length",
        "maximum_allocated_bytes",
        "maximum_reserved_bytes",
        "kv_bytes_per_token_per_row",
    }
    result: dict[str, Any] = {
        key: sum(int(item.get(key, 0)) for item in payloads) for key in summed
    }
    result.update(
        {
            key: max(int(item.get(key, 0)) for item in payloads)
            for key in maximum
        }
    )
    result["decode_wall_seconds"] = sum(
        float(item.get("decode_wall_seconds", 0.0)) for item in payloads
    )
    result["forwarded_to_useful_ratio"] = result["forwarded_row_steps"] / max(
        result["useful_output_tokens"], 1
    )
    result["useful_output_tokens_per_second"] = result[
        "useful_output_tokens"
    ] / max(result["decode_wall_seconds"], 1e-12)
    occupancy_weights = [
        int(item.get("compaction_refill_events", 0)) for item in payloads
    ]
    occupancy_denominator = sum(occupancy_weights)
    result["mean_wave_occupancy"] = (
        sum(
            float(item.get("mean_wave_occupancy") or 0.0) * weight
            for item, weight in zip(payloads, occupancy_weights)
        )
        / occupancy_denominator
        if occupancy_denominator
        else None
    )
    limits = {item.get("maximum_decode_kv_bytes") for item in payloads}
    if len(limits) != 1:
        raise ValueError("decode groups used different frozen KV-cache budgets")
    result["maximum_decode_kv_bytes"] = next(iter(limits))
    result["decode_group_count"] = len(payloads)
    return result


def execute_pack(
    *,
    config: Mapping[str, Any],
    model_key: str,
    loaded: Any,
    pack: Mapping[str, Any],
    trace_rows: Mapping[str, Mapping[str, Any]],
    pack_root: Path,
    generation: Mapping[str, Any],
    batch_size: int,
    compaction_quantum: int,
    prefill_chunk_size: int,
    traces_per_decode_group: int,
    maximum_decode_kv_bytes: int | None,
    mode: str,
    execution_freeze_digest: str,
    model_execution_settings_hash: str,
    interrupt_after_decode_groups: int | None = None,
) -> dict[str, Any]:
    layers = list(map(int, config["models"][model_key]["selected_hidden_state_layers"]))
    trace_subset = {
        str(trace_id): trace_rows[str(trace_id)] for trace_id in pack["trace_ids"]
    }
    if valid_pack_artifact(
        pack,
        pack_root,
        trace_rows=trace_subset,
        expected_freeze_digest=execution_freeze_digest,
        expected_settings_hash=model_execution_settings_hash,
        expected_layers=layers,
        expected_model_revision=loaded.model_revision,
    ):
        marker = json.loads(_marker_path(pack_root).read_text(encoding="utf-8"))
        return {**marker, "status": "SKIPPED_VALID", "resumed_from_marker": True}
    if int(traces_per_decode_group) < 1:
        raise ValueError("traces_per_decode_group must be positive")
    started = time.perf_counter()
    model, tokenizer = loaded.model, loaded.tokenizer
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    configured_context = int(config["models"][model_key]["max_context_length"])
    native_context = int(
        getattr(model.config, "max_position_embeddings", configured_context)
    )
    if configured_context > native_context:
        raise RuntimeError(
            f"configured context {configured_context} exceeds native model limit {native_context}"
        )
    utilization_samples: list[float] = []
    utilization_stop = threading.Event()

    def sample_utilization() -> None:
        while not utilization_stop.wait(1.0):
            try:
                utilization_samples.append(float(torch.cuda.utilization()))
            except Exception:
                return

    utilization_thread = threading.Thread(
        target=sample_utilization, name="safeprefix-gpu-utilization", daemon=True
    )
    if torch.cuda.is_available():
        utilization_thread.start()
    feature_rows: dict[str, Any] = {}
    prefill_seconds = 0.0
    verifier = resolve_verifier(str(config["rollout"]["verifier"]))
    hashes = parser_verifier_hashes()
    rows: list[dict[str, Any]] = []
    verifier_seconds = 0.0
    decode_group_metrics: list[dict[str, Any]] = []
    trace_order = sorted(
        map(str, pack["trace_ids"]),
        key=lambda trace_id: (
            -max(map(int, trace_rows[trace_id]["eligible_checkpoint_offsets"])),
            trace_id,
        ),
    )
    groups = [
        trace_order[index : index + int(traces_per_decode_group)]
        for index in range(0, len(trace_order), int(traces_per_decode_group))
    ]
    try:
        for group_index, trace_group in enumerate(groups):
            requests: list[ProductionRolloutRequest] = []
            for trace_id in trace_group:
                row = dict(trace_rows[trace_id])
                prompt_ids = list(map(int, row["prompt_token_ids"]))
                completion_ids = list(map(int, row["completion_token_ids"]))
                eligible_offsets = list(map(int, row["eligible_checkpoint_offsets"]))
                token_offsets = sorted(
                    set(
                        [offset - 1 for offset in eligible_offsets]
                        + [len(prompt_ids) + len(completion_ids) - 1]
                    )
                )
                prefill_started = time.perf_counter()
                forced = teacher_force_token_ids_chunked(
                    model,
                    prompt_ids + completion_ids,
                    prompt_count=len(prompt_ids),
                    chunk_size=int(prefill_chunk_size),
                    selected_layers=layers,
                    selected_token_offsets=token_offsets,
                    selected_checkpoint_offsets=eligible_offsets,
                )
                prefill_seconds += time.perf_counter() - prefill_started
                feature_rows[trace_id] = _feature_payload(
                    forced,
                    eligible_offsets,
                    layers,
                    loaded.model_revision,
                    len(prompt_ids),
                )
                trace_keys = [
                    key
                    for key in pack["logical_rollout_keys"]
                    if str(key["trace_id"]) == trace_id
                ]
                for checkpoint_index, token_offset in enumerate(eligible_offsets):
                    checkpoint = forced.cache_checkpoint(
                        token_offset,
                        model_id=loaded.model_id,
                        model_revision=loaded.model_revision,
                        tokenizer_id=loaded.tokenizer_id,
                        tokenizer_revision=loaded.tokenizer_revision,
                        tokenizer_metadata=tokenizer_checkpoint_metadata(tokenizer),
                        generation_metadata=dict(generation),
                        clone_cache=False,
                    )
                    matching = [
                        key
                        for key in trace_keys
                        if int(key["checkpoint_index"]) == checkpoint_index
                    ]
                    if [int(key["rollout_index"]) for key in matching] != [
                        0,
                        1,
                        2,
                        3,
                    ]:
                        raise AssertionError(
                            "pack checkpoint does not contain rollout indices 0..3"
                        )
                    for key in matching:
                        rollout_index = int(key["rollout_index"])
                        requests.append(
                            ProductionRolloutRequest(
                                branch_id=(
                                    f"{model_key}:{trace_id}:{checkpoint_index}:"
                                    f"{rollout_index}"
                                ),
                                rollout_seed=int(key["rollout_seed"]),
                                rollout_index=rollout_index,
                                checkpoint=checkpoint,
                                metadata={
                                    "pack_id": pack["pack_id"],
                                    "trace_id": trace_id,
                                    "checkpoint_index": checkpoint_index,
                                    "checkpoint_token_offset": token_offset,
                                },
                            )
                        )
                # Requests retain read-only views into this complete cache until
                # the group finishes; no per-checkpoint K/V deep copies exist.
                del forced
            results, group_metrics = decode_execution_pack(
                model,
                tokenizer,
                requests,
                generation=generation,
                maximum_batch_size=int(batch_size),
                compaction_quantum=int(compaction_quantum),
                maximum_context_length=configured_context,
                maximum_decode_kv_bytes=maximum_decode_kv_bytes,
            )
            decode_group_metrics.append(group_metrics.to_dict())
            verifier_started = time.perf_counter()
            for result in results:
                metadata = result.request.metadata
                source = dict(trace_rows[str(metadata["trace_id"])])
                parsed = parse_answer_region(result.text)
                truncation = result.stop_reason == "length"
                verifier_pass = bool(
                    parsed.success
                    and not truncation
                    and verifier(
                        parsed.parsed_answer,
                        decode_reference_answer(source["reference_answer"]),
                        {},
                    )
                )
                rows.append(
                    {
                        "schema_version": PRODUCTION_SCHEMA_VERSION,
                        "configuration_hash": pack["configuration_hash"],
                        "engine_revision": pack["engine_revision"],
                        "execution_freeze_digest": execution_freeze_digest,
                        "model_execution_settings_hash": model_execution_settings_hash,
                        "pack_id": pack["pack_id"],
                        "pack_hash": pack["pack_hash"],
                        "model_key": model_key,
                        "model_id": loaded.model_id,
                        "model_revision": loaded.model_revision,
                        "tokenizer_id": loaded.tokenizer_id,
                        "tokenizer_revision": loaded.tokenizer_revision,
                        "source_bucket": source["source_bucket"],
                        "source_dataset": source["source_dataset"],
                        "source_subset": source["source_subset"],
                        "source_trace_id": source["source_trace_id"],
                        "problem_id": source["problem_id"],
                        "trace_id": metadata["trace_id"],
                        "pipeline_split": source["pipeline_split"],
                        "dataset_first_visible_error_index": int(
                            source["first_error_index"]
                        ),
                        "first_visible_error_zero_based": int(
                            source["first_error_zero_based"]
                        ),
                        "checkpoint_index": int(metadata["checkpoint_index"]),
                        "checkpoint_token_offset": int(
                            metadata["checkpoint_token_offset"]
                        ),
                        "checkpoint_protocol": result.request.checkpoint.protocol_version,
                        "checkpoint_cache_storage": result.request.checkpoint.cache_storage,
                        "rollout_index": int(result.request.rollout_index),
                        "rollout_seed": int(result.request.rollout_seed),
                        "generated_token_ids": list(map(int, result.token_ids)),
                        "generated_token_count": len(result.token_ids),
                        "generated_text": result.text,
                        "stop_reason": result.stop_reason,
                        "truncation_flag": truncation,
                        "parser_status": "success" if parsed.success else "failure",
                        "parser_method": parsed.method,
                        "parser_confidence": parsed.confidence,
                        "parsed_answer": (
                            None if not parsed.success else str(parsed.parsed_answer)
                        ),
                        "parser_sha256": hashes["parser_sha256"],
                        "verifier_pass": verifier_pass,
                        "verifier_sha256": hashes["verifier_sha256"],
                        "binary_outcome": verifier_pass,
                        "latency_seconds": float(result.latency_seconds),
                        "mode": mode,
                        "decode_group_index": group_index,
                    }
                )
            verifier_seconds += time.perf_counter() - verifier_started
            del results, requests
            if (
                interrupt_after_decode_groups is not None
                and group_index + 1 >= int(interrupt_after_decode_groups)
                and group_index + 1 < len(groups)
            ):
                raise RuntimeError(
                    "INTENTIONAL_SMOKE_INTERRUPTION_BEFORE_ATOMIC_PACK_COMMIT"
                )
    finally:
        utilization_stop.set()
        if utilization_thread.is_alive():
            utilization_thread.join(timeout=2.0)
    if len(rows) != int(pack["rollout_count"]):
        raise AssertionError("constructed result count differs from immutable pack")
    decode_metrics = _merge_decode_metric_payloads(decode_group_metrics)
    integrity = validate_pack_rows(
        pack,
        rows,
        trace_rows=trace_subset,
        expected_freeze_digest=execution_freeze_digest,
        expected_settings_hash=model_execution_settings_hash,
        expected_parser_sha256=hashes["parser_sha256"],
        expected_verifier_sha256=hashes["verifier_sha256"],
    )
    if not integrity["passed"]:
        raise AssertionError(f"pack integrity failed before persistence: {integrity}")
    feature_integrity = _validate_feature_payload(
        pack,
        feature_rows,
        trace_subset,
        selected_layers=layers,
        model_revision=loaded.model_revision,
    )
    if not feature_integrity["passed"]:
        raise AssertionError(
            f"feature integrity failed before persistence: {feature_integrity}"
        )
    pack_root.mkdir(parents=True, exist_ok=True)
    rollout_path = pack_root / "rollouts.parquet"
    feature_path = pack_root / "checkpoint_features.pt"
    write_started = time.perf_counter()
    atomic_parquet(rollout_path, pd.DataFrame(rows))
    _atomic_torch(feature_path, feature_rows)
    write_seconds = time.perf_counter() - write_started
    written_bytes = rollout_path.stat().st_size + feature_path.stat().st_size
    marker = {
        "status": "COMPLETE",
        "completed_at": now_iso(),
        "pack_id": pack["pack_id"],
        "pack_hash": pack["pack_hash"],
        "configuration_hash": pack["configuration_hash"],
        "engine_revision": pack["engine_revision"],
        "execution_freeze_digest": execution_freeze_digest,
        "model_execution_settings_hash": model_execution_settings_hash,
        "row_count": len(rows),
        "rollouts_sha256": sha256_file(rollout_path),
        "features_sha256": sha256_file(feature_path),
        "integrity": integrity,
        "feature_integrity": feature_integrity,
        "prefill_seconds": prefill_seconds,
        "verifier_seconds": verifier_seconds,
        "artifact_write_seconds": write_seconds,
        "artifact_bytes": written_bytes,
        "artifact_bytes_per_second": written_bytes / max(write_seconds, 1e-12),
        "gpu_utilization_sample_count": len(utilization_samples),
        "gpu_utilization_mean_percent": (
            sum(utilization_samples) / len(utilization_samples)
            if utilization_samples
            else None
        ),
        "gpu_utilization_max_percent": max(utilization_samples, default=None),
        "total_wall_seconds": time.perf_counter() - started,
        "decode_metrics": decode_metrics,
        "decode_groups": decode_group_metrics,
        "gpu_total_memory_bytes": (
            int(torch.cuda.get_device_properties(0).total_memory)
            if torch.cuda.is_available()
            else None
        ),
    }
    atomic_json(_marker_path(pack_root), marker)
    _commit_modal_volume()
    if not valid_pack_artifact(
        pack,
        pack_root,
        trace_rows=trace_subset,
        expected_freeze_digest=execution_freeze_digest,
        expected_settings_hash=model_execution_settings_hash,
        expected_layers=layers,
        expected_model_revision=loaded.model_revision,
    ):
        raise AssertionError("pack failed validation after atomic persistence")
    return marker


def select_validation_pack(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_key: str,
    count: int,
    mode: str,
    configuration_hash: str,
    revision: str,
    base_seed: int,
) -> dict[str, Any]:
    ordered = sorted(
        rows, key=lambda item: stable_hash([mode, model_key, item["trace_id"]])
    )
    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in ordered:
        by_source[str(row["source_bucket"])].append(row)
    source_order = [
        "crv_arithmetic",
        "processbench_math",
        "processbench_olympiadbench",
        "processbench_omnimath",
    ]
    chosen: list[Mapping[str, Any]] = []
    # Fixed profiles deliberately cover high checkpoint count, short prefix,
    # long prefix, and a central case while retaining all four sources when the
    # validation count permits it.
    for profile_index, source in enumerate(source_order):
        candidates = by_source.get(source, [])
        if not candidates or len(chosen) >= count:
            continue
        if profile_index == 0:
            selected = max(
                candidates,
                key=lambda row: (
                    len(row["eligible_checkpoint_offsets"]),
                    int(row["full_token_count"]),
                    str(row["trace_id"]),
                ),
            )
        elif profile_index == 1:
            selected = min(
                candidates,
                key=lambda row: (
                    max(map(int, row["eligible_checkpoint_offsets"])),
                    str(row["trace_id"]),
                ),
            )
        elif profile_index == 2:
            selected = max(
                candidates,
                key=lambda row: (
                    max(map(int, row["eligible_checkpoint_offsets"])),
                    str(row["trace_id"]),
                ),
            )
        else:
            ranked = sorted(
                candidates,
                key=lambda row: (
                    int(row["full_token_count"]),
                    str(row["trace_id"]),
                ),
            )
            selected = ranked[len(ranked) // 2]
        chosen.append(selected)
    chosen_ids = {str(row["trace_id"]) for row in chosen}
    for row in ordered:
        if len(chosen) >= count:
            break
        if str(row["trace_id"]) not in chosen_ids:
            chosen.append(row)
            chosen_ids.add(str(row["trace_id"]))
    if len(chosen) < count:
        raise RuntimeError(f"{mode} requested {count} traces, found {len(chosen)}")
    pack = build_execution_packs(
        chosen,
        model_key=model_key,
        base_seed=base_seed,
        configuration_hash=configuration_hash,
        engine_revision=revision,
        traces_per_pack=count,
    )[0]
    identity = {**pack, "pack_id": f"{mode}-{pack['pack_id']}"}
    identity["pack_hash"] = stable_hash(
        {key: value for key, value in identity.items() if key != "pack_hash"}
    )
    return identity


def aggregate_production(
    config: Mapping[str, Any], *, run_id: str
) -> dict[str, Any]:
    root = run_root(config, run_id)
    frozen = json.loads(
        (root / "frozen_execution_manifest.json").read_text(encoding="utf-8")
    )
    freeze_digest = str(frozen["freeze_digest"])
    per_model: dict[str, Any] = {}
    all_checks: list[dict[str, Any]] = []
    compute_by_model: dict[str, Any] = {}
    resume_ledger: list[dict[str, Any]] = []
    global_logical_keys: set[tuple[str, str, int, int]] = set()
    total_rollouts = 0
    for model_key in config["selected_models"]:
        trace_path, packs_path, _ = manifest_paths(config, run_id, str(model_key))
        packs = read_jsonl(packs_path)
        trace_rows = {
            str(row["trace_id"]): row for row in read_jsonl(trace_path)
        }
        settings_hash = str(frozen["model_settings"][model_key]["settings_hash"])
        layers = list(
            map(int, config["models"][model_key]["selected_hidden_state_layers"])
        )
        model_aggregates: list[dict[str, Any]] = []
        model_markers: list[dict[str, Any]] = []
        model_logical: set[tuple[str, str, int, int]] = set()
        model_trace_ids: set[str] = set()
        source_counts: Counter[str] = Counter()
        source_successes: Counter[str] = Counter()
        rollout_count = 0
        successful_rollouts = 0
        parser_failures = 0
        truncations = 0
        natural_no_answer = 0
        generated_tokens = 0
        for pack in packs:
            pack_root = _pack_output_root(
                root, "production", str(model_key), str(pack["pack_id"])
            )
            pack_trace_rows = {
                str(trace_id): trace_rows[str(trace_id)]
                for trace_id in pack["trace_ids"]
            }
            if not valid_pack_artifact(
                pack,
                pack_root,
                trace_rows=pack_trace_rows,
                expected_freeze_digest=freeze_digest,
                expected_settings_hash=settings_hash,
                expected_layers=layers,
                expected_model_revision=config["models"][model_key].get("revision"),
            ):
                raise RuntimeError(f"missing or invalid production pack: {pack['pack_id']}")
            rows = _load_pack_rows(pack_root)
            marker = json.loads(
                _marker_path(pack_root).read_text(encoding="utf-8")
            )
            model_markers.append(marker)
            resume_ledger.append(
                {
                    "model_key": model_key,
                    "pack_id": pack["pack_id"],
                    "pack_hash": pack["pack_hash"],
                    "status": "COMPLETE_VALID",
                    "row_count": marker["row_count"],
                    "completed_at": marker["completed_at"],
                }
            )
            hashes = parser_verifier_hashes()
            check = {
                "pack_id": pack["pack_id"],
                **validate_pack_rows(
                    pack,
                    rows,
                    trace_rows=pack_trace_rows,
                    expected_freeze_digest=freeze_digest,
                    expected_settings_hash=settings_hash,
                    expected_parser_sha256=hashes["parser_sha256"],
                    expected_verifier_sha256=hashes["verifier_sha256"],
                ),
            }
            all_checks.append(check)
            if not check["passed"]:
                raise AssertionError(f"pack semantic validation failed: {check}")
            for row in rows:
                logical = (
                    str(row["model_key"]),
                    str(row["trace_id"]),
                    int(row["checkpoint_index"]),
                    int(row["rollout_index"]),
                )
                if logical in global_logical_keys:
                    raise AssertionError(
                        f"duplicate logical rollout across packs: {logical}"
                    )
                global_logical_keys.add(logical)
                model_logical.add(logical)
                model_trace_ids.add(str(row["trace_id"]))
                source = str(row["source_bucket"])
                source_counts[source] += 1
                source_successes[source] += int(bool(row["binary_outcome"]))
                rollout_count += 1
                successful_rollouts += int(bool(row["binary_outcome"]))
                parser_failures += int(row["parser_status"] != "success")
                truncations += int(bool(row["truncation_flag"]))
                natural_no_answer += int(
                    row["parser_status"] != "success"
                    and not bool(row["truncation_flag"])
                )
                generated_tokens += int(row["generated_token_count"])
            # Every checkpoint and its four outcomes are wholly contained in
            # one immutable pack, so aggregation can be streamed pack-wise.
            model_aggregates.extend(aggregate_checkpoint_outcomes(rows))
            del rows
        if model_trace_ids != set(trace_rows):
            raise AssertionError(
                f"production trace coverage differs for {model_key}: "
                f"expected={len(trace_rows)} observed={len(model_trace_ids)}"
            )
        if rollout_count != sum(int(pack["rollout_count"]) for pack in packs):
            raise AssertionError(f"rollout coverage count differs for {model_key}")
        atomic_parquet(
            root / "aggregated_checkpoint_outcomes" / f"{model_key}.parquet",
            pd.DataFrame(model_aggregates),
        )
        per_model[str(model_key)] = {
            "rollouts": rollout_count,
            "checkpoints": len(model_aggregates),
            "traces": len(model_trace_ids),
            "successful_rollouts": successful_rollouts,
            "parser_failures": parser_failures,
            "natural_no_answer_completions": natural_no_answer,
            "truncations": truncations,
            "generated_tokens": generated_tokens,
            "by_source": {
                source: {
                    "rollouts": source_counts[source],
                    "successful_rollouts": source_successes[source],
                }
                for source in sorted(source_counts)
            },
        }
        compute_by_model[str(model_key)] = {
            "pack_count": len(model_markers),
            "teacher_forced_prefill_seconds": sum(
                float(item["prefill_seconds"]) for item in model_markers
            ),
            "suffix_decode_seconds": sum(
                float(item["decode_metrics"]["decode_wall_seconds"])
                for item in model_markers
            ),
            "verifier_seconds": sum(
                float(item["verifier_seconds"]) for item in model_markers
            ),
            "artifact_write_seconds": sum(
                float(item.get("artifact_write_seconds", 0.0))
                for item in model_markers
            ),
            "total_pack_wall_seconds": sum(
                float(item["total_wall_seconds"]) for item in model_markers
            ),
            "useful_output_tokens": sum(
                int(item["decode_metrics"]["useful_output_tokens"])
                for item in model_markers
            ),
            "forwarded_row_steps": sum(
                int(item["decode_metrics"]["forwarded_row_steps"])
                for item in model_markers
            ),
            "maximum_allocated_bytes": max(
                (
                    int(item["decode_metrics"]["maximum_allocated_bytes"])
                    for item in model_markers
                ),
                default=0,
            ),
            "maximum_reserved_bytes": max(
                (
                    int(item["decode_metrics"]["maximum_reserved_bytes"])
                    for item in model_markers
                ),
                default=0,
            ),
            "artifact_bytes": sum(
                int(item.get("artifact_bytes", 0)) for item in model_markers
            ),
        }
        total_rollouts += rollout_count
    source_access = json.loads(
        (root / "source_manifest_access_ledger.json").read_text(encoding="utf-8")
    )
    if any(
        bool(source_access.get(field))
        for field in (
            "native_configuration_development_opened",
            "prompt_pilot_manifest_opened",
            "native_final_test_manifest_opened",
            "native_or_final_outputs_opened",
        )
    ):
        raise AssertionError("prohibited native/final source access was recorded")
    payload = {
        "status": "INTEGRITY_VALIDATED",
        "completed_at": now_iso(),
        "models_complete": len(per_model),
        "models_required": len(config["selected_models"]),
        "per_model": per_model,
        "total_rollouts": total_rollouts,
        "total_checkpoints": sum(row["checkpoints"] for row in per_model.values()),
        "total_generated_tokens": sum(row["generated_tokens"] for row in per_model.values()),
        "pack_checks": all_checks,
        "raw_outcomes_aggregated": True,
        "repairability_discretized": False,
        "boundary_training_occurred": False,
        "native_final_test_access_count": 0,
        "source_access_ledger": source_access,
        "compute_by_model": compute_by_model,
    }
    atomic_json(root / "compute_and_infrastructure_report.json", {
        "status": "COMPLETE",
        "per_model": compute_by_model,
        "worker_failures": sum(
            1
            for path in (root / "event_log").glob("*.json")
            if "FAILED" in path.name
        ),
        "logical_resume_unit": "immutable_execution_pack",
    })
    atomic_json(root / "resume_ledger.json", resume_ledger)
    parser_summary = {
        model: {
            "parser_success": values["rollouts"] - values["parser_failures"],
            "parser_failures": values["parser_failures"],
            "truncations": values["truncations"],
            "successful_rollouts": values["successful_rollouts"],
        }
        for model, values in per_model.items()
    }
    atomic_json(root / "parser_verifier_summary.json", parser_summary)
    atomic_text(
        root / "OPTIMIZATION_REPORT.md",
        "# SafePrefix production optimization report\n\n"
        "The frozen engine uses cross-checkpoint heterogeneous batching, "
        "deterministic branch-specific counter RNG, chunked teacher forcing, "
        "read-only checkpoint views copied into independent decode caches, static "
        "per-wave KV preallocation, one-softmax top-p sampling, deterministic "
        "completion compaction/refill, model-major dynamic dispatch across eight "
        "warm H100 workers, and immutable pack-level recovery. These optimizations "
        "change execution shape and resource use only; prompts, token IDs, "
        "checkpoint eligibility, four branch seeds, sampling law, parser, verifier, "
        "pack membership, and logical record semantics remain fixed.\n",
    )
    atomic_json(root / "integrity_validation_report.json", payload)
    atomic_json(root / "final_summary.json", payload)
    return payload


def render_final_report(config: Mapping[str, Any], *, run_id: str) -> Path:
    root = run_root(config, run_id)
    summary_path = root / "integrity_validation_report.json"
    if not summary_path.exists():
        raise RuntimeError("cannot write final report before integrity aggregation")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "INTEGRITY_VALIDATED":
        raise RuntimeError("integrity validation did not complete")
    lines = [
        "# SafePrefix full teacher-forced rollout suite",
        "",
        "- Status: **COMPLETE — INTEGRITY VALIDATED**",
        f"- Run ID: `{run_id}`",
        f"- Total checkpoints: `{summary['total_checkpoints']}`",
        f"- Total real rollouts: `{summary['total_rollouts']}`",
        f"- Generated tokens: `{summary['total_generated_tokens']}`",
        "- Repairability discretized: `False`",
        "- Boundary-model training: `not run`",
        "- Native evaluation/final-test access: `not run (0 accesses)`",
        "",
        "| Model | Traces | Checkpoints | Rollouts | Successes | Parser failures | Truncations | Tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, values in summary["per_model"].items():
        lines.append(
            f"| {model} | {values['traces']} | {values['checkpoints']} | {values['rollouts']} | "
            f"{values['successful_rollouts']} | {values['parser_failures']} | "
            f"{values['truncations']} | {values['generated_tokens']} |"
        )
    lines.extend(
        [
            "",
            "The suite preserves four individual Bernoulli verifier outcomes per checkpoint and their raw success count. It does not derive first-unsafe labels or select a repair boundary.",
            "",
        ]
    )
    target = root / "FINAL_REPORT.md"
    atomic_text(target, "\n".join(lines))
    return target
