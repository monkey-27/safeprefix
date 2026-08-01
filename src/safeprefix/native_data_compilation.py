"""Native-evaluation data compilation without boundary training or repair.

The module deliberately separates three artifacts:

* immutable attempted-problem records;
* initially failed native traces and checkpoint representations;
* four independent from-scratch prompt continuations for each failed trace.

No checkpoint receives a derived unsafe or repairability label here.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from safeprefix.manifests import decode_reference_answer
from safeprefix.models.loader import primary_device
from safeprefix.models.teacher_forcing import teacher_force_token_ids_chunked
from safeprefix.parsing.answer_parsers import parse_answer_region, parse_is_acceptable_for_finish
from safeprefix.parsing.reasoning_region import split_reasoning_and_answer_from_tokens
from safeprefix.parsing.segmenters import HybridReasoningSegmenter
from safeprefix.prompting.chat_format import render_chat_prompt
from safeprefix.prompting.templates import problem_instruction
from safeprefix.reproducibility import atomic_json, atomic_jsonl, stable_hash, stable_seed
from safeprefix.rollout.verifier import ExactAnswerVerifier


ATTEMPT_REQUIRED = {
    "attempt_key", "model_key", "problem_id", "problem_group_hash", "native_bucket",
    "prompt_token_ids", "completion_token_ids", "completion_text", "parser", "verifier_pass",
    "initial_generation_seed", "stop_reason", "truncation_flag",
}
ROLLOUT_REQUIRED = {
    "rollout_key", "model_key", "trace_id", "rollout_index", "rollout_seed",
    "generated_token_ids", "generated_text", "parser", "verifier_pass", "stop_reason",
    "truncation_flag",
}


@dataclass(frozen=True)
class ExecutionPack:
    pack_id: str
    model_key: str
    problem_group_hashes: tuple[str, ...]
    configuration_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "problem_group_hashes": list(self.problem_group_hashes)}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_frozen_native_manifest(path: str | Path, specification: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = Path(path)
    if file_sha256(source) != str(specification["sha256"]):
        raise RuntimeError("frozen native manifest hash mismatch")
    rows = _read_jsonl(source)
    if len(rows) != int(specification["expected_rows"]):
        raise RuntimeError("frozen native manifest row count mismatch")
    groups = [str(row["problem_group_hash"]) for row in rows]
    if len(groups) != len(set(groups)) or len(set(groups)) != int(specification["expected_unique_problem_groups"]):
        raise RuntimeError("frozen native manifest has duplicate or missing problem groups")
    counts = Counter(str(row["manifest_bucket"]) for row in rows)
    expected = {str(key): int(value) for key, value in specification["expected_bucket_counts"].items()}
    if dict(counts) != expected:
        raise RuntimeError(f"frozen native bucket counts differ: {dict(counts)} != {expected}")
    required = {"problem_id", "problem_group_hash", "problem_text", "reference_answer", "manifest_role", "manifest_bucket"}
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise RuntimeError(f"frozen native row {index} lacks {sorted(missing)}")
        if row["manifest_role"] != "native_configuration_development":
            raise RuntimeError("non-native-development row entered the native manifest")
        if decode_reference_answer(row["reference_answer"]) in (None, "", []):
            raise RuntimeError("native manifest contains a row without an exact reference")
    return rows


def validate_protected_groups(rows: Iterable[Mapping[str, Any]], protected_path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(protected_path).read_text(encoding="utf-8"))
    native = {str(row["problem_group_hash"]) for row in rows}
    overlaps = {
        role: sorted(native & set(map(str, groups)))
        for role, groups in payload["protected_problem_groups"].items()
    }
    nonempty = {key: value for key, value in overlaps.items() if value}
    if nonempty:
        raise RuntimeError(f"native manifest overlaps protected groups: {nonempty}")
    return {
        "protected_group_counts": {key: len(value) for key, value in payload["protected_problem_groups"].items()},
        "overlap_counts": {key: len(value) for key, value in overlaps.items()},
        "passed": True,
    }


def build_protected_group_manifest(source_root: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Persist only identities for every role protected from native development."""

    root = Path(source_root)
    role_files = {
        "teacher_forced_train": "teacher_forced_train.jsonl",
        "teacher_forced_dev": "teacher_forced_dev.jsonl",
        "prompt_segmentation": "prompt_segmentation.jsonl",
        "final_test_reservations": "final_test_reservations.jsonl",
    }
    protected: dict[str, list[str]] = {}
    hashes: dict[str, str] = {}
    for role, filename in role_files.items():
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen protected-role manifest: {path}")
        values = _read_jsonl(path)
        groups = sorted({str(row["problem_group_hash"]) for row in values})
        if len(groups) != len(values):
            raise RuntimeError(f"protected role {role} contains duplicate problem groups")
        protected[role] = groups
        hashes[role] = file_sha256(path)
    payload = {
        "schema_version": 1,
        "protected_problem_groups": protected,
        "source_manifest_sha256": hashes,
        "contains_prompt_text_or_outputs": False,
    }
    atomic_json(output_path, payload)
    return payload


def build_execution_packs(
    rows: Iterable[Mapping[str, Any]], *, model_key: str, pack_size: int, configuration_hash: str
) -> list[ExecutionPack]:
    if int(pack_size) < 1:
        raise ValueError("pack_size must be positive")
    values = list(rows)
    packs = []
    for start in range(0, len(values), int(pack_size)):
        groups = tuple(str(row["problem_group_hash"]) for row in values[start : start + int(pack_size)])
        packs.append(ExecutionPack(
            pack_id=stable_hash(["native-attempt-pack-v1", model_key, configuration_hash, groups])[:24],
            model_key=str(model_key),
            problem_group_hashes=groups,
            configuration_hash=str(configuration_hash),
        ))
    flattened = [group for pack in packs for group in pack.problem_group_hashes]
    expected = [str(row["problem_group_hash"]) for row in values]
    if flattened != expected or len(flattened) != len(set(flattened)):
        raise AssertionError("execution packs changed ordering or duplicated membership")
    return packs


def initial_seed(config: Mapping[str, Any], model_key: str, problem_group_hash: str) -> int:
    return stable_seed(int(config["generation"]["base_seed"]), model_key, problem_group_hash, "native-initial")


def rollout_seed(config: Mapping[str, Any], model_key: str, trace_id: str, rollout_index: int) -> int:
    indices = list(map(int, config["generation"]["rollout_indices"]))
    if int(rollout_index) not in indices:
        raise ValueError("rollout index is not one of the frozen four indices")
    return stable_seed(int(config["generation"]["base_seed"]), model_key, trace_id, "from-scratch", int(rollout_index))


def validate_exact_once(rows: Iterable[Mapping[str, Any]], key: str, required: set[str]) -> None:
    values = list(rows)
    keys = []
    for index, row in enumerate(values):
        missing = required - set(row)
        if missing:
            raise RuntimeError(f"row {index} lacks required fields {sorted(missing)}")
        keys.append(str(row[key]))
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"duplicate logical {key} records")


def select_smoke_rows(rows: Iterable[Mapping[str, Any]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_bucket: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_bucket.setdefault(str(row["manifest_bucket"]), []).append(dict(row))
    output = []
    maximum = int(config["smoke"]["maximum_attempts_per_model"])
    for bucket in config["smoke"]["preferred_bucket_order"]:
        output.extend(by_bucket.get(str(bucket), []))
    return output[:maximum]


def render_native_prompt(tokenizer: Any, row: Mapping[str, Any], config: Mapping[str, Any], model_entry: Mapping[str, Any]) -> str:
    user = problem_instruction(str(row["problem_text"]), str(config["prompting"]["condition"]))
    return render_chat_prompt(
        tokenizer,
        user,
        override=model_entry.get("chat_template_override"),
        template_kwargs=model_entry.get("chat_template_kwargs"),
    )


def attempt_record(
    *, config: Mapping[str, Any], model_key: str, loaded: Any, row: Mapping[str, Any], prompt: str,
    prompt_token_ids: list[int], completion: Any, parser: Any, verifier_pass: bool,
) -> dict[str, Any]:
    group = str(row["problem_group_hash"])
    seed = initial_seed(config, model_key, group)
    return {
        "attempt_key": stable_hash(["native-attempt-v1", model_key, group, seed]),
        "model_key": model_key,
        "model_id": loaded.model_id,
        "model_revision": loaded.model_revision,
        "tokenizer_id": loaded.tokenizer_id,
        "tokenizer_revision": loaded.tokenizer_revision,
        "problem_id": str(row["problem_id"]),
        "problem_group_hash": group,
        "native_bucket": str(row["manifest_bucket"]),
        "source_dataset": str(row.get("source_dataset", "")),
        "source_subset": str(row.get("source_subset", "")),
        "prompt_text": prompt,
        "prompt_token_ids": list(map(int, prompt_token_ids)),
        "completion_token_ids": list(map(int, completion.token_ids)),
        "completion_text": completion.text,
        "token_log_probabilities": list(map(float, completion.token_log_probabilities)),
        "parser": parser.to_dict(),
        "verifier_pass": bool(verifier_pass),
        "initial_generation_seed": seed,
        "stop_reason": str(completion.finish_reason),
        "truncation_flag": str(completion.finish_reason) == "length",
        "generated_token_count": len(completion.token_ids),
        "generation_latency_seconds": float(completion.latency_seconds),
        "reference_answer": decode_reference_answer(row["reference_answer"]),
        "parser_version": str(config["parser"]["version"]),
        "verifier_version": str(config["verifier"]["version"]),
        "configuration_hash": stable_hash(config),
    }


def extract_failed_trace_features(
    *, model: Any, tokenizer: Any, attempt: Mapping[str, Any], selected_layers: list[int],
    segmenter: HybridReasoningSegmenter, chunk_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    completion_ids = list(map(int, attempt["completion_token_ids"]))
    prompt_ids = list(map(int, attempt["prompt_token_ids"]))
    reasoning, alignment = split_reasoning_and_answer_from_tokens(tokenizer, str(attempt["completion_text"]), completion_ids)
    spans = segmenter.segment_with_alignment(reasoning.text, alignment)
    if not spans:
        raise RuntimeError("parseable failed trace has no reasoning checkpoint span")
    prompt_count = len(prompt_ids)
    checkpoint_offsets = [prompt_count] + [prompt_count + int(span.token_end) for span in spans]
    selected_offsets = [offset - 1 for offset in checkpoint_offsets]
    span_ranges = [(prompt_count + span.token_start, prompt_count + span.token_end) for span in spans]
    forced = teacher_force_token_ids_chunked(
        model,
        prompt_ids + completion_ids,
        prompt_count=prompt_count,
        chunk_size=int(chunk_size),
        selected_layers=selected_layers,
        selected_token_offsets=selected_offsets,
        selected_span_ranges=span_ranges,
    )
    tensor_payload = {
        "schema_version": 1,
        "extraction_method": "posthoc_teacher_force_exact_saved_native_tokens",
        "selected_layers": list(map(int, selected_layers)),
        "checkpoint_offsets": checkpoint_offsets,
        "final_token_hidden_states": forced.selected_hidden_states,
        "span_mean_hidden_states": forced.selected_span_mean_hidden_states,
        "token_log_probabilities": forced.token_log_probabilities,
    }
    metadata = {
        "extraction_method": "posthoc_teacher_force_exact_saved_native_tokens",
        "reasoning_region": reasoning.to_dict(),
        "alignment_method": alignment.method,
        "spans": [span.to_dict() for span in spans],
        "checkpoint_offsets": checkpoint_offsets,
        "checkpoint_count": len(checkpoint_offsets),
        "selected_hidden_state_layers": list(map(int, selected_layers)),
        "representation_types": ["checkpoint_final_token", "reasoning_span_mean"],
        "full_kv_cache_persisted": False,
        "full_token_hidden_states_persisted": False,
    }
    return tensor_payload, metadata


def save_feature_payload(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(destination)


def write_cpu_preflight(
    *, rows: list[dict[str, Any]], config: Mapping[str, Any], output_dir: str | Path,
    protected_report: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    packs = {
        model_key: [pack.to_dict() for pack in build_execution_packs(
            rows,
            model_key=model_key,
            pack_size=int(config["native_data_compilation"]["attempt_pack_size"]),
            configuration_hash=stable_hash(config),
        )]
        for model_key in config["selected_models"]
    }
    summary = {
        "passed": True,
        "row_count": len(rows),
        "unique_problem_groups": len({str(row["problem_group_hash"]) for row in rows}),
        "bucket_counts": dict(Counter(str(row["manifest_bucket"]) for row in rows)),
        "execution_pack_counts": {key: len(value) for key, value in packs.items()},
        "protected_group_audit": dict(protected_report),
        "full_inference_authorized": bool(config["native_data_compilation"]["full_inference_authorized"]),
        "boundary_training_enabled": bool(config["native_data_compilation"]["boundary_training_enabled"]),
        "final_test_access_enabled": bool(config["native_data_compilation"]["final_test_access_enabled"]),
        "launch_blockers": list(config["launch_gates"]["blockers"]),
    }
    atomic_jsonl(root / "immutable_native_manifest.jsonl", rows)
    for model_key, values in packs.items():
        atomic_jsonl(root / "execution_packs" / f"{model_key}.jsonl", values)
    atomic_json(root / "cpu_preflight.json", summary)
    return summary


def validate_smoke_artifacts(root: str | Path, model_key: str) -> dict[str, Any]:
    model_root = Path(root) / "models" / model_key
    attempts = _read_jsonl(model_root / "attempts.jsonl")
    failures = _read_jsonl(model_root / "failed_traces.jsonl")
    rollouts = _read_jsonl(model_root / "from_scratch_rollouts.jsonl")
    validate_exact_once(attempts, "attempt_key", ATTEMPT_REQUIRED)
    validate_exact_once(rollouts, "rollout_key", ROLLOUT_REQUIRED)
    if len(failures) != 1:
        raise RuntimeError("model smoke must retain exactly one failed trace")
    trace_id = str(failures[0]["trace_id"])
    trace_rollouts = [row for row in rollouts if str(row["trace_id"]) == trace_id]
    if sorted(int(row["rollout_index"]) for row in trace_rollouts) != [0, 1, 2, 3]:
        raise RuntimeError("model smoke lacks the frozen four from-scratch rollouts")
    feature_path = model_root / str(failures[0]["feature_path"])
    if not feature_path.is_file() or feature_path.stat().st_size == 0:
        raise RuntimeError("model smoke lacks the hidden-state feature artifact")
    return {
        "passed": True,
        "model_key": model_key,
        "attempt_count": len(attempts),
        "failed_trace_count": len(failures),
        "from_scratch_rollout_count": len(rollouts),
        "checkpoint_count": int(failures[0]["checkpoint_metadata"]["checkpoint_count"]),
        "feature_path": str(feature_path),
        "no_boundary_training": True,
        "no_repair_evaluation": True,
        "no_final_test_access": True,
    }
