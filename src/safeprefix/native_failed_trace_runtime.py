"""GPU runtime for native initial attempts and prompt-root regenerations."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from safeprefix.models.generation import capture_prompt_checkpoint
from safeprefix.native_failed_trace_acquisition import (
    INITIAL_REQUIRED,
    REGEN_REQUIRED,
    file_sha256,
    initial_seed,
    regeneration_seed,
    validate_logical_rows,
)
from safeprefix.parsing.answer_parsers import parse_answer_region
from safeprefix.prompting.chat_format import render_chat_prompt
from safeprefix.prompting.templates import problem_instruction
from safeprefix.reproducibility import atomic_json, atomic_jsonl, stable_hash
from safeprefix.rollout.production_engine import ProductionRolloutRequest, decode_execution_pack
from safeprefix.rollout.verifier import ExactAnswerVerifier


def acquisition_root(runs_root: str | Path, run_id: str) -> Path:
    if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
        raise ValueError("run_id must be one safe path component")
    return Path(runs_root) / run_id / "artifacts/native_failed_trace_acquisition"


def _atomic_torchless_marker(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_json(path, dict(payload))


def _pack_valid(pack_root: Path, *, expected_pack_id: str, filename: str, expected_rows: int) -> bool:
    marker_path = pack_root / "complete.json"
    data_path = pack_root / filename
    if not marker_path.is_file() or not data_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        return (
            marker.get("status") == "COMPLETE"
            and marker.get("pack_id") == expected_pack_id
            and int(marker.get("row_count", -1)) == int(expected_rows)
            and marker.get("data_sha256") == file_sha256(data_path)
        )
    except Exception:
        return False


def _render_prompt(tokenizer: Any, source: Mapping[str, Any], config: Mapping[str, Any], model_entry: Mapping[str, Any]) -> str:
    user = problem_instruction(str(source["problem_text"]), str(config["prompting"]["condition"]))
    return render_chat_prompt(
        tokenizer, user,
        override=model_entry.get("chat_template_override"),
        template_kwargs=model_entry.get("chat_template_kwargs"),
    )


def _prompt_checkpoint(
    *, loaded: Any, prompt_token_ids: Sequence[int], generation: Mapping[str, Any],
) -> tuple[Any, float]:
    tensor = torch.tensor([list(map(int, prompt_token_ids))], dtype=torch.long)
    checkpoint, prefill_seconds, _ = capture_prompt_checkpoint(
        loaded.model, loaded.tokenizer, tensor,
        model_id=loaded.model_id, model_revision=loaded.model_revision,
        tokenizer_id=loaded.tokenizer_id, tokenizer_revision=loaded.tokenizer_revision,
        generation=generation,
    )
    return checkpoint, float(prefill_seconds)


def _parse_and_verify(
    *, text: str, stop_reason: str, gold_answer: Any, verifier: ExactAnswerVerifier,
) -> tuple[Any, str, str, bool, str]:
    parser = parse_answer_region(text)
    if not text.strip():
        return parser, "empty", "not_run_empty", False, "empty_response"
    if str(stop_reason) == "length":
        return parser, "parsed" if parser.success else "failed", "not_run_truncated", False, "truncated"
    if not parser.success:
        return parser, "failed", "not_run_parse_failure", False, "answer_extraction_failure"
    # A verifier exception is an infrastructure error.  It is deliberately not
    # converted into a model outcome and therefore retries the immutable pack.
    passed = bool(verifier(parser.parsed_answer, gold_answer, {}))
    return parser, "parsed", "completed", passed, "valid_correct" if passed else "valid_incorrect"


def execute_attempt_pack(
    *, config: Mapping[str, Any], runs_root: str | Path, run_id: str,
    model_key: str, pack: Mapping[str, Any], loaded: Any,
    source_index: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    root = acquisition_root(runs_root, run_id)
    pack_id = str(pack["pack_id"])
    source_ids = list(map(str, pack["source_ids"]))
    pack_root = root / "attempts/raw" / model_key / str(pack["stratum"]) / pack_id
    if _pack_valid(pack_root, expected_pack_id=pack_id, filename="attempts.jsonl", expected_rows=len(source_ids)):
        return json.loads((pack_root / "complete.json").read_text(encoding="utf-8")) | {"status": "SKIPPED_VALID"}
    generation = dict(config["generation"]["shared"])
    model_entry = config["models"][model_key]
    configuration_hash = stable_hash(config)
    requests: list[ProductionRolloutRequest] = []
    contexts: dict[str, dict[str, Any]] = {}
    invalid_rows: list[dict[str, Any]] = []
    prefill_seconds = 0.0
    started = time.perf_counter()
    for source_id in source_ids:
        source = dict(source_index[source_id])
        prompt = _render_prompt(loaded.tokenizer, source, config, model_entry)
        seed = initial_seed(config, model_key, source_id)
        attempt_key = stable_hash([
            "native-attempt-v2", configuration_hash, model_key, source_id, seed,
        ])
        try:
            prompt_ids = loaded.tokenizer(
                prompt, add_special_tokens=False, return_tensors="pt"
            )["input_ids"][0].tolist()
        except Exception as exc:
            invalid_rows.append({
                "schema_version": 1, "configuration_hash": configuration_hash,
                "pack_id": pack_id, "attempt_key": attempt_key,
                "model_key": model_key, "model_id": loaded.model_id,
                "model_revision": loaded.model_revision,
                "tokenizer_id": loaded.tokenizer_id,
                "tokenizer_revision": loaded.tokenizer_revision,
                "dataset_id": source["dataset_id"], "dataset_revision": source["dataset_revision"],
                "source_config": source["source_config"], "source_split": source["source_split"],
                "source_id": source["source_id"], "stratum": source["stratum"],
                "difficulty_level": source.get("difficulty_level"),
                "source_order_rank": int(source["source_order_rank"]),
                "normalized_problem_hash": source["normalized_problem_hash"],
                "problem_text": source["problem_text"], "gold_answer": source["gold_answer"],
                "rendered_prompt": prompt, "prompt_token_ids": [],
                "initial_generation_seed": int(seed), "completion_token_ids": [],
                "raw_initial_response": "", "extracted_answer": None,
                "parser": parse_answer_region("").to_dict(), "parser_status": "not_run",
                "verifier_status": "not_run_tokenizer_failure", "verifier_result": False,
                "initial_status": "tokenizer_or_context_failure",
                "generated_token_count": 0, "stop_reason": "tokenizer_failure",
                "truncation_flag": False, "latency_seconds": 0.0,
                "parser_version": config["parser"]["version"],
                "verifier_version": config["verifier"]["version"],
                "cohort_inclusion_status": "excluded",
                "cohort_exclusion_reason": "tokenizer_or_context_failure",
                "gold_answer_supplied_to_model": False,
                "failed_trace_supplied_to_model": False,
                "verifier_feedback_supplied_to_model": False,
                "infrastructure_detail": f"{type(exc).__name__}: {exc}",
            })
            continue
        if len(prompt_ids) + int(generation["max_new_tokens"]) > int(model_entry["max_context_length"]):
            invalid_rows.append({
                "schema_version": 1, "configuration_hash": configuration_hash,
                "pack_id": pack_id, "attempt_key": attempt_key,
                "model_key": model_key, "model_id": loaded.model_id,
                "model_revision": loaded.model_revision,
                "tokenizer_id": loaded.tokenizer_id,
                "tokenizer_revision": loaded.tokenizer_revision,
                "dataset_id": source["dataset_id"], "dataset_revision": source["dataset_revision"],
                "source_config": source["source_config"], "source_split": source["source_split"],
                "source_id": source["source_id"], "stratum": source["stratum"],
                "difficulty_level": source.get("difficulty_level"),
                "source_order_rank": int(source["source_order_rank"]),
                "normalized_problem_hash": source["normalized_problem_hash"],
                "problem_text": source["problem_text"], "gold_answer": source["gold_answer"],
                "rendered_prompt": prompt, "prompt_token_ids": prompt_ids,
                "initial_generation_seed": int(seed), "completion_token_ids": [],
                "raw_initial_response": "", "extracted_answer": None,
                "parser": parse_answer_region("").to_dict(), "parser_status": "not_run",
                "verifier_status": "not_run_context_overflow", "verifier_result": False,
                "initial_status": "tokenizer_or_context_failure",
                "generated_token_count": 0, "stop_reason": "prompt_plus_max_tokens_exceeds_context",
                "truncation_flag": False, "latency_seconds": 0.0,
                "parser_version": config["parser"]["version"],
                "verifier_version": config["verifier"]["version"],
                "cohort_inclusion_status": "excluded",
                "cohort_exclusion_reason": "tokenizer_or_context_failure",
                "gold_answer_supplied_to_model": False,
                "failed_trace_supplied_to_model": False,
                "verifier_feedback_supplied_to_model": False,
                "infrastructure_detail": (
                    f"prompt_tokens={len(prompt_ids)} max_new_tokens={generation['max_new_tokens']} "
                    f"max_context={model_entry['max_context_length']}"
                ),
            })
            continue
        branch_id = f"native-initial-v2:{model_key}:{source_id}"
        checkpoint, elapsed = _prompt_checkpoint(loaded=loaded, prompt_token_ids=prompt_ids, generation=generation)
        prefill_seconds += elapsed
        requests.append(ProductionRolloutRequest(
            branch_id=branch_id, rollout_seed=seed, rollout_index=0,
            checkpoint=checkpoint,
            metadata={"role": "initial", "source_id": source_id, "pack_id": pack_id},
        ))
        contexts[branch_id] = {"source": source, "prompt": prompt, "prompt_token_ids": prompt_ids, "seed": seed}
    results, metrics = decode_execution_pack(
        loaded.model, loaded.tokenizer, requests, generation=generation,
        maximum_batch_size=int(config["production_execution"]["branch_batch_sizes"][model_key]),
        compaction_quantum=int(config["production_execution"]["compaction_quantum"]),
        maximum_context_length=int(model_entry["max_context_length"]),
        maximum_decode_kv_bytes=int(config["production_execution"]["maximum_decode_kv_bytes"][model_key]),
    )
    verifier = ExactAnswerVerifier(
        float(config["verifier"]["absolute_tolerance"]),
        float(config["verifier"]["relative_tolerance"]),
    )
    rows: list[dict[str, Any]] = list(invalid_rows)
    for result in results:
        context = contexts[result.request.branch_id]
        source = context["source"]
        parser, parser_status, verifier_status, passed, status = _parse_and_verify(
            text=result.text, stop_reason=result.stop_reason,
            gold_answer=source["gold_answer"], verifier=verifier,
        )
        attempt_key = stable_hash([
            "native-attempt-v2", configuration_hash, model_key,
            source["source_id"], context["seed"],
        ])
        rows.append({
            "schema_version": 1,
            "configuration_hash": configuration_hash,
            "pack_id": pack_id,
            "attempt_key": attempt_key,
            "model_key": model_key,
            "model_id": loaded.model_id,
            "model_revision": loaded.model_revision,
            "tokenizer_id": loaded.tokenizer_id,
            "tokenizer_revision": loaded.tokenizer_revision,
            "dataset_id": source["dataset_id"],
            "dataset_revision": source["dataset_revision"],
            "source_config": source["source_config"],
            "source_split": source["source_split"],
            "source_id": source["source_id"],
            "stratum": source["stratum"],
            "difficulty_level": source.get("difficulty_level"),
            "source_order_rank": int(source["source_order_rank"]),
            "normalized_problem_hash": source["normalized_problem_hash"],
            "problem_text": source["problem_text"],
            "gold_answer": source["gold_answer"],
            "rendered_prompt": context["prompt"],
            "prompt_token_ids": context["prompt_token_ids"],
            "initial_generation_seed": int(context["seed"]),
            "completion_token_ids": list(map(int, result.token_ids)),
            "raw_initial_response": result.text,
            "extracted_answer": parser.parsed_answer,
            "parser": parser.to_dict(),
            "parser_status": parser_status,
            "verifier_status": verifier_status,
            "verifier_result": bool(passed),
            "initial_status": status,
            "generated_token_count": len(result.token_ids),
            "stop_reason": result.stop_reason,
            "truncation_flag": result.stop_reason == "length",
            "latency_seconds": float(result.latency_seconds),
            "parser_version": config["parser"]["version"],
            "verifier_version": config["verifier"]["version"],
            "cohort_inclusion_status": "pending_ordered_quota_freeze",
            "cohort_exclusion_reason": None if status == "valid_incorrect" else status,
            "gold_answer_supplied_to_model": False,
            "failed_trace_supplied_to_model": False,
            "verifier_feedback_supplied_to_model": False,
        })
    validate_logical_rows(rows, required=INITIAL_REQUIRED, key="attempt_key")
    pack_root.mkdir(parents=True, exist_ok=True)
    data_path = pack_root / "attempts.jsonl"
    atomic_jsonl(data_path, rows)
    marker = {
        "status": "COMPLETE", "pack_id": pack_id, "model_key": model_key,
        "stratum": pack["stratum"], "row_count": len(rows),
        "valid_initial_failures": sum(row["initial_status"] == "valid_incorrect" for row in rows),
        "data_sha256": file_sha256(data_path),
        "prefill_seconds": prefill_seconds,
        "decode_metrics": metrics.to_dict(),
        "wall_seconds": time.perf_counter() - started,
    }
    _atomic_torchless_marker(pack_root / "complete.json", marker)
    return marker


def execute_regeneration_pack(
    *, config: Mapping[str, Any], runs_root: str | Path, run_id: str,
    model_key: str, pack: Mapping[str, Any], loaded: Any,
    cohort_index: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    root = acquisition_root(runs_root, run_id)
    pack_id = str(pack["pack_id"])
    trace_ids = list(map(str, pack["trace_ids"]))
    expected_rows = 4 * len(trace_ids)
    pack_root = root / "regenerations/raw" / model_key / pack_id
    if _pack_valid(pack_root, expected_pack_id=pack_id, filename="regenerations.jsonl", expected_rows=expected_rows):
        return json.loads((pack_root / "complete.json").read_text(encoding="utf-8")) | {"status": "SKIPPED_VALID"}
    generation = dict(config["generation"]["shared"])
    configuration_hash = stable_hash(config)
    requests: list[ProductionRolloutRequest] = []
    contexts: dict[str, dict[str, Any]] = {}
    prefill_seconds = 0.0
    started = time.perf_counter()
    for trace_id in trace_ids:
        trace = dict(cohort_index[trace_id])
        checkpoint, elapsed = _prompt_checkpoint(
            loaded=loaded, prompt_token_ids=trace["prompt_token_ids"], generation=generation,
        )
        prefill_seconds += elapsed
        for rollout_index in map(int, config["generation"]["rollout_indices"]):
            seed = regeneration_seed(config, model_key, trace_id, rollout_index)
            branch_id = f"native-full-regeneration-v2:{model_key}:{trace_id}:{rollout_index}"
            requests.append(ProductionRolloutRequest(
                branch_id=branch_id, rollout_seed=seed, rollout_index=rollout_index,
                checkpoint=checkpoint,
                metadata={"role": "full_regeneration", "trace_id": trace_id, "pack_id": pack_id},
            ))
            contexts[branch_id] = {"trace": trace, "seed": seed, "rollout_index": rollout_index}
    results, metrics = decode_execution_pack(
        loaded.model, loaded.tokenizer, requests, generation=generation,
        maximum_batch_size=int(config["production_execution"]["branch_batch_sizes"][model_key]),
        compaction_quantum=int(config["production_execution"]["compaction_quantum"]),
        maximum_context_length=int(config["models"][model_key]["max_context_length"]),
        maximum_decode_kv_bytes=int(config["production_execution"]["maximum_decode_kv_bytes"][model_key]),
    )
    verifier = ExactAnswerVerifier(
        float(config["verifier"]["absolute_tolerance"]),
        float(config["verifier"]["relative_tolerance"]),
    )
    rows: list[dict[str, Any]] = []
    for result in results:
        context = contexts[result.request.branch_id]
        trace = context["trace"]
        parser, parser_status, verifier_status, passed, output_status = _parse_and_verify(
            text=result.text, stop_reason=result.stop_reason,
            gold_answer=trace["gold_answer"], verifier=verifier,
        )
        rollout_index = int(context["rollout_index"])
        rows.append({
            "schema_version": 1,
            "configuration_hash": configuration_hash,
            "pack_id": pack_id,
            "rollout_key": stable_hash(["native-full-regeneration-v2", configuration_hash, model_key, trace["trace_id"], rollout_index]),
            "model_key": model_key,
            "model_id": loaded.model_id,
            "model_revision": loaded.model_revision,
            "tokenizer_id": loaded.tokenizer_id,
            "tokenizer_revision": loaded.tokenizer_revision,
            "trace_id": trace["trace_id"],
            "attempt_key": trace["attempt_key"],
            "source_id": trace["source_id"],
            "stratum": trace["stratum"],
            "rollout_index": rollout_index,
            "rollout_seed": int(context["seed"]),
            "generated_token_ids": list(map(int, result.token_ids)),
            "raw_regenerated_response": result.text,
            "extracted_answer": parser.parsed_answer,
            "parser": parser.to_dict(),
            "parser_status": parser_status,
            "verifier_status": verifier_status,
            "binary_verifier_outcome": int(bool(passed)),
            "model_output_status": output_status,
            "generated_token_count": len(result.token_ids),
            "stop_reason": result.stop_reason,
            "truncation_flag": result.stop_reason == "length",
            "latency_seconds": float(result.latency_seconds),
            "parser_version": config["parser"]["version"],
            "verifier_version": config["verifier"]["version"],
            "original_failed_trace_supplied_to_model": False,
            "verifier_feedback_supplied_to_model": False,
            "gold_answer_supplied_to_model": False,
        })
    validate_logical_rows(rows, required=REGEN_REQUIRED, key="rollout_key")
    by_trace: dict[str, list[int]] = {}
    for row in rows:
        by_trace.setdefault(str(row["trace_id"]), []).append(int(row["rollout_index"]))
    if any(sorted(values) != [0, 1, 2, 3] for values in by_trace.values()):
        raise RuntimeError("regeneration pack does not contain indices 0..3 per trace")
    pack_root.mkdir(parents=True, exist_ok=True)
    data_path = pack_root / "regenerations.jsonl"
    atomic_jsonl(data_path, rows)
    marker = {
        "status": "COMPLETE", "pack_id": pack_id, "model_key": model_key,
        "trace_count": len(trace_ids), "row_count": len(rows),
        "successful_rollouts": sum(int(row["binary_verifier_outcome"]) for row in rows),
        "data_sha256": file_sha256(data_path),
        "prefill_seconds": prefill_seconds,
        "decode_metrics": metrics.to_dict(),
        "wall_seconds": time.perf_counter() - started,
    }
    _atomic_torchless_marker(pack_root / "complete.json", marker)
    return marker
