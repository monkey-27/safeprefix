"""Exact suffix-only execution for K=32 checkpoint bundles."""

from __future__ import annotations

import gc
import json
import hashlib
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import pandas as pd
import torch

from safeprefix.manifests import decode_reference_answer
from safeprefix.models.cache_checkpoint import tokenizer_checkpoint_metadata
from safeprefix.models.teacher_forcing import teacher_force_token_ids_chunked
from safeprefix.parsing.answer_parsers import parse_answer_region
from safeprefix.reproducibility import atomic_json, atomic_parquet, now_iso
from safeprefix.rollout.production_engine import decode_execution_pack
from safeprefix.rollout.verifier import resolve_verifier
from safeprefix.threshold_selection_tf_v1.data import row_artifact_hash
from safeprefix.threshold_selection_tf_v1.runtime import ThresholdRolloutRequest


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _commit_output_volume() -> None:
    if not os.environ.get("MODAL_TASK_ID"):
        return
    name = os.environ.get("SAFEPREFIX_K_DENSIFICATION_OUTPUT_VOLUME")
    if not name:
        raise RuntimeError("K-densification output volume is not configured")
    import modal

    modal.Volume.from_name(name).commit()


def _block_root(artifact_root: Path, job: Mapping[str, Any]) -> Path:
    return (
        Path(artifact_root)
        / "raw_blocks"
        / str(job["model_key"])
        / str(job["checkpoint_key"])
        / f"slots-{int(job['block_start']):02d}-{int(job['block_end']):02d}"
    )


def _valid_block(root: Path, job: Mapping[str, Any]) -> bool:
    marker_path = root / "complete.json"
    outcomes_path = root / "outcomes.parquet"
    if not marker_path.is_file() or not outcomes_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text())
        outcomes = pd.read_parquet(outcomes_path)
    except Exception:
        return False
    expected = list(range(int(job["block_start"]), int(job["block_end"]) + 1))
    return bool(
        marker.get("status") == "COMPLETE"
        and marker.get("job_id") == job["job_id"]
        and marker.get("outcomes_sha256") == sha256_file(outcomes_path)
        and len(outcomes) == 4
        and sorted(outcomes["rollout_index"].astype(int).tolist()) == expected
        and outcomes["checkpoint_key"].astype(str).eq(str(job["checkpoint_key"])).all()
        and outcomes["infrastructure_status"].eq("executed").all()
        and outcomes["artifact_hash"].nunique() == 4
    )


def _gpu_memory() -> dict[str, float]:
    free, total = torch.cuda.mem_get_info()
    return {
        "free_bytes": float(free),
        "total_bytes": float(total),
        "free_fraction": float(free / total),
    }


def execute_k32_checkpoint_bundle(
    config: Mapping[str, Any],
    *,
    artifact_root: str | Path,
    jobs: Sequence[Mapping[str, Any]],
    checkpoint: Mapping[str, Any],
    generation_rows: Sequence[Mapping[str, Any]],
    source_trace: Mapping[str, Any],
    loaded: Any,
) -> dict[str, Any]:
    ordered_jobs = sorted(jobs, key=lambda value: int(value["block_start"]))
    if not ordered_jobs:
        raise RuntimeError("checkpoint bundle contains no jobs")
    checkpoint_keys = {str(job["checkpoint_key"]) for job in ordered_jobs}
    model_keys = {str(job["model_key"]) for job in ordered_jobs}
    if len(checkpoint_keys) != 1 or len(model_keys) != 1:
        raise RuntimeError("checkpoint bundle crosses checkpoint or model identity")
    if checkpoint_keys != {str(checkpoint["checkpoint_key"])}:
        raise RuntimeError("checkpoint bundle identity differs")
    expected_by_job: dict[str, list[int]] = {}
    job_by_slot: dict[int, Mapping[str, Any]] = {}
    for job in ordered_jobs:
        expected = list(range(int(job["block_start"]), int(job["block_end"]) + 1))
        if len(expected) != 4:
            raise RuntimeError("checkpoint bundle job is not an exact four-slot block")
        expected_by_job[str(job["job_id"])] = expected
        for slot in expected:
            if slot in job_by_slot:
                raise RuntimeError("checkpoint bundle jobs overlap rollout slots")
            job_by_slot[slot] = job
    generation_by_slot = {int(row["rollout_slot"]): row for row in generation_rows}
    if len(generation_by_slot) != len(generation_rows) or set(generation_by_slot) != set(job_by_slot):
        raise RuntimeError("bundle generation rows do not match the exact claimed slots")
    if any(str(row["checkpoint_key"]) != str(checkpoint["checkpoint_key"]) for row in generation_rows):
        raise RuntimeError("bundle generation identity differs")

    results_by_job: dict[str, dict[str, Any]] = {}
    jobs_to_generate: list[Mapping[str, Any]] = []
    for job in ordered_jobs:
        root = _block_root(Path(artifact_root), job)
        if _valid_block(root, job):
            outcomes = pd.read_parquet(root / "outcomes.parquet")
            marker = json.loads((root / "complete.json").read_text())
            results_by_job[str(job["job_id"])] = {
                **marker,
                "status": "SKIPPED_VALID",
                "rows": outcomes.to_dict("records"),
            }
        else:
            jobs_to_generate.append(job)
    if not jobs_to_generate:
        return {"status": "SKIPPED_VALID", "job_results": results_by_job}

    model_key = next(iter(model_keys))
    model_config = config["models"][model_key]
    if (
        str(loaded.model_id) != str(model_config["hf_model_id"])
        or str(loaded.model_revision) != str(model_config["revision"])
        or str(loaded.tokenizer_revision) != str(model_config["tokenizer_revision"])
    ):
        raise RuntimeError("loaded model/tokenizer identity differs from the frozen configuration")
    started = time.perf_counter()
    model, tokenizer = loaded.model, loaded.tokenizer
    generation = dict(config["generation"])
    configured_context = int(model_config["max_context_length"])
    native_context = int(getattr(model.config, "max_position_embeddings", configured_context))
    if configured_context > native_context:
        raise RuntimeError("configured context exceeds the pinned model context")
    prompt_ids = list(map(int, source_trace["prompt_token_ids"]))
    completion_ids = list(map(int, source_trace["completion_token_ids"]))
    token_offset = int(checkpoint["checkpoint_token_offset"])
    if str(source_trace["trace_id"]) != str(checkpoint["trace_id"]):
        raise RuntimeError("source trace identity differs")
    if token_offset not in set(map(int, source_trace["eligible_checkpoint_offsets"])):
        raise RuntimeError("selected checkpoint is not source-eligible")
    full_trace_ids = prompt_ids + completion_ids
    if not len(prompt_ids) <= token_offset <= len(full_trace_ids):
        raise RuntimeError("checkpoint offset is outside the causal source prefix")
    # Only the causal prefix through this checkpoint can affect its cache and
    # saved next-token logits.  Forcing the post-checkpoint suffix would retain
    # an irrelevant full-trace KV allocation and can violate the 10% memory
    # reserve when 16 continuation branches are admitted together.
    checkpoint_prefix_ids = full_trace_ids[:token_offset]
    prefill_started = time.perf_counter()
    forced = teacher_force_token_ids_chunked(
        model,
        checkpoint_prefix_ids,
        prompt_count=len(prompt_ids),
        chunk_size=int(config["execution"]["prefill_chunk_size"]),
        selected_layers=(),
        selected_token_offsets=(),
        selected_checkpoint_offsets=[token_offset],
    )
    prefill_seconds = time.perf_counter() - prefill_started
    cache_checkpoint = forced.cache_checkpoint(
        token_offset,
        model_id=loaded.model_id,
        model_revision=loaded.model_revision,
        tokenizer_id=loaded.tokenizer_id,
        tokenizer_revision=loaded.tokenizer_revision,
        tokenizer_metadata=tokenizer_checkpoint_metadata(tokenizer),
        generation_metadata=generation,
        clone_cache=False,
    )
    del forced
    gc.collect()
    torch.cuda.empty_cache()
    requested_slots = {
        slot
        for job in jobs_to_generate
        for slot in expected_by_job[str(job["job_id"])]
    }
    requested_generation_rows = [
        generation_by_slot[slot] for slot in sorted(requested_slots)
    ]
    requests = [
        ThresholdRolloutRequest(
            branch_id=f"k32:{checkpoint['checkpoint_key']}:{int(row['rollout_slot'])}",
            rollout_seed=int(row["rollout_seed"]),
            rollout_index=int(row["rollout_slot"]),
            checkpoint=cache_checkpoint,
            metadata={
                "stage": "k32_confirmation_suffix",
                "checkpoint_key": str(checkpoint["checkpoint_key"]),
                "job_id": str(job_by_slot[int(row["rollout_slot"])]["job_id"]),
                "trace_id": str(checkpoint["trace_id"]),
                "checkpoint_ordinal": int(checkpoint["checkpoint_ordinal"]),
                "checkpoint_token_offset": token_offset,
            },
        )
        for row in requested_generation_rows
    ]
    memory_before = _gpu_memory()
    if memory_before["free_fraction"] < float(
        config["execution"]["minimum_free_gpu_memory_fraction"]
    ):
        raise RuntimeError("less than 10 percent GPU memory is free before block decode")
    results, decode_metrics = decode_execution_pack(
        model,
        tokenizer,
        requests,
        generation=generation,
        maximum_batch_size=int(config["execution"]["branch_batch_sizes"][model_key]),
        compaction_quantum=int(config["execution"]["compaction_quantum"]),
        maximum_context_length=configured_context,
        maximum_decode_kv_bytes=int(config["execution"]["maximum_decode_kv_bytes"][model_key]),
    )
    verifier = resolve_verifier(str(generation["verifier"]))
    rows: list[dict[str, Any]] = []
    for result in results:
        parsed = parse_answer_region(result.text)
        truncation = str(result.stop_reason) == "length"
        verifier_pass = bool(
            parsed.success
            and not truncation
            and verifier(
                parsed.parsed_answer,
                decode_reference_answer(source_trace["reference_answer"]),
                {},
            )
        )
        row = {
            "schema_version": 2,
            "scientific": True,
            "analysis_role": "calibration_k32_confirmation",
            "model_key": model_key,
            "model_id": loaded.model_id,
            "model_revision": loaded.model_revision,
            "tokenizer_id": loaded.tokenizer_id,
            "tokenizer_revision": loaded.tokenizer_revision,
            "trace_id": str(checkpoint["trace_id"]),
            "source_trace_id": str(checkpoint["source_trace_id"]),
            "problem_id": str(checkpoint["problem_id"]),
            "problem_group": str(checkpoint["problem_group"]),
            "domain": str(checkpoint["domain"]),
            "checkpoint_key": str(checkpoint["checkpoint_key"]),
            "checkpoint_id": str(checkpoint["checkpoint_id"]),
            "checkpoint_ordinal": int(checkpoint["checkpoint_ordinal"]),
            "checkpoint_token_offset": token_offset,
            "prefix_token_hash": str(checkpoint["prefix_token_hash"]),
            "prompt_token_hash": str(checkpoint["prompt_token_hash"]),
            "continuation_policy_hash": str(checkpoint["continuation_policy_hash"]),
            "verifier_version": str(checkpoint["verifier_version"]),
            "job_id": str(job_by_slot[int(result.request.rollout_index)]["job_id"]),
            "rollout_index": int(result.request.rollout_index),
            "rollout_seed": int(result.request.rollout_seed),
            "raw_suffix": result.text,
            "generated_token_ids": list(map(int, result.token_ids)),
            "generated_token_count": len(result.token_ids),
            "normalized_final_answer": None if not parsed.success else str(parsed.parsed_answer),
            "parser_status": "success" if parsed.success else "failure",
            "parser_method": parsed.method,
            "truncation_status": bool(truncation),
            "stop_reason": str(result.stop_reason),
            "verifier_outcome": verifier_pass,
            "binary_outcome": verifier_pass,
            "infrastructure_status": "executed",
            "latency_seconds": float(result.latency_seconds),
        }
        row["artifact_hash"] = row_artifact_hash(row)
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("rollout_index", kind="mergesort")
    if sorted(frame["rollout_index"].astype(int)) != sorted(requested_slots):
        raise RuntimeError("generated bundle does not contain the exact requested slots")
    expected_seeds = {
        int(row["rollout_slot"]): int(row["rollout_seed"])
        for row in requested_generation_rows
    }
    if any(expected_seeds[int(row.rollout_index)] != int(row.rollout_seed) for row in frame.itertuples()):
        raise RuntimeError("generated bundle seed differs from the frozen manifest")
    memory_after = _gpu_memory()
    for job in jobs_to_generate:
        job_id = str(job["job_id"])
        expected = expected_by_job[job_id]
        job_frame = frame.loc[frame["rollout_index"].astype(int).isin(expected)].copy()
        job_frame = job_frame.sort_values("rollout_index", kind="mergesort")
        if sorted(job_frame["rollout_index"].astype(int)) != expected:
            raise RuntimeError("generated bundle cannot reconstruct an exact four-slot job")
        root = _block_root(Path(artifact_root), job)
        root.mkdir(parents=True, exist_ok=True)
        outcomes_path = root / "outcomes.parquet"
        atomic_parquet(outcomes_path, job_frame)
        marker = {
            "status": "COMPLETE",
            "completed_at": now_iso(),
            "job_id": job_id,
            "checkpoint_key": str(job["checkpoint_key"]),
            "model_key": model_key,
            "block_start": int(job["block_start"]),
            "block_end": int(job["block_end"]),
            "outcome_rows": len(job_frame),
            "outcomes_sha256": sha256_file(outcomes_path),
            "prefill_seconds": prefill_seconds,
            "decode_metrics": decode_metrics.to_dict(),
            "bundle_job_count": len(jobs_to_generate),
            "bundle_request_count": len(requests),
            "gpu_memory_before_decode": memory_before,
            "gpu_memory_after_decode": memory_after,
            "wall_seconds": time.perf_counter() - started,
            "native_evaluation_used": False,
            "teacher_forced_test_used": False,
        }
        atomic_json(root / "complete.json", marker)
        _commit_output_volume()
        if not _valid_block(root, job):
            raise RuntimeError("K32 block failed post-write validation")
        results_by_job[job_id] = {**marker, "rows": job_frame.to_dict("records")}
    return {"status": "COMPLETE", "job_results": results_by_job}


def execute_k32_block(
    config: Mapping[str, Any],
    *,
    artifact_root: str | Path,
    job: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    generation_rows: Sequence[Mapping[str, Any]],
    source_trace: Mapping[str, Any],
    loaded: Any,
) -> dict[str, Any]:
    """Backward-compatible one-block wrapper around checkpoint-bundle execution."""
    result = execute_k32_checkpoint_bundle(
        config,
        artifact_root=artifact_root,
        jobs=[job],
        checkpoint=checkpoint,
        generation_rows=generation_rows,
        source_trace=source_trace,
        loaded=loaded,
    )
    return result["job_results"][str(job["job_id"])]
