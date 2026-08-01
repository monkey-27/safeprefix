"""Microbatched suffix rollouts from one shared exact checkpoint cache."""

from __future__ import annotations

import time
from typing import Any, Mapping, MutableMapping

from safeprefix.manifests import RolloutRecord
from safeprefix.models.cache_checkpoint import CacheCheckpoint
from safeprefix.models.generation import continue_from_checkpoint, continue_from_checkpoint_scheduled

from .verifier import Verifier, parse_and_verify


def generate_branches(
    model: Any,
    tokenizer: Any,
    *,
    model_id: str,
    problem_id: str,
    trace_id: str,
    cache_checkpoint: CacheCheckpoint,
    checkpoint_index: int,
    checkpoint_token_offset: int,
    rollout_seeds: list[int],
    generation_config: Mapping[str, Any],
    reference_answer: Any,
    verifier: Verifier,
    microbatch_size: int | None = None,
    batching_policy: str = "fixed_padded",
    metrics: MutableMapping[str, Any] | None = None,
) -> list[RolloutRecord]:
    branch_ids = [f"{trace_id}:{checkpoint_index}:{seed}" for seed in rollout_seeds]
    decode_started = time.perf_counter()
    if microbatch_size is None:
        completions = continue_from_checkpoint(
            model,
            tokenizer,
            cache_checkpoint,
            branch_seeds=rollout_seeds,
            branch_ids=branch_ids,
            generation=generation_config,
        )
        decode_seconds = time.perf_counter() - decode_started
        executed_tokens = sum(len(item.token_ids) for item in completions)
        padded_branches = 0
        microbatch_count = 1
    else:
        fixed = continue_from_checkpoint_scheduled(
            model,
            tokenizer,
            cache_checkpoint,
            branch_seeds=rollout_seeds,
            branch_ids=branch_ids,
            generation=generation_config,
            microbatch_size=microbatch_size,
            batching_policy=batching_policy,
        )
        completions = fixed.completions
        decode_seconds = fixed.decode_wall_seconds
        executed_tokens = fixed.executed_output_tokens
        padded_branches = fixed.padded_branch_count
        microbatch_count = fixed.microbatch_count
    records = []
    verifier_seconds = 0.0
    for seed, completion in zip(rollout_seeds, completions):
        verifier_started = time.perf_counter()
        parsed, passed, _method = parse_and_verify(completion.text, reference_answer, verifier)
        verifier_seconds += time.perf_counter() - verifier_started
        records.append(RolloutRecord(
            model_id=model_id,
            problem_id=problem_id,
            trace_id=trace_id,
            checkpoint_index=checkpoint_index,
            checkpoint_token_offset=checkpoint_token_offset,
            rollout_seed=seed,
            generated_token_count=len(completion.token_ids),
            generated_text=completion.text,
            parsed_answer=parsed,
            verifier_pass=passed,
            latency_seconds=completion.latency_seconds,
        ))
    if metrics is not None:
        metrics["decode_seconds"] = float(metrics.get("decode_seconds", 0.0)) + decode_seconds
        metrics["verifier_seconds"] = float(metrics.get("verifier_seconds", 0.0)) + verifier_seconds
        metrics["requested_branches"] = int(metrics.get("requested_branches", 0)) + len(completions)
        metrics["padded_branches"] = int(metrics.get("padded_branches", 0)) + padded_branches
        metrics["microbatches"] = int(metrics.get("microbatches", 0)) + microbatch_count
        metrics["requested_output_tokens"] = int(metrics.get("requested_output_tokens", 0)) + sum(len(item.token_ids) for item in completions)
        metrics["executed_output_tokens"] = int(metrics.get("executed_output_tokens", 0)) + executed_tokens
        metrics.setdefault("suffix_lengths", []).extend(len(item.token_ids) for item in completions)
    return records
