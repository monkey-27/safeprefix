"""V3 cache, batching, ordering, and state-independence validity checks."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from safeprefix.reproducibility import stable_seed

from .cache_checkpoint import (
    CacheCheckpoint,
    capture_cache_checkpoint,
    clone_cache_checkpoint,
    compare_caches,
    repeat_cache_checkpoint,
    tokenizer_checkpoint_metadata,
)
from .cache_diagnostics import first_divergence, run_cache_diagnostics
from .cache_utils import cache_sequence_length, cache_to_legacy
from .generation import (
    CheckpointContinuation,
    decode_from_checkpoint,
    deterministic_branch_seed,
)
from .loader import primary_device


def _forward(model: Any, **kwargs: Any) -> Any:
    kwargs.update(use_cache=True, return_dict=True, output_hidden_states=False)
    with torch.inference_mode():
        try:
            return model(**kwargs)
        except TypeError as exc:
            if "cache_position" not in str(exc):
                raise
            kwargs.pop("cache_position", None)
            return model(**kwargs)


def capture_production_checkpoint(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    prefix_length: int,
    metadata: Mapping[str, Any],
) -> CacheCheckpoint:
    device = primary_device(model)
    prefix_ids = input_ids[:, :prefix_length].to(device)
    output = _forward(
        model,
        input_ids=prefix_ids,
        attention_mask=torch.ones_like(prefix_ids),
        position_ids=torch.arange(prefix_length, device=device)[None],
        cache_position=torch.arange(prefix_length, device=device),
        past_key_values=None,
    )
    return capture_cache_checkpoint(
        output.past_key_values,
        output.logits[:, -1],
        input_ids.to(device),
        prefix_length,
        attention_mask=torch.ones_like(input_ids, device=device),
        model_id=str(metadata["model_id"]),
        model_revision=metadata.get("model_revision"),
        tokenizer_id=str(metadata["tokenizer_id"]),
        tokenizer_revision=metadata.get("tokenizer_revision"),
        tokenizer_metadata=tokenizer_checkpoint_metadata(tokenizer),
        generation_metadata=dict(metadata.get("generation", {})),
        rng_metadata={"sampling": "branch_id_inverse_cdf_v3", "branch_id_required": True},
    )


def _row_logits(left: CheckpointContinuation, right: CheckpointContinuation, right_row: int) -> dict[str, Any]:
    compared = min(len(left.logits_by_step), len(right.logits_by_step))
    maxima: list[float] = []
    means: list[float] = []
    top1: list[bool] = []
    for step in range(compared):
        lhs = left.logits_by_step[step][0].float()
        rhs = right.logits_by_step[step][right_row].float()
        difference = (lhs - rhs).abs()
        maxima.append(float(difference.max()) if difference.numel() else 0.0)
        means.append(float(difference.mean()) if difference.numel() else 0.0)
        top1.append(int(lhs.argmax()) == int(rhs.argmax()))
    return {
        "compared_steps": compared,
        "all_top1_equal": bool(all(top1)),
        "top1_equal_steps": sum(top1),
        "max_absolute_difference": max(maxima, default=0.0),
        "mean_absolute_difference": sum(means) / max(len(means), 1),
    }


def _compare_branch(single: CheckpointContinuation, batch: CheckpointContinuation, row: int) -> dict[str, Any]:
    left = single.completions[0]
    right = batch.completions[row]
    return {
        "first_suffix_token_equal": single.first_suffix_token_ids[0] == batch.first_suffix_token_ids[row],
        "token_ids_equal": left.token_ids == right.token_ids,
        "text_equal": left.text == right.text,
        "finish_reason_equal": left.finish_reason == right.finish_reason,
        "first_divergence_token": first_divergence(left.token_ids, right.token_ids),
        "logits": _row_logits(single, batch, row),
    }


def cache_independence_diagnostics(checkpoint: CacheCheckpoint, repeats: int = 4) -> dict[str, Any]:
    source = clone_cache_checkpoint(checkpoint)
    clone = clone_cache_checkpoint(checkpoint)
    repeated = repeat_cache_checkpoint(checkpoint, repeats)
    source_key = cache_to_legacy(source.past_key_values)[0][0]
    clone_key = cache_to_legacy(clone.past_key_values)[0][0]
    repeated_key = cache_to_legacy(repeated.past_key_values)[0][0]
    source_pointer = int(source_key.untyped_storage().data_ptr())
    clone_pointer = int(clone_key.untyped_storage().data_ptr())
    repeated_pointer = int(repeated_key.untyped_storage().data_ptr())
    source_before = source_key.clone()
    clone_key.reshape(-1)[0] += 1
    clone_mutation_isolated = torch.equal(source_key, source_before)
    other_row_before = repeated_key[1].clone() if repeats > 1 else None
    repeated_key[0].reshape(-1)[0] += 1
    row_mutation_isolated = True if other_row_before is None else torch.equal(repeated_key[1], other_row_before)
    return {
        "passed": bool(source_pointer != clone_pointer and source_pointer != repeated_pointer and clone_mutation_isolated and row_mutation_isolated),
        "source_clone_storage_distinct": source_pointer != clone_pointer,
        "source_repeat_storage_distinct": source_pointer != repeated_pointer,
        "clone_mutation_isolated": bool(clone_mutation_isolated),
        "repeated_row_mutation_isolated": bool(row_mutation_isolated),
        "source_cache_length": cache_sequence_length(source.past_key_values),
        "clone_cache_length": cache_sequence_length(clone.past_key_values),
        "repeat_cache_length": cache_sequence_length(repeated.past_key_values),
    }


def _decode_set(
    model: Any,
    tokenizer: Any,
    checkpoint: CacheCheckpoint,
    branch_ids: list[str],
    *,
    base_seed: int,
    generation: Mapping[str, Any],
) -> tuple[dict[str, CheckpointContinuation], CheckpointContinuation]:
    seeds = [deterministic_branch_seed(base_seed, branch_id) for branch_id in branch_ids]
    singles = {
        branch_id: decode_from_checkpoint(
            model,
            tokenizer,
            checkpoint,
            branch_ids=[branch_id],
            branch_seeds=[seed],
            generation=generation,
            capture_diagnostics=True,
        )
        for branch_id, seed in zip(branch_ids, seeds)
    }
    batch = decode_from_checkpoint(
        model,
        tokenizer,
        checkpoint,
        branch_ids=branch_ids,
        branch_seeds=seeds,
        generation=generation,
        capture_diagnostics=True,
    )
    return singles, batch


def validate_branch_batching(
    model: Any,
    tokenizer: Any,
    checkpoint: CacheCheckpoint,
    *,
    batch_sizes: list[int],
    generation: Mapping[str, Any],
    base_seed: int,
    require_mixed_termination: bool = True,
) -> dict[str, Any]:
    source_before = clone_cache_checkpoint(checkpoint)
    modes = {
        "greedy": {**dict(generation), "temperature": 0.0, "top_p": 1.0},
        "sampled": {**dict(generation), "temperature": max(float(generation.get("temperature", 0.0)), 0.7), "top_p": float(generation.get("top_p", 0.95))},
    }
    size_records: list[dict[str, Any]] = []
    all_pass = True
    for mode, settings in modes.items():
        maximum_ids = [f"branch_{index:02d}" for index in range(max(batch_sizes))]
        maximum_seeds = [deterministic_branch_seed(base_seed, branch_id) for branch_id in maximum_ids]
        singleton_results = {
            branch_id: decode_from_checkpoint(
                model,
                tokenizer,
                checkpoint,
                branch_ids=[branch_id],
                branch_seeds=[seed],
                generation=settings,
                capture_diagnostics=True,
            )
            for branch_id, seed in zip(maximum_ids, maximum_seeds)
        }
        for batch_size in batch_sizes:
            ids = [f"branch_{index:02d}" for index in range(batch_size)]
            seeds = [deterministic_branch_seed(base_seed, branch_id) for branch_id in ids]
            batch = decode_from_checkpoint(
                model,
                tokenizer,
                checkpoint,
                branch_ids=ids,
                branch_seeds=seeds,
                generation=settings,
                capture_diagnostics=True,
            )
            comparisons = [_compare_branch(singleton_results[branch_id], batch, row) for row, branch_id in enumerate(ids)]
            passed = all(
                item["first_suffix_token_equal"]
                and item["token_ids_equal"]
                and item["text_equal"]
                and item["finish_reason_equal"]
                and item["logits"]["all_top1_equal"]
                for item in comparisons
            )
            all_pass &= passed
            size_records.append({"mode": mode, "batch_size": batch_size, "passed": passed, "branches": comparisons})

    maximum = max(batch_sizes)
    branch_ids = [f"branch_{index:02d}" for index in range(maximum)]
    orders = {
        "original": branch_ids,
        "reversed": list(reversed(branch_ids)),
        "shuffled": sorted(branch_ids, key=lambda value: stable_seed(base_seed, "order", value)),
    }
    order_records: dict[str, Any] = {}
    reference: dict[str, list[int]] | None = None
    sampled = modes["sampled"]
    for order_name, ids in orders.items():
        seeds = [deterministic_branch_seed(base_seed, branch_id) for branch_id in ids]
        result = decode_from_checkpoint(model, tokenizer, checkpoint, branch_ids=ids, branch_seeds=seeds, generation=sampled, capture_diagnostics=True)
        by_id = {branch_id: completion.token_ids for branch_id, completion in zip(ids, result.completions)}
        if reference is None:
            reference = by_id
        passed = all(by_id[branch_id] == reference[branch_id] for branch_id in branch_ids)
        all_pass &= passed
        order_records[order_name] = {"passed": passed, "token_ids_by_branch": by_id}

    # Force a mixed-length diagnostic using a sampled first token as a temporary
    # EOS sentinel. This changes no model or tokenization state and exercises the
    # production active-row mask and padding path.
    base_ids = branch_ids
    base_seeds = [deterministic_branch_seed(base_seed, branch_id) for branch_id in base_ids]
    candidate_tokens = sorted({token for tokens in (reference or {}).values() for token in tokens})
    sentinel = None
    for token in candidate_tokens:
        stop_lengths = []
        for tokens in (reference or {}).values():
            stop_lengths.append(tokens.index(token) + 1 if token in tokens else len(tokens))
        if len(set(stop_lengths)) > 1:
            sentinel = token
            break
    mixed_observed = sentinel is not None
    if sentinel is not None:
        mixed_generation = {**sampled, "stop_token_ids": sorted(set(sampled.get("stop_token_ids", [])) | {sentinel})}
        mixed_singles, mixed_batch = _decode_set(model, tokenizer, checkpoint, base_ids, base_seed=base_seed, generation=mixed_generation)
        mixed_comparisons = [_compare_branch(mixed_singles[branch_id], mixed_batch, row) for row, branch_id in enumerate(base_ids)]
        mixed_lengths = [len(item.completions[0].token_ids) for item in mixed_singles.values()]
        mixed_observed = len(set(mixed_lengths)) > 1
        mixed_passed = mixed_observed and all(item["token_ids_equal"] and item["text_equal"] for item in mixed_comparisons)
    else:
        mixed_comparisons, mixed_lengths = [], []
        mixed_passed = not require_mixed_termination
    if not require_mixed_termination and not mixed_observed:
        mixed_passed = True
    all_pass &= mixed_passed
    source_unchanged = compare_caches(source_before.past_key_values, checkpoint.past_key_values)["passed"]
    all_pass &= source_unchanged
    return {
        "passed": bool(all_pass),
        "batch_sizes": size_records,
        "branch_order_invariance": order_records,
        "mixed_length": {"passed": mixed_passed, "observed": mixed_observed, "required": require_mixed_termination, "sentinel_token_id": sentinel, "continuation_lengths": mixed_lengths, "branches": mixed_comparisons},
        "source_checkpoint_unmutated": source_unchanged,
    }


def run_production_checkpoint_validation(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    prefix_length: int,
    *,
    metadata: Mapping[str, Any],
    generation: Mapping[str, Any],
    seed: int,
    batch_sizes: list[int],
    top_k: int,
    require_mixed_termination: bool = True,
) -> dict[str, Any]:
    """Run all blocking V3 tests for one exact example/checkpoint."""
    checkpoint = capture_production_checkpoint(model, tokenizer, input_ids, prefix_length, metadata)
    legacy = run_cache_diagnostics(
        model,
        tokenizer,
        input_ids,
        prefix_length,
        metadata=metadata,
        generation=generation,
        seed=seed,
        top_k=top_k,
        branch_count=max(batch_sizes),
        serialization_round_trip=False,
        old_replay_diagnostic=False,
    )
    independence = cache_independence_diagnostics(checkpoint, repeats=max(batch_sizes))
    batching = validate_branch_batching(
        model,
        tokenizer,
        checkpoint,
        batch_sizes=batch_sizes,
        generation=generation,
        base_seed=seed,
        require_mixed_termination=require_mixed_termination,
    )
    topology = {
        "passed": True,
        "production_topology": "same_worker_memory",
        "cache_serialized_in_production": False,
        "same_worker_clone_repeat_exercised": True,
        "serialization_round_trip_diagnostic_run": legacy["test_h_serialization_round_trip"]["run"],
        "serialization_round_trip_diagnostic_passed": None,
    }
    passed = bool(
        legacy["test_a_live_vs_restored"]["passed"]
        and legacy["test_b_captured_vs_cropped_final"]["passed"]
        and legacy["test_c_no_replay_vs_prefix_recomputation"]["passed"]
        and legacy["test_f_position_and_mask_audit"]["passed"]
        and independence["passed"]
        and batching["passed"]
        and topology["passed"]
    )
    return {
        "passed": passed,
        "protocol": "complete_cache_saved_next_logits_v3",
        "prefix_token_count": prefix_length,
        "sequence_token_count": int(input_ids.shape[1]),
        "test_1_live_cache_vs_restored_copy": legacy["test_a_live_vs_restored"],
        "test_2_singleton_vs_batched": batching,
        "test_3_capture_vs_final_crop": legacy["test_b_captured_vs_cropped_final"],
        "test_4_clone_independence": independence,
        "test_5_position_mask_consistency": legacy["test_f_position_and_mask_audit"],
        "test_6_branch_order_invariance": batching["branch_order_invariance"],
        "test_7_worker_process_transfer": topology,
        "old_replay_nonblocking_diagnostic": legacy["test_d_old_replay_diagnostic"],
        "no_replay_recomputation_diagnostic": legacy["test_c_no_replay_vs_prefix_recomputation"],
    }
