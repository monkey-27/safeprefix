"""Correctness diagnostics for complete-cache SafePrefix checkpoints."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch

from safeprefix.reproducibility import stable_seed

from .cache_checkpoint import (
    CacheCheckpoint,
    capture_cache_checkpoint,
    clone_cache_checkpoint,
    compare_caches,
    load_cache_checkpoint,
    save_cache_checkpoint,
    tokenizer_checkpoint_metadata,
)
from .cache_utils import (
    cache_sequence_length,
    clone_past_key_values,
    forward_replay_diagnostic,
    prepare_replay_diagnostic_inputs,
    slice_past_key_values,
)
from .generation import CheckpointContinuation, decode_from_checkpoint
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


def _logit_comparison(left: torch.Tensor, right: torch.Tensor, top_k: int) -> dict[str, Any]:
    lhs, rhs = left.detach().float().cpu(), right.detach().float().cpu()
    if lhs.shape != rhs.shape:
        return {"shape_equal": False, "exact_equal": False, "top1_equal": False, "top_k_overlap": 0.0, "max_absolute_difference": float("inf"), "mean_absolute_difference": float("inf")}
    difference = (lhs - rhs).abs()
    k = min(int(top_k), lhs.shape[-1])
    left_top = set(torch.topk(lhs[0], k).indices.tolist())
    right_top = set(torch.topk(rhs[0], k).indices.tolist())
    return {
        "shape_equal": True,
        "exact_equal": bool(torch.equal(left, right)),
        "top1_equal": int(lhs[0].argmax()) == int(rhs[0].argmax()),
        "top_k_overlap": len(left_top & right_top) / max(k, 1),
        "max_absolute_difference": float(difference.max()) if difference.numel() else 0.0,
        "mean_absolute_difference": float(difference.mean()) if difference.numel() else 0.0,
    }


def first_divergence(left: list[int], right: list[int]) -> int | None:
    for index, (lhs, rhs) in enumerate(zip(left, right)):
        if lhs != rhs:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def _continuation_comparison(left: CheckpointContinuation, right: CheckpointContinuation, top_k: int) -> dict[str, Any]:
    left_tokens = left.completions[0].token_ids
    right_tokens = right.completions[0].token_ids
    logits = (
        _logit_comparison(left.logits_after_first_suffix_token, right.logits_after_first_suffix_token, top_k)
        if left.logits_after_first_suffix_token is not None and right.logits_after_first_suffix_token is not None
        else {"available": False}
    )
    step_logits = [
        _logit_comparison(lhs[:1], rhs[:1], top_k)
        for lhs, rhs in zip(left.logits_by_step, right.logits_by_step)
    ]
    return {
        "first_suffix_token_equal": left.first_suffix_token_ids == right.first_suffix_token_ids,
        "continuation_equal": left_tokens == right_tokens,
        "verifier_ready_output_equal": left.completions[0].text == right.completions[0].text,
        "left_text": left.completions[0].text,
        "right_text": right.completions[0].text,
        "left_token_ids": left_tokens,
        "right_token_ids": right_tokens,
        "first_divergence_token": first_divergence(left_tokens, right_tokens),
        "logits_after_first_suffix_token": logits,
        "all_compared_step_top1_equal": bool(step_logits) and all(item["top1_equal"] for item in step_logits),
        "compared_logit_steps": len(step_logits),
        "maximum_step_logit_difference": max((item["max_absolute_difference"] for item in step_logits), default=0.0),
        "mean_step_logit_difference": (
            sum(item["mean_absolute_difference"] for item in step_logits) / len(step_logits)
            if step_logits else 0.0
        ),
        "step_logit_comparisons": step_logits,
        "position_audit_equal": left.position_audit == right.position_audit,
        "left_position_audit": left.position_audit,
        "right_position_audit": right.position_audit,
    }


def _checkpoint(
    model: Any,
    tokenizer: Any,
    full_ids: torch.Tensor,
    prefix_length: int,
    output: Any,
    metadata: Mapping[str, Any],
) -> CacheCheckpoint:
    return capture_cache_checkpoint(
        output.past_key_values,
        output.logits[:, -1],
        full_ids,
        prefix_length,
        attention_mask=torch.ones_like(full_ids),
        model_id=str(metadata["model_id"]),
        model_revision=metadata.get("model_revision"),
        tokenizer_id=str(metadata["tokenizer_id"]),
        tokenizer_revision=metadata.get("tokenizer_revision"),
        tokenizer_metadata=tokenizer_checkpoint_metadata(tokenizer),
        generation_metadata=dict(metadata.get("generation", {})),
    )


def _decode_pair(
    model: Any,
    tokenizer: Any,
    left: CacheCheckpoint,
    right: CacheCheckpoint,
    generation: Mapping[str, Any],
    seed: int,
    top_k: int,
) -> dict[str, Any]:
    greedy = dict(generation, temperature=0.0, top_p=1.0)
    sampled = dict(generation)
    if float(sampled.get("temperature", 0.0)) <= 0:
        sampled.update(temperature=0.7, top_p=0.95)
    left_greedy = decode_from_checkpoint(model, tokenizer, left, branch_seeds=[seed], generation=greedy, capture_diagnostics=True)
    right_greedy = decode_from_checkpoint(model, tokenizer, right, branch_seeds=[seed], generation=greedy, capture_diagnostics=True)
    left_sampled = decode_from_checkpoint(model, tokenizer, left, branch_seeds=[seed], generation=sampled, capture_diagnostics=True)
    right_sampled = decode_from_checkpoint(model, tokenizer, right, branch_seeds=[seed], generation=sampled, capture_diagnostics=True)
    return {
        "greedy": _continuation_comparison(left_greedy, right_greedy, top_k),
        "seeded": _continuation_comparison(left_sampled, right_sampled, top_k),
    }


def run_cache_diagnostics(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    prefix_length: int,
    *,
    metadata: Mapping[str, Any],
    generation: Mapping[str, Any],
    seed: int,
    top_k: int = 20,
    branch_count: int = 4,
    serialization_round_trip: bool = True,
    old_replay_diagnostic: bool = True,
) -> dict[str, Any]:
    """Run tests A–D and F–H for one immutable example/checkpoint."""
    if model.training:
        raise RuntimeError("diagnostics require model.eval()")
    device = primary_device(model)
    ids = input_ids.to(device=device, dtype=torch.long)
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError("input_ids must have shape [1, sequence]")
    if not 1 <= prefix_length <= ids.shape[1]:
        raise ValueError("prefix_length outside sequence")
    prefix_ids = ids[:, :prefix_length]
    prefix_attention = torch.ones_like(prefix_ids)
    prefix_positions = torch.arange(prefix_length, device=device, dtype=torch.long)[None]
    prefix_cache_positions = torch.arange(prefix_length, device=device, dtype=torch.long)

    # The checkpoint is constructed exactly once from a normal prefix prefill.
    prefix_output = _forward(
        model,
        input_ids=prefix_ids,
        attention_mask=prefix_attention,
        position_ids=prefix_positions,
        cache_position=prefix_cache_positions,
        past_key_values=None,
    )
    checkpoint = _checkpoint(model, tokenizer, ids, prefix_length, prefix_output, metadata)
    live_copy = clone_cache_checkpoint(checkpoint, device=device)

    if serialization_round_trip:
        with tempfile.TemporaryDirectory(prefix="safeprefix-cache-") as directory:
            serialized_path = Path(directory) / "checkpoint.pt"
            save_cache_checkpoint(checkpoint, serialized_path)
            restored = load_cache_checkpoint(serialized_path, device=device)
    else:
        # Exact production topology: caches stay on one worker and restoration
        # uses the same clone/repeat path as branch construction.
        from .cache_checkpoint import repeat_cache_checkpoint

        restored = repeat_cache_checkpoint(checkpoint, 1)
    serialization_cache = compare_caches(live_copy.past_key_values, restored.past_key_values)
    serialization_logits = _logit_comparison(live_copy.next_token_logits, restored.next_token_logits, top_k)
    path_a = _decode_pair(model, tokenizer, live_copy, restored, generation, seed, top_k)
    test_a_passed = (
        serialization_cache["passed"]
        and serialization_logits["exact_equal"]
        and path_a["greedy"]["first_suffix_token_equal"]
        and path_a["greedy"]["logits_after_first_suffix_token"].get("top1_equal", False)
        and path_a["greedy"]["all_compared_step_top1_equal"]
        and path_a["greedy"]["continuation_equal"]
        and path_a["greedy"]["verifier_ready_output_equal"]
        and path_a["seeded"]["first_suffix_token_equal"]
        and path_a["seeded"]["all_compared_step_top1_equal"]
        and path_a["seeded"]["continuation_equal"]
        and path_a["seeded"]["verifier_ready_output_equal"]
        and path_a["greedy"]["position_audit_equal"]
        and path_a["seeded"]["position_audit_equal"]
    )

    # Test B appends later tokens to the exact captured cache, then crops them.
    captured_before_append = clone_past_key_values(prefix_output.past_key_values)
    if prefix_length < ids.shape[1]:
        append_cache = clone_past_key_values(prefix_output.past_key_values)
        remaining = ids[:, prefix_length:]
        final_output = _forward(
            model,
            input_ids=remaining,
            attention_mask=torch.ones_like(ids),
            position_ids=torch.arange(prefix_length, ids.shape[1], device=device, dtype=torch.long)[None],
            cache_position=torch.arange(prefix_length, ids.shape[1], device=device, dtype=torch.long),
            past_key_values=append_cache,
        )
        cropped_final = slice_past_key_values(final_output.past_key_values, prefix_length)
    else:
        cropped_final = clone_past_key_values(prefix_output.past_key_values)
    path_b = compare_caches(captured_before_append, cropped_final)

    # Test C recomputes the same prefix through the same prefill path, then uses
    # complete caches and saved logits on both sides. No checkpoint token replay.
    recomputed_output = _forward(
        model,
        input_ids=prefix_ids,
        attention_mask=prefix_attention,
        position_ids=prefix_positions,
        cache_position=prefix_cache_positions,
        past_key_values=None,
    )
    recomputed_checkpoint = _checkpoint(model, tokenizer, ids, prefix_length, recomputed_output, metadata)
    path_c_logits = _logit_comparison(checkpoint.next_token_logits, recomputed_checkpoint.next_token_logits, top_k)
    path_c_cache = compare_caches(checkpoint.past_key_values, recomputed_checkpoint.past_key_values)
    path_c_continuations = _decode_pair(model, tokenizer, recomputed_checkpoint, restored, generation, seed, top_k)
    test_c_passed = (
        path_c_logits["exact_equal"]
        and path_c_cache["passed"]
        and path_c_continuations["greedy"]["continuation_equal"]
        and path_c_continuations["seeded"]["continuation_equal"]
    )

    # Test D deliberately preserves the rejected V1 replay path as an optional,
    # nonblocking diagnostic. V3 does not spend 100-example production compute
    # repeating evidence already established by V2.
    if old_replay_diagnostic:
        full_output = _forward(
            model,
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            position_ids=torch.arange(ids.shape[1], device=device, dtype=torch.long)[None],
            cache_position=torch.arange(ids.shape[1], device=device, dtype=torch.long),
            past_key_values=None,
        )
        replay_output = forward_replay_diagnostic(
            model,
            prepare_replay_diagnostic_inputs(full_output.past_key_values, ids, prefix_length),
        )
        replay_checkpoint = _checkpoint(model, tokenizer, ids, prefix_length, replay_output, metadata)
        path_d_logits = _logit_comparison(recomputed_output.logits[:, -1], replay_output.logits[:, -1], top_k)
        path_d_continuations = _decode_pair(model, tokenizer, recomputed_checkpoint, replay_checkpoint, generation, seed, top_k)
        path_d: dict[str, Any] = {"blocking": False, "run": True, "logits": path_d_logits, "continuations": path_d_continuations}
    else:
        path_d = {"blocking": False, "run": False, "reason": "V2 replay diagnostic retained; not repeated in the V3 production gate"}

    # Test G compares one branch with the first row of an actual microbatch.
    single = decode_from_checkpoint(model, tokenizer, checkpoint, branch_seeds=[seed], generation=generation, capture_diagnostics=True)
    batch_seeds = [seed] + [stable_seed(seed, "branch", index) for index in range(1, branch_count)]
    batched = decode_from_checkpoint(model, tokenizer, checkpoint, branch_seeds=batch_seeds, generation=generation, capture_diagnostics=True)
    single_tokens = single.completions[0].token_ids
    batch_first_tokens = batched.completions[0].token_ids
    path_g = {
        "branches": branch_count,
        "first_suffix_token_equal": single.first_suffix_token_ids[0] == batched.first_suffix_token_ids[0],
        "first_branch_continuation_equal": single_tokens == batch_first_tokens,
        "first_divergence_token": first_divergence(single_tokens, batch_first_tokens),
        "first_branch_logits_after_first": _logit_comparison(single.logits_after_first_suffix_token, batched.logits_after_first_suffix_token[:1], top_k),
    }
    path_g["passed"] = bool(path_g["first_suffix_token_equal"] and path_g["first_branch_continuation_equal"] and path_g["first_branch_logits_after_first"]["top1_equal"])

    position_and_mask = {
        "prefix_token_ids": checkpoint.prefix_token_ids.detach().cpu().tolist(),
        "prefix_length": checkpoint.prefix_token_count,
        "attention_mask_shape": list(checkpoint.attention_mask.shape),
        "attention_mask_values": checkpoint.attention_mask.detach().cpu().tolist(),
        "position_ids": checkpoint.position_ids.detach().cpu().tolist(),
        "cache_position": checkpoint.cache_position.detach().cpu().tolist(),
        "cache_sequence_length": cache_sequence_length(checkpoint.past_key_values),
        "cache_format": type(checkpoint.past_key_values).__name__,
        "cache_dtype": checkpoint.dtype,
        "cache_device": str(next(value.device for layer in __import__("safeprefix.models.cache_utils", fromlist=["cache_to_legacy"]).cache_to_legacy(checkpoint.past_key_values) for value in layer[:2] if isinstance(value, torch.Tensor))),
        "model_eval": not model.training,
        "dropout_disabled": not model.training,
        "generation": dict(generation),
        "tokenizer_eos_token_id": tokenizer.eos_token_id,
        "tokenizer_pad_token_id": tokenizer.pad_token_id,
    }
    position_passed = (
        position_and_mask["prefix_length"] == position_and_mask["cache_sequence_length"]
        and checkpoint.position_ids.unique().tolist() == [prefix_length]
        and checkpoint.cache_position.tolist() == [prefix_length]
        and position_and_mask["attention_mask_shape"] == [1, prefix_length]
        and position_and_mask["model_eval"]
    )

    primary_passed = bool(test_a_passed and path_b["passed"] and test_c_passed and path_g["passed"] and position_passed)
    return {
        "protocol": checkpoint.protocol_version,
        "prefix_token_count": prefix_length,
        "sequence_token_count": int(ids.shape[1]),
        "test_a_live_vs_restored": {"passed": test_a_passed, "cache": serialization_cache, "saved_logits": serialization_logits, "continuations": path_a},
        "test_b_captured_vs_cropped_final": path_b,
        "test_c_no_replay_vs_prefix_recomputation": {"passed": test_c_passed, "cache": path_c_cache, "saved_logits": path_c_logits, "continuations": path_c_continuations},
        "test_d_old_replay_diagnostic": path_d,
        "test_f_position_and_mask_audit": {"passed": position_passed, **position_and_mask},
        "test_g_branch_batching": path_g,
        "test_h_serialization_round_trip": {"passed": serialization_cache["passed"] and serialization_logits["exact_equal"], "run": serialization_round_trip, "cache": serialization_cache, "saved_logits": serialization_logits, "production_interprocess_cache_transfer": False},
        "primary_gate_passed": primary_passed,
    }
