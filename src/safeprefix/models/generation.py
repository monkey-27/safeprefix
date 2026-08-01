"""Native completion generation and cache-restored suffix continuation."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F

from safeprefix.reproducibility import set_seed, stable_seed

from .cache_checkpoint import (
    CacheCheckpoint,
    capture_cache_checkpoint,
    clone_cache_checkpoint,
    repeat_cache_checkpoint,
    tokenizer_checkpoint_metadata,
)
from .cache_utils import cache_sequence_length, select_past_key_values_batch
from .loader import primary_device


@dataclass(frozen=True)
class GeneratedCompletion:
    token_ids: list[int]
    text: str
    token_log_probabilities: list[float]
    finish_reason: str
    latency_seconds: float
    past_key_values: Any = None
    hidden_states_by_layer: dict[int, torch.Tensor] | None = None
    next_token_logits_by_step: list[torch.Tensor] | None = None


def generate_completion(
    model: Any,
    tokenizer: Any,
    prompt_ids: torch.Tensor,
    generation: Mapping[str, Any],
    seed: int,
) -> GeneratedCompletion:
    set_seed(seed)
    prompt_ids = prompt_ids.to(primary_device(model))
    attention = torch.ones_like(prompt_ids)
    started = time.perf_counter()
    with torch.inference_mode():
        result = model.generate(
            input_ids=prompt_ids,
            attention_mask=attention,
            max_new_tokens=int(generation.get("max_new_tokens", 512)),
            do_sample=float(generation.get("temperature", 0.0)) > 0,
            temperature=max(float(generation.get("temperature", 0.0)), 1e-6),
            top_p=float(generation.get("top_p", 1.0)),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )
    tokens = result.sequences[0, prompt_ids.shape[1] :].tolist()
    log_probs = []
    for token, scores in zip(tokens, result.scores):
        log_probs.append(float(F.log_softmax(scores[0].float(), dim=-1)[int(token)].cpu()))
    eos = tokenizer.eos_token_id
    return GeneratedCompletion(
        tokens,
        tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False),
        log_probs,
        "eos" if tokens and eos is not None and tokens[-1] == eos else "length",
        time.perf_counter() - started,
        getattr(result, "past_key_values", None),
    )


def _uniform(seed: int, step: int) -> float:
    """Generate a branch-keyed device-independent open-interval uniform.

    A counter-based digest avoids constructing and seeding one CPU torch
    generator per branch and token.  The stream is invariant to batch shape,
    queue order, worker scheduling, and dynamic compaction.
    """
    payload = f"safeprefix-uniform-v1:{int(seed)}:{int(step)}".encode("ascii")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return (integer + 0.5) / float(2**64)


def deterministic_branch_seed(base_seed: int, branch_id: str) -> int:
    """Return a stream seed tied to identity, never batch row or scheduling."""
    if not branch_id:
        raise ValueError("branch_id must be nonempty")
    return stable_seed("safeprefix-v3-branch-stream", int(base_seed), str(branch_id))


def processed_logits(logits: torch.Tensor, generation: Mapping[str, Any]) -> torch.Tensor:
    """Apply the one shared greedy/sampling logits-processing stack."""
    result = logits.float().clone()
    banned = generation.get("banned_token_ids", generation.get("bad_token_ids", []))
    for token in banned or []:
        if isinstance(token, (tuple, list)):
            if len(token) != 1:
                raise ValueError("only single-token banned sequences are supported")
            token = token[0]
        token_id = int(token)
        if 0 <= token_id < result.shape[-1]:
            result[:, token_id] = -torch.inf
    temperature = float(generation.get("temperature", 0.0))
    if temperature > 0:
        result = result / temperature
    top_k = int(generation.get("top_k", 0) or 0)
    if 0 < top_k < result.shape[-1]:
        threshold = torch.topk(result, top_k, dim=-1).values[:, -1, None]
        result = result.masked_fill(result < threshold, -torch.inf)
    top_p = float(generation.get("top_p", 1.0))
    if temperature > 0 and top_p < 1.0:
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        sorted_logits, sorted_indices = torch.sort(result, descending=True, dim=-1)
        probabilities = torch.softmax(sorted_logits, dim=-1)
        remove = probabilities.cumsum(dim=-1) - probabilities >= top_p
        sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
        result = torch.full_like(result, -torch.inf).scatter(-1, sorted_indices, sorted_logits)
    if not bool(torch.isfinite(result).any(dim=-1).all()):
        raise ValueError("logits processing banned every possible token")
    return result


def select_tokens_from_saved_logits(
    logits: torch.Tensor,
    generation: Mapping[str, Any],
    seeds: list[int],
    *,
    step: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select tokens using explicit uniforms, never by reprocessing a prefix token."""
    if logits.ndim != 2 or logits.shape[0] != len(seeds):
        raise ValueError("one seed is required for each logits row")
    filtered = processed_logits(logits, generation)
    if float(generation.get("temperature", 0.0)) <= 0:
        tokens = filtered.argmax(dim=-1)
    else:
        probabilities = torch.softmax(filtered, dim=-1)
        cumulative = probabilities.cumsum(dim=-1)
        uniforms = torch.tensor(
            [_uniform(seed, step) for seed in seeds],
            device=cumulative.device,
            dtype=cumulative.dtype,
        )[:, None]
        tokens = torch.searchsorted(cumulative, uniforms, right=False).squeeze(-1)
        tokens = tokens.clamp(max=cumulative.shape[-1] - 1).to(dtype=torch.long)
    log_probabilities = F.log_softmax(filtered, dim=-1).gather(-1, tokens[:, None]).squeeze(-1)
    return tokens, log_probabilities


@dataclass(frozen=True)
class CheckpointContinuation:
    completions: list[GeneratedCompletion]
    first_suffix_token_ids: list[int]
    logits_after_first_suffix_token: torch.Tensor | None
    position_audit: list[dict[str, Any]]
    branch_ids: list[str]
    logits_by_step: list[torch.Tensor]


@dataclass(frozen=True)
class FixedMicrobatchResult:
    """Requested completions plus exact padded execution accounting."""

    completions: list[GeneratedCompletion]
    branch_ids: list[str]
    microbatch_size: int
    microbatch_count: int
    requested_branch_count: int
    padded_branch_count: int
    requested_output_tokens: int
    executed_output_tokens: int
    decode_wall_seconds: float
    position_audits: list[dict[str, Any]]


DynamicMicrobatchResult = FixedMicrobatchResult


@dataclass(frozen=True)
class PromptGenerationResult:
    completion: GeneratedCompletion
    prefill_wall_seconds: float
    decode: FixedMicrobatchResult
    prompt_checkpoint: CacheCheckpoint
    prompt_hidden_states: dict[int, torch.Tensor]


def _forward_suffix_token(
    model: Any,
    *,
    token_ids: torch.Tensor,
    cache: Any,
    attention_mask: torch.Tensor,
    position: int,
    output_hidden_states: bool = False,
) -> tuple[Any, dict[str, Any]]:
    if model.training:
        raise RuntimeError("cache continuation requires model.eval()")
    batch = int(token_ids.shape[0])
    position_ids = torch.full((batch, 1), position, dtype=torch.long, device=token_ids.device)
    cache_position = torch.tensor([position], dtype=torch.long, device=token_ids.device)
    audit = {
        "input_token_ids": token_ids.detach().cpu().tolist(),
        "attention_mask_shape": list(attention_mask.shape),
        "attention_mask_values_equal_one": bool(torch.all(attention_mask == 1)),
        "position_ids": position_ids.detach().cpu().tolist(),
        "cache_position": cache_position.detach().cpu().tolist(),
        "cache_sequence_length_before": cache_sequence_length(cache),
        "expected_position": position,
        "batch_size": batch,
        "device": str(token_ids.device),
        "model_eval": not model.training,
        "use_cache": True,
        "return_dict": True,
        "output_hidden_states": bool(output_hidden_states),
    }
    if audit["cache_sequence_length_before"] != position:
        raise AssertionError("cache length and suffix position disagree")
    if attention_mask.shape != (batch, position + 1):
        raise AssertionError("attention mask has an off-by-one error")
    kwargs = {
        "input_ids": token_ids[:, None],
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "cache_position": cache_position,
        "past_key_values": cache,
        "use_cache": True,
        "return_dict": True,
        "output_hidden_states": bool(output_hidden_states),
    }
    with torch.inference_mode():
        try:
            output = model(**kwargs)
        except TypeError as exc:
            if "cache_position" not in str(exc):
                raise
            kwargs.pop("cache_position")
            output = model(**kwargs)
            audit["cache_position_fallback"] = True
    if cache_sequence_length(output.past_key_values) != position + 1:
        raise AssertionError("model did not append exactly one suffix cache row")
    return output, audit


def decode_from_checkpoint(
    model: Any,
    tokenizer: Any,
    checkpoint: CacheCheckpoint,
    *,
    branch_seeds: Iterable[int],
    branch_ids: Iterable[str] | None = None,
    generation: Mapping[str, Any],
    capture_diagnostics: bool = False,
    return_cache: bool = False,
    selected_hidden_layers: Iterable[int] = (),
    retained_branch_count: int | None = None,
) -> CheckpointContinuation:
    """Decode from complete cache plus saved logits; never replay the final prefix token."""
    seeds = list(map(int, branch_seeds))
    identities = list(branch_ids) if branch_ids is not None else [f"branch_{index}" for index in range(len(seeds))]
    if len(identities) != len(seeds):
        raise ValueError("one persistent branch ID is required for every branch seed")
    if len(set(identities)) != len(identities):
        raise ValueError("persistent branch IDs must be unique")
    if not seeds:
        return CheckpointContinuation([], [], None, [], [], [])
    checkpoint.validate()
    device = primary_device(model)
    if checkpoint.next_token_logits.device != device:
        raise ValueError("checkpoint must be on the model device")
    state = repeat_cache_checkpoint(checkpoint, len(seeds))
    cache, logits, attention = state.past_key_values, state.next_token_logits, state.attention_mask
    prefix_length, batch = state.prefix_token_count, len(seeds)
    generated: list[list[int]] = [[] for _ in seeds]
    log_probs: list[list[float]] = [[] for _ in seeds]
    active = torch.ones(batch, dtype=torch.bool, device=device)
    eos_ids = tokenizer.eos_token_id
    eos_set = {int(eos_ids)} if isinstance(eos_ids, int) else set(map(int, eos_ids or []))
    stop_ids = eos_set | set(map(int, generation.get("stop_token_ids", [])))
    max_new = int(generation.get("max_new_tokens", 512))
    if max_new < 0:
        raise ValueError("max_new_tokens cannot be negative")
    started = time.perf_counter()
    first_tokens: list[int] = []
    first_logits_after: torch.Tensor | None = None
    audits: list[dict[str, Any]] = []
    logits_by_step: list[torch.Tensor] = []
    selected_layers = list(map(int, selected_hidden_layers))
    hidden_by_row: list[dict[int, list[torch.Tensor]]] = [
        {layer: [] for layer in selected_layers} for _ in seeds
    ]
    processed_counts = [0 for _ in seeds]
    retained = batch if retained_branch_count is None else int(retained_branch_count)
    if not 0 <= retained <= batch:
        raise ValueError("retained_branch_count is outside the branch batch")
    next_logits_by_row: list[list[torch.Tensor]] = [[] for _ in seeds]
    for step in range(max_new):
        active_before = active.clone()
        next_tokens, next_log_probs = select_tokens_from_saved_logits(logits, generation, seeds, step=step)
        if step == 0:
            first_tokens = list(map(int, next_tokens.detach().cpu().tolist()))
        active_before_cpu = active_before.detach().cpu().tolist()
        next_tokens_cpu = next_tokens.detach().cpu().tolist()
        next_log_probs_cpu = next_log_probs.detach().cpu().tolist()
        for row in range(batch):
            if bool(active_before_cpu[row]):
                token = int(next_tokens_cpu[row])
                generated[row].append(token)
                log_probs[row].append(float(next_log_probs_cpu[row]))
                if token in stop_ids:
                    active[row] = False
        must_step = step + 1 < max_new and bool(active.any())
        if capture_diagnostics and step == 0:
            must_step = True
        if return_cache or selected_layers:
            must_step = True
        if not must_step:
            break
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else next(iter(eos_set), 0))
        feed = torch.where(active_before, next_tokens, torch.full_like(next_tokens, pad))
        attention = torch.cat([attention, active_before[:, None].to(dtype=attention.dtype)], dim=1)
        output, audit = _forward_suffix_token(
            model,
            token_ids=feed,
            cache=cache,
            attention_mask=attention,
            position=prefix_length + step,
            output_hidden_states=bool(selected_layers),
        )
        audits.append(audit)
        cache, logits = output.past_key_values, output.logits[:, -1]
        for row in range(batch):
            if bool(active_before[row]):
                processed_counts[row] += 1
                for layer in selected_layers:
                    actual = layer if layer >= 0 else len(output.hidden_states) + layer
                    if not 0 <= actual < len(output.hidden_states):
                        raise ValueError(f"selected hidden layer is invalid: {layer}")
                    hidden_by_row[row][layer].append(
                        output.hidden_states[actual][row, -1].detach().cpu().clone()
                    )
                if return_cache and row < retained:
                    next_logits_by_row[row].append(logits[row].detach().clone())
        if capture_diagnostics:
            logits_by_step.append(logits.detach().cpu().clone())
        if step == 0 and capture_diagnostics:
            first_logits_after = logits.detach().cpu().clone()
        if not bool(active.any()):
            break
    elapsed = time.perf_counter() - started
    completions = []
    for row, (tokens, probabilities) in enumerate(zip(generated, log_probs)):
        if return_cache and processed_counts[row] != len(tokens):
            raise AssertionError("native cache did not process every generated token")
        row_cache = (
            select_past_key_values_batch(cache, row, prefix_length + processed_counts[row])
            if return_cache else None
        )
        row_hidden = None
        if selected_layers:
            row_hidden = {
                layer: torch.stack(hidden_by_row[row][layer])
                if hidden_by_row[row][layer]
                else torch.empty((0,))
                for layer in selected_layers
            }
        completions.append(
            GeneratedCompletion(
                tokens,
                tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                probabilities,
                (
                    "eos" if tokens and tokens[-1] in eos_set
                    else "stop" if tokens and tokens[-1] in stop_ids
                    else "length"
                ),
                elapsed,
                row_cache,
                row_hidden,
                next_logits_by_row[row] if return_cache and row < retained else None,
            )
        )
    return CheckpointContinuation(completions, first_tokens, first_logits_after, audits, identities, logits_by_step)


def continue_from_checkpoint(
    model: Any,
    tokenizer: Any,
    checkpoint: CacheCheckpoint,
    *,
    branch_seeds: Iterable[int],
    branch_ids: Iterable[str] | None = None,
    generation: Mapping[str, Any],
) -> list[GeneratedCompletion]:
    """Production SafePrefix V2 continuation using no final-token replay."""
    return decode_from_checkpoint(
        model,
        tokenizer,
        checkpoint,
        branch_seeds=branch_seeds,
        branch_ids=branch_ids,
        generation=generation,
    ).completions


def continue_from_checkpoint_fixed_microbatch(
    model: Any,
    tokenizer: Any,
    checkpoint: CacheCheckpoint,
    *,
    branch_seeds: Iterable[int],
    branch_ids: Iterable[str],
    generation: Mapping[str, Any],
    microbatch_size: int,
    capture_diagnostics: bool = False,
    return_cache: bool = False,
    selected_hidden_layers: Iterable[int] = (),
) -> FixedMicrobatchResult:
    """Decode every request with one fixed padded batch shape.

    The final partial microbatch is padded with deterministic dummy streams.
    Dummy completions are discarded but included in executed-token accounting.
    SafePrefix and all generative baselines use this same production policy.
    """
    seeds = list(map(int, branch_seeds))
    identities = list(map(str, branch_ids))
    if len(seeds) != len(identities) or not seeds:
        raise ValueError("fixed microbatch decoding requires aligned nonempty seeds and branch IDs")
    if len(set(identities)) != len(identities):
        raise ValueError("fixed microbatch branch IDs must be unique")
    if int(microbatch_size) < 1:
        raise ValueError("microbatch_size must be positive")
    size = int(microbatch_size)
    requested: list[GeneratedCompletion] = []
    padded = 0
    executed_tokens = 0
    microbatches = 0
    position_audits: list[dict[str, Any]] = []
    started = time.perf_counter()
    for start in range(0, len(seeds), size):
        chunk_seeds = seeds[start : start + size]
        chunk_ids = identities[start : start + size]
        requested_in_chunk = len(chunk_seeds)
        for slot in range(requested_in_chunk, size):
            padding_id = f"__padding__{start // size:06d}_{slot:04d}"
            chunk_ids.append(padding_id)
            chunk_seeds.append(
                stable_seed(
                    "safeprefix-fixed-padding",
                    checkpoint.model_id,
                    checkpoint.model_revision,
                    checkpoint.checkpoint_token_offset,
                    padding_id,
                )
            )
            padded += 1
        decoded = decode_from_checkpoint(
            model,
            tokenizer,
            checkpoint,
            branch_seeds=chunk_seeds,
            branch_ids=chunk_ids,
            generation=generation,
            capture_diagnostics=capture_diagnostics,
            return_cache=return_cache,
            selected_hidden_layers=selected_hidden_layers,
            retained_branch_count=requested_in_chunk if return_cache else None,
        )
        requested.extend(decoded.completions[:requested_in_chunk])
        if decoded.position_audit:
            position_audits.append(decoded.position_audit[0])
        executed_tokens += sum(len(item.token_ids) for item in decoded.completions)
        microbatches += 1
    return FixedMicrobatchResult(
        completions=requested,
        branch_ids=identities,
        microbatch_size=size,
        microbatch_count=microbatches,
        requested_branch_count=len(identities),
        padded_branch_count=padded,
        requested_output_tokens=sum(len(item.token_ids) for item in requested),
        executed_output_tokens=executed_tokens,
        decode_wall_seconds=time.perf_counter() - started,
        position_audits=position_audits,
    )


def continue_from_checkpoint_dynamic_microbatch(
    model: Any,
    tokenizer: Any,
    checkpoint: CacheCheckpoint,
    *,
    branch_seeds: Iterable[int],
    branch_ids: Iterable[str],
    generation: Mapping[str, Any],
    maximum_microbatch_size: int,
    capture_diagnostics: bool = False,
    return_cache: bool = False,
    selected_hidden_layers: Iterable[int] = (),
) -> DynamicMicrobatchResult:
    """Decode exact request batches without padded dummy branches.

    The scheduler forms the largest available checkpoint-local microbatch up
    to ``maximum_microbatch_size``.  It preserves every real request, branch
    seed, logits operation, checkpoint, and verifier input.  The only removed
    work is synthetic padding that has no experimental label.
    """
    seeds = list(map(int, branch_seeds))
    identities = list(map(str, branch_ids))
    if len(seeds) != len(identities) or not seeds:
        raise ValueError("dynamic microbatch decoding requires aligned nonempty seeds and branch IDs")
    if len(set(identities)) != len(identities):
        raise ValueError("dynamic microbatch branch IDs must be unique")
    size = int(maximum_microbatch_size)
    if size < 1:
        raise ValueError("maximum_microbatch_size must be positive")
    completions: list[GeneratedCompletion] = []
    position_audits: list[dict[str, Any]] = []
    microbatches = 0
    started = time.perf_counter()
    for start in range(0, len(seeds), size):
        decoded = decode_from_checkpoint(
            model,
            tokenizer,
            checkpoint,
            branch_seeds=seeds[start : start + size],
            branch_ids=identities[start : start + size],
            generation=generation,
            capture_diagnostics=capture_diagnostics,
            return_cache=return_cache,
            selected_hidden_layers=selected_hidden_layers,
        )
        completions.extend(decoded.completions)
        position_audits.extend(decoded.position_audit[:1])
        microbatches += 1
    output_tokens = sum(len(item.token_ids) for item in completions)
    return DynamicMicrobatchResult(
        completions=completions,
        branch_ids=identities,
        microbatch_size=size,
        microbatch_count=microbatches,
        requested_branch_count=len(identities),
        padded_branch_count=0,
        requested_output_tokens=output_tokens,
        executed_output_tokens=output_tokens,
        decode_wall_seconds=time.perf_counter() - started,
        position_audits=position_audits,
    )


def continue_from_checkpoint_scheduled(
    model: Any,
    tokenizer: Any,
    checkpoint: CacheCheckpoint,
    *,
    branch_seeds: Iterable[int],
    branch_ids: Iterable[str],
    generation: Mapping[str, Any],
    microbatch_size: int,
    batching_policy: str,
    capture_diagnostics: bool = False,
    return_cache: bool = False,
    selected_hidden_layers: Iterable[int] = (),
) -> FixedMicrobatchResult:
    common = dict(
        branch_seeds=branch_seeds,
        branch_ids=branch_ids,
        generation=generation,
        capture_diagnostics=capture_diagnostics,
        return_cache=return_cache,
        selected_hidden_layers=selected_hidden_layers,
    )
    if batching_policy == "dynamic_unpadded":
        return continue_from_checkpoint_dynamic_microbatch(
            model, tokenizer, checkpoint,
            maximum_microbatch_size=microbatch_size,
            **common,
        )
    if batching_policy == "fixed_padded":
        return continue_from_checkpoint_fixed_microbatch(
            model, tokenizer, checkpoint,
            microbatch_size=microbatch_size,
            **common,
        )
    raise ValueError(f"unsupported batching policy: {batching_policy}")


def run_matched_shape_smoke(
    model: Any,
    tokenizer: Any,
    checkpoint: CacheCheckpoint,
    *,
    microbatch_size: int,
    continuation_tokens: int,
    seed: int,
    branch_id: str,
) -> dict[str, Any]:
    """Check restored continuation under the exact fixed production shape.

    This narrow pre-production gate intentionally does not compare singleton
    and batched execution. Both paths use the same padded microbatch policy.
    """
    checkpoint.validate()
    restored = clone_cache_checkpoint(checkpoint)
    conditions: dict[str, Any] = {}
    passed = True
    for name, temperature in (("greedy", 0.0), ("seeded", 0.7)):
        generation = dict(checkpoint.generation_metadata)
        generation.update(
            temperature=temperature,
            top_p=1.0 if temperature == 0.0 else float(generation.get("top_p", 0.95)),
            max_new_tokens=int(continuation_tokens),
        )
        kwargs = {
            "branch_seeds": [int(seed)],
            "branch_ids": [str(branch_id)],
            "generation": generation,
            "microbatch_size": int(microbatch_size),
            "capture_diagnostics": True,
        }
        live = continue_from_checkpoint_fixed_microbatch(model, tokenizer, checkpoint, **kwargs)
        copied = continue_from_checkpoint_fixed_microbatch(model, tokenizer, restored, **kwargs)
        live_tokens = live.completions[0].token_ids
        copied_tokens = copied.completions[0].token_ids
        token_equal = live_tokens == copied_tokens
        first_equal = (not live_tokens and not copied_tokens) or bool(
            live_tokens and copied_tokens and live_tokens[0] == copied_tokens[0]
        )
        audits_equal = live.position_audits == copied.position_audits
        expected_shape = [int(microbatch_size), checkpoint.prefix_token_count + 1]
        position_ok = bool(live.position_audits) and all(
            audit["cache_sequence_length_before"] == checkpoint.prefix_token_count
            and audit["cache_position"] == [checkpoint.prefix_token_count]
            and audit["attention_mask_shape"] == expected_shape
            and audit["attention_mask_values_equal_one"]
            for audit in live.position_audits + copied.position_audits
        )
        condition_passed = first_equal and token_equal and audits_equal and position_ok
        passed = passed and condition_passed
        conditions[name] = {
            "passed": condition_passed,
            "first_suffix_token_equal": first_equal,
            "continuation_token_equal": token_equal,
            "position_mask_cache_length_equal": audits_equal and position_ok,
            "live_token_ids": live_tokens,
            "restored_token_ids": copied_tokens,
            "microbatch_size": int(microbatch_size),
        }
    return {
        "passed": passed,
        "prefix_token_count": checkpoint.prefix_token_count,
        "checkpoint_token_offset": checkpoint.checkpoint_token_offset,
        "cache_protocol": checkpoint.protocol_version,
        "microbatch_size": int(microbatch_size),
        "conditions": conditions,
    }


def capture_prompt_checkpoint(
    model: Any,
    tokenizer: Any,
    prompt_ids: torch.Tensor,
    *,
    model_id: str,
    model_revision: str | None,
    tokenizer_id: str,
    tokenizer_revision: str | None,
    generation: Mapping[str, Any],
    selected_hidden_layers: Iterable[int] = (),
) -> tuple[CacheCheckpoint, float, dict[int, torch.Tensor]]:
    """Prefill one prompt and capture the same no-replay checkpoint as repair."""
    if model.training:
        raise RuntimeError("production prompt generation requires model.eval()")
    ids = prompt_ids.to(device=primary_device(model), dtype=torch.long)
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1:
        raise ValueError("prompt_ids must have shape [1, positive_length]")
    length = int(ids.shape[1])
    mask = torch.ones_like(ids)
    selected_layers = list(map(int, selected_hidden_layers))
    kwargs = {
        "input_ids": ids,
        "attention_mask": mask,
        "position_ids": torch.arange(length, device=ids.device, dtype=torch.long)[None],
        "cache_position": torch.arange(length, device=ids.device, dtype=torch.long),
        "past_key_values": None,
        "use_cache": True,
        "return_dict": True,
        "output_hidden_states": bool(selected_layers),
    }
    started = time.perf_counter()
    with torch.inference_mode():
        try:
            output = model(**kwargs)
        except TypeError as exc:
            if "cache_position" not in str(exc):
                raise
            kwargs.pop("cache_position")
            output = model(**kwargs)
    elapsed = time.perf_counter() - started
    checkpoint = capture_cache_checkpoint(
        output.past_key_values,
        output.logits[:, -1],
        ids,
        length,
        attention_mask=mask,
        model_id=model_id,
        model_revision=model_revision,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        tokenizer_metadata=tokenizer_checkpoint_metadata(tokenizer),
        generation_metadata=generation,
    )
    hidden = {}
    for layer in selected_layers:
        actual = layer if layer >= 0 else len(output.hidden_states) + layer
        if not 0 <= actual < len(output.hidden_states):
            raise ValueError(f"selected hidden layer is invalid: {layer}")
        hidden[layer] = output.hidden_states[actual][0, -1].detach().cpu().clone()
    return checkpoint, elapsed, hidden


def generate_prompt_completion_fixed_microbatch(
    model: Any,
    tokenizer: Any,
    prompt_ids: torch.Tensor,
    generation: Mapping[str, Any],
    seed: int,
    *,
    branch_id: str,
    microbatch_size: int,
    model_id: str,
    model_revision: str | None,
    tokenizer_id: str,
    tokenizer_revision: str | None,
    selected_hidden_layers: Iterable[int] = (),
) -> PromptGenerationResult:
    checkpoint, prefill, prompt_hidden = capture_prompt_checkpoint(
        model,
        tokenizer,
        prompt_ids,
        model_id=model_id,
        model_revision=model_revision,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        generation=generation,
        selected_hidden_layers=selected_hidden_layers,
    )
    decoded = continue_from_checkpoint_fixed_microbatch(
        model,
        tokenizer,
        checkpoint,
        branch_seeds=[seed],
        branch_ids=[branch_id],
        generation=generation,
        microbatch_size=microbatch_size,
        return_cache=True,
        selected_hidden_layers=selected_hidden_layers,
    )
    return PromptGenerationResult(decoded.completions[0], prefill, decoded, checkpoint, prompt_hidden)


def generate_prompt_completion_scheduled(
    model: Any,
    tokenizer: Any,
    prompt_ids: torch.Tensor,
    generation: Mapping[str, Any],
    seed: int,
    *,
    branch_id: str,
    microbatch_size: int,
    batching_policy: str,
    model_id: str,
    model_revision: str | None,
    tokenizer_id: str,
    tokenizer_revision: str | None,
    selected_hidden_layers: Iterable[int] = (),
) -> PromptGenerationResult:
    checkpoint, prefill, prompt_hidden = capture_prompt_checkpoint(
        model,
        tokenizer,
        prompt_ids,
        model_id=model_id,
        model_revision=model_revision,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        generation=generation,
        selected_hidden_layers=selected_hidden_layers,
    )
    decoded = continue_from_checkpoint_scheduled(
        model,
        tokenizer,
        checkpoint,
        branch_seeds=[seed],
        branch_ids=[branch_id],
        generation=generation,
        microbatch_size=microbatch_size,
        batching_policy=batching_policy,
        return_cache=True,
        selected_hidden_layers=selected_hidden_layers,
    )
    return PromptGenerationResult(decoded.completions[0], prefill, decoded, checkpoint, prompt_hidden)
