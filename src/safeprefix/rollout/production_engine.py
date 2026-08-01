"""Deterministic heterogeneous-cache rollout engine for production labels.

The engine batches branches from different source checkpoints.  At frozen
quantum boundaries it removes finished rows, admits pending rows, and rebuilds
one right-aligned cache batch.  This makes compaction/refill deterministic and
pack-local rather than dependent on wall-clock worker scheduling.

The mathematical generation policy is unchanged: the first suffix token comes
from each checkpoint's saved logits, subsequent logits come from BF16 SDPA,
and every branch uses the existing SHA-derived ``(seed, local_step)`` stream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Mapping, Sequence

import torch

from safeprefix.models.cache_checkpoint import CacheCheckpoint
from safeprefix.models.cache_utils import (
    CacheRowSource,
    cache_to_legacy,
    cache_sequence_length,
    collate_right_aligned_cache_rows,
    preallocate_decode_cache,
)
from safeprefix.models.generation import _uniform
from safeprefix.models.loader import primary_device


@dataclass(frozen=True)
class ProductionRolloutRequest:
    branch_id: str
    rollout_seed: int
    rollout_index: int
    checkpoint: CacheCheckpoint
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.branch_id:
            raise ValueError("branch_id must be nonempty")
        if not 0 <= int(self.rollout_index) < 4:
            raise ValueError("production rollout_index must be in [0, 3]")
        self.checkpoint.validate()


@dataclass(frozen=True)
class ProductionRolloutResult:
    request: ProductionRolloutRequest
    token_ids: list[int]
    text: str
    stop_reason: str
    latency_seconds: float


@dataclass
class ProductionDecodeMetrics:
    useful_output_tokens: int = 0
    forwarded_row_steps: int = 0
    model_forward_calls: int = 0
    compaction_refill_events: int = 0
    admitted_branches: int = 0
    completed_branches: int = 0
    maximum_batch_size_observed: int = 0
    maximum_physical_cache_length: int = 0
    maximum_allocated_bytes: int = 0
    maximum_reserved_bytes: int = 0
    decode_wall_seconds: float = 0.0
    wave_occupancies: list[float] = field(default_factory=list)
    preallocated_cache_waves: int = 0
    refilled_branches: int = 0
    heterogeneous_prefix_waves: int = 0
    completion_removal_events: int = 0
    kv_bytes_per_token_per_row: int = 0
    maximum_decode_kv_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        waste = self.forwarded_row_steps - self.useful_output_tokens
        return {
            "useful_output_tokens": self.useful_output_tokens,
            "forwarded_row_steps": self.forwarded_row_steps,
            "wasted_forwarded_row_steps": waste,
            "forwarded_to_useful_ratio": self.forwarded_row_steps / max(self.useful_output_tokens, 1),
            "model_forward_calls": self.model_forward_calls,
            "compaction_refill_events": self.compaction_refill_events,
            "admitted_branches": self.admitted_branches,
            "completed_branches": self.completed_branches,
            "maximum_batch_size_observed": self.maximum_batch_size_observed,
            "maximum_physical_cache_length": self.maximum_physical_cache_length,
            "maximum_allocated_bytes": self.maximum_allocated_bytes,
            "maximum_reserved_bytes": self.maximum_reserved_bytes,
            "decode_wall_seconds": self.decode_wall_seconds,
            "useful_output_tokens_per_second": self.useful_output_tokens / max(self.decode_wall_seconds, 1e-12),
            "mean_wave_occupancy": (
                sum(self.wave_occupancies) / len(self.wave_occupancies)
                if self.wave_occupancies else None
            ),
            "preallocated_cache_waves": self.preallocated_cache_waves,
            "refilled_branches": self.refilled_branches,
            "heterogeneous_prefix_waves": self.heterogeneous_prefix_waves,
            "completion_removal_events": self.completion_removal_events,
            "kv_bytes_per_token_per_row": self.kv_bytes_per_token_per_row,
            "maximum_decode_kv_bytes": self.maximum_decode_kv_bytes,
        }


@dataclass
class _ActiveBranch:
    request: ProductionRolloutRequest
    source_cache: Any
    source_row: int
    next_logits: torch.Tensor
    logical_length: int
    generated: list[int] = field(default_factory=list)
    admitted_at: float = 0.0


def _processed_logits_without_host_sync(
    logits: torch.Tensor, generation: Mapping[str, Any]
) -> torch.Tensor:
    """Match the frozen logits stack while avoiding token-level host sync."""
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
    return result


def _sample_with_uniforms(
    logits: torch.Tensor,
    generation: Mapping[str, Any],
    uniforms: torch.Tensor,
) -> torch.Tensor:
    temperature = float(generation.get("temperature", 0.0))
    if temperature <= 0:
        return _processed_logits_without_host_sync(logits, generation).argmax(dim=-1)
    # Compute top-p probabilities once.  The reference helper first builds a
    # truncated logit tensor and then applies a second softmax; renormalizing
    # the already-computed truncated probabilities is algebraically identical
    # and avoids a second vocabulary-wide exponential plus a large logit
    # scatter on every decode step.
    base_generation = dict(generation)
    top_p = float(base_generation.pop("top_p", 1.0))
    filtered = _processed_logits_without_host_sync(logits, base_generation)
    if top_p < 1.0:
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        probabilities = _exact_nucleus_probabilities(
            filtered,
            top_p=top_p,
            initial_top_k=int(generation.get("nucleus_initial_top_k", 2048)),
        )
    else:
        probabilities = torch.softmax(filtered, dim=-1)
    cumulative = probabilities.cumsum(dim=-1)
    selected = torch.searchsorted(
        cumulative, uniforms[:, None].contiguous(), right=False
    ).squeeze(-1)
    return selected.clamp(max=cumulative.shape[-1] - 1).to(dtype=torch.long)


def policy_sampling_probabilities(
    logits: torch.Tensor, generation: Mapping[str, Any]
) -> torch.Tensor:
    """Return the exact categorical law used by the production sampler.

    This is public so outcome-blind diagnostics such as the entropy rewind
    baseline cannot drift onto a subtly different top-p/top-k implementation.
    It performs no sampling and does not consume RNG state.
    """

    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if logits.ndim != 2:
        raise ValueError("logits must have shape [vocab] or [batch, vocab]")
    temperature = float(generation.get("temperature", 0.0))
    if temperature <= 0:
        filtered = _processed_logits_without_host_sync(logits, generation)
        selected = filtered.argmax(dim=-1)
        return torch.zeros_like(filtered).scatter(-1, selected[:, None], 1.0)
    base_generation = dict(generation)
    top_p = float(base_generation.pop("top_p", 1.0))
    filtered = _processed_logits_without_host_sync(logits, base_generation)
    if top_p < 1.0:
        return _exact_nucleus_probabilities(
            filtered,
            top_p=top_p,
            initial_top_k=int(generation.get("nucleus_initial_top_k", 2048)),
        )
    return torch.softmax(filtered, dim=-1)


def _full_sort_nucleus_probabilities(
    filtered_logits: torch.Tensor, *, top_p: float
) -> torch.Tensor:
    """Reference-equivalent full-sort nucleus probabilities.

    This path is retained for diffuse distributions and exact ties at the
    nucleus boundary.  It preserves the frozen token-ID-order inverse-CDF
    sampler after filtering.
    """

    sorted_logits, sorted_indices = torch.sort(
        filtered_logits, descending=True, dim=-1
    )
    sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
    remove = (
        sorted_probabilities.cumsum(dim=-1) - sorted_probabilities
    ) >= float(top_p)
    sorted_probabilities = sorted_probabilities.masked_fill(remove, 0.0)
    sorted_probabilities /= sorted_probabilities.sum(dim=-1, keepdim=True)
    return torch.zeros_like(sorted_probabilities).scatter(
        -1, sorted_indices, sorted_probabilities
    )


def _exact_nucleus_probabilities(
    filtered_logits: torch.Tensor,
    *,
    top_p: float,
    initial_top_k: int,
) -> torch.Tensor:
    """Build the exact nucleus without normally sorting the full vocabulary.

    For instruction models, substantially fewer than 2,048 tokens normally
    contain 95% of the probability mass.  We therefore compute the ordinary
    full-vocabulary softmax, select a candidate head with ``topk``, and double
    that head only when its mass is insufficient.  Once the candidate head
    contains the complete nucleus, filtering and renormalization are exactly
    the same as the frozen full-sort implementation.  A tie at the inclusion
    boundary falls back to the reference sort so token-ID assignment remains
    deterministic even for adversarial equal logits.
    """

    if filtered_logits.ndim != 2:
        raise ValueError("nucleus logits must have shape [batch, vocabulary]")
    if not 0.0 < float(top_p) <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    vocabulary = int(filtered_logits.shape[-1])
    if vocabulary < 1:
        raise ValueError("nucleus sampling requires a nonempty vocabulary")
    candidate_count = min(vocabulary, max(1, int(initial_top_k)))
    base_probabilities = torch.softmax(filtered_logits, dim=-1)
    while True:
        candidate_probabilities, candidate_indices = torch.topk(
            base_probabilities,
            k=candidate_count,
            dim=-1,
            largest=True,
            sorted=True,
        )
        candidate_mass = candidate_probabilities.sum(dim=-1)
        enough = candidate_mass >= float(top_p)
        if bool(enough.all()) or candidate_count == vocabulary:
            break
        candidate_count = min(vocabulary, candidate_count * 2)
    if not bool(enough.all()):
        return _full_sort_nucleus_probabilities(
            filtered_logits, top_p=float(top_p)
        )

    cumulative_before = (
        candidate_probabilities.cumsum(dim=-1) - candidate_probabilities
    )
    keep = cumulative_before < float(top_p)
    # The first excluded candidate exists because the retained cumulative mass
    # reaches top_p. If an equal-probability tie crosses that boundary, topk's
    # unspecified tie ordering could select different token IDs from sort.
    retained_counts = keep.sum(dim=-1)
    last_indices = (retained_counts - 1).clamp(min=0, max=candidate_count - 1)
    next_indices = retained_counts.clamp(min=0, max=candidate_count - 1)
    last_values = candidate_probabilities.gather(-1, last_indices[:, None]).squeeze(-1)
    next_values = candidate_probabilities.gather(-1, next_indices[:, None]).squeeze(-1)
    valid_boundary = (retained_counts > 0) & (retained_counts < candidate_count)
    tie_at_boundary = valid_boundary & (last_values == next_values)
    if bool(tie_at_boundary.any()):
        return _full_sort_nucleus_probabilities(
            filtered_logits, top_p=float(top_p)
        )

    retained_probabilities = candidate_probabilities.masked_fill(~keep, 0.0)
    retained_probabilities /= retained_probabilities.sum(dim=-1, keepdim=True)
    probabilities = torch.zeros_like(base_probabilities)
    probabilities.scatter_(-1, candidate_indices, retained_probabilities)
    return probabilities


def _uniform_matrix(
    branches: Sequence[_ActiveBranch], quantum: int, device: torch.device
) -> torch.Tensor:
    rows = [
        [
            _uniform(branch.request.rollout_seed, len(branch.generated) + step)
            for step in range(quantum)
        ]
        for branch in branches
    ]
    return torch.tensor(rows, dtype=torch.float32, device=device)


def _new_active(request: ProductionRolloutRequest, now: float) -> _ActiveBranch:
    checkpoint = request.checkpoint
    return _ActiveBranch(
        request=request,
        source_cache=checkpoint.past_key_values,
        source_row=0,
        next_logits=checkpoint.next_token_logits[0].detach(),
        logical_length=int(checkpoint.prefix_token_count),
        admitted_at=now,
    )


def _kv_bytes_per_token_per_row(cache: Any) -> int:
    layers = cache_to_legacy(cache)
    if not layers:
        raise ValueError("cannot estimate an empty cache")
    sequence = cache_sequence_length(cache)
    batch = int(layers[0][0].shape[0])
    if sequence < 1 or batch < 1:
        raise ValueError("cache byte estimate requires nonempty batch and sequence")
    total = 0
    for layer in layers:
        for tensor in layer[:2]:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("cache key/value components must be tensors")
            total += tensor.numel() * tensor.element_size()
    exact, remainder = divmod(total, batch * sequence)
    if remainder:
        raise ValueError("cache storage is not an integral per-token row size")
    return int(exact)


def _model_forward(
    model: Any,
    *,
    token_ids: torch.Tensor,
    cache: Any,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    cache_position: torch.Tensor,
) -> Any:
    kwargs = {
        "input_ids": token_ids[:, None],
        "attention_mask": attention_mask,
        "position_ids": position_ids[:, None],
        "cache_position": cache_position,
        "past_key_values": cache,
        "use_cache": True,
        "return_dict": True,
        "output_hidden_states": False,
    }
    with torch.inference_mode():
        try:
            return model(**kwargs)
        except TypeError as exc:
            if "cache_position" not in str(exc):
                raise
            kwargs.pop("cache_position")
            return model(**kwargs)


def decode_execution_pack(
    model: Any,
    tokenizer: Any,
    requests: Sequence[ProductionRolloutRequest],
    *,
    generation: Mapping[str, Any],
    maximum_batch_size: int,
    compaction_quantum: int,
    maximum_context_length: int,
    maximum_decode_kv_bytes: int | None = None,
) -> tuple[list[ProductionRolloutResult], ProductionDecodeMetrics]:
    """Decode one immutable pack with deterministic heterogeneous refill."""
    if model.training:
        raise RuntimeError("production rollout decoding requires model.eval()")
    if maximum_batch_size < 1 or compaction_quantum < 1:
        raise ValueError("batch size and compaction quantum must be positive")
    if not requests:
        return [], ProductionDecodeMetrics()
    identities = [request.branch_id for request in requests]
    if len(identities) != len(set(identities)):
        raise ValueError("execution pack contains duplicate branch IDs")
    max_new = int(generation.get("max_new_tokens", 4096))
    if max_new < 1:
        raise ValueError("max_new_tokens must be positive")
    for request in requests:
        if request.checkpoint.prefix_token_count + max_new > maximum_context_length:
            raise ValueError(
                f"branch {request.branch_id} prefix plus continuation exceeds configured context"
            )
    # Longest prefixes are admitted first.  Refill can therefore always place a
    # pending prefix inside the current or freshly compacted physical cache.
    pending = sorted(
        requests,
        key=lambda request: (-request.checkpoint.prefix_token_count, request.branch_id),
    )
    completed: dict[str, ProductionRolloutResult] = {}
    active: list[_ActiveBranch] = []
    metrics = ProductionDecodeMetrics()
    metrics.kv_bytes_per_token_per_row = _kv_bytes_per_token_per_row(
        pending[0].checkpoint.past_key_values
    )
    metrics.maximum_decode_kv_bytes = (
        None if maximum_decode_kv_bytes is None else int(maximum_decode_kv_bytes)
    )
    device = primary_device(model)
    pad_id = int(
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    )
    eos = tokenizer.eos_token_id
    eos_set = {int(eos)} if isinstance(eos, int) else set(map(int, eos or []))
    stop_ids = eos_set | set(map(int, generation.get("stop_token_ids", [])))
    started = time.perf_counter()

    while pending or active:
        now = time.perf_counter()
        admission = 0
        while pending and len(active) < maximum_batch_size:
            candidate = pending[0]
            candidate_count = len(active) + 1
            maximum_final_length = max(
                [
                    branch.request.checkpoint.prefix_token_count + max_new
                    for branch in active
                ]
                + [candidate.checkpoint.prefix_token_count + max_new]
            )
            estimated_one_cache = (
                candidate_count
                * maximum_final_length
                * metrics.kv_bytes_per_token_per_row
            )
            if (
                maximum_decode_kv_bytes is not None
                and estimated_one_cache > int(maximum_decode_kv_bytes)
            ):
                if not active:
                    raise RuntimeError(
                        "one branch exceeds the frozen deterministic KV-cache budget"
                    )
                break
            active.append(_new_active(pending.pop(0), now))
            admission += 1
        if admission:
            if metrics.admitted_branches:
                metrics.refilled_branches += admission
            metrics.admitted_branches += admission
        if not active:
            break
        metrics.compaction_refill_events += 1
        metrics.maximum_batch_size_observed = max(metrics.maximum_batch_size_observed, len(active))
        sources = [
            CacheRowSource(branch.source_cache, branch.source_row, branch.logical_length)
            for branch in active
        ]
        if len({int(source.valid_length) for source in sources}) > 1:
            metrics.heterogeneous_prefix_waves += 1
        cache, initial_attention, logical_lengths_list = collate_right_aligned_cache_rows(sources)
        physical_length = cache_sequence_length(cache)
        metrics.maximum_physical_cache_length = max(metrics.maximum_physical_cache_length, physical_length)
        remaining = [max_new - len(branch.generated) for branch in active]
        wave_steps = min(compaction_quantum, max(remaining))
        if physical_length + wave_steps > maximum_context_length:
            raise RuntimeError("compacted cache wave exceeds the configured context length")
        batch = len(active)
        attention = torch.zeros(
            (batch, physical_length + wave_steps),
            dtype=initial_attention.dtype,
            device=device,
        )
        attention[:, :physical_length] = initial_attention
        use_preallocated = bool(generation.get("preallocate_kv_cache", False))
        if use_preallocated:
            cache = preallocate_decode_cache(
                model, cache, physical_length + wave_steps
            )
            metrics.preallocated_cache_waves += 1
        logical_lengths = torch.tensor(logical_lengths_list, dtype=torch.long, device=device)
        local_lengths = torch.tensor(
            [len(branch.generated) for branch in active], dtype=torch.long, device=device
        )
        limit_lengths = torch.tensor(remaining, dtype=torch.long, device=device)
        uniforms = _uniform_matrix(active, wave_steps, device)
        logits = torch.stack([branch.next_logits.to(device=device) for branch in active])
        token_buffer = torch.full((batch, wave_steps), pad_id, dtype=torch.long, device=device)
        valid_buffer = torch.zeros((batch, wave_steps), dtype=torch.bool, device=device)
        stop_buffer = torch.zeros((batch, wave_steps), dtype=torch.bool, device=device)
        active_mask = torch.ones(batch, dtype=torch.bool, device=device)

        for step in range(wave_steps):
            within_limit = step < limit_lengths
            active_before = active_mask & within_limit
            selected = _sample_with_uniforms(logits, generation, uniforms[:, step])
            selected = torch.where(active_before, selected, torch.full_like(selected, pad_id))
            token_buffer[:, step] = selected
            valid_buffer[:, step] = active_before
            if stop_ids:
                is_stop = torch.zeros_like(active_before)
                for token_id in stop_ids:
                    is_stop |= selected == int(token_id)
                is_stop &= active_before
            else:
                is_stop = torch.zeros_like(active_before)
            reaches_limit = active_before & (step + 1 >= limit_lengths)
            stop_buffer[:, step] = is_stop | reaches_limit
            active_mask &= ~(is_stop | reaches_limit)
            attention[:, physical_length + step] = active_before.to(dtype=attention.dtype)
            positions = torch.where(
                active_before,
                logical_lengths + step,
                torch.zeros_like(logical_lengths),
            )
            output = _model_forward(
                model,
                token_ids=selected,
                cache=cache,
                attention_mask=(
                    attention
                    if use_preallocated
                    else attention[:, : physical_length + step + 1]
                ),
                position_ids=positions,
                cache_position=torch.tensor([physical_length + step], dtype=torch.long, device=device),
            )
            cache = output.past_key_values
            logits = output.logits[:, -1]
            metrics.model_forward_calls += 1
            metrics.forwarded_row_steps += batch
        if not use_preallocated and cache_sequence_length(cache) != physical_length + wave_steps:
            raise AssertionError("model cache did not advance by the frozen decode quantum")

        tokens_cpu = token_buffer.detach().cpu().tolist()
        valid_cpu = valid_buffer.detach().cpu().tolist()
        stop_cpu = stop_buffer.detach().cpu().tolist()
        next_active: list[_ActiveBranch] = []
        completed_in_wave = 0
        for row, branch in enumerate(active):
            added: list[int] = []
            stopped_at: int | None = None
            for step in range(wave_steps):
                if not bool(valid_cpu[row][step]):
                    continue
                token = int(tokens_cpu[row][step])
                added.append(token)
                if bool(stop_cpu[row][step]):
                    stopped_at = step
                    break
            branch.generated.extend(added)
            metrics.useful_output_tokens += len(added)
            stopped_token = added[-1] if added else None
            if stopped_at is not None:
                if stopped_token in eos_set:
                    reason = "eos"
                elif stopped_token in stop_ids:
                    reason = "stop"
                else:
                    reason = "length"
                completed[branch.request.branch_id] = ProductionRolloutResult(
                    request=branch.request,
                    token_ids=list(branch.generated),
                    text=tokenizer.decode(
                        branch.generated,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    ),
                    stop_reason=reason,
                    latency_seconds=time.perf_counter() - branch.admitted_at,
                )
                metrics.completed_branches += 1
                completed_in_wave += 1
            else:
                branch.source_cache = cache
                branch.source_row = row
                branch.next_logits = logits[row].detach()
                branch.logical_length += len(added)
                next_active.append(branch)
        if completed_in_wave:
            metrics.completion_removal_events += 1
        metrics.wave_occupancies.append(len(next_active) / max(batch, 1))
        active = next_active
        if torch.cuda.is_available():
            metrics.maximum_allocated_bytes = max(
                metrics.maximum_allocated_bytes, int(torch.cuda.max_memory_allocated(device))
            )
            metrics.maximum_reserved_bytes = max(
                metrics.maximum_reserved_bytes, int(torch.cuda.max_memory_reserved(device))
            )

    metrics.decode_wall_seconds = time.perf_counter() - started
    if len(completed) != len(requests):
        missing = sorted(set(identities) - set(completed))
        raise AssertionError(f"production decoder lost branches: {missing[:5]}")
    return [completed[identity] for identity in identities], metrics
