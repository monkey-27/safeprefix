"""Complete-cache checkpoints with saved next-token logits.

A checkpoint represents the model state *after* every token in the preserved
prefix has already been processed. Continuation samples the first suffix token
from ``next_token_logits`` and never replays the checkpoint-final token.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import torch

from .cache_utils import (
    cache_format,
    cache_sequence_length,
    cache_to_legacy,
    clone_past_key_values,
    repeat_past_key_values,
    restore_cache_format,
    slice_past_key_values,
    view_past_key_values,
)


PROTOCOL_VERSION = "complete_cache_saved_next_logits_v2"


def tokenizer_checkpoint_metadata(tokenizer: Any) -> dict[str, Any]:
    """Capture tokenizer state that can affect exact continuation semantics."""
    return {
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "padding_side": getattr(tokenizer, "padding_side", None),
        "truncation_side": getattr(tokenizer, "truncation_side", None),
        "chat_template": getattr(tokenizer, "chat_template", None),
    }


def _cache_dtype(cache: Any) -> str:
    for layer in cache_to_legacy(cache):
        for value in layer[:2]:
            if isinstance(value, torch.Tensor):
                return str(value.dtype).removeprefix("torch.")
    return "unknown"


def _clone_tensor(value: torch.Tensor, device: torch.device | str | None = None) -> torch.Tensor:
    result = value.detach().clone()
    return result if device is None else result.to(device=device)


def _move_cache(cache: Any, device: torch.device | str) -> Any:
    legacy = tuple(
        tuple(_clone_tensor(value, device) if isinstance(value, torch.Tensor) else value for value in layer)
        for layer in cache_to_legacy(cache)
    )
    return restore_cache_format(cache_format(cache), legacy)


@dataclass
class CacheCheckpoint:
    """Exact state required to start a suffix without prefix-token replay."""

    past_key_values: Any
    next_token_logits: torch.Tensor
    prefix_token_count: int
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    cache_position: torch.Tensor
    prefix_token_ids: torch.Tensor
    model_id: str
    model_revision: str | None
    tokenizer_id: str
    tokenizer_revision: str | None
    dtype: str
    checkpoint_token_offset: int
    tokenizer_metadata: dict[str, Any] = field(default_factory=dict)
    generation_metadata: dict[str, Any] = field(default_factory=dict)
    rng_metadata: dict[str, Any] = field(default_factory=dict)
    cache_storage: str = "independent"
    protocol_version: str = PROTOCOL_VERSION

    def validate(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"unsupported checkpoint protocol: {self.protocol_version}")
        if self.prefix_token_count < 1:
            raise ValueError("checkpoint prefix must contain at least one token")
        if self.checkpoint_token_offset != self.prefix_token_count:
            raise ValueError("checkpoint offset and prefix token count differ")
        if cache_sequence_length(self.past_key_values) != self.prefix_token_count:
            raise ValueError("cache sequence length does not equal complete prefix length")
        batch = int(self.next_token_logits.shape[0])
        if self.next_token_logits.ndim != 2 or batch < 1:
            raise ValueError("next_token_logits must have shape [batch, vocabulary]")
        expected_prefix = (batch, self.prefix_token_count)
        if tuple(self.attention_mask.shape) != expected_prefix:
            raise ValueError(f"attention mask shape {tuple(self.attention_mask.shape)} != {expected_prefix}")
        if tuple(self.prefix_token_ids.shape) != expected_prefix:
            raise ValueError(f"prefix token shape {tuple(self.prefix_token_ids.shape)} != {expected_prefix}")
        if tuple(self.position_ids.shape) != (batch, 1):
            raise ValueError("next position_ids must have shape [batch, 1]")
        if tuple(self.cache_position.shape) != (1,):
            raise ValueError("next cache_position must have shape [1]")
        if not bool(torch.all(self.attention_mask == 1)):
            raise ValueError("primary checkpoints currently require an all-ones prefix mask")
        if not bool(torch.all(self.position_ids == self.prefix_token_count)):
            raise ValueError("next position_ids contain an off-by-one error")
        if int(self.cache_position.item()) != self.prefix_token_count:
            raise ValueError("next cache_position contains an off-by-one error")
        if _cache_dtype(self.past_key_values) != self.dtype:
            raise ValueError("recorded cache dtype does not match tensors")
        if self.cache_storage not in {"independent", "readonly_teacher_forced_view"}:
            raise ValueError(f"unsupported checkpoint cache storage: {self.cache_storage}")

    @property
    def batch_size(self) -> int:
        return int(self.next_token_logits.shape[0])


def capture_cache_checkpoint(
    full_cache: Any,
    next_token_logits: torch.Tensor,
    full_input_ids: torch.Tensor,
    prefix_token_count: int,
    *,
    attention_mask: torch.Tensor | None = None,
    model_id: str,
    model_revision: str | None,
    tokenizer_id: str,
    tokenizer_revision: str | None,
    tokenizer_metadata: Mapping[str, Any] | None = None,
    generation_metadata: Mapping[str, Any] | None = None,
    rng_metadata: Mapping[str, Any] | None = None,
    clone_cache: bool = True,
) -> CacheCheckpoint:
    """Capture a complete prefix cache and its native-dtype next-token logits."""
    if full_input_ids.ndim != 2 or full_input_ids.shape[0] != 1:
        raise ValueError("full_input_ids must have shape [1, sequence]")
    if not 1 <= prefix_token_count <= full_input_ids.shape[1]:
        raise ValueError("prefix_token_count outside full token sequence")
    if next_token_logits.ndim == 1:
        next_token_logits = next_token_logits.unsqueeze(0)
    if next_token_logits.ndim != 2 or next_token_logits.shape[0] != 1:
        raise ValueError("next_token_logits must represent one checkpoint")
    device = next_token_logits.device
    ids = full_input_ids[:, :prefix_token_count].to(device=device).detach().clone()
    if attention_mask is None:
        mask = torch.ones_like(ids)
    else:
        mask = attention_mask[:, :prefix_token_count].to(device=device).detach().clone()
    checkpoint_cache = (
        slice_past_key_values(full_cache, prefix_token_count)
        if clone_cache
        else view_past_key_values(full_cache, prefix_token_count)
    )
    checkpoint = CacheCheckpoint(
        past_key_values=checkpoint_cache,
        next_token_logits=next_token_logits.detach().clone(),
        prefix_token_count=prefix_token_count,
        attention_mask=mask,
        position_ids=torch.full((1, 1), prefix_token_count, dtype=torch.long, device=device),
        cache_position=torch.tensor([prefix_token_count], dtype=torch.long, device=device),
        prefix_token_ids=ids,
        model_id=str(model_id),
        model_revision=model_revision,
        tokenizer_id=str(tokenizer_id),
        tokenizer_revision=tokenizer_revision,
        dtype=_cache_dtype(checkpoint_cache),
        checkpoint_token_offset=prefix_token_count,
        tokenizer_metadata=dict(tokenizer_metadata or {}),
        generation_metadata=dict(generation_metadata or {}),
        rng_metadata=dict(rng_metadata or {"sampling": "explicit_uniform_inverse_cdf", "seed_supplied_per_branch": True}),
        cache_storage=(
            "independent" if clone_cache else "readonly_teacher_forced_view"
        ),
    )
    checkpoint.validate()
    return checkpoint


def clone_cache_checkpoint(checkpoint: CacheCheckpoint, *, device: torch.device | str | None = None) -> CacheCheckpoint:
    """Return an independent writable checkpoint copy without dtype conversion."""
    target = checkpoint.next_token_logits.device if device is None else device
    cloned = replace(
        checkpoint,
        past_key_values=_move_cache(checkpoint.past_key_values, target),
        next_token_logits=_clone_tensor(checkpoint.next_token_logits, target),
        attention_mask=_clone_tensor(checkpoint.attention_mask, target),
        position_ids=_clone_tensor(checkpoint.position_ids, target),
        cache_position=_clone_tensor(checkpoint.cache_position, target),
        prefix_token_ids=_clone_tensor(checkpoint.prefix_token_ids, target),
        tokenizer_metadata=dict(checkpoint.tokenizer_metadata),
        generation_metadata=dict(checkpoint.generation_metadata),
        rng_metadata=dict(checkpoint.rng_metadata),
        cache_storage="independent",
    )
    cloned.validate()
    return cloned


def repeat_cache_checkpoint(checkpoint: CacheCheckpoint, repeats: int) -> CacheCheckpoint:
    """Create one independent cache row per rollout branch."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    repeated = replace(
        checkpoint,
        past_key_values=repeat_past_key_values(checkpoint.past_key_values, repeats),
        next_token_logits=checkpoint.next_token_logits.repeat_interleave(repeats, dim=0).detach().clone(),
        attention_mask=checkpoint.attention_mask.repeat_interleave(repeats, dim=0).detach().clone(),
        position_ids=checkpoint.position_ids.repeat_interleave(repeats, dim=0).detach().clone(),
        prefix_token_ids=checkpoint.prefix_token_ids.repeat_interleave(repeats, dim=0).detach().clone(),
        cache_position=checkpoint.cache_position.detach().clone(),
        tokenizer_metadata=dict(checkpoint.tokenizer_metadata),
        generation_metadata=dict(checkpoint.generation_metadata),
        rng_metadata=dict(checkpoint.rng_metadata),
        cache_storage="independent",
    )
    repeated.validate()
    return repeated


def checkpoint_payload(checkpoint: CacheCheckpoint) -> dict[str, Any]:
    """Create a CPU serialization payload while preserving every tensor dtype."""
    checkpoint.validate()
    legacy = tuple(
        tuple(value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value for value in layer)
        for layer in cache_to_legacy(checkpoint.past_key_values)
    )
    return {
        "protocol_version": checkpoint.protocol_version,
        "cache_format": cache_format(checkpoint.past_key_values),
        "cache_legacy": legacy,
        "next_token_logits": checkpoint.next_token_logits.detach().cpu().clone(),
        "prefix_token_count": checkpoint.prefix_token_count,
        "attention_mask": checkpoint.attention_mask.detach().cpu().clone(),
        "position_ids": checkpoint.position_ids.detach().cpu().clone(),
        "cache_position": checkpoint.cache_position.detach().cpu().clone(),
        "prefix_token_ids": checkpoint.prefix_token_ids.detach().cpu().clone(),
        "model_id": checkpoint.model_id,
        "model_revision": checkpoint.model_revision,
        "tokenizer_id": checkpoint.tokenizer_id,
        "tokenizer_revision": checkpoint.tokenizer_revision,
        "dtype": checkpoint.dtype,
        "checkpoint_token_offset": checkpoint.checkpoint_token_offset,
        "tokenizer_metadata": dict(checkpoint.tokenizer_metadata),
        "generation_metadata": dict(checkpoint.generation_metadata),
        "rng_metadata": dict(checkpoint.rng_metadata),
        "cache_storage": "independent",
    }


def checkpoint_from_payload(payload: Mapping[str, Any], *, device: torch.device | str = "cpu") -> CacheCheckpoint:
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("serialized checkpoint protocol mismatch")
    legacy = tuple(
        tuple(value.to(device=device) if isinstance(value, torch.Tensor) else value for value in layer)
        for layer in payload["cache_legacy"]
    )
    checkpoint = CacheCheckpoint(
        past_key_values=restore_cache_format(str(payload["cache_format"]), legacy),
        next_token_logits=payload["next_token_logits"].to(device=device),
        prefix_token_count=int(payload["prefix_token_count"]),
        attention_mask=payload["attention_mask"].to(device=device),
        position_ids=payload["position_ids"].to(device=device),
        cache_position=payload["cache_position"].to(device=device),
        prefix_token_ids=payload["prefix_token_ids"].to(device=device),
        model_id=str(payload["model_id"]),
        model_revision=payload.get("model_revision"),
        tokenizer_id=str(payload["tokenizer_id"]),
        tokenizer_revision=payload.get("tokenizer_revision"),
        dtype=str(payload["dtype"]),
        checkpoint_token_offset=int(payload["checkpoint_token_offset"]),
        tokenizer_metadata=dict(payload.get("tokenizer_metadata", {})),
        generation_metadata=dict(payload.get("generation_metadata", {})),
        rng_metadata=dict(payload.get("rng_metadata", {})),
        cache_storage=str(payload.get("cache_storage", "independent")),
        protocol_version=str(payload["protocol_version"]),
    )
    checkpoint.validate()
    return checkpoint


def save_cache_checkpoint(checkpoint: CacheCheckpoint, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=destination.name, suffix=".tmp", dir=destination.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        torch.save(checkpoint_payload(checkpoint), temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_cache_checkpoint(path: str | Path, *, device: torch.device | str = "cpu") -> CacheCheckpoint:
    return checkpoint_from_payload(torch.load(Path(path), map_location="cpu", weights_only=False), device=device)


def compare_caches(left: Any, right: Any) -> dict[str, Any]:
    """Return per-layer key/value exact and absolute-difference diagnostics.

    Cache tensors can legitimately contain paired NaNs in unused numerical
    lanes.  Those lanes represent the same state and compare equal here, just
    as ``torch.testing.assert_close(..., equal_nan=True)`` treats them.  An
    unpaired NaN remains a blocking discrepancy and is reported explicitly.
    """
    left_layers, right_layers = cache_to_legacy(left), cache_to_legacy(right)
    if len(left_layers) != len(right_layers):
        raise ValueError("cache layer count differs")
    rows: list[dict[str, Any]] = []
    first_difference: int | None = None
    for layer_index, (left_layer, right_layer) in enumerate(zip(left_layers, right_layers)):
        for value_index, label in ((0, "key"), (1, "value")):
            lhs, rhs = left_layer[value_index], right_layer[value_index]
            if not isinstance(lhs, torch.Tensor) or not isinstance(rhs, torch.Tensor):
                raise TypeError("cache keys and values must be tensors")
            same_shape = tuple(lhs.shape) == tuple(rhs.shape)
            same_dtype = lhs.dtype == rhs.dtype
            if same_shape:
                if lhs.is_floating_point() or lhs.is_complex():
                    left_nan = torch.isnan(lhs)
                    right_nan = torch.isnan(rhs)
                    paired_nan = left_nan & right_nan
                    unpaired_nan = left_nan ^ right_nan
                else:
                    paired_nan = torch.zeros_like(lhs, dtype=torch.bool)
                    unpaired_nan = torch.zeros_like(lhs, dtype=torch.bool)
                equal = (lhs == rhs) | paired_nan
                exact_rate = float(equal.float().mean().cpu()) if lhs.numel() else 1.0
                difference = (lhs.float() - rhs.float()).abs()
                difference = torch.where(paired_nan, torch.zeros_like(difference), difference)
                difference = torch.where(unpaired_nan, torch.full_like(difference, float("inf")), difference)
                maximum = float(difference.max().cpu()) if difference.numel() else 0.0
                mean = float(difference.mean().cpu()) if difference.numel() else 0.0
                paired_nan_count = int(paired_nan.sum().cpu())
                unpaired_nan_count = int(unpaired_nan.sum().cpu())
            else:
                exact_rate, maximum, mean = 0.0, float("inf"), float("inf")
                paired_nan_count, unpaired_nan_count = 0, 0
            if (not same_shape or not same_dtype or exact_rate < 1.0) and first_difference is None:
                first_difference = layer_index
            rows.append(
                {
                    "layer": layer_index,
                    "component": label,
                    "left_shape": list(lhs.shape),
                    "right_shape": list(rhs.shape),
                    "left_dtype": str(lhs.dtype),
                    "right_dtype": str(rhs.dtype),
                    "shape_equal": same_shape,
                    "dtype_equal": same_dtype,
                    "exact_equality_rate": exact_rate,
                    "paired_nan_count": paired_nan_count,
                    "unpaired_nan_count": unpaired_nan_count,
                    "max_absolute_difference": maximum,
                    "mean_absolute_difference": mean,
                }
            )
    return {
        "passed": first_difference is None,
        "first_layer_with_discrepancy": first_difference,
        "layers": len(left_layers),
        "rows": rows,
    }
