"""Low-level Qwen/Llama cache operations.

The SafePrefix V2 production path slices a cache *through* the complete
checkpoint and pairs it with the already-computed next-token logits.  The
``prepare_replay_diagnostic_inputs`` helper retains the rejected V1
slice-one-early/replay protocol solely for numerical-path diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch


def cache_to_legacy(cache: Any) -> tuple[tuple[Any, ...], ...]:
    if isinstance(cache, (tuple, list)):
        return tuple(tuple(layer) for layer in cache)
    if hasattr(cache, "to_legacy_cache"):
        return tuple(tuple(layer) for layer in cache.to_legacy_cache())
    if hasattr(cache, "layers"):
        return tuple((layer.keys, layer.values) for layer in cache.layers)
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return tuple(zip(cache.key_cache, cache.value_cache))
    raise TypeError(f"unsupported cache type: {type(cache).__name__}")


def cache_format(cache: Any) -> str:
    if isinstance(cache, tuple):
        return "legacy_tuple"
    if isinstance(cache, list):
        return "legacy_list"
    return f"dynamic:{type(cache).__module__}.{type(cache).__qualname__}"


def restore_cache_type(original: Any, legacy: tuple[tuple[Any, ...], ...]) -> Any:
    if isinstance(original, tuple):
        return legacy
    if isinstance(original, list):
        return list(legacy)
    factory = getattr(type(original), "from_legacy_cache", None)
    if callable(factory):
        return factory(legacy)
    # Modern DynamicCache versions expose layers but may not expose a factory.
    try:
        from transformers import DynamicCache

        return DynamicCache.from_legacy_cache(legacy)
    except Exception as exc:
        raise TypeError(f"cannot reconstruct cache type {type(original).__name__}") from exc


def restore_cache_format(format_name: str, legacy: tuple[tuple[Any, ...], ...]) -> Any:
    """Reconstruct a serialized standard cache without model-specific casting."""
    if format_name == "legacy_tuple":
        return legacy
    if format_name == "legacy_list":
        return list(legacy)
    if format_name.startswith("dynamic:"):
        try:
            from transformers import DynamicCache

            return DynamicCache.from_legacy_cache(legacy)
        except Exception as exc:
            raise TypeError(f"cannot reconstruct serialized cache {format_name}") from exc
    raise TypeError(f"unsupported serialized cache format: {format_name}")


def cache_sequence_length(cache: Any) -> int:
    layers = cache_to_legacy(cache)
    if not layers:
        return 0
    key = layers[0][0]
    if not isinstance(key, torch.Tensor) or key.ndim < 3:
        raise TypeError("cache keys must be rank-3 or rank-4 tensors")
    return int(key.shape[-2])


def _slice_tensor(value: Any, prefix_length: int, *, clone: bool = True) -> Any:
    if not isinstance(value, torch.Tensor):
        return value
    if value.ndim < 3:
        return value.detach().clone() if clone else value.detach()
    if prefix_length > value.shape[-2]:
        raise ValueError("requested prefix exceeds cache sequence length")
    result = value[..., :prefix_length, :].detach()
    return result.clone() if clone else result


def slice_past_key_values(cache: Any, prefix_length: int) -> Any:
    if prefix_length < 0 or prefix_length > cache_sequence_length(cache):
        raise ValueError("invalid cache prefix length")
    sliced = tuple(
        tuple(_slice_tensor(value, prefix_length) if index < 2 else value for index, value in enumerate(layer))
        for layer in cache_to_legacy(cache)
    )
    return restore_cache_type(cache, sliced)


def view_past_key_values(cache: Any, prefix_length: int) -> Any:
    """Return a read-only prefix view backed by the complete teacher-forced cache.

    Production collation copies every source row into independent writable
    storage before decode.  Keeping immutable checkpoint prefixes as views
    therefore preserves exact K/V values while avoiding one deep copy per
    dataset step.  Callers must never pass these views directly to a mutating
    model forward.
    """

    if prefix_length < 0 or prefix_length > cache_sequence_length(cache):
        raise ValueError("invalid cache prefix length")
    sliced = tuple(
        tuple(
            _slice_tensor(value, prefix_length, clone=False)
            if index < 2
            else value
            for index, value in enumerate(layer)
        )
        for layer in cache_to_legacy(cache)
    )
    return restore_cache_type(cache, sliced)


def repeat_past_key_values(cache: Any, repeats: int) -> Any:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    repeated = tuple(
        tuple(value.repeat_interleave(repeats, dim=0) if index < 2 and isinstance(value, torch.Tensor) else value for index, value in enumerate(layer))
        for layer in cache_to_legacy(cache)
    )
    return restore_cache_type(cache, repeated)


def select_past_key_values_batch(cache: Any, row: int, prefix_length: int | None = None) -> Any:
    """Extract one independent cache row, optionally cropped to its true length."""
    legacy = cache_to_legacy(cache)
    if not legacy:
        raise ValueError("cannot select from an empty cache")
    batch = int(legacy[0][0].shape[0])
    if not 0 <= int(row) < batch:
        raise IndexError("cache batch row is out of range")
    length = cache_sequence_length(cache) if prefix_length is None else int(prefix_length)
    if not 0 <= length <= cache_sequence_length(cache):
        raise ValueError("selected cache prefix length is invalid")
    selected = tuple(
        tuple(
            value[int(row) : int(row) + 1, ..., :length, :].detach().clone()
            if index < 2 and isinstance(value, torch.Tensor)
            else value
            for index, value in enumerate(layer)
        )
        for layer in legacy
    )
    return restore_cache_type(cache, selected)


def clone_past_key_values(cache: Any) -> Any:
    return slice_past_key_values(cache, cache_sequence_length(cache))


@dataclass(frozen=True)
class CacheRowSource:
    """One logical history inside an optionally padded physical cache."""

    cache: Any
    row: int
    valid_length: int


def collate_right_aligned_cache_rows(
    sources: Iterable[CacheRowSource],
) -> tuple[Any, torch.Tensor, list[int]]:
    """Collate heterogeneous logical histories into one masked cache batch.

    Cached key/value vectors have already received RoPE at their logical token
    positions.  Moving them to a right-aligned physical cache slot is therefore
    safe as long as leading slots are masked and future ``position_ids`` remain
    logical while ``cache_position`` addresses the shared physical append slot.
    This operation never casts tensors and returns independent writable storage.
    """
    values = list(sources)
    if not values:
        raise ValueError("at least one cache row is required")
    formats = {cache_format(item.cache) for item in values}
    # A survivor from a preallocated decode wave is a StaticCache while a new
    # checkpoint is normally a DynamicCache.  Both expose the same lossless
    # legacy K/V tensors, so canonicalize mixed modern Cache classes through a
    # DynamicCache.  Tuple/list caches remain strict because changing those
    # representations can alter older model dispatch paths.
    modern = all(value.startswith("dynamic:") for value in formats)
    if len(formats) != 1 and not modern:
        raise ValueError(f"cannot collate incompatible cache formats: {sorted(formats)}")
    lengths = [int(item.valid_length) for item in values]
    if any(length < 1 for length in lengths):
        raise ValueError("cache rows must contain at least one valid token")
    maximum = max(lengths)
    first_layers = cache_to_legacy(values[0].cache)
    if not first_layers:
        raise ValueError("cannot collate an empty cache")
    output_layers: list[tuple[Any, ...]] = []
    for layer_index in range(len(first_layers)):
        template_layer = first_layers[layer_index]
        output_values: list[Any] = []
        for component_index, template in enumerate(template_layer):
            if component_index >= 2 or not isinstance(template, torch.Tensor):
                output_values.append(template)
                continue
            shape = list(template.shape)
            shape[0] = len(values)
            shape[-2] = maximum
            collated = torch.zeros(shape, dtype=template.dtype, device=template.device)
            for output_row, source in enumerate(values):
                layer = cache_to_legacy(source.cache)[layer_index]
                tensor = layer[component_index]
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError("cache key/value components must be tensors")
                if tensor.dtype != template.dtype or tensor.device != template.device:
                    raise ValueError("cache collation may not cast or move tensors")
                if not 0 <= int(source.row) < int(tensor.shape[0]):
                    raise IndexError("source cache row is out of range")
                length = int(source.valid_length)
                if length > int(tensor.shape[-2]):
                    raise ValueError("logical cache length exceeds physical cache length")
                collated[output_row : output_row + 1, ..., -length:, :].copy_(
                    tensor[int(source.row) : int(source.row) + 1, ..., -length:, :]
                )
            output_values.append(collated)
        output_layers.append(tuple(output_values))
    output_format = (
        "dynamic:transformers.cache_utils.DynamicCache"
        if modern
        else next(iter(formats))
    )
    cache = restore_cache_format(output_format, tuple(output_layers))
    device = output_layers[0][0].device
    mask = torch.zeros((len(values), maximum), dtype=torch.long, device=device)
    for row, length in enumerate(lengths):
        mask[row, -length:] = 1
    return cache, mask, lengths


def preallocate_decode_cache(model: Any, cache: Any, max_cache_len: int) -> Any:
    """Copy a compacted cache into a version-compatible StaticCache.

    DynamicCache performs a full ``torch.cat`` for every generated token and
    layer.  A decode wave has a frozen physical length, so it can instead use a
    preallocated StaticCache without changing token positions, masks, logits,
    or sampling.  Transformers 4.55 and 4.57 expose different constructors;
    both are handled explicitly and any unsupported implementation fails the
    production smoke rather than silently falling back.
    """

    from transformers import StaticCache

    legacy = cache_to_legacy(cache)
    if not legacy:
        raise ValueError("cannot preallocate an empty cache")
    current = int(legacy[0][0].shape[-2])
    if max_cache_len < current:
        raise ValueError("static cache capacity is shorter than the compacted prefix")
    template = legacy[0][0]
    batch = int(template.shape[0])
    common = {
        "config": model.config,
        "max_cache_len": int(max_cache_len),
    }
    try:
        # Transformers 4.55 production environment.
        result = StaticCache(
            **common,
            max_batch_size=batch,
            device=template.device,
            dtype=template.dtype,
        )
    except TypeError:
        # Transformers 4.57+ infers batch/device/dtype lazily on first update.
        result = StaticCache(**common)
    positions = torch.arange(current, dtype=torch.long, device=template.device)
    for layer_index, layer in enumerate(legacy):
        key, value = layer[:2]
        result.update(key, value, layer_index, {"cache_position": positions})
    return result


@dataclass(frozen=True)
class RestoreInputs:
    cache: Any
    replay_token_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    cache_position: torch.Tensor
    prefix_length: int


def prepare_replay_diagnostic_inputs(
    cache: Any,
    full_input_ids: torch.Tensor,
    prefix_length: int,
    *,
    repeats: int = 1,
) -> RestoreInputs:
    if full_input_ids.ndim != 2 or full_input_ids.shape[0] != 1:
        raise ValueError("full_input_ids must have shape [1, sequence]")
    if not 1 <= prefix_length <= full_input_ids.shape[1]:
        raise ValueError("checkpoint prefix must contain at least one token")
    early = slice_past_key_values(cache, prefix_length - 1)
    if repeats > 1:
        early = repeat_past_key_values(early, repeats)
    device = full_input_ids.device
    replay = full_input_ids[:, prefix_length - 1 : prefix_length].repeat(repeats, 1)
    attention = torch.ones((repeats, prefix_length), dtype=torch.long, device=device)
    positions = torch.full((repeats, 1), prefix_length - 1, dtype=torch.long, device=device)
    cache_position = torch.tensor([prefix_length - 1], dtype=torch.long, device=device)
    return RestoreInputs(early, replay, attention, positions, cache_position, prefix_length)


def forward_replay_diagnostic(model: Any, inputs: RestoreInputs) -> Any:
    kwargs = {
        "input_ids": inputs.replay_token_ids,
        "attention_mask": inputs.attention_mask,
        "position_ids": inputs.position_ids,
        "cache_position": inputs.cache_position,
        "past_key_values": inputs.cache,
        "use_cache": True,
        "return_dict": True,
    }
    try:
        return model(**kwargs)
    except TypeError as exc:
        if "cache_position" not in str(exc):
            raise
        kwargs.pop("cache_position")
        return model(**kwargs)


# Backward-compatible names exist only so older artifacts and callers fail
# gradually. New production code must not import these aliases.
prepare_restore_inputs = prepare_replay_diagnostic_inputs
forward_restored = forward_replay_diagnostic
