"""Checkpoint-level hidden-state feature construction."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch


def build_checkpoint_features(
    layer_vectors: Mapping[int, Mapping[int, torch.Tensor]],
    checkpoint_offsets: list[int],
    *,
    final_offset: int,
    prefix_total_tokens: int,
    nll_by_token: torch.Tensor | None = None,
) -> torch.Tensor:
    layers = list(layer_vectors)
    if not layers or not checkpoint_offsets:
        raise ValueError("hidden layers and checkpoints must be nonempty")
    final = torch.cat([layer_vectors[layer][final_offset].float() for layer in layers])
    rows = []
    previous = None
    for offset in checkpoint_offsets:
        current = torch.cat([layer_vectors[layer][offset].float() for layer in layers])
        difference = torch.zeros_like(current) if previous is None else current - previous
        position = torch.tensor([
            offset / max(prefix_total_tokens, 1),
            float(offset),
            float(max(prefix_total_tokens - offset, 0)),
        ])
        optional = torch.empty(0)
        if nll_by_token is not None:
            left, right = max(0, offset - 8), min(len(nll_by_token), offset + 8)
            optional = torch.tensor([float(nll_by_token[left:right].mean())])
        rows.append(torch.cat([current, difference, final, position, optional]))
        previous = current
    return torch.stack(rows)


def build_representation_features(
    layer_vectors: Mapping[int, Mapping[int, torch.Tensor]],
    span_mean_vectors: Mapping[int, Mapping[tuple[int, int], torch.Tensor]],
    checkpoint_offsets: list[int],
    span_ranges: Sequence[tuple[int, int]],
    *,
    final_offset: int,
    prefix_total_tokens: int,
    representation: str,
    nll_by_token: torch.Tensor | None = None,
) -> torch.Tensor:
    """Construct one frozen representation candidate without another forward.

    Layer ordering is the configured middle, upper-middle, and final trio.
    Root has no reasoning span, so its span-mean component equals the prompt's
    final token. This preserves dimensionality without inventing a span.
    """
    layers = list(layer_vectors)
    if len(layers) < 3:
        raise ValueError("representation comparison requires middle, upper-middle, and final layers")
    if representation.startswith("middle_"):
        chosen = [layers[0]]
    elif representation.startswith("upper_middle_"):
        chosen = [layers[1]]
    elif representation.startswith("final_layer_"):
        chosen = [layers[-1]]
    elif representation.startswith("three_layer_concat_"):
        chosen = [layers[0], layers[1], layers[-1]]
    else:
        raise ValueError(f"unknown hidden representation: {representation}")
    use_mean = "span_mean" in representation
    use_both = "final_plus_span_mean" in representation

    def final_token(offset: int) -> torch.Tensor:
        return torch.cat([layer_vectors[layer][offset].float() for layer in chosen])

    def span_mean(index: int, offset: int) -> torch.Tensor:
        if index == 0:
            return final_token(offset)
        key = tuple(map(int, span_ranges[index - 1]))
        return torch.cat([span_mean_vectors[layer][key].float() for layer in chosen])

    def vector(index: int, offset: int) -> torch.Tensor:
        token = final_token(offset)
        mean = span_mean(index, offset)
        if use_both:
            return torch.cat([token, mean])
        return mean if use_mean else token

    terminal = final_token(final_offset)
    if use_both:
        terminal = torch.cat([terminal, terminal])
    rows = []
    previous = None
    for index, offset in enumerate(checkpoint_offsets):
        current = vector(index, offset)
        difference = torch.zeros_like(current) if previous is None else current - previous
        position = torch.tensor([offset / max(prefix_total_tokens, 1), float(offset), float(max(prefix_total_tokens - offset, 0))])
        optional = torch.empty(0)
        if nll_by_token is not None:
            left, right = max(0, offset - 8), min(len(nll_by_token), offset + 8)
            optional = torch.tensor([float(nll_by_token[left:right].mean())])
        rows.append(torch.cat([current, difference, terminal, position, optional]))
        previous = current
    return torch.stack(rows)
