"""Post-hoc views over an exact native fixed-batch generation state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from safeprefix.parsing.reasoning_region import split_reasoning_and_answer_from_tokens
from safeprefix.parsing.segmenters import HybridReasoningSegmenter, ReasoningSpan
from safeprefix.parsing.token_alignment import align_saved_token_ids

from .cache_checkpoint import CacheCheckpoint, capture_cache_checkpoint, tokenizer_checkpoint_metadata
from .hidden_features import build_checkpoint_features
from .loader import primary_device


@dataclass
class NativeTraceState:
    generated: Any
    prompt_token_ids: list[int]
    spans: list[ReasoningSpan]
    checkpoint_offsets: list[int]
    features: torch.Tensor

    def checkpoint(
        self,
        checkpoint_index: int,
        *,
        model: Any,
        tokenizer: Any,
        model_id: str,
        model_revision: str | None,
        tokenizer_id: str,
        tokenizer_revision: str | None,
        generation: dict[str, Any],
    ) -> CacheCheckpoint:
        if checkpoint_index == 0:
            return self.generated.prompt_checkpoint
        if not 0 <= checkpoint_index < len(self.checkpoint_offsets):
            raise IndexError("native checkpoint index is out of range")
        token_end = self.spans[checkpoint_index - 1].token_end
        completion = self.generated.completion
        if completion.past_key_values is None or completion.next_token_logits_by_step is None:
            raise RuntimeError("native cache or per-token logits were not retained")
        full_ids = torch.tensor(
            [self.prompt_token_ids + completion.token_ids],
            dtype=torch.long,
            device=primary_device(model),
        )
        return capture_cache_checkpoint(
            completion.past_key_values,
            completion.next_token_logits_by_step[token_end - 1],
            full_ids,
            self.checkpoint_offsets[checkpoint_index],
            attention_mask=torch.ones_like(full_ids),
            model_id=model_id,
            model_revision=model_revision,
            tokenizer_id=tokenizer_id,
            tokenizer_revision=tokenizer_revision,
            tokenizer_metadata=tokenizer_checkpoint_metadata(tokenizer),
            generation_metadata=generation,
        )


def build_native_trace_state(
    generated: Any,
    tokenizer: Any,
    prompt_token_ids: list[int],
    selected_layers: list[int],
    segmenter: HybridReasoningSegmenter,
) -> NativeTraceState:
    completion = generated.completion
    reasoning, reasoning_alignment = split_reasoning_and_answer_from_tokens(
        tokenizer, completion.text, completion.token_ids
    )
    spans = segmenter.segment_with_alignment(reasoning.text, reasoning_alignment)
    full_alignment = align_saved_token_ids(tokenizer, completion.token_ids, completion.text)
    visible_count = max(
        (index + 1 for index, (start, end) in enumerate(full_alignment.offsets) if end > start),
        default=0,
    )
    if visible_count == 0:
        raise ValueError("native completion has no visible tokens")
    prompt_count = len(prompt_token_ids)
    offsets = [prompt_count] + [prompt_count + span.token_end for span in spans]
    final_offset = prompt_count + visible_count - 1
    if completion.hidden_states_by_layer is None or completion.next_token_logits_by_step is None:
        raise RuntimeError("production native generation did not retain hidden states and checkpoint logits")
    layer_vectors: dict[int, dict[int, torch.Tensor]] = {}
    feature_offsets = [offset - 1 for offset in offsets]
    for layer in selected_layers:
        generated_hidden = completion.hidden_states_by_layer[layer]
        vectors = {prompt_count - 1: generated.prompt_hidden_states[layer]}
        for offset in set(feature_offsets + [final_offset]):
            if offset >= prompt_count:
                vectors[offset] = generated_hidden[offset - prompt_count]
        layer_vectors[layer] = vectors
    nll = torch.zeros(prompt_count + len(completion.token_ids), dtype=torch.float32)
    if completion.token_log_probabilities:
        nll[prompt_count : prompt_count + len(completion.token_log_probabilities)] = -torch.tensor(
            completion.token_log_probabilities
        )
    features = build_checkpoint_features(
        layer_vectors,
        feature_offsets,
        final_offset=final_offset,
        prefix_total_tokens=prompt_count + len(completion.token_ids),
        nll_by_token=nll,
    )
    return NativeTraceState(generated, list(prompt_token_ids), spans, offsets, features)
