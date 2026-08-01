"""Teacher-force a recorded completion and retain exact cache/feature metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from .cache_checkpoint import CacheCheckpoint, capture_cache_checkpoint
from .loader import primary_device


@dataclass
class TeacherForcedOutput:
    token_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_token_count: int
    past_key_values: Any
    token_log_probabilities: torch.Tensor
    selected_hidden_states: dict[int, dict[int, torch.Tensor]]
    selected_span_mean_hidden_states: dict[int, dict[tuple[int, int], torch.Tensor]]
    checkpoint_next_token_logits: dict[int, torch.Tensor]
    final_logits: torch.Tensor

    def cache_checkpoint(
        self,
        prefix_token_count: int,
        *,
        model_id: str,
        model_revision: str | None,
        tokenizer_id: str,
        tokenizer_revision: str | None,
        tokenizer_metadata: dict[str, Any] | None = None,
        generation_metadata: dict[str, Any] | None = None,
        rng_metadata: dict[str, Any] | None = None,
        clone_cache: bool = True,
    ) -> CacheCheckpoint:
        if prefix_token_count not in self.checkpoint_next_token_logits:
            raise KeyError(
                f"next-token logits for prefix {prefix_token_count} were not retained; "
                "pass selected_checkpoint_offsets when teacher forcing"
            )
        return capture_cache_checkpoint(
            self.past_key_values,
            self.checkpoint_next_token_logits[prefix_token_count],
            self.token_ids,
            prefix_token_count,
            attention_mask=self.attention_mask,
            model_id=model_id,
            model_revision=model_revision,
            tokenizer_id=tokenizer_id,
            tokenizer_revision=tokenizer_revision,
            tokenizer_metadata=tokenizer_metadata,
            generation_metadata=generation_metadata,
            rng_metadata=rng_metadata,
            clone_cache=clone_cache,
        )


def _prompt_and_completion_ids(tokenizer: Any, prompt: str, completion: str) -> tuple[list[int], int]:
    """Serialize the exact autoregressive boundary used at generation time.

    Tokenizing ``prompt + completion`` in one call can merge a token across the
    boundary. A native generation cannot do that: prompt IDs already exist
    before the first completion token is sampled. Teacher forcing therefore
    encodes the two regions independently and concatenates their IDs.
    """
    prefix = list(map(int, tokenizer(prompt, add_special_tokens=False)["input_ids"]))
    suffix = list(map(int, tokenizer(completion, add_special_tokens=False)["input_ids"]))
    return prefix + suffix, len(prefix)


def teacher_force(
    model: Any,
    tokenizer: Any,
    prompt: str,
    completion: str,
    *,
    selected_layers: Iterable[int] = (),
    selected_token_offsets: Iterable[int] = (),
    selected_checkpoint_offsets: Iterable[int] = (),
    selected_span_ranges: Iterable[tuple[int, int]] = (),
) -> TeacherForcedOutput:
    ids, prompt_count = _prompt_and_completion_ids(tokenizer, prompt, completion)
    return teacher_force_token_ids(
        model,
        ids,
        prompt_count=prompt_count,
        selected_layers=selected_layers,
        selected_token_offsets=selected_token_offsets,
        selected_checkpoint_offsets=selected_checkpoint_offsets,
        selected_span_ranges=selected_span_ranges,
    )


def teacher_force_token_ids(
    model: Any,
    token_ids: list[int] | torch.Tensor,
    *,
    prompt_count: int,
    selected_layers: Iterable[int] = (),
    selected_token_offsets: Iterable[int] = (),
    selected_checkpoint_offsets: Iterable[int] = (),
    selected_span_ranges: Iterable[tuple[int, int]] = (),
) -> TeacherForcedOutput:
    device = primary_device(model)
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.to(device=device, dtype=torch.long)
        if token_ids.ndim == 1: token_ids = token_ids.unsqueeze(0)
    else:
        token_ids = torch.tensor([list(map(int, token_ids))], dtype=torch.long, device=device)
    if not 0 <= prompt_count <= token_ids.shape[1]:
        raise ValueError("prompt_count outside exact token sequence")
    attention = torch.ones_like(token_ids)
    layers = list(dict.fromkeys(int(value) for value in selected_layers))
    offsets = sorted(set(int(value) for value in selected_token_offsets))
    checkpoints = sorted(set(int(value) for value in selected_checkpoint_offsets))
    span_ranges = sorted(set((int(left), int(right)) for left, right in selected_span_ranges))
    if any(not 0 <= value < token_ids.shape[1] for value in offsets):
        raise ValueError("selected token offset outside teacher-forced sequence")
    if any(not 1 <= value <= token_ids.shape[1] for value in checkpoints):
        raise ValueError("selected checkpoint offset outside teacher-forced sequence")
    if any(not 0 <= left < right <= token_ids.shape[1] for left, right in span_ranges):
        raise ValueError("selected span range outside teacher-forced sequence")
    with torch.inference_mode():
        output = model(
            input_ids=token_ids,
            attention_mask=attention,
            use_cache=True,
            output_hidden_states=bool(layers),
            return_dict=True,
        )
    shifted = F.log_softmax(output.logits[:, :-1].float(), dim=-1)
    targets = token_ids[:, 1:].unsqueeze(-1)
    token_log_probs = shifted.gather(-1, targets).squeeze(-1)[0]
    features: dict[int, dict[int, torch.Tensor]] = {}
    span_means: dict[int, dict[tuple[int, int], torch.Tensor]] = {}
    if layers:
        hidden = output.hidden_states
        for layer in layers:
            actual = layer if layer >= 0 else len(hidden) + layer
            if not 0 <= actual < len(hidden):
                raise ValueError(f"selected hidden layer is invalid: {layer}")
            features[layer] = {offset: hidden[actual][0, offset].detach().cpu() for offset in offsets}
            span_means[layer] = {
                (left, right): hidden[actual][0, left:right].float().mean(dim=0).detach().cpu()
                for left, right in span_ranges
            }
    return TeacherForcedOutput(
        token_ids=token_ids,
        attention_mask=attention,
        prompt_token_count=int(prompt_count),
        past_key_values=output.past_key_values,
        token_log_probabilities=token_log_probs.detach().cpu(),
        selected_hidden_states=features,
        selected_span_mean_hidden_states=span_means,
        checkpoint_next_token_logits={
            prefix: output.logits[:, prefix - 1].detach().clone() for prefix in checkpoints
        },
        final_logits=output.logits[0, -1].detach().cpu(),
    )


def teacher_force_token_ids_chunked(
    model: Any,
    token_ids: list[int] | torch.Tensor,
    *,
    prompt_count: int,
    chunk_size: int = 256,
    selected_layers: Iterable[int] = (),
    selected_token_offsets: Iterable[int] = (),
    selected_checkpoint_offsets: Iterable[int] = (),
    selected_span_ranges: Iterable[tuple[int, int]] = (),
) -> TeacherForcedOutput:
    """Teacher-force long traces without retaining full-sequence logits.

    Causal K/V states are accumulated in deterministic contiguous chunks.  The
    model sees the same token sequence, masks, logical positions, and weights;
    only the prefill execution schedule changes.  Checkpoint logits and hidden
    vectors are copied at their exact global token offsets, while token NLLs
    are reduced chunk-wise.  This avoids the multi-gigabyte
    ``[sequence,vocabulary]`` FP32 log-softmax used by the simple reference
    implementation.
    """

    if int(chunk_size) < 1:
        raise ValueError("teacher-forcing chunk_size must be positive")
    device = primary_device(model)
    if isinstance(token_ids, torch.Tensor):
        ids = token_ids.to(device=device, dtype=torch.long)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
    else:
        ids = torch.tensor([list(map(int, token_ids))], dtype=torch.long, device=device)
    total = int(ids.shape[1])
    if ids.ndim != 2 or ids.shape[0] != 1 or not 0 <= int(prompt_count) <= total:
        raise ValueError("chunked teacher forcing expects one valid token sequence")
    layers = list(dict.fromkeys(int(value) for value in selected_layers))
    offsets = sorted(set(int(value) for value in selected_token_offsets))
    checkpoints = sorted(set(int(value) for value in selected_checkpoint_offsets))
    spans = sorted(set((int(left), int(right)) for left, right in selected_span_ranges))
    if any(not 0 <= value < total for value in offsets):
        raise ValueError("selected token offset outside teacher-forced sequence")
    if any(not 1 <= value <= total for value in checkpoints):
        raise ValueError("selected checkpoint offset outside teacher-forced sequence")
    if any(not 0 <= left < right <= total for left, right in spans):
        raise ValueError("selected span range outside teacher-forced sequence")

    attention = torch.ones_like(ids)
    token_log_probs = torch.empty(max(total - 1, 0), dtype=torch.float32)
    selected_hidden: dict[int, dict[int, torch.Tensor]] = {layer: {} for layer in layers}
    span_sums: dict[int, dict[tuple[int, int], torch.Tensor]] = {layer: {} for layer in layers}
    span_counts: dict[tuple[int, int], int] = {span: 0 for span in spans}
    checkpoint_logits: dict[int, torch.Tensor] = {}
    cache = None
    final_logits: torch.Tensor | None = None

    for start in range(0, total, int(chunk_size)):
        end = min(total, start + int(chunk_size))
        positions = torch.arange(start, end, dtype=torch.long, device=device)
        kwargs = {
            "input_ids": ids[:, start:end],
            "attention_mask": attention[:, :end],
            "position_ids": positions.unsqueeze(0),
            "cache_position": positions,
            "past_key_values": cache,
            "use_cache": True,
            "output_hidden_states": bool(layers),
            "return_dict": True,
        }
        with torch.inference_mode():
            try:
                output = model(**kwargs)
            except TypeError as exc:
                if "cache_position" not in str(exc):
                    raise
                kwargs.pop("cache_position")
                output = model(**kwargs)
        cache = output.past_key_values
        logits = output.logits[0]
        prediction_end = min(end, total - 1)
        if prediction_end > start:
            local = logits[: prediction_end - start].float()
            targets = ids[0, start + 1 : prediction_end + 1]
            gathered = local.gather(-1, targets[:, None]).squeeze(-1)
            values = gathered - torch.logsumexp(local, dim=-1)
            token_log_probs[start:prediction_end] = values.detach().cpu()
        for prefix in checkpoints:
            global_offset = prefix - 1
            if start <= global_offset < end:
                checkpoint_logits[prefix] = logits[
                    global_offset - start
                ].detach().clone().unsqueeze(0)
        if layers:
            hidden = output.hidden_states
            for requested_layer in layers:
                actual = requested_layer if requested_layer >= 0 else len(hidden) + requested_layer
                if not 0 <= actual < len(hidden):
                    raise ValueError(f"selected hidden layer is invalid: {requested_layer}")
                current = hidden[actual][0]
                for global_offset in offsets:
                    if start <= global_offset < end:
                        selected_hidden[requested_layer][global_offset] = (
                            current[global_offset - start].detach().cpu()
                        )
                for span in spans:
                    left, right = max(span[0], start), min(span[1], end)
                    if left >= right:
                        continue
                    contribution = current[left - start : right - start].float().sum(dim=0).detach().cpu()
                    previous = span_sums[requested_layer].get(span)
                    span_sums[requested_layer][span] = contribution if previous is None else previous + contribution
                    if requested_layer == layers[0]:
                        span_counts[span] += right - left
        final_logits = logits[-1].detach().cpu()
        del output, logits

    missing_logits = sorted(set(checkpoints) - set(checkpoint_logits))
    missing_hidden = {
        layer: sorted(set(offsets) - set(values))
        for layer, values in selected_hidden.items()
        if set(offsets) - set(values)
    }
    if missing_logits or missing_hidden or cache is None or final_logits is None:
        raise AssertionError(
            f"chunked teacher forcing lost requested state: logits={missing_logits}, hidden={missing_hidden}"
        )
    span_means = {
        layer: {
            span: values[span] / max(span_counts[span], 1)
            for span in spans
        }
        for layer, values in span_sums.items()
    }
    return TeacherForcedOutput(
        token_ids=ids,
        attention_mask=attention,
        prompt_token_count=int(prompt_count),
        past_key_values=cache,
        token_log_probabilities=token_log_probs,
        selected_hidden_states=selected_hidden,
        selected_span_mean_hidden_states=span_means,
        checkpoint_next_token_logits=checkpoint_logits,
        final_logits=final_logits,
    )
