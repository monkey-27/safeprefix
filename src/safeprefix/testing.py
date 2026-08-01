"""Explicitly non-scientific CPU fixtures used by tests and --mock stages."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from torch import nn


class CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 3
    eos_token = ""
    pad_token = ""
    chat_template = "mock"
    init_kwargs = {"_commit_hash": "mock-character-tokenizer-v1"}

    def __call__(self, text: str, *, add_special_tokens: bool = False, return_offsets_mapping: bool = False, return_tensors: str | None = None, **_: Any) -> dict[str, Any]:
        ids = [4 + ord(character) for character in text]
        result: dict[str, Any] = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        if return_tensors == "pt":
            result["input_ids"] = torch.tensor([ids], dtype=torch.long)
            result["attention_mask"] = torch.ones((1, len(ids)), dtype=torch.long)
        return result

    def decode(self, ids: list[int], **_: Any) -> str:
        return "".join(chr(value - 4) for value in ids if value >= 4)

    def apply_chat_template(self, messages: list[dict[str, str]], *, tokenize: bool = False, add_generation_prompt: bool = True, **_: Any) -> Any:
        text = "".join(f"{item['role'].title()}: {item['content']}\n" for item in messages)
        if add_generation_prompt:
            text += "Assistant: "
        return self(text)["input_ids"] if tokenize else text


class ToyCausalModel(nn.Module):
    """Causal model whose cache holds token IDs in a standard legacy layout."""

    def __init__(self, vocab_size: int = 512, hidden_size: int = 8) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.config = SimpleNamespace(_commit_hash="mock-toy-causal-v1", eos_token_id=3)

    def forward(self, input_ids: torch.Tensor, past_key_values: Any = None, output_hidden_states: bool = False, **_: Any) -> Any:
        batch, length = input_ids.shape
        if past_key_values is None:
            previous = torch.empty((batch, 0), dtype=input_ids.dtype, device=input_ids.device)
        else:
            previous = past_key_values[0][0][:, 0, :, 0].long()
        sequence = torch.cat([previous, input_ids], dim=1)
        logits = torch.full((batch, length, self.vocab_size), -20.0, device=input_ids.device)
        target = (input_ids + 1).remainder(self.vocab_size)
        logits.scatter_(-1, target.unsqueeze(-1), 20.0)
        cache_tensor = sequence[:, None, :, None].float()
        cache = ((cache_tensor, cache_tensor + 0.5),)
        hidden_states = None
        if output_hidden_states:
            base = sequence[:, -length:].float().unsqueeze(-1).repeat(1, 1, self.hidden_size)
            hidden_states = tuple(base + layer for layer in range(4))
        return SimpleNamespace(logits=logits, past_key_values=cache, hidden_states=hidden_states)


class StochasticToyCausalModel(ToyCausalModel):
    """Toy cache model with a broad next-token distribution for EOS batching tests."""

    def forward(self, input_ids: torch.Tensor, past_key_values: Any = None, output_hidden_states: bool = False, **kwargs: Any) -> Any:
        output = super().forward(input_ids, past_key_values=past_key_values, output_hidden_states=output_hidden_states, **kwargs)
        output.logits.fill_(-4.0)
        for offset, score in enumerate((2.0, 1.5, 1.0, 0.5)):
            target = (input_ids + offset + 1).remainder(self.vocab_size)
            output.logits.scatter_(-1, target.unsqueeze(-1), score)
        return output


def mock_traces(count: int = 24) -> list[dict[str, Any]]:
    rows = []
    for index in range(count):
        error = None if index % 4 == 0 else index % 3
        steps = [f"Step {step + 1}: deterministic mock reasoning {index}.{step}." for step in range(4)]
        steps[-1] += f"\nFinal answer: {index}"
        rows.append({
            "problem_id": f"mock-{index:04d}", "source_dataset": "mock/safeprefix",
            "source_subset": "cpu_smoke", "source_generator": "mock-generator",
            "problem_text": f"Compute the mock value for item {index}.", "reasoning_steps": steps,
            "final_answer_text": str(index), "reference_answer": index,
            "first_error_index": error, "index_base": "zero" if error is not None else None,
            "final_answer_correct": error is None, "metadata": {"mock": True},
        })
    return rows
