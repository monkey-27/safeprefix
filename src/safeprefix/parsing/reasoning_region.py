"""Separate visible reasoning tokens from the final answer region."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from .answer_parsers import AnswerRegion, parse_answer_region
from .token_alignment import TokenAlignment, align_saved_token_ids, tokenize_with_offsets


@dataclass(frozen=True)
class ReasoningRegion:
    text: str
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    answer: AnswerRegion

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["answer"] = self.answer.to_dict()
        return value


def split_reasoning_and_answer(tokenizer: Any, completion: str) -> ReasoningRegion:
    answer = parse_answer_region(completion)
    if not answer.success or answer.char_start is None:
        raise ValueError(answer.failure_reason or "answer parsing failed")
    alignment = tokenize_with_offsets(tokenizer, completion)
    # Exclude every token that touches the answer marker. A checkpoint can never
    # contain any part of the final-answer region.
    token_end = alignment.token_index_at_or_before(answer.char_start)
    char_end = alignment.boundary_char(token_end)
    return ReasoningRegion(completion[:char_end], 0, char_end, 0, token_end, answer)


def split_reasoning_and_answer_from_tokens(tokenizer: Any, completion: str, token_ids: list[int]) -> tuple[ReasoningRegion, TokenAlignment]:
    answer = parse_answer_region(completion)
    if not answer.success or answer.char_start is None:
        raise ValueError(answer.failure_reason or "answer parsing failed")
    alignment = align_saved_token_ids(tokenizer, token_ids, completion)
    reasoning_boundary = alignment.resolve_boundary(answer.char_start, "before")
    token_end = int(reasoning_boundary["token_offset"])
    char_end = int(reasoning_boundary["resolved_char_offset"])
    content_start = answer.content_char_start if answer.content_char_start is not None else answer.char_start
    content_end = answer.content_char_end if answer.content_char_end is not None else answer.char_end
    answer_start = alignment.resolve_boundary(int(content_start), "before")
    answer_end = alignment.resolve_boundary(int(content_end), "after")
    answer = replace(answer, token_start=int(answer_start["token_offset"]), token_end=int(answer_end["token_offset"]))
    alignment.assert_partition([(0, token_end), (token_end, len(alignment.input_ids))])
    if answer.token_start is not None and answer.token_end is not None and answer.token_start > answer.token_end:
        raise ValueError("answer token region is reversed")
    region = ReasoningRegion(completion[:char_end], 0, char_end, 0, token_end, answer)
    return region, alignment.prefix(token_end, char_end)
