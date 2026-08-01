"""Post-hoc visible-reasoning segmenters returning exact token offsets."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .token_alignment import TokenAlignment, tokenize_with_offsets


@dataclass(frozen=True)
class ReasoningSpan:
    span_index: int
    text: str
    char_start: int
    char_end: int
    token_start: int
    token_end: int

    def __post_init__(self) -> None:
        if not (0 <= self.char_start <= self.char_end and 0 <= self.token_start <= self.token_end):
            raise ValueError("invalid span offsets")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def span_index_containing_char(spans: Iterable[ReasoningSpan], char_offset: int) -> int | None:
    """Return the unique post-hoc span containing a source error boundary.

    A hybrid span may intentionally merge several short source steps. The
    containing span is then unsafe as a whole, while the checkpoint before it
    remains safe. An offset at a shared boundary belongs to the later span.
    """
    values = list(spans)
    matches = [span.span_index for span in values if span.char_start <= char_offset < span.char_end]
    if len(matches) > 1:
        raise ValueError("reasoning spans overlap at the annotated error")
    return matches[0] if matches else None


def _latex_depths(text: str) -> list[int]:
    depth = 0
    values = [0] * (len(text) + 1)
    dollar = False
    for index, char in enumerate(text):
        escaped = index > 0 and text[index - 1] == "\\"
        if char == "$" and not escaped:
            dollar = not dollar
        if not dollar and char == "{" and not escaped:
            depth += 1
        elif not dollar and char == "}" and not escaped:
            depth = max(0, depth - 1)
        values[index + 1] = depth + int(dollar)
    return values


def _token_spans(text: str, alignment: TokenAlignment, token_boundaries: Iterable[int]) -> list[ReasoningSpan]:
    boundaries = sorted(set([0, *map(int, token_boundaries), len(alignment.input_ids)]))
    spans = []
    for start_token, end_token in zip(boundaries, boundaries[1:]):
        if end_token <= start_token:
            continue
        start_char = 0 if start_token == 0 else alignment.boundary_char(start_token)
        end_char = alignment.boundary_char(end_token)
        spans.append(ReasoningSpan(len(spans), text[start_char:end_char], start_char, end_char, start_token, end_token))
    return spans


def source_step_spans(
    text: str,
    tokenizer: Any,
    source_step_ranges: Iterable[tuple[int, int]],
) -> list[ReasoningSpan]:
    """Map frozen source-step ends to exact tokenizer boundaries.

    This is intentionally a measurement-only segmenter for teacher-forced
    traces that already carry human/dataset step annotations.  It is not a
    candidate for native SafePrefix inference, where source steps do not
    exist.  Ranges may extend into the final-answer region; those ends are
    clipped to ``text`` before exact token-boundary mapping.
    """
    alignment = tokenize_with_offsets(tokenizer, text)
    if not alignment.input_ids:
        return []
    character_ends = sorted(
        {
            min(int(end), len(text))
            for start, end in source_step_ranges
            if int(start) < len(text) and min(int(end), len(text)) > 0
        }
    )
    token_ends = [alignment.token_index_at_or_before(value) for value in character_ends]
    return _token_spans(text, alignment, token_ends)


class NaturalParagraphSegmenter:
    def segment(self, text: str, tokenizer: Any) -> list[ReasoningSpan]:
        return self.segment_with_alignment(text, tokenize_with_offsets(tokenizer, text))

    def segment_with_alignment(self, text: str, alignment: TokenAlignment) -> list[ReasoningSpan]:
        char_boundaries = [match.end() for match in re.finditer(r"\n\s*\n+", text)]
        token_boundaries = [alignment.token_index_at_or_before(value) for value in char_boundaries]
        return _token_spans(text, alignment, token_boundaries)


class FixedTokenSegmenter:
    def __init__(self, block_tokens: int = 64) -> None:
        if block_tokens not in {64, 128}:
            raise ValueError("fixed-token control supports only 64 or 128 tokens")
        self.block_tokens = block_tokens

    def segment(self, text: str, tokenizer: Any) -> list[ReasoningSpan]:
        return self.segment_with_alignment(text, tokenize_with_offsets(tokenizer, text))

    def segment_with_alignment(self, text: str, alignment: TokenAlignment) -> list[ReasoningSpan]:
        return _token_spans(text, alignment, range(self.block_tokens, len(alignment.input_ids), self.block_tokens))


class HybridReasoningSegmenter:
    def __init__(self, min_tokens: int = 24, max_tokens: int = 128) -> None:
        if not 1 <= min_tokens <= max_tokens:
            raise ValueError("require 1 <= min_tokens <= max_tokens")
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    def segment(self, text: str, tokenizer: Any) -> list[ReasoningSpan]:
        return self.segment_with_alignment(text, tokenize_with_offsets(tokenizer, text))

    def segment_with_alignment(self, text: str, alignment: TokenAlignment) -> list[ReasoningSpan]:
        total = len(alignment.input_ids)
        if total == 0:
            return []
        depths = _latex_depths(text)
        candidates = {total}
        patterns = [r"\n\s*\n+", r"(?<=[.!?])\s+(?=[A-Z0-9])", r"\n(?=\s*(?:\d+[.)]|[-*]))", r"(?<=\\])\s+(?=[A-Z])"]
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                if depths[match.end()] == 0:
                    candidates.add(alignment.token_index_at_or_before(match.end()))
        candidates = {value for value in candidates if 0 < value <= total}
        boundaries: list[int] = []
        start = 0
        while start < total:
            legal = sorted(value for value in candidates if start + self.min_tokens <= value <= min(total, start + self.max_tokens))
            if legal:
                end = legal[-1]
            else:
                later = sorted(value for value in candidates if value > start)
                end = min(start + self.max_tokens, total) if not later else min(later[0], start + self.max_tokens)
            if total - end < self.min_tokens and end < total:
                end = total
            boundaries.append(end)
            start = end
        return _token_spans(text, alignment, boundaries)
