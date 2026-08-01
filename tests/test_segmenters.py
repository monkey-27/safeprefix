from safeprefix.parsing.segmenters import (
    FixedTokenSegmenter,
    HybridReasoningSegmenter,
    NaturalParagraphSegmenter,
    span_index_containing_char,
)
from safeprefix.testing import CharacterTokenizer


def _assert_exact(text, spans) -> None:
    assert "".join(span.text for span in spans) == text
    assert spans[0].char_start == 0
    assert spans[-1].char_end == len(text)
    assert all(left.token_end == right.token_start for left, right in zip(spans, spans[1:]))


def test_natural_paragraphs_are_lossless() -> None:
    text = "First paragraph.\n\nSecond paragraph.\n\nThird."
    spans = NaturalParagraphSegmenter().segment(text, CharacterTokenizer())
    _assert_exact(text, spans)
    assert len(spans) == 3


def test_hybrid_merges_short_and_splits_long_without_latex_split() -> None:
    text = "Short.\n\n" + "A longer reasoning sentence. " * 8 + r" Keep $x + y = 2$ intact."
    spans = HybridReasoningSegmenter(min_tokens=12, max_tokens=64).segment(text, CharacterTokenizer())
    _assert_exact(text, spans)
    assert all(span.token_end - span.token_start <= 64 for span in spans)


def test_fixed_token_control() -> None:
    text = "x" * 150
    spans = FixedTokenSegmenter(64).segment(text, CharacterTokenizer())
    assert [span.token_end - span.token_start for span in spans] == [64, 64, 22]
    _assert_exact(text, spans)


def test_error_inside_merged_source_step_maps_to_containing_span() -> None:
    text = "short one. short two. " + "Long final reasoning sentence. " * 3
    spans = HybridReasoningSegmenter(min_tokens=20, max_tokens=64).segment(text, CharacterTokenizer())
    error_start = text.index("short two")
    selected = span_index_containing_char(spans, error_start)
    assert selected is not None
    assert spans[selected].char_start <= error_start < spans[selected].char_end
