import pytest

from safeprefix.parsing.reasoning_region import split_reasoning_and_answer, split_reasoning_and_answer_from_tokens
from safeprefix.parsing.token_alignment import align_boundaries, align_saved_token_ids, tokenize_with_offsets
from safeprefix.testing import CharacterTokenizer


def test_character_alignment_is_exact() -> None:
    tokenizer = CharacterTokenizer()
    alignment = tokenize_with_offsets(tokenizer, "abcd")
    assert alignment.token_index_at_or_before(2) == 2
    assert alignment.boundary_char(2) == 2
    assert align_boundaries(tokenizer, "abcd", [1, 3])[1]["exact"]


def test_answer_tokens_are_not_reasoning_checkpoints() -> None:
    tokenizer = CharacterTokenizer()
    value = split_reasoning_and_answer(tokenizer, r"Reasoning.\n\n\boxed{4}")
    assert "boxed" not in value.text
    assert value.token_end == value.answer.char_start


def test_alignment_rejects_out_of_range_offsets() -> None:
    with pytest.raises(ValueError):
        tokenize_with_offsets(CharacterTokenizer(), "x").boundary_char(2)


def test_saved_ids_are_preserved_during_alignment() -> None:
    tokenizer = CharacterTokenizer()
    ids = tokenizer("exact", add_special_tokens=False)["input_ids"]
    alignment = align_saved_token_ids(tokenizer, ids, "exact")
    assert list(alignment.input_ids) == ids
    assert alignment.boundary_char(len(ids)) == len("exact")
    assert alignment.method == "direct_single_token_decode"


@pytest.mark.parametrize(
    "text",
    [
        "Unicode: π and café.",
        "multiple   spaces\n\nand newlines",
        r"LaTeX $x=\frac{1}{2}$ and punctuation!",
        "1. First paragraph.\n\n2. Second paragraph.",
        r"unfinished \boxed{value",
        "Reasoning.\nFinal answer: 12",
    ],
)
def test_direct_saved_id_alignment_reconstructs_varied_text(text: str) -> None:
    tokenizer = CharacterTokenizer()
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    alignment = align_saved_token_ids(tokenizer, ids, text)
    alignment.assert_partition([(0, len(ids) // 2), (len(ids) // 2, len(ids))])
    assert list(alignment.input_ids) == ids
    assert alignment.boundary_char(len(ids)) == len(text)


def test_boundary_snapping_has_predefined_direction_and_displacement() -> None:
    alignment = tokenize_with_offsets(CharacterTokenizer(), "abcd")
    # Construct a two-character token to exercise an interior boundary.
    wide = type(alignment)((1, 2, 3), ((0, 2), (2, 3), (3, 4)), 4, "fixture")
    before = wide.resolve_boundary(1, "before")
    after = wide.resolve_boundary(1, "after")
    assert before == {
        "requested_char_offset": 1,
        "token_offset": 0,
        "resolved_char_offset": 0,
        "exact": False,
        "snap_policy": "before",
        "character_displacement": -1,
    }
    assert after["token_offset"] == 1
    assert after["resolved_char_offset"] == 2
    assert after["character_displacement"] == 1


def test_special_eos_token_is_preserved_as_zero_width() -> None:
    tokenizer = CharacterTokenizer()
    text = "done"
    ids = tokenizer(text, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    alignment = align_saved_token_ids(tokenizer, ids, text)
    assert alignment.offsets[-1] == (len(text), len(text))
    alignment.assert_partition([(0, len(ids) - 1), (len(ids) - 1, len(ids))])


def test_reasoning_and_answer_tokens_do_not_overlap_and_reconstruct() -> None:
    tokenizer = CharacterTokenizer()
    text = r"Reasoning with punctuation.\n\n\boxed{12}"
    ids = tokenizer(text, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    reasoning, prefix_alignment = split_reasoning_and_answer_from_tokens(tokenizer, text, ids)
    assert reasoning.token_end <= reasoning.answer.token_start
    assert reasoning.answer.token_start <= reasoning.answer.token_end
    assert list(prefix_alignment.input_ids) == ids[: reasoning.token_end]
    full = align_saved_token_ids(tokenizer, ids, text)
    full.assert_partition([(0, reasoning.token_end), (reasoning.token_end, len(ids))])


def test_prefix_decode_fallback_handles_split_unicode_bytes() -> None:
    class SplitUnicodeTokenizer:
        all_special_ids = [99]

        def decode(self, ids, **kwargs):
            visible = [value for value in ids if value != 99]
            mapping = {
                (): "", (1,): "�", (2,): "�", (3,): "!",
                (1, 2): "π", (1, 2, 3): "π!",
            }
            return mapping[tuple(visible)]

    alignment = align_saved_token_ids(SplitUnicodeTokenizer(), [1, 2, 3, 99], "π!")
    assert alignment.method == "direct_saved_prefix_decode"
    assert alignment.offsets == ((0, 0), (0, 1), (1, 2), (2, 2))
    assert alignment.resolve_boundary(1, "before")["exact"]


def test_partition_rejects_dropped_or_duplicated_tokens() -> None:
    alignment = tokenize_with_offsets(CharacterTokenizer(), "abcd")
    with pytest.raises(ValueError):
        alignment.assert_partition([(0, 2), (3, 4)])


def test_fast_tokenizer_unicode_overlap_preserves_valid_token_boundaries() -> None:
    class OverlappingUnicodeOffsetTokenizer:
        def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
            assert text == "a÷b"
            assert add_special_tokens is False
            assert return_offsets_mapping is True
            return {
                "input_ids": [1, 2, 3, 4],
                "offset_mapping": [(0, 1), (1, 2), (1, 2), (2, 3)],
            }

    alignment = tokenize_with_offsets(OverlappingUnicodeOffsetTokenizer(), "a÷b")
    assert alignment.input_ids == (1, 2, 3, 4)
    assert alignment.offsets == ((0, 1), (1, 2), (2, 2), (2, 3))
    assert alignment.method == "retokenized_text_overlap_normalized"
    assert alignment.token_index_at_or_before(1) == 1
    assert alignment.token_index_at_or_before(2) == 3
