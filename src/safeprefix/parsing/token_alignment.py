"""Exact, auditable character-to-token offset alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class TokenAlignment:
    input_ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]
    text_length: int
    method: str = "unknown"

    def __post_init__(self) -> None:
        if len(self.input_ids) != len(self.offsets):
            raise ValueError("token IDs and offsets have different lengths")
        previous = 0
        for start, end in self.offsets:
            if start < previous or end < start or end > self.text_length:
                raise ValueError("offset mapping is not monotonic")
            previous = end

    def token_index_at_or_before(self, char_offset: int) -> int:
        if not 0 <= char_offset <= self.text_length:
            raise ValueError("character offset outside text")
        result = 0
        for index, (start, end) in enumerate(self.offsets):
            if end <= char_offset:
                result = index + 1
            elif start < char_offset < end:
                break
            elif start >= char_offset:
                break
        return result

    def token_index_at_or_after(self, char_offset: int) -> int:
        if not 0 <= char_offset <= self.text_length:
            raise ValueError("character offset outside text")
        for index, (start, end) in enumerate(self.offsets):
            if char_offset <= start:
                return index
            if start < char_offset < end:
                return index + 1
        return len(self.offsets)

    def boundary_char(self, token_offset: int) -> int:
        if not 0 <= token_offset <= len(self.offsets):
            raise ValueError("token offset outside sequence")
        if token_offset == 0:
            return 0
        return self.offsets[token_offset - 1][1]

    def prefix(self, token_count: int, text_length: int | None = None) -> "TokenAlignment":
        if not 0 <= token_count <= len(self.input_ids):
            raise ValueError("token_count outside sequence")
        length = self.boundary_char(token_count) if text_length is None else int(text_length)
        return TokenAlignment(self.input_ids[:token_count], self.offsets[:token_count], length, self.method)

    def resolve_boundary(self, char_offset: int, policy: str = "before") -> dict[str, int | bool | str]:
        """Map a textual boundary to a real token boundary with an audit record."""

        if policy not in {"before", "after"}:
            raise ValueError("boundary snap policy must be before or after")
        token_offset = (
            self.token_index_at_or_before(char_offset)
            if policy == "before"
            else self.token_index_at_or_after(char_offset)
        )
        resolved = self.boundary_char(token_offset)
        return {
            "requested_char_offset": int(char_offset),
            "token_offset": int(token_offset),
            "resolved_char_offset": int(resolved),
            "exact": bool(resolved == char_offset),
            "snap_policy": policy,
            "character_displacement": int(resolved - char_offset),
        }

    def assert_partition(self, intervals: Iterable[tuple[int, int]]) -> None:
        """Assert that token intervals cover every saved ID exactly once."""

        values = list(intervals)
        if not values:
            raise ValueError("token partition cannot be empty")
        cursor = 0
        reconstructed: list[int] = []
        for start, end in values:
            if start != cursor or not start <= end <= len(self.input_ids):
                raise ValueError("token partition has a gap, overlap, or invalid offset")
            reconstructed.extend(self.input_ids[start:end])
            cursor = end
        if cursor != len(self.input_ids) or tuple(reconstructed) != self.input_ids:
            raise ValueError("token partition does not reconstruct the saved token IDs")


def tokenize_with_offsets(tokenizer: Any, text: str) -> TokenAlignment:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise TypeError("a fast tokenizer with offset_mapping support is required")
    if ids and isinstance(ids[0], list):
        ids, offsets = ids[0], offsets[0]
    cleaned: list[tuple[int, int]] = []
    previous_end = 0
    overlap_normalized = False
    for raw_start, raw_end in offsets:
        start, end = int(raw_start), int(raw_end)
        # Byte-level fast tokenizers may assign the same Unicode character
        # span to more than one byte-fragment token (for example both pieces
        # of Qwen's ``÷`` can report ``(108, 109)``). Preserve every token ID
        # while assigning later fragments a zero-width span at the character's
        # end. Boundaries before/after the character then remain valid prefixes.
        if start < previous_end:
            start = previous_end
            overlap_normalized = True
        if end < start:
            end = start
            overlap_normalized = True
        cleaned.append((start, end))
        previous_end = end
    method = "retokenized_text_overlap_normalized" if overlap_normalized else "retokenized_text"
    return TokenAlignment(tuple(map(int, ids)), tuple(cleaned), len(text), method)


def _decode(tokenizer: Any, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def _longest_common_prefix(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def align_saved_token_ids(
    tokenizer: Any,
    token_ids: Iterable[int],
    decoded_text: str,
    *,
    piece_cache: dict[int, str] | None = None,
) -> TokenAlignment:
    """Align exact generated IDs directly to their decoded characters.

    No encoded replacement is ever substituted for a saved ID.  The fast path
    decodes each saved token independently.  Byte fragments that cannot be
    decoded independently use saved-prefix decoding and longest stable prefixes;
    this may assign zero-width spans to partial-byte tokens, which is precisely
    why textual boundaries are snapped and audited.
    """

    saved = list(map(int, token_ids))
    full_decoded = _decode(tokenizer, saved)
    if full_decoded != decoded_text:
        raise ValueError("saved token IDs do not decode exactly to the recorded completion")
    cache = piece_cache if piece_cache is not None else {}
    pieces = []
    for token_id in saved:
        if token_id not in cache:
            cache[token_id] = _decode(tokenizer, [token_id])
        pieces.append(cache[token_id])
    if "".join(pieces) == decoded_text:
        offsets = []
        cursor = 0
        for piece in pieces:
            offsets.append((cursor, cursor + len(piece)))
            cursor += len(piece)
        return TokenAlignment(tuple(saved), tuple(offsets), len(decoded_text), "direct_single_token_decode")

    # Fallback for byte-level Unicode sequences whose component tokens decode
    # to replacement characters until the complete byte sequence is present.
    offsets: list[tuple[int, int]] = []
    previous_boundary = 0
    for index in range(len(saved)):
        current = _decode(tokenizer, saved[: index + 1])
        boundary = _longest_common_prefix(current, decoded_text)
        if boundary < previous_boundary:
            raise ValueError("saved-prefix decoding produced a non-monotonic stable character boundary")
        offsets.append((previous_boundary, boundary))
        previous_boundary = boundary
    if previous_boundary != len(decoded_text):
        raise ValueError("saved-prefix alignment did not cover the complete recorded text")
    return TokenAlignment(tuple(saved), tuple(offsets), len(decoded_text), "direct_saved_prefix_decode")


def align_boundaries(tokenizer: Any, text: str, char_offsets: Iterable[int]) -> list[dict[str, int | bool]]:
    alignment = tokenize_with_offsets(tokenizer, text)
    result = []
    for requested in char_offsets:
        result.append(alignment.resolve_boundary(int(requested), "before"))
    return result
