"""Deterministic final-answer extraction with explicit provenance."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class AnswerRegion:
    success: bool
    method: str
    text: str | None
    char_start: int | None
    char_end: int | None
    parsed_answer: Any = None
    failure_reason: str | None = None
    confidence: str = "none"
    rule_priority: int | None = None
    content_char_start: int | None = None
    content_char_end: int | None = None
    token_start: int | None = None
    token_end: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _boxed_regions(text: str) -> list[tuple[int, int, str]]:
    regions: list[tuple[int, int, str]] = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        depth = 1
        cursor = match.end()
        while cursor < len(text) and depth:
            previous = cursor - 1
            slash_count = 0
            while previous >= 0 and text[previous] == "\\":
                slash_count += 1
                previous -= 1
            escaped = bool(slash_count % 2)
            if text[cursor] == "{" and not escaped:
                depth += 1
            elif text[cursor] == "}" and not escaped:
                depth -= 1
            cursor += 1
        if depth == 0:
            regions.append((match.start(), cursor, text[match.end() : cursor - 1].strip()))
    return regions


def _value_region(value: str, absolute_start: int) -> tuple[int, int, str] | None:
    """Select an exact answer substring from an assertive final fragment."""

    leading = len(value) - len(value.lstrip())
    value = value.strip()
    absolute_start += leading
    if not value:
        return None
    # Prefer the last inline/display mathematical expression.  Its exact source
    # substring is preserved; delimiters are excluded only as syntax wrappers.
    math_matches = list(re.finditer(r"(?s)\$\$([^$]+)\$\$|\\\[([^\]]+)\\\]|\$([^$]+)\$|\\\((.+?)\\\)", value))
    if math_matches:
        match = math_matches[-1]
        group_index = next(index for index in range(1, 5) if match.group(index) is not None)
        content = match.group(group_index).strip()
        raw_start = match.start(group_index) + len(match.group(group_index)) - len(match.group(group_index).lstrip())
        raw_end = raw_start + len(content)
        equals = list(re.finditer(r"(?<![<>])=(?!=)", content))
        if equals:
            rhs = content[equals[-1].end() :].strip()
            if rhs:
                rhs_start = raw_start + equals[-1].end() + len(content[equals[-1].end() :]) - len(content[equals[-1].end() :].lstrip())
                return absolute_start + rhs_start, absolute_start + rhs_start + len(rhs), rhs
        return absolute_start + raw_start, absolute_start + raw_end, content
    # Terminal scalar forms cover arithmetic and word-problem answers while
    # avoiding arbitrary prose.  Take the last scalar after an optional equals.
    scalars = list(
        re.finditer(
            r"(?ix)(?:[$£€]\s*)?[-+]?\s*(?:\d[\d,]*(?:\.\d+)?|\.\d+)"
            r"(?:\s*/\s*[-+]?\s*(?:\d[\d,]*(?:\.\d+)?|\.\d+))?%?",
            value,
        )
    )
    if scalars:
        match = scalars[-1]
        selected = match.group(0).strip()
        start = match.start() + len(match.group(0)) - len(match.group(0).lstrip())
        return absolute_start + start, absolute_start + start + len(selected), selected
    symbolic = re.search(r"(?i)\b(?:undefined|true|false|yes|no|[A-E])\b[.!]?\s*$", value)
    if symbolic:
        selected = symbolic.group(0).rstrip(".! ")
        return absolute_start + symbolic.start(), absolute_start + symbolic.start() + len(selected), selected
    return None


def _final_marker(text: str) -> tuple[int, int, int, int, str] | None:
    matches = list(re.finditer(r"(?im)^\s*(?:#{1,6}\s*)?(?:final\s+answer|answer)\s*:?[ \t]*(.*)$", text))
    if not matches:
        return None
    for match in reversed(matches):
        value = match.group(1)
        value_start = match.start(1)
        if not value.strip():
            following = re.search(r"(?m)^\s*(\S[^\n]*)$", text[match.end() :])
            if following:
                value = following.group(1)
                value_start = match.end() + following.start(1)
        selected = _value_region(value, value_start)
        if selected:
            content_start, content_end, content = selected
            return match.start(), max(match.end(), content_end), content_start, content_end, content
    return None


def _isolated_line(text: str) -> tuple[int, int, int, int, str] | None:
    lines = list(re.finditer(r"(?m)^([^\n]+)$", text.rstrip()))
    for match in reversed(lines):
        candidate = match.group(1).strip()
        if not candidate or len(candidate) > 160:
            continue
        wrapper = re.fullmatch(r"(?:\$\$?|\\\[|\\\()?(.+?)(?:\$\$?|\\\]|\\\))?\.?", candidate, re.S)
        inner = wrapper.group(1).strip() if wrapper else candidate
        if re.fullmatch(
            r"(?ix)(?:[-+$£€]?\s*[\d,./%]+|[-+]?\\frac\s*\{[^{}]+\}\s*\{[^{}]+\}"
            r"|[-+]?\\sqrt\s*\{[^{}]+\}|[-+]?\s*[a-z]\s*=\s*[^\n]+"
            r"|\([^\n]{1,120}\)|\[[^\n]{1,120}\]|True|False|yes|no|[A-E])",
            inner,
        ):
            inner_start = match.start(1) + match.group(1).find(inner)
            return match.start(), match.end(), inner_start, inner_start + len(inner), inner
    return None


def _final_assertion(text: str) -> tuple[int, int, int, int, str] | None:
    """Parse only the final nonempty line when it is explicitly assertive."""

    lines = list(re.finditer(r"(?m)^([^\n]+)$", text.rstrip()))
    if not lines:
        return None
    match = lines[-1]
    line = match.group(1)
    if len(line) > 700:
        return None
    assertion = re.search(
        r"(?ix)\b(?:therefore|thus|hence|so|consequently|we\s+(?:can\s+now\s+)?conclude|"
        r"the\s+(?:final\s+)?(?:answer|result|solution|value|expression|remainder)|"
        r"as\s+the\s+answer\s+must\s+be|this\s+implies|there\s+are)\b",
        line,
    )
    if not assertion:
        return None
    selected = _value_region(line[assertion.start() :], match.start(1) + assertion.start())
    if not selected:
        return None
    content_start, content_end, content = selected
    return match.start(), match.end(), content_start, content_end, content


def parse_answer_region(
    text: str,
    dataset_parser: Callable[[str], tuple[Any, tuple[int, int]] | None] | None = None,
) -> AnswerRegion:
    boxed = _boxed_regions(text)
    if boxed:
        start, end, value = boxed[-1]
        content_start = text.find(value, start, end) if value else end - 1
        return AnswerRegion(
            True, "last_boxed", value, start, end, value,
            confidence="high", rule_priority=1,
            content_char_start=content_start, content_char_end=content_start + len(value),
        )
    marked = _final_marker(text)
    if marked:
        start, end, content_start, content_end, value = marked
        return AnswerRegion(
            True, "final_answer_marker", value, start, end, value,
            confidence="high", rule_priority=2,
            content_char_start=content_start, content_char_end=content_end,
        )
    isolated = _isolated_line(text)
    if isolated:
        start, end, content_start, content_end, value = isolated
        return AnswerRegion(
            True, "isolated_answer_line", value, start, end, value,
            confidence="medium", rule_priority=4,
            content_char_start=content_start, content_char_end=content_end,
        )
    if dataset_parser:
        parsed = dataset_parser(text)
        if parsed is not None:
            value, (start, end) = parsed
            return AnswerRegion(
                True, "dataset_specific", text[start:end], start, end, value,
                confidence="high", rule_priority=5, content_char_start=start, content_char_end=end,
            )
    asserted = _final_assertion(text)
    if asserted:
        start, end, content_start, content_end, value = asserted
        return AnswerRegion(
            True, "final_assertion", value, start, end, value,
            confidence="medium", rule_priority=6,
            content_char_start=content_start, content_char_end=content_end,
        )
    return AnswerRegion(
        False, "none", None, None, None,
        failure_reason="no deterministic answer region", confidence="none",
    )


def parse_is_acceptable_for_finish(region: AnswerRegion, finish_reason: str) -> bool:
    """Reject weak answer fallbacks when generation stopped at its length cap.

    A complete box or explicit answer marker is self-delimiting.  An isolated
    equation or an assertive prose fragment at the final token of a capped
    completion is not reliable evidence that the response actually completed.
    """

    if not region.success:
        return False
    if str(finish_reason) != "length":
        return True
    return region.method in {"last_boxed", "final_answer_marker", "dataset_specific"}
