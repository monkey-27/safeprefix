"""Deterministic terminal verifiers; no explanatory feedback reaches the model."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Protocol

from safeprefix.manifests import decode_reference_answer
from safeprefix.parsing.answer_parsers import parse_answer_region


class Verifier(Protocol):
    def __call__(self, parsed_answer: Any, reference_answer: Any, metadata: dict[str, Any]) -> bool: ...


def normalize_scalar(value: Any) -> str:
    return re.sub(r"\s+", "", str(value)).replace(",", "").strip().casefold()


def numeric_value(value: Any) -> Decimal | None:
    text = normalize_scalar(value).replace("$", "").replace("£", "").replace("€", "")
    percent = text.endswith("%")
    if percent:
        text = text[:-1]
    try:
        result = Decimal(text)
        return result / 100 if percent else result
    except (InvalidOperation, ValueError):
        return None


@dataclass(frozen=True)
class ExactAnswerVerifier:
    absolute_tolerance: float = 1e-6
    relative_tolerance: float = 1e-6

    def __call__(self, parsed_answer: Any, reference_answer: Any, metadata: dict[str, Any] | None = None) -> bool:
        reference_answer = decode_reference_answer(reference_answer)
        left, right = numeric_value(parsed_answer), numeric_value(reference_answer)
        if left is not None and right is not None:
            return math.isclose(float(left), float(right), abs_tol=self.absolute_tolerance, rel_tol=self.relative_tolerance)
        return normalize_scalar(parsed_answer) == normalize_scalar(reference_answer)


def parse_and_verify(text: str, reference_answer: Any, verifier: Verifier) -> tuple[Any, bool, str]:
    region = parse_answer_region(text)
    if not region.success:
        return None, False, "parse_failure"
    return region.parsed_answer, bool(verifier(region.parsed_answer, reference_answer, {})), region.method


def resolve_verifier(name: str, config: dict[str, Any] | None = None) -> Verifier:
    config = config or {}
    if name == "exact_answer":
        return ExactAnswerVerifier(float(config.get("absolute_tolerance", 1e-6)), float(config.get("relative_tolerance", 1e-6)))
    raise KeyError(f"unknown verifier: {name}")
