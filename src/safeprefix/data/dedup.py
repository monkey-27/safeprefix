"""Exact and transparent fuzzy problem deduplication."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Iterable

from safeprefix.manifests import NormalizedTrace

from .common import normalize_problem_text


def normalized_exact_hash(text: str) -> str:
    return hashlib.sha256(normalize_problem_text(text).encode()).hexdigest()


def latex_whitespace_hash(text: str) -> str:
    value = re.sub(r"\\(?:left|right|quad|qquad)\b|\\[,;!]", "", text)
    value = value.replace("$", "").replace("\\(", "").replace("\\)", "")
    value = re.sub(r"\s+", "", value).casefold()
    return hashlib.sha256(value.encode()).hexdigest()


def exact_groups(rows: Iterable[NormalizedTrace]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        groups[normalized_exact_hash(row.problem_text)].append(row.problem_id)
    return {key: value for key, value in groups.items() if len(value) > 1}


def fuzzy_report(rows: list[NormalizedTrace], threshold: float = 0.92) -> list[dict[str, object]]:
    try:
        from rapidfuzz.fuzz import ratio
    except ImportError:
        from difflib import SequenceMatcher

        ratio = lambda left, right: 100 * SequenceMatcher(None, left, right).ratio()  # noqa: E731
    normalized = [(row.problem_id, normalize_problem_text(row.problem_text)) for row in rows]
    candidates: list[dict[str, object]] = []
    buckets: dict[tuple[int, str], list[tuple[str, str]]] = defaultdict(list)
    for item in normalized:
        text = item[1]
        buckets[(len(text) // 80, text[:24])].append(item)
    for bucket in buckets.values():
        for index, (left_id, left) in enumerate(bucket):
            for right_id, right in bucket[index + 1 :]:
                score = float(ratio(left, right)) / 100.0
                if score >= threshold:
                    candidates.append({"left_problem_id": left_id, "right_problem_id": right_id, "similarity": score})
    return sorted(candidates, key=lambda row: (-float(row["similarity"]), str(row["left_problem_id"])))
