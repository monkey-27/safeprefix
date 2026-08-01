"""Deterministic gold-answer joins for ProcessBench source problems."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable, Mapping

from safeprefix.data.dedup import latex_whitespace_hash
from safeprefix.manifests import encode_reference_answer
from safeprefix.parsing.answer_parsers import parse_answer_region


def source_problem_key(text: str) -> str:
    return latex_whitespace_hash(str(text))


def extract_reference_answer(source_key: str, row: Mapping[str, Any]) -> Any:
    if source_key == "gsm8k":
        answer = str(row.get("answer", ""))
        match = re.search(r"####\s*([^\n]+)\s*$", answer)
        return match.group(1).strip().replace(",", "") if match else None
    if source_key == "math":
        parsed = parse_answer_region(str(row.get("solution", "")))
        return parsed.parsed_answer if parsed.success else None
    if source_key == "olympiadbench":
        answer = row.get("answer", row.get("final_answer"))
        if isinstance(answer, (list, tuple)):
            return answer[0] if len(answer) == 1 else list(answer)
        return answer
    if source_key == "omnimath":
        return row.get("answer", row.get("final_answer"))
    raise KeyError(source_key)


def source_required_columns(source_key: str) -> tuple[str, ...]:
    return {
        "gsm8k": ("question", "answer"),
        "math": ("problem", "solution"),
        "olympiadbench": ("question", "problem", "answer", "final_answer"),
        "omnimath": ("problem", "question", "answer", "final_answer"),
    }[source_key]


def source_problem_text(source_key: str, row: Mapping[str, Any]) -> str | None:
    fields = {
        "gsm8k": ("question",),
        "math": ("problem",),
        "olympiadbench": ("question", "problem"),
        "omnimath": ("problem", "question"),
    }[source_key]
    for field in fields:
        value = row.get(field)
        if value not in (None, ""):
            return str(value)
    return None


def build_reference_index(source_key: str, rows: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    candidates: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        problem = source_problem_text(source_key, row)
        answer = extract_reference_answer(source_key, row)
        if problem is None or answer in (None, ""):
            continue
        candidates[source_problem_key(problem)].append(answer)
    conflicts = {
        key: values for key, values in candidates.items()
        if len({repr(value) for value in values}) > 1
    }
    index = {
        key: values[0] for key, values in candidates.items()
        if key not in conflicts
    }
    return index, conflicts


def processbench_source_key(subset: str) -> str | None:
    normalized = str(subset).casefold().replace("-", "_")
    if "gsm8k" in normalized:
        return "gsm8k"
    if "olympiad" in normalized:
        return "olympiadbench"
    if "omnimath" in normalized or "omni_math" in normalized:
        return "omnimath"
    if normalized == "math":
        return "math"
    return None


def join_processbench_references(
    rows: Iterable[Mapping[str, Any]],
    indices: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    joined: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "matched": 0})
    unmatched: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        if "processbench" not in str(row.get("source_dataset", "")).casefold():
            joined.append(row)
            continue
        source_key = processbench_source_key(str(row.get("source_subset", "")))
        if source_key is None:
            joined.append(row)
            continue
        counts[source_key]["total"] += 1
        answer = indices.get(source_key, {}).get(source_problem_key(str(row.get("problem_text", ""))))
        if answer is None:
            unmatched.append({
                "problem_id": row.get("problem_id"),
                "source_subset": row.get("source_subset"),
                "problem_text": row.get("problem_text"),
            })
        else:
            row["reference_answer"] = encode_reference_answer(answer)
            row["reference_join_source"] = source_key
            counts[source_key]["matched"] += 1
        joined.append(row)
    report = {
        "by_source": dict(counts),
        "total": sum(value["total"] for value in counts.values()),
        "matched": sum(value["matched"] for value in counts.values()),
        "unmatched_count": len(unmatched),
        "unmatched": unmatched,
        "unmatched_sample": unmatched[:50],
        "method": "normalized_exact_problem_text_hash",
        "fuzzy_assignments": 0,
    }
    return joined, report
