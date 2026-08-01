"""Shared normalization and dataset loading helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from safeprefix.manifests import NormalizedTrace
from safeprefix.reproducibility import stable_hash


def normalize_problem_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\r\n", "\n")
    return re.sub(r"\s+", " ", text).strip().casefold()


def source_trace_identity(row: NormalizedTrace) -> str:
    """Identify a recorded trace without changing its problem-level group."""
    return stable_hash([
        row.source_dataset, row.source_subset, row.problem_id,
        row.source_generator, row.reasoning_steps, row.final_answer_text,
    ])[:24]


def as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    text = str(value).strip().casefold()
    if text in {"true", "correct", "yes", "1"}:
        return True
    if text in {"false", "incorrect", "no", "0"}:
        return False
    return None


def read_local_rows(path: Path) -> list[dict[str, Any]]:
    import json

    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        for key in ("data", "examples", "rows"):
            if isinstance(value.get(key), list):
                return list(value[key])
        return [value]
    if not isinstance(value, list):
        raise ValueError(f"expected a JSON list or object: {path}")
    return value


def load_hf_rows(dataset_id: str, subset: str | None, split: str, revision: str | None) -> tuple[list[dict[str, Any]], str | None]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, subset, split=split, revision=revision)
    resolved_revision = getattr(dataset, "_fingerprint", None)
    return [dict(row) for row in dataset], resolved_revision


def split_final_answer(steps: list[str], explicit: Any = None) -> tuple[list[str], str]:
    if explicit not in (None, ""):
        return steps, str(explicit)
    if not steps:
        return [], ""
    return steps[:-1], steps[-1]


def validate_unique_ids(rows: Iterable[NormalizedTrace]) -> None:
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row.source_dataset, row.problem_id)
        if key in seen:
            raise ValueError(f"duplicate normalized trace ID: {key}")
        seen.add(key)
