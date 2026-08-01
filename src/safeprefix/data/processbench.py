"""Adapter for the official Qwen/ProcessBench schema."""

from __future__ import annotations

from typing import Any, Iterable

from safeprefix.manifests import NormalizedTrace


def normalize_row(row: dict[str, Any], subset: str) -> NormalizedTrace:
    required = {"id", "problem", "steps", "label", "final_answer_correct"}
    missing = required - row.keys()
    if missing:
        raise ValueError(f"ProcessBench row missing fields: {sorted(missing)}")
    steps = [str(value) for value in row["steps"]]
    label = int(row["label"])
    # ProcessBench uses -1 for a fully correct trace and a zero-based step index otherwise.
    first_error = None if label < 0 else label
    final_answer = row.get("final_answer")
    if final_answer in (None, ""):
        from safeprefix.parsing.answer_parsers import parse_answer_region

        parsed = parse_answer_region("\n\n".join(steps))
        final_answer = parsed.parsed_answer if parsed.success else ""
    return NormalizedTrace(
        problem_id=str(row["id"]),
        source_dataset="Qwen/ProcessBench",
        source_subset=subset,
        source_generator=str(row.get("generator")) if row.get("generator") is not None else None,
        problem_text=str(row["problem"]),
        reasoning_steps=steps,
        final_answer_text=str(final_answer),
        reference_answer=row.get("reference_answer"),
        first_error_index=first_error,
        index_base="zero" if first_error is not None else None,
        final_answer_correct=bool(row["final_answer_correct"]),
        metadata={key: value for key, value in row.items() if key not in required},
    )


def normalize_rows(rows: Iterable[dict[str, Any]], subset: str) -> tuple[list[NormalizedTrace], list[dict[str, Any]]]:
    accepted: list[NormalizedTrace] = []
    excluded: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        try:
            accepted.append(normalize_row(row, subset))
        except Exception as exc:
            excluded.append({"source_dataset": "Qwen/ProcessBench", "source_subset": subset, "row_index": index, "reason": f"{type(exc).__name__}: {exc}"})
    return accepted, excluded
