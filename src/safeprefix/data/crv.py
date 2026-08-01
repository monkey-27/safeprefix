"""Adapter for facebook/crv annotated arithmetic, Boolean, and GSM8K files."""

from __future__ import annotations

from typing import Any, Iterable

from safeprefix.manifests import NormalizedTrace


def materialize_annotated_files(
    dataset_id: str,
    *,
    revision: str | None,
    local_root: str,
) -> tuple[list[str], str | None]:
    """Download only public annotated JSON files from the file-backed CRV repo."""
    from huggingface_hub import HfApi, snapshot_download

    root = snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=["arithmetic_expressions/*.annotated.json", "boolean_expressions/*.annotated.json", "gsm8k_expressions/*.annotated.json"],
        local_dir=local_root,
    )
    files = sorted(str(path) for path in __import__("pathlib").Path(root).glob("**/*.annotated.json"))
    resolved = HfApi().dataset_info(dataset_id, revision=revision).sha
    return files, resolved


def _step_parts(row: dict[str, Any]) -> tuple[list[str], list[bool | None]]:
    raw_steps = row.get("step_expressions") or row.get("step_level") or []
    texts: list[str] = []
    labels: list[bool | None] = []
    for step in raw_steps:
        if isinstance(step, str):
            texts.append(step)
            labels.append(None)
        elif isinstance(step, dict):
            texts.append(str(step.get("step_content", step.get("content", ""))))
            value = step.get("step_label")
            labels.append(value if isinstance(value, bool) else None)
        else:
            raise TypeError("CRV step must be text or a mapping")
    return texts, labels


def normalize_row(row: dict[str, Any], subset: str, row_index: int = 0) -> NormalizedTrace:
    problem = row.get("original_expression") or row.get("question") or row.get("problem")
    if not problem:
        raise ValueError("CRV row has no original_expression/question/problem")
    steps, labels = _step_parts(row)
    if not steps:
        raise ValueError("CRV row has no step_expressions/step_level")
    first_error = next((index for index, value in enumerate(labels) if value is False), None)
    correct = row.get("correct_value", row.get("answer"))
    predicted = row.get("predicted_value", row.get("predicted_truth_value"))
    identifier = row.get("expression_id", row.get("id", row_index))
    final_correct = None if correct is None or predicted is None else predicted == correct
    return NormalizedTrace(
        problem_id=f"{subset}:{identifier}",
        source_dataset="facebook/crv",
        source_subset=subset,
        # CRV's annotated files do not consistently carry generation
        # provenance. Preserve absence rather than inferring a model name from
        # the repository documentation or file path.
        source_generator=str(row["source_generator"]) if row.get("source_generator") is not None else None,
        problem_text=str(problem),
        reasoning_steps=steps,
        final_answer_text="" if predicted is None else str(predicted),
        reference_answer=correct,
        first_error_index=first_error,
        index_base="zero" if first_error is not None else None,
        final_answer_correct=final_correct,
        metadata={"step_labels": labels, "raw_total_steps": row.get("total_steps")},
    )


def normalize_rows(rows: Iterable[dict[str, Any]], subset: str) -> tuple[list[NormalizedTrace], list[dict[str, Any]]]:
    accepted: list[NormalizedTrace] = []
    excluded: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        try:
            accepted.append(normalize_row(row, subset, index))
        except Exception as exc:
            excluded.append({"source_dataset": "facebook/crv", "source_subset": subset, "row_index": index, "reason": f"{type(exc).__name__}: {exc}"})
    return accepted, excluded
