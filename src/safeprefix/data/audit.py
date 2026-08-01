"""Dataset audit aggregation and report rendering."""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any, Iterable, Mapping

from safeprefix.manifests import NormalizedTrace

from .dedup import exact_groups, fuzzy_report
from .dedup import latex_whitespace_hash


def _distribution(values: Iterable[int]) -> dict[str, float | int | None]:
    sequence = list(values)
    if not sequence:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(sequence), "min": min(sequence), "median": statistics.median(sequence),
        "mean": statistics.fmean(sequence), "max": max(sequence),
    }


def audit_rows(
    rows: list[NormalizedTrace],
    token_lengths: Mapping[str, Mapping[str, int]] | None = None,
    first_error_token_positions: Mapping[str, Mapping[str, int]] | None = None,
    first_error_token_fractions: Mapping[str, Mapping[str, float]] | None = None,
    evaluation_hashes: set[str] | None = None,
    fuzzy_similarity_threshold: float = 0.92,
) -> dict[str, Any]:
    token_lengths = token_lengths or {}
    first_error_token_positions = first_error_token_positions or {}
    first_error_token_fractions = first_error_token_fractions or {}
    evaluation_hashes = evaluation_hashes or set()
    errors = [row.first_error_zero_based for row in rows if row.first_error_zero_based is not None]
    step_counts = [len(row.reasoning_steps) for row in rows]
    source_counts = Counter((row.source_dataset, row.source_subset) for row in rows)
    generator_counts = Counter(row.source_generator or "unknown" for row in rows)
    correctness = Counter(str(row.final_answer_correct) for row in rows)
    root_only = sum(value == 0 for value in errors)
    by_tokenizer = {
        name: _distribution(lengths.values()) for name, lengths in token_lengths.items()
    }
    return {
        "trace_count": len(rows),
        "unique_problem_count": len({latex_whitespace_hash(row.problem_text) for row in rows}),
        "source_counts": {f"{a}/{b}": n for (a, b), n in source_counts.items()},
        "source_generator_counts": dict(generator_counts),
        "final_answer_correctness": dict(correctness),
        "first_error_distribution": _distribution(errors),
        "clean_steps_before_first_error": _distribution(errors),
        "root_only_percentage": 100.0 * root_only / len(errors) if errors else 0.0,
        "reasoning_step_count_distribution": _distribution(step_counts),
        "estimated_candidate_checkpoint_count": sum(value + 1 for value in errors),
        "token_lengths": by_tokenizer,
        "first_error_token_positions": {name: _distribution(values.values()) for name, values in first_error_token_positions.items()},
        "first_error_token_fractions": {name: _distribution(values.values()) for name, values in first_error_token_fractions.items()},
        "exact_duplicate_groups": exact_groups(rows),
        "near_duplicate_pairs": fuzzy_report(rows, threshold=fuzzy_similarity_threshold),
        "fuzzy_similarity_threshold": fuzzy_similarity_threshold,
        "evaluation_overlap_count": sum(latex_whitespace_hash(row.problem_text) in evaluation_hashes for row in rows),
    }


def render_report(summary: Mapping[str, Any], exclusions: list[Mapping[str, Any]]) -> str:
    lines = [
        "# SafePrefix data audit", "",
        f"- Traces: `{summary['trace_count']}`",
        f"- Unique problems: `{summary['unique_problem_count']}`",
        f"- Root-only failed traces: `{summary['root_only_percentage']:.2f}%`",
        f"- Candidate checkpoints (estimate): `{summary['estimated_candidate_checkpoint_count']}`",
        f"- Exact duplicate groups: `{len(summary['exact_duplicate_groups'])}`",
        f"- Near-duplicate pairs flagged: `{len(summary['near_duplicate_pairs'])}`",
        f"- Evaluation-pool overlaps: `{summary['evaluation_overlap_count']}`",
        f"- Exclusions (never silently dropped): `{len(exclusions)}`", "",
        "## Source counts", "",
    ]
    lines.extend(f"- `{key}`: {value}" for key, value in sorted(summary["source_counts"].items()))
    lines.extend(["", "## Exclusions", ""])
    if exclusions:
        lines.extend(f"- row `{item.get('row_index')}` from `{item.get('source_dataset')}`: {item.get('reason')}" for item in exclusions[:100])
    else:
        lines.append("None.")
    return "\n".join(lines) + "\n"
