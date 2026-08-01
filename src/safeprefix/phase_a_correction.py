"""Deterministic diagnostics for the SafePrefix Phase-A correction cycle.

This module intentionally does not call the current answer parser when it
classifies the *original* failures.  The resulting taxonomy is therefore a
frozen pre-change diagnostic rather than a retrospective evaluation of parser
rules added later in the correction cycle.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping

from .reproducibility import stable_hash


_FINAL_MARKER = re.compile(r"(?im)^\s*(?:final\s+answer|answer)\s*:\s*(.*)$")
_FINAL_PHRASE = re.compile(
    r"(?is)(?:therefore|thus|hence|so|consequently)?\s*,?\s*"
    r"(?:the\s+)?(?:final\s+)?(?:answer|result|value)\s+is\s*:?[ \t]*([^\n]+)\s*$"
)
_ISOLATED = re.compile(
    r"(?ix)"
    r"(?:[-+$£€]?\s*[\d,]+(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?%?"
    r"|\\?[a-z]+(?:\s*\{[^\n]+\})?"
    r"|\$[^\n$]+\$"
    r"|\([^\n]{1,120}\)"
    r"|true|false|yes|no|[A-E])"
)


def complete_boxed_regions(text: str) -> list[tuple[int, int, str]]:
    """Return syntactically complete ``\\boxed{...}`` regions.

    Escaping is based on the parity of the preceding backslash run, so a brace
    after ``\\\\`` is structural while a brace after ``\\`` is literal.
    """

    regions: list[tuple[int, int, str]] = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        depth = 1
        cursor = match.end()
        while cursor < len(text) and depth:
            char = text[cursor]
            slash_count = 0
            previous = cursor - 1
            while previous >= 0 and text[previous] == "\\":
                slash_count += 1
                previous -= 1
            escaped = bool(slash_count % 2)
            if char == "{" and not escaped:
                depth += 1
            elif char == "}" and not escaped:
                depth -= 1
            cursor += 1
        if depth == 0:
            regions.append((match.start(), cursor, text[match.end() : cursor - 1].strip()))
    return regions


def has_unclosed_box(text: str) -> bool:
    return len(list(re.finditer(r"\\boxed\s*\{", text))) > len(complete_boxed_regions(text))


def _last_nonempty_line(text: str) -> str:
    return next((line.strip() for line in reversed(text.splitlines()) if line.strip()), "")


def deterministic_answer_evidence(text: str) -> dict[str, Any]:
    """Find format evidence without using a reference answer or correctness."""

    boxes = complete_boxed_regions(text)
    markers = list(_FINAL_MARKER.finditer(text))
    last_line = _last_nonempty_line(text)
    phrases = list(_FINAL_PHRASE.finditer(text.rstrip()))
    marker_value = markers[-1].group(1).strip() if markers else ""
    return {
        "complete_box_count": len(boxes),
        "has_complete_box": bool(boxes),
        "has_unclosed_box": has_unclosed_box(text),
        "has_final_answer_marker": bool(markers and marker_value),
        "has_empty_final_answer_marker": bool(markers and not marker_value),
        "has_final_phrase": bool(phrases and phrases[-1].group(1).strip()),
        "has_isolated_answer_line": bool(last_line and len(last_line) <= 180 and _ISOLATED.fullmatch(last_line)),
        "last_nonempty_line": last_line,
    }


def classify_original_answer_failure(row: Mapping[str, Any]) -> dict[str, Any]:
    """Assign the required primary category to an original Phase-A row."""

    text = str(row.get("completion_text", ""))
    evidence = deterministic_answer_evidence(text)
    parser = dict(row.get("answer_parser") or {})
    parser_success = bool(parser.get("success"))
    finish = str(row.get("finish_reason", ""))

    if parser_success and parser.get("method") == "last_boxed":
        category = "complete boxed answer parsed correctly"
        correction_bucket = "not_an_original_parser_failure"
    elif evidence["has_complete_box"]:
        category = "complete boxed answer missed"
        correction_bucket = "A_complete_answer_recoverable"
    elif evidence["has_final_answer_marker"]:
        category = "answer written after Final answer:"
        correction_bucket = "A_complete_answer_recoverable"
    elif re.search(r"(?im)^\s*answer\s*:", text):
        category = "answer written after Answer:"
        correction_bucket = "A_complete_answer_recoverable"
    elif evidence["has_isolated_answer_line"]:
        category = "answer given as final isolated line"
        correction_bucket = "A_complete_answer_recoverable"
    elif evidence["has_final_phrase"]:
        category = "valid answer in another common deterministic format"
        correction_bucket = "A_complete_answer_recoverable"
    elif evidence["has_unclosed_box"]:
        category = "malformed or unclosed box"
        correction_bucket = "D_other_malformed"
    elif finish == "length":
        category = "output-length limit reached"
        correction_bucket = "B_genuinely_ended_before_complete_answer"
    elif finish == "eos":
        category = "response ended at EOS without final answer"
        correction_bucket = "C_complete_without_identifiable_answer"
    else:
        category = "no final answer despite sufficient generation budget"
        correction_bucket = "C_complete_without_identifiable_answer"

    return {
        "answer_primary_category": category,
        "correction_bucket": correction_bucket,
        **evidence,
    }


def audit_identity(row: Mapping[str, Any]) -> str:
    return stable_hash(
        [row.get("problem_id"), row.get("model_name"), row.get("prompt_condition"), row.get("generation_seed")]
    )[:20]


def parser_development_split(rows: Iterable[Mapping[str, Any]], seed: int) -> tuple[list[str], list[str]]:
    """Create a stable, approximately stratified 50/50 parser split."""

    strata: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("model_name")),
            str(row.get("prompt_condition")),
            str(row.get("finish_reason")),
            str(row.get("source_dataset")),
        )
        strata[key].append(audit_identity(row))
    development: list[str] = []
    holdout: list[str] = []
    for key, identifiers in sorted(strata.items()):
        ordered = sorted(identifiers, key=lambda value: stable_hash([seed, "parser-split", key, value]))
        cut = (len(ordered) + 1) // 2
        development.extend(ordered[:cut])
        holdout.extend(ordered[cut:])
    return sorted(development), sorted(holdout)


def task_family(row: Mapping[str, Any]) -> str:
    dataset = str(row.get("source_dataset", "")).casefold()
    subset = str(row.get("source_subset", "")).casefold()
    if "crv" in dataset or "arith" in subset:
        return "synthetic_arithmetic"
    if "gsm8k" in subset:
        return "grade_school"
    if "olympiad" in subset or "omnimath" in subset or "omni-math" in subset:
        return "competition"
    return "math_level"


def completion_length_bin(rows: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    values = sorted((len(row.get("completion_token_ids") or []), audit_identity(row)) for row in rows)
    count = max(len(values), 1)
    return {
        identifier: ("short" if index < count / 3 else "medium" if index < 2 * count / 3 else "long")
        for index, (_, identifier) in enumerate(values)
    }


def select_blinded_audit(rows: list[Mapping[str, Any]], count: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select a deterministic audit using a multi-margin greedy design.

    Exact model and prompt balance are hard constraints. Task family, length,
    and parser status are minimized jointly subject to the available cells.
    """

    if count % 12:
        raise ValueError("audit count must be divisible by 12 for four-model/three-prompt balance")
    lengths = completion_length_bin(rows)
    candidates = []
    for raw in rows:
        row = dict(raw)
        identifier = audit_identity(row)
        row["audit_id"] = identifier
        row["task_family"] = task_family(row)
        row["length_bin"] = lengths[identifier]
        row["parser_status"] = "success" if bool((row.get("answer_parser") or {}).get("success")) else "failure"
        candidates.append(row)

    targets = {
        "model_name": count / 4,
        "prompt_condition": count / 3,
        "task_family": count / 4,
        "length_bin": count / 3,
        # Parser failures are deliberately enriched while still blinded. The
        # attainable target is capped because failures are only about 10%.
        "parser_status:failure": min(count / 4, sum(row["parser_status"] == "failure" for row in candidates)),
    }
    counts: Counter[tuple[str, str]] = Counter()
    selected: list[dict[str, Any]] = []
    remaining = {row["audit_id"]: row for row in candidates}
    while len(selected) < count:
        best: tuple[float, str, dict[str, Any]] | None = None
        for identifier, row in remaining.items():
            if counts[("model_name", row["model_name"])] >= targets["model_name"]:
                continue
            if counts[("prompt_condition", row["prompt_condition"])] >= targets["prompt_condition"]:
                continue
            score = 0.0
            for dimension in ("model_name", "prompt_condition", "task_family", "length_bin"):
                value = str(row[dimension])
                target = float(targets[dimension])
                before = abs(counts[(dimension, value)] - target)
                after = abs(counts[(dimension, value)] + 1 - target)
                score += before - after
            if row["parser_status"] == "failure":
                target = float(targets["parser_status:failure"])
                before = abs(counts[("parser_status", "failure")] - target)
                after = abs(counts[("parser_status", "failure")] + 1 - target)
                score += before - after
            tie = stable_hash([seed, "blinded-audit", identifier])
            candidate = (score, tie, row)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        if best is None:
            raise RuntimeError("could not satisfy exact model/prompt audit margins")
        row = best[2]
        selected.append(row)
        del remaining[row["audit_id"]]
        for dimension in ("model_name", "prompt_condition", "task_family", "length_bin", "parser_status"):
            counts[(dimension, str(row[dimension]))] += 1

    annotation_rows = []
    manifest_rows = []
    for row in selected:
        annotation_rows.append(
            {
                "audit_id": row["audit_id"],
                "trace_text": row["completion_text"],
                "complete_answer_present": "",
                "mathematically_interpretable": "",
                "reasoning_answer_boundary_valid": "",
                "semantic_segmentation_valid": "",
                "genuinely_truncated": "",
                "parser_missed_existing_answer": "",
                "malformed_answer_marker": "",
                "no_identifiable_answer": "",
                "prompt_noncompliance": "",
                "complete_but_unverifiable_formatting": "",
                "primary_failure_category": "",
                "notes": "",
                "human_confirmed": "",
            }
        )
        manifest_rows.append(
            {
                "audit_id": row["audit_id"],
                "problem_id": row["problem_id"],
                "model_name": row["model_name"],
                "prompt_condition": row["prompt_condition"],
                "task_family": row["task_family"],
                "length_bin": row["length_bin"],
                "parser_status": row["parser_status"],
                "finish_reason": row["finish_reason"],
            }
        )
    return annotation_rows, manifest_rows


def margin_counts(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[key]) for row in rows).items()))
