"""Deterministic, pass-blind manifests for production cache validation."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from safeprefix.reproducibility import stable_hash


def _domain(row: Mapping[str, Any]) -> str:
    text = " ".join(
        str(row.get(key, "")) for key in ("source_dataset", "source_subset", "problem_text")
    ).casefold()
    if any(marker in text for marker in ("arithmetic", "gsm8k", "calculate", "compute")):
        return "arithmetic"
    return "word_problem"


def _structural_exclusion(row: Mapping[str, Any]) -> str | None:
    if not str(row.get("problem_id", "")).strip():
        return "missing_problem_id"
    if not str(row.get("problem_text", "")).strip():
        return "missing_problem_text"
    steps = row.get("reasoning_steps")
    if hasattr(steps, "tolist"):
        steps = steps.tolist()
    if not isinstance(steps, (list, tuple)) or not any(str(step).strip() for step in steps):
        return "missing_reasoning_steps"
    return None


def build_common_validation_cohort(
    rows: Iterable[Mapping[str, Any]],
    *,
    count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Choose problems before any cache result exists, stratified by domain/length."""
    if count < 1:
        raise ValueError("validation cohort count must be positive")
    exclusions: list[dict[str, Any]] = []
    unique: dict[str, dict[str, Any]] = {}
    for source_index, original in enumerate(rows):
        row = dict(original)
        if hasattr(row.get("reasoning_steps"), "tolist"):
            row["reasoning_steps"] = row["reasoning_steps"].tolist()
        reason = _structural_exclusion(row)
        problem_id = str(row.get("problem_id", f"missing-{source_index}"))
        if reason:
            exclusions.append({"problem_id": problem_id, "reason": reason, "source_index": source_index})
            continue
        normalized = " ".join(str(row["problem_text"]).casefold().split())
        problem_hash = stable_hash(normalized)
        if problem_hash in unique:
            exclusions.append({"problem_id": problem_id, "reason": "duplicate_normalized_problem", "source_index": source_index})
            continue
        row["_problem_hash"] = problem_hash
        row["_domain"] = _domain(row)
        row["_character_length"] = len(str(row["problem_text"])) + sum(len(str(step)) for step in row["reasoning_steps"])
        row["_selection_rank"] = stable_hash(["safeprefix-cache-v3", int(seed), problem_hash])
        unique[problem_hash] = row
    candidates = list(unique.values())
    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} structurally valid unique problems are available; need {count}")
    lengths = sorted(row["_character_length"] for row in candidates)
    first = lengths[len(lengths) // 3]
    second = lengths[(2 * len(lengths)) // 3]
    for row in candidates:
        length = row["_character_length"]
        row["_length_bin"] = "short" if length <= first else "medium" if length <= second else "long"
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in candidates:
        buckets.setdefault((row["_domain"], row["_length_bin"]), []).append(row)
    for values in buckets.values():
        values.sort(key=lambda row: row["_selection_rank"])
    selected: list[dict[str, Any]] = []
    keys = sorted(buckets)
    while len(selected) < count:
        progress = False
        for key in keys:
            if buckets[key] and len(selected) < count:
                selected.append(buckets[key].pop(0))
                progress = True
        if not progress:
            break
    if len(selected) != count:
        raise AssertionError("deterministic stratified selection did not fill the requested cohort")
    result = []
    for index, row in enumerate(selected):
        clean = {key: value for key, value in row.items() if not key.startswith("_")}
        clean.update(
            validation_example_index=index,
            validation_example_id=f"cache-v3-{index:04d}-{row['_problem_hash'][:12]}",
            problem_hash=row["_problem_hash"],
            domain=row["_domain"],
            character_length=row["_character_length"],
            length_bin=row["_length_bin"],
        )
        result.append(clean)
    return result, exclusions


def materialize_model_manifest(
    cohort: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    *,
    model_key: str,
    max_context_length: int,
    continuation_tokens: int,
    multiple_checkpoint_examples: int,
    checkpoint_fractions: Iterable[float],
    min_prefix_tokens: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Tokenize the fixed common cohort and assign exact checkpoint offsets."""
    fractions = [float(value) for value in checkpoint_fractions]
    if fractions != sorted(fractions) or not all(0.0 < value < 1.0 for value in fractions):
        raise ValueError("checkpoint fractions must be ordered and strictly between zero and one")
    records: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for row_index, original in enumerate(cohort):
        row = dict(original)
        prompt_text = str(row["problem_text"]).rstrip() + "\n\n"
        reasoning_text = "\n\n".join(str(step) for step in row["reasoning_steps"] if str(step).strip())
        prompt_ids = list(tokenizer(prompt_text, add_special_tokens=True)["input_ids"])
        reasoning_ids = list(tokenizer(reasoning_text, add_special_tokens=False)["input_ids"])
        token_ids = [int(value) for value in prompt_ids + reasoning_ids]
        reason = None
        if len(prompt_ids) < 1:
            reason = "empty_tokenized_prompt"
        elif len(reasoning_ids) < 1:
            reason = "empty_tokenized_reasoning"
        elif len(token_ids) + continuation_tokens > int(max_context_length):
            reason = "exceeds_context_with_continuation"
        elif len(token_ids) <= min_prefix_tokens:
            reason = "insufficient_prefix_tokens"
        if reason:
            exclusions.append({"model_key": model_key, "validation_example_id": row["validation_example_id"], "problem_id": row["problem_id"], "reason": reason})
            continue
        reasoning_start = len(prompt_ids)
        usable = max(1, len(reasoning_ids) - 1)
        exact_offsets = [max(reasoning_start, min(len(token_ids) - 1, reasoning_start + round(usable * fraction))) for fraction in fractions]
        exact_offsets = sorted(set(int(value) for value in exact_offsets))
        multiple = row_index < int(multiple_checkpoint_examples)
        if multiple:
            checkpoint_offsets = exact_offsets
            checkpoint_kinds = ["early", "middle", "late"][: len(exact_offsets)]
        else:
            choice = row_index % len(exact_offsets)
            checkpoint_offsets = [exact_offsets[choice]]
            checkpoint_kinds = [["early", "middle", "late"][min(choice, 2)]]
        records.append(
            {
                "model_key": model_key,
                "validation_example_id": row["validation_example_id"],
                "problem_id": row["problem_id"],
                "problem_hash": row["problem_hash"],
                "source_dataset": row.get("source_dataset"),
                "source_subset": row.get("source_subset"),
                "domain": row["domain"],
                "length_bin": row["length_bin"],
                "reasoning_span_count": len(row["reasoning_steps"]),
                "prompt_token_count": len(prompt_ids),
                "reasoning_token_count": len(reasoning_ids),
                "sequence_token_count": len(token_ids),
                "token_ids": token_ids,
                "token_ids_sha256": stable_hash(token_ids),
                "checkpoint_offsets": checkpoint_offsets,
                "checkpoint_kinds": checkpoint_kinds,
                "multiple_checkpoint_example": multiple,
                "long_prefix_stress": row["length_bin"] == "long",
            }
        )
    return records, exclusions
