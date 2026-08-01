"""Frozen cohort, split, pack, and integrity rules for rollout-only production.

This module deliberately contains no model training or native-evaluation code.
The logical recovery unit is a complete immutable execution pack.  A partial
pack is never accepted as completed because changing the remaining rows would
change the BF16 batch schedule on resume.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import ast
import io
import math
import tokenize
from typing import Any, Iterable, Mapping, Sequence

from safeprefix.manifests import decode_reference_answer
from safeprefix.reproducibility import stable_hash, stable_seed
from safeprefix.scalable import source_bucket


def eligible_source_bucket(row: Mapping[str, Any]) -> str | None:
    bucket = source_bucket(row)
    return bucket if bucket in {
        "crv_arithmetic",
        "processbench_math",
        "processbench_olympiadbench",
        "processbench_omnimath",
    } else None


def first_error_bin(value: int, boundaries: Iterable[int] = (0, 1, 3, 6)) -> str:
    error = int(value)
    edges = sorted(set(map(int, boundaries)))
    if not edges or edges[0] != 0:
        raise ValueError("first-error bins must begin at zero")
    for left, right in zip(edges, edges[1:]):
        if left <= error < right:
            return f"{left}-{right - 1}"
    return f"{edges[-1]}+"


def clean_checkpoint_indices(first_error_zero_based: int, checkpoint_count: int) -> list[int]:
    """Return root plus every state after a complete clean source step."""
    error = int(first_error_zero_based)
    if error < 0 or error >= int(checkpoint_count) - 1:
        raise ValueError("first error is outside dataset-defined checkpoint sequence")
    result = list(range(error + 1))
    if not result or result[0] != 0 or result[-1] != error:
        raise AssertionError("clean checkpoint enumeration is inconsistent")
    return result


def rollout_seed(base_seed: int, trace_id: str, checkpoint_index: int, rollout_index: int) -> int:
    if not 0 <= int(rollout_index) < 4:
        raise ValueError("production suite has exactly four rollout seed indices")
    return stable_seed(int(base_seed), str(trace_id), int(checkpoint_index), int(rollout_index))


def _valid_failure(row: Mapping[str, Any]) -> bool:
    correct = row.get("final_answer_correct")
    error = row.get("first_error_index")
    return correct is not None and not bool(correct) and error is not None and not (
        isinstance(error, float) and math.isnan(error)
    )


def reasoning_steps_list(value: Any) -> list[str]:
    """Normalize list-like steps, including the legacy frozen JSONL encoding.

    The configuration-pilot manifests were atomically serialized from NumPy
    arrays.  Their JSON ``reasoning_steps`` field is consequently NumPy's
    printable array representation (adjacent quoted Python string literals),
    not a JSON list.  Tokenizing those literals individually recovers the exact
    elements; ``ast.literal_eval`` on the whole value would incorrectly join
    adjacent literals into one string.
    """

    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if not isinstance(converted, list):
            raise TypeError("reasoning_steps.tolist() did not return a list")
        return [str(item) for item in converted]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        # Tokenize a bracketed list/array representation before trying to
        # evaluate it as Python.  NumPy prints string arrays as adjacent string
        # literals without commas (``['a'\n 'b']``); Python deliberately
        # concatenates those literals, so ``literal_eval`` would silently turn
        # an N-step trace into one giant step.
        literals: list[str] = []
        if stripped[0] in "[(" and stripped[-1] in ")]":
            try:
                tokens = tokenize.generate_tokens(io.StringIO(stripped).readline)
                for token in tokens:
                    if token.type == tokenize.STRING:
                        item = ast.literal_eval(token.string)
                        if not isinstance(item, str):
                            raise TypeError("reasoning-step literal is not a string")
                        literals.append(item)
            except (SyntaxError, tokenize.TokenError, ValueError) as exc:
                raise ValueError("cannot decode frozen reasoning_steps") from exc
            if literals:
                return literals
        try:
            decoded = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            decoded = None
        if isinstance(decoded, (list, tuple)) and all(
            isinstance(item, str) for item in decoded
        ):
            return list(decoded)
    raise TypeError(f"cannot normalize reasoning_steps from {type(value).__name__}")


def select_frozen_manifest_failures(
    train_rows: Sequence[Mapping[str, Any]],
    dev_rows: Sequence[Mapping[str, Any]],
    *,
    error_bins: Iterable[int] = (0, 1, 3, 6),
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Select every verifier-eligible failure from pre-frozen train/dev roles.

    This function never invents a new split.  The configuration pilot froze
    the underlying problem partition before any downstream output was
    generated, so production preserves those role assignments exactly.
    """

    selected: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    seen_trace_ids: set[str] = set()
    groups_by_split: dict[str, set[str]] = defaultdict(set)
    source_roles: Counter[tuple[str, str]] = Counter()
    for split, rows, expected_role in (
        ("train", train_rows, "teacher_forced_train"),
        ("dev", dev_rows, "teacher_forced_dev"),
    ):
        for raw in rows:
            row = dict(raw)
            source_trace_id = str(row.get("source_trace_id") or "")
            reason = None
            if str(row.get("manifest_role")) != expected_role:
                raise ValueError(
                    f"frozen manifest role drift for {source_trace_id}: "
                    f"{row.get('manifest_role')!r} != {expected_role!r}"
                )
            bucket = eligible_source_bucket(row)
            if bucket is None:
                reason = "outside_designated_sources"
            elif not _valid_failure(row):
                reason = "not_failed_with_visible_error_annotation"
            reference = decode_reference_answer(row.get("reference_answer"))
            if reason is None and reference in (None, "", []):
                reason = "missing_exact_terminal_reference"
            try:
                steps = reasoning_steps_list(row.get("reasoning_steps"))
            except (TypeError, ValueError) as exc:
                steps = []
                if reason is None:
                    reason = f"invalid_reasoning_steps:{type(exc).__name__}"
            error = None
            if reason is None:
                error = int(row["first_error_index"]) - int(
                    row.get("index_base") == "one"
                )
                if not 0 <= error < len(steps):
                    reason = "first_error_outside_reasoning_steps"
            if reason is not None:
                exclusions.append(
                    {
                        "pipeline_split": split,
                        "manifest_role": expected_role,
                        "source_trace_id": source_trace_id,
                        "problem_id": row.get("problem_id"),
                        "source_bucket": bucket or row.get("manifest_bucket"),
                        "reason": reason,
                    }
                )
                continue
            assert bucket is not None and error is not None
            if source_trace_id in seen_trace_ids:
                raise ValueError(f"duplicate frozen source trace ID: {source_trace_id}")
            seen_trace_ids.add(source_trace_id)
            group = str(row.get("problem_group_hash") or row.get("problem_id"))
            groups_by_split[split].add(group)
            source_roles[(split, bucket)] += 1
            normalized = {
                **row,
                "reasoning_steps": steps,
                "source_bucket": bucket,
                "first_error_zero_based": error,
                "first_error_bin": first_error_bin(error, error_bins),
                "rollout_eligible": True,
                "semantic_safety_only": False,
                "pipeline_split": split,
                "production_problem_group": group,
                "trace_id": stable_hash(
                    ["safeprefix-full-tf-trace-v2", source_trace_id]
                )[:24],
            }
            selected.append(normalized)
    overlap = groups_by_split["train"] & groups_by_split["dev"]
    if overlap:
        raise ValueError(
            f"frozen teacher-forced train/dev leakage: {sorted(overlap)[:10]}"
        )
    required_sources = {
        "crv_arithmetic",
        "processbench_math",
        "processbench_olympiadbench",
        "processbench_omnimath",
    }
    present = {row["source_bucket"] for row in selected}
    if present != required_sources:
        raise RuntimeError(
            f"frozen eligible cohort source coverage differs: {sorted(present)}"
        )
    selected.sort(
        key=lambda row: (
            str(row["pipeline_split"]),
            str(row["source_bucket"]),
            str(row["source_trace_id"]),
        )
    )
    checkpoints = sum(int(row["first_error_zero_based"]) + 1 for row in selected)
    exclusion_counts = Counter(row["reason"] for row in exclusions)
    summary = {
        "status": "COMPLETE",
        "cohort_source": "authoritative_frozen_teacher_forced_manifests",
        "eligible_traces": len(selected),
        "unique_problem_groups": len(
            groups_by_split["train"] | groups_by_split["dev"]
        ),
        "split_counts": dict(Counter(row["pipeline_split"] for row in selected)),
        "source_split_counts": {
            f"{split}:{source}": count
            for (split, source), count in sorted(source_roles.items())
        },
        "anticipated_checkpoints_per_model": checkpoints,
        "anticipated_rollouts_per_model": checkpoints * 4,
        "exclusion_counts": dict(exclusion_counts),
        "new_split_created": False,
    }
    return selected, summary, exclusions


def select_and_split_failures(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed: int,
    dev_fraction: float = 0.15,
    error_bins: Iterable[int] = (0, 1, 3, 6),
    mock: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select every eligible failure and split immutable problem groups.

    Groups are stratified by source and a frozen first-error-position bin.
    Duplicate traces from one normalized problem always remain in one split.
    """
    if not 0.0 < float(dev_fraction) < 1.0:
        raise ValueError("development fraction must be in (0, 1)")
    selected: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    for raw in rows:
        row = dict(raw)
        bucket = eligible_source_bucket(row)
        if mock and str(row.get("source_dataset", "")).startswith("mock/"):
            bucket = "mock"
        if bucket is None:
            exclusions["outside_designated_sources"] += 1
            continue
        if not _valid_failure(row):
            exclusions["not_eligible_failed_annotated_trace"] += 1
            continue
        error = int(row["first_error_index"]) - int(row.get("index_base") == "one")
        steps = reasoning_steps_list(row.get("reasoning_steps"))
        if not 0 <= error < len(steps):
            exclusions["first_error_outside_reasoning_steps"] += 1
            continue
        row.update(
            source_bucket=bucket,
            first_error_zero_based=error,
            first_error_bin=first_error_bin(error, error_bins),
            rollout_eligible=True,
            semantic_safety_only=False,
            trace_id=stable_hash(
                [
                    "safeprefix-full-tf-trace-v1",
                    row.get("source_trace_id") or row.get("problem_id"),
                ]
            )[:24],
        )
        selected.append(row)
    if not selected:
        raise RuntimeError("no eligible teacher-forced failures were found")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        groups[str(row.get("problem_group_hash") or row["problem_id"])].append(row)
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    cross_stratum_groups = 0
    for group_id, group_rows in groups.items():
        observed = sorted({(str(row["source_bucket"]), str(row["first_error_bin"])) for row in group_rows})
        if len(observed) > 1:
            cross_stratum_groups += 1
        strata[observed[0]].append(group_id)
    assignments: dict[str, str] = {}
    for stratum, group_ids in sorted(strata.items()):
        ordered = sorted(group_ids, key=lambda value: stable_hash(["safeprefix-full-tf-split-v1", seed, *stratum, value]))
        count = int(round(len(ordered) * float(dev_fraction)))
        if len(ordered) > 1:
            count = min(max(count, 1), len(ordered) - 1)
        else:
            count = 0
        development = set(ordered[:count])
        for group_id in ordered:
            assignments[group_id] = "dev" if group_id in development else "train"
    output = []
    for row in selected:
        group_id = str(row.get("problem_group_hash") or row["problem_id"])
        value = dict(row)
        value["pipeline_split"] = assignments[group_id]
        value["production_problem_group"] = group_id
        output.append(value)
    overlaps: dict[str, set[str]] = defaultdict(set)
    for row in output:
        overlaps[str(row["production_problem_group"])].add(str(row["pipeline_split"]))
    leaking = {key: sorted(value) for key, value in overlaps.items() if len(value) > 1}
    if leaking:
        raise AssertionError(f"problem-level split leakage: {leaking}")
    output.sort(key=lambda row: (str(row["pipeline_split"]), str(row["source_bucket"]), str(row["source_trace_id"])))
    split_counts = Counter(str(row["pipeline_split"]) for row in output)
    source_counts = Counter((str(row["pipeline_split"]), str(row["source_bucket"])) for row in output)
    checkpoints = sum(int(row["first_error_zero_based"]) + 1 for row in output)
    summary = {
        "status": "COMPLETE",
        "eligible_traces": len(output),
        "unique_problem_groups": len(groups),
        "split_counts": dict(split_counts),
        "source_split_counts": {f"{split}:{source}": count for (split, source), count in sorted(source_counts.items())},
        "anticipated_checkpoints_per_model": checkpoints,
        "anticipated_rollouts_per_model": checkpoints * 4,
        "cross_stratum_problem_groups": cross_stratum_groups,
        "exclusion_counts": dict(exclusions),
        "dev_fraction_requested": float(dev_fraction),
        "dev_fraction_observed": split_counts.get("dev", 0) / len(output),
    }
    return output, summary


@dataclass(frozen=True)
class LogicalRolloutKey:
    """One exact production continuation identity."""

    model_key: str
    trace_id: str
    checkpoint_index: int
    rollout_index: int
    rollout_seed: int

    def as_tuple(self) -> tuple[str, str, int, int, int]:
        return (
            self.model_key,
            self.trace_id,
            self.checkpoint_index,
            self.rollout_index,
            self.rollout_seed,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "trace_id": self.trace_id,
            "checkpoint_index": self.checkpoint_index,
            "rollout_index": self.rollout_index,
            "rollout_seed": self.rollout_seed,
        }


def logical_rollout_keys(
    row: Mapping[str, Any], *, model_key: str, base_seed: int
) -> list[LogicalRolloutKey]:
    error = int(row.get("first_error_zero_based", row.get("first_error_span")))
    checkpoint_count = len(reasoning_steps_list(row.get("reasoning_steps"))) + 1
    trace_id = str(row["trace_id"])
    return [
        LogicalRolloutKey(
            model_key=str(model_key),
            trace_id=trace_id,
            checkpoint_index=checkpoint,
            rollout_index=rollout_index,
            rollout_seed=rollout_seed(
                base_seed, trace_id, checkpoint, rollout_index
            ),
        )
        for checkpoint in clean_checkpoint_indices(error, checkpoint_count)
        for rollout_index in range(4)
    ]


def build_execution_packs(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_key: str,
    base_seed: int,
    configuration_hash: str,
    engine_revision: str,
    traces_per_pack: int,
) -> list[dict[str, Any]]:
    """Build deterministic, approximately work-balanced trace packs.

    All four branches for every checkpoint remain in the same pack.  Pack
    membership is independent of input ordering and runtime worker timing.
    """
    if traces_per_pack < 1:
        raise ValueError("traces_per_pack must be positive")
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            -(int(row["first_error_zero_based"]) + 1),
            str(row["source_bucket"]),
            str(row["trace_id"]),
        ),
    )
    if len({str(row["trace_id"]) for row in ordered}) != len(ordered):
        raise ValueError("execution-pack input contains duplicate trace IDs")
    pack_count = max(1, math.ceil(len(ordered) / int(traces_per_pack)))
    bins: list[list[dict[str, Any]]] = [[] for _ in range(pack_count)]
    work = [0 for _ in range(pack_count)]
    for row in ordered:
        eligible_checkpoints = int(row["first_error_zero_based"]) + 1
        row_work = int(
            row.get("estimated_rollout_work", eligible_checkpoints * 4)
        )
        if row_work < eligible_checkpoints * 4:
            raise ValueError("estimated rollout work cannot undercount logical branches")
        available = [index for index, values in enumerate(bins) if len(values) < traces_per_pack]
        target = min(available, key=lambda index: (work[index], len(bins[index]), index))
        bins[target].append(row)
        work[target] += row_work

    packs: list[dict[str, Any]] = []
    for index, members in enumerate(bins):
        members = sorted(members, key=lambda row: str(row["trace_id"]))
        keys = [
            key.to_dict()
            for row in members
            for key in logical_rollout_keys(row, model_key=model_key, base_seed=base_seed)
        ]
        identity = {
            "schema_version": 1,
            "configuration_hash": configuration_hash,
            "engine_revision": engine_revision,
            "model_key": model_key,
            "pack_index": index,
            "trace_ids": [str(row["trace_id"]) for row in members],
            "logical_rollout_keys": keys,
        }
        pack_hash = stable_hash(identity)
        packs.append(
            {
                **identity,
                "pack_id": f"{model_key}-{index:05d}-{pack_hash[:12]}",
                "pack_hash": pack_hash,
                "trace_count": len(members),
                "checkpoint_count": len(keys) // 4,
                "rollout_count": len(keys),
                "estimated_work": work[index],
            }
        )
    observed = [
        (
            key["model_key"], key["trace_id"], int(key["checkpoint_index"]),
            int(key["rollout_index"]), int(key["rollout_seed"]),
        )
        for pack in packs for key in pack["logical_rollout_keys"]
    ]
    if len(observed) != len(set(observed)):
        raise AssertionError("logical rollout key crossed execution packs")
    return packs


def assign_packs_to_workers(
    packs: Sequence[Mapping[str, Any]], worker_count: int
) -> dict[int, list[str]]:
    """Deterministic LPT assignment used for each eight-GPU model wave."""
    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    assigned: dict[int, list[str]] = {index: [] for index in range(worker_count)}
    work = [0 for _ in range(worker_count)]
    for pack in sorted(
        packs,
        key=lambda value: (-int(value["estimated_work"]), str(value["pack_id"])),
    ):
        worker = min(range(worker_count), key=lambda index: (work[index], index))
        assigned[worker].append(str(pack["pack_id"]))
        work[worker] += int(pack["estimated_work"])
    return assigned


def _observed_key(row: Mapping[str, Any]) -> tuple[str, str, int, int, int]:
    return (
        str(row["model_key"]),
        str(row["trace_id"]),
        int(row["checkpoint_index"]),
        int(row["rollout_index"]),
        int(row["rollout_seed"]),
    )


def validate_pack_rows(
    pack: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    trace_rows: Mapping[str, Mapping[str, Any]] | None = None,
    expected_freeze_digest: str | None = None,
    expected_settings_hash: str | None = None,
    expected_parser_sha256: str | None = None,
    expected_verifier_sha256: str | None = None,
) -> dict[str, Any]:
    """Require exact-once completion of every logical row in one pack."""
    expected = {
        (
            str(key["model_key"]), str(key["trace_id"]),
            int(key["checkpoint_index"]), int(key["rollout_index"]),
            int(key["rollout_seed"]),
        )
        for key in pack["logical_rollout_keys"]
    }
    observed_list = [_observed_key(row) for row in rows]
    observed = set(observed_list)
    duplicates = len(observed_list) - len(observed)
    missing = expected - observed
    unexpected = observed - expected
    wrong_pack = [row for row in rows if str(row.get("pack_id")) != str(pack["pack_id"])]
    wrong_hash = [row for row in rows if str(row.get("pack_hash")) != str(pack["pack_hash"])]
    required = {
        "generated_token_ids", "generated_token_count", "generated_text",
        "stop_reason", "truncation_flag", "parser_status", "parser_method",
        "verifier_pass", "binary_outcome", "checkpoint_token_offset",
        "first_visible_error_zero_based", "configuration_hash", "engine_revision",
    }
    schema_failures = [index for index, row in enumerate(rows) if required - set(row)]
    pack_membership_failures: list[str] = []
    if trace_rows is not None:
        missing_pack_traces = [
            str(trace_id)
            for trace_id in pack["trace_ids"]
            if str(trace_id) not in trace_rows
        ]
        if missing_pack_traces:
            pack_membership_failures.append("unknown_pack_trace")
        authoritative_membership = {
            (
                str(pack["model_key"]),
                str(trace_id),
                checkpoint_index,
                rollout_index,
            )
            for trace_id in map(str, pack["trace_ids"])
            for checkpoint_index in range(
                len(trace_rows.get(trace_id, {}).get("eligible_checkpoint_offsets", []))
            )
            for rollout_index in range(4)
        }
        frozen_membership = {
            (
                str(key["model_key"]),
                str(key["trace_id"]),
                int(key["checkpoint_index"]),
                int(key["rollout_index"]),
            )
            for key in pack["logical_rollout_keys"]
        }
        if authoritative_membership != frozen_membership:
            pack_membership_failures.append("authoritative_checkpoint_coverage")
    semantic_failures: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        reasons: list[str] = []
        if str(row.get("configuration_hash")) != str(pack["configuration_hash"]):
            reasons.append("configuration_hash")
        if str(row.get("engine_revision")) != str(pack["engine_revision"]):
            reasons.append("engine_revision")
        token_ids = row.get("generated_token_ids")
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if not isinstance(token_ids, (list, tuple)) or int(
            row.get("generated_token_count", -1)
        ) != (len(token_ids) if isinstance(token_ids, (list, tuple)) else -1):
            reasons.append("generated_token_count")
        if bool(row.get("binary_outcome")) != bool(row.get("verifier_pass")):
            reasons.append("binary_outcome")
        truncation = str(row.get("stop_reason")) == "length"
        if bool(row.get("truncation_flag")) != truncation:
            reasons.append("truncation_flag")
        if (truncation or str(row.get("parser_status")) != "success") and bool(
            row.get("binary_outcome")
        ):
            reasons.append("invalid_success")
        if expected_freeze_digest is not None and str(
            row.get("execution_freeze_digest")
        ) != str(expected_freeze_digest):
            reasons.append("execution_freeze_digest")
        if expected_settings_hash is not None and str(
            row.get("model_execution_settings_hash")
        ) != str(expected_settings_hash):
            reasons.append("model_execution_settings_hash")
        if expected_parser_sha256 is not None and str(
            row.get("parser_sha256")
        ) != str(expected_parser_sha256):
            reasons.append("parser_sha256")
        if expected_verifier_sha256 is not None and str(
            row.get("verifier_sha256")
        ) != str(expected_verifier_sha256):
            reasons.append("verifier_sha256")
        if trace_rows is not None:
            trace = trace_rows.get(str(row.get("trace_id")))
            if trace is None:
                reasons.append("unknown_trace")
            else:
                checkpoint = int(row.get("checkpoint_index", -1))
                offsets = list(map(int, trace["eligible_checkpoint_offsets"]))
                if not 0 <= checkpoint < len(offsets):
                    reasons.append("checkpoint_index")
                elif int(row.get("checkpoint_token_offset", -1)) != offsets[checkpoint]:
                    reasons.append("checkpoint_token_offset")
                expected_fields = {
                    "problem_id": trace["problem_id"],
                    "source_bucket": trace["source_bucket"],
                    "source_trace_id": trace["source_trace_id"],
                    "pipeline_split": trace["pipeline_split"],
                    "first_visible_error_zero_based": trace["first_error_zero_based"],
                }
                for field, expected_value in expected_fields.items():
                    if str(row.get(field)) != str(expected_value):
                        reasons.append(field)
        if reasons:
            semantic_failures.append({"row": index, "reasons": sorted(set(reasons))})
    passed = not (
        duplicates
        or missing
        or unexpected
        or wrong_pack
        or wrong_hash
        or schema_failures
        or pack_membership_failures
        or semantic_failures
    )
    return {
        "passed": passed,
        "expected_rows": len(expected),
        "observed_rows": len(observed_list),
        "duplicate_rows": duplicates,
        "missing_rows": len(missing),
        "unexpected_rows": len(unexpected),
        "wrong_pack_rows": len(wrong_pack),
        "wrong_pack_hash_rows": len(wrong_hash),
        "schema_failure_rows": schema_failures,
        "pack_membership_failures": pack_membership_failures,
        "semantic_failure_rows": semantic_failures,
    }


def aggregate_checkpoint_outcomes(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate four raw Bernoulli outcomes without deriving another label."""
    groups: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["model_key"]), str(row["trace_id"]), int(row["checkpoint_index"]))].append(row)
    output: list[dict[str, Any]] = []
    for (model_key, trace_id, checkpoint), values in sorted(groups.items()):
        values = sorted(values, key=lambda row: int(row["rollout_index"]))
        indices = [int(row["rollout_index"]) for row in values]
        if indices != [0, 1, 2, 3]:
            raise AssertionError(
                f"checkpoint {model_key}/{trace_id}/{checkpoint} does not contain rollout indices 0..3"
            )
        outcomes = [bool(row["binary_outcome"]) for row in values]
        output.append(
            {
                "model_key": model_key,
                "trace_id": trace_id,
                "checkpoint_index": checkpoint,
                "problem_id": values[0]["problem_id"],
                "source_bucket": values[0]["source_bucket"],
                "checkpoint_token_offset": int(values[0]["checkpoint_token_offset"]),
                "first_visible_error_zero_based": int(values[0]["first_visible_error_zero_based"]),
                "rollout_seeds": [int(row["rollout_seed"]) for row in values],
                "binary_outcomes": outcomes,
                "success_count": sum(outcomes),
                "trial_count": 4,
                "repairability_discretized": False,
            }
        )
    return output


def worker_shard(source_trace_id: str, shard_count: int) -> int:
    if int(shard_count) < 1:
        raise ValueError("shard_count must be positive")
    return int(stable_hash(["safeprefix-full-tf-worker-v1", str(source_trace_id)])[:16], 16) % int(shard_count)


def expected_rollout_keys(rows: Iterable[Mapping[str, Any]], *, base_seed: int) -> set[tuple[str, int, int, int]]:
    keys: set[tuple[str, int, int, int]] = set()
    for row in rows:
        trace_id = str(row["trace_id"])
        for checkpoint in clean_checkpoint_indices(int(row["first_error_span"]), len(row["checkpoint_offsets"])):
            for rollout_index in range(4):
                key = (trace_id, checkpoint, rollout_index, rollout_seed(base_seed, trace_id, checkpoint, rollout_index))
                if key in keys:
                    raise AssertionError("duplicate expected rollout key")
                keys.add(key)
    return keys
