"""Frozen native-failure acquisition and full-regeneration bookkeeping.

This module intentionally has no reasoning segmentation, checkpoint feature,
KV-persistence, or boundary-model dependency.  Cohort membership is determined
only by the first valid, parseable, verifier-incorrect native response in a
frozen source order.  Four fresh prompt-root generations are recorded later.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from safeprefix.data.common import normalize_problem_text
from safeprefix.data.dedup import latex_whitespace_hash, normalized_exact_hash
from safeprefix.parsing.answer_parsers import parse_answer_region
from safeprefix.reproducibility import atomic_json, atomic_jsonl, stable_hash, stable_seed


STRATA = ("gsm1k", "math_level_3", "math_level_4")
VALID_INITIAL_FAILURE = "valid_incorrect"
INITIAL_REQUIRED = {
    "attempt_key", "model_key", "source_id", "stratum", "source_order_rank",
    "initial_generation_seed", "raw_initial_response", "parser_status",
    "verifier_status", "initial_status", "configuration_hash",
}
REGEN_REQUIRED = {
    "rollout_key", "model_key", "trace_id", "rollout_index", "rollout_seed",
    "raw_regenerated_response", "parser_status", "verifier_status",
    "binary_verifier_outcome", "configuration_hash",
}


@dataclass(frozen=True)
class SourceProblem:
    source_id: str
    dataset_id: str
    dataset_revision: str
    source_config: str
    source_split: str
    stratum: str
    difficulty_level: int | None
    problem_text: str
    gold_answer: Any
    gold_derivation: str
    normalized_problem_hash: str
    latex_whitespace_hash: str
    source_order_key: str
    source_order_rank: int = -1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AttemptPack:
    pack_id: str
    model_key: str
    stratum: str
    pack_index: int
    source_ids: tuple[str, ...]
    configuration_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "source_ids": list(self.source_ids)}


@dataclass(frozen=True)
class RegenerationPack:
    pack_id: str
    model_key: str
    pack_index: int
    trace_ids: tuple[str, ...]
    configuration_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "trace_ids": list(self.trace_ids)}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        return []
    return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]


def _math_gold(solution: str) -> tuple[Any | None, str]:
    parsed = parse_answer_region(solution)
    if parsed.success and parsed.method == "last_boxed":
        return parsed.parsed_answer, "official_solution_last_boxed"
    return None, "official_solution_missing_complete_boxed_answer"


def load_public_source_rows(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load only GSM1K and official MATH test levels 3/4 at pinned revisions."""

    from datasets import load_dataset

    rows: list[dict[str, Any]] = []
    raw_counts: Counter[str] = Counter()
    revisions: dict[str, Any] = {}
    gsm = config["sources"]["gsm1k"]
    gsm_data = load_dataset(
        str(gsm["hf_dataset_id"]), str(gsm["config"]), split=str(gsm["split"]),
        revision=str(gsm["revision"]),
    )
    revisions["gsm1k"] = {
        "dataset_id": gsm["hf_dataset_id"], "revision": gsm["revision"],
        "config": gsm["config"], "split": gsm["split"],
    }
    for index, raw in enumerate(gsm_data):
        stratum = "gsm1k"
        raw_counts[stratum] += 1
        text = str(raw.get(str(gsm["question_field"]), ""))
        answer = raw.get(str(gsm["answer_field"]))
        rows.append({
            "source_id": f"ScaleAI/gsm1k:test:{index}",
            "dataset_id": str(gsm["hf_dataset_id"]),
            "dataset_revision": str(gsm["revision"]),
            "source_config": str(gsm["config"]),
            "source_split": str(gsm["split"]),
            "stratum": stratum,
            "difficulty_level": None,
            "problem_text": text,
            "gold_answer": answer,
            "gold_derivation": "dataset_answer_field",
        })

    math_spec = config["sources"]["math"]
    revisions["math"] = {
        "dataset_id": math_spec["hf_dataset_id"], "revision": math_spec["revision"],
        "configs": list(math_spec["configs"]), "split": math_spec["split"],
    }
    retained = set(map(int, math_spec["retained_levels"]))
    for subset in math_spec["configs"]:
        dataset = load_dataset(
            str(math_spec["hf_dataset_id"]), str(subset),
            split=str(math_spec["split"]), revision=str(math_spec["revision"]),
        )
        for index, raw in enumerate(dataset):
            match = re.fullmatch(r"Level\s+([1-5])", str(raw.get(math_spec["level_field"], "")))
            if not match or int(match.group(1)) not in retained:
                continue
            level = int(match.group(1))
            stratum = f"math_level_{level}"
            raw_counts[stratum] += 1
            solution = str(raw.get(str(math_spec["solution_field"]), ""))
            gold, derivation = _math_gold(solution)
            rows.append({
                "source_id": f"EleutherAI/hendrycks_math:test:{subset}:{index}",
                "dataset_id": str(math_spec["hf_dataset_id"]),
                "dataset_revision": str(math_spec["revision"]),
                "source_config": str(subset),
                "source_split": str(math_spec["split"]),
                "stratum": stratum,
                "difficulty_level": level,
                "problem_text": str(raw.get(str(math_spec["problem_field"]), "")),
                "gold_answer": gold,
                "gold_derivation": derivation,
            })
    return rows, {"raw_counts": dict(raw_counts), "dataset_revisions": revisions}


def _load_protected_rows(specification: Mapping[str, Any], repo_root: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(repo_root)
    output: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {}
    # The enriched table also crosswalks identity-only reservation manifests.
    group_to_text: dict[str, str] = {}
    for item in specification["protected_sources"]:
        path = Path(str(item["path"]))
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise FileNotFoundError(f"missing protected source: {path}")
        if str(item["format"]) == "parquet":
            values = pd.read_parquet(path).to_dict("records")
        elif str(item["format"]) == "jsonl":
            values = read_jsonl(path)
        else:
            raise ValueError(f"unsupported protected source format: {item['format']}")
        for row in values:
            if row.get("problem_group_hash") and row.get("problem_text"):
                group_to_text[str(row["problem_group_hash"])] = str(row["problem_text"])
        output.extend({"role": str(item["role"]), **row} for row in values)
        provenance[str(item["role"])] = {
            "path": str(path), "sha256": file_sha256(path), "rows": len(values),
        }
    for row in output:
        if not row.get("problem_text") and row.get("problem_group_hash") in group_to_text:
            row["problem_text"] = group_to_text[str(row["problem_group_hash"])]
    for relative in specification.get("previous_native_smoke_attempts", []):
        path = root / str(relative)
        if not path.is_file():
            raise FileNotFoundError(f"missing previous native smoke artifact: {path}")
        values = read_jsonl(path)
        for row in values:
            if not row.get("problem_text") and row.get("problem_group_hash") in group_to_text:
                row["problem_text"] = group_to_text[str(row["problem_group_hash"])]
        output.extend({"role": "previous_native_smoke", **row} for row in values)
        provenance[f"previous_native_smoke:{path.name}:{path.parent.name}"] = {
            "path": str(path), "sha256": file_sha256(path), "rows": len(values),
        }
    return output, provenance


def _fuzzy_match(query: str, choices: Sequence[str], threshold: float) -> tuple[int, float] | None:
    if not query or not choices:
        return None
    from rapidfuzz import fuzz, process

    match = process.extractOne(query, choices, scorer=fuzz.ratio, score_cutoff=100.0 * threshold)
    if match is None:
        return None
    _value, score, index = match
    return int(index), float(score) / 100.0


def build_source_census(config: Mapping[str, Any], *, repo_root: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    raw_rows, source_info = load_public_source_rows(config)
    census = config["source_census"]
    protected, protected_provenance = _load_protected_rows(census, repo_root)
    protected_text_rows = [row for row in protected if str(row.get("problem_text", "")).strip()]
    protected_norms = list(dict.fromkeys(normalize_problem_text(str(row["problem_text"])) for row in protected_text_rows))
    norm_roles: dict[str, set[str]] = defaultdict(set)
    id_roles: dict[str, set[str]] = defaultdict(set)
    latex_roles: dict[str, set[str]] = defaultdict(set)
    for row in protected_text_rows:
        role = str(row.get("role", "unknown"))
        norm_roles[normalize_problem_text(str(row["problem_text"]))].add(role)
        latex_roles[latex_whitespace_hash(str(row["problem_text"]))].add(role)
        identity = row.get("source_id") or row.get("problem_id")
        if identity:
            id_roles[str(identity)].add(role)
    protected_exact = set(protected_norms)
    protected_latex = {latex_whitespace_hash(str(row["problem_text"])) for row in protected_text_rows}
    protected_ids = {str(row.get("source_id") or row.get("problem_id")) for row in protected if row.get("source_id") or row.get("problem_id")}
    image_patterns = [re.compile(str(value), re.IGNORECASE) for value in census["image_dependency_patterns"]]
    threshold = float(census["near_duplicate_threshold"])
    exclusions: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    accepted_norms: list[str] = []
    accepted_latex: set[str] = set()
    accepted_ids: set[str] = set()

    for raw in raw_rows:
        row = dict(raw)
        source_id = str(row["source_id"])
        text = str(row.get("problem_text", ""))
        normalized = normalize_problem_text(text)
        latex_hash = latex_whitespace_hash(text) if text.strip() else ""
        reason: str | None = None
        details: dict[str, Any] = {}
        if not text.strip():
            reason = "corrupted_or_empty_problem"
        elif row.get("gold_answer") in (None, "", []):
            reason = "missing_deterministic_gold_answer"
        elif any(pattern.search(text) for pattern in image_patterns):
            reason = "unavailable_image_or_diagram_dependency"
        elif source_id in protected_ids:
            reason = "protected_canonical_id_overlap"
            details["protected_roles"] = sorted(id_roles.get(source_id, {"unknown"}))
        elif normalized in protected_exact or latex_hash in protected_latex:
            reason = "protected_normalized_text_overlap"
            details["protected_roles"] = sorted(
                norm_roles.get(normalized, set()) | latex_roles.get(latex_hash, set())
                or {"latex_normalized_match"}
            )
        elif source_id in accepted_ids or normalized in accepted_norms or latex_hash in accepted_latex:
            reason = "duplicate_within_or_across_source_reservoir"
        else:
            fuzzy = _fuzzy_match(normalized, protected_norms, threshold)
            if fuzzy is not None:
                index, similarity = fuzzy
                reason = "protected_near_duplicate_overlap"
                matched = protected_norms[index]
                details.update(
                    similarity=similarity,
                    matched_protected_text=matched,
                    protected_roles=sorted(norm_roles.get(matched, {"unknown"})),
                )
            else:
                within = _fuzzy_match(normalized, accepted_norms, threshold)
                if within is not None:
                    index, similarity = within
                    reason = "near_duplicate_within_or_across_source_reservoir"
                    details.update(similarity=similarity, matched_source_text=accepted_norms[index])
        if reason:
            exclusions.append({
                "source_id": source_id, "stratum": row["stratum"],
                "exclusion_reason": reason, **details,
            })
            continue
        order_key = stable_hash([
            "native-source-order-v1", int(census["source_order_seed"]),
            row["stratum"], source_id, normalized_exact_hash(text),
        ])
        row.update(
            normalized_problem_hash=normalized_exact_hash(text),
            latex_whitespace_hash=latex_hash,
            source_order_key=order_key,
        )
        eligible.append(row)
        accepted_ids.add(source_id)
        accepted_norms.append(normalized)
        accepted_latex.add(latex_hash)

    ordered: list[dict[str, Any]] = []
    for stratum in STRATA:
        values = sorted(
            (row for row in eligible if row["stratum"] == stratum),
            key=lambda row: (row["source_order_key"], row["source_id"]),
        )
        for rank, row in enumerate(values):
            ordered.append({**row, "source_order_rank": rank})
    summary = {
        **source_info,
        "eligible_counts": dict(Counter(str(row["stratum"]) for row in ordered)),
        "exclusion_counts": dict(Counter(str(row["exclusion_reason"]) for row in exclusions)),
        "exclusion_counts_by_stratum": {
            stratum: dict(Counter(str(row["exclusion_reason"]) for row in exclusions if row["stratum"] == stratum))
            for stratum in STRATA
        },
        "protected_source_provenance": protected_provenance,
        "overlap_exclusion_counts_by_protected_role": dict(Counter(
            role
            for row in exclusions
            for role in row.get("protected_roles", [])
        )),
        "protected_text_count": len(protected_norms),
        "near_duplicate_threshold": threshold,
        "source_order_seed": int(census["source_order_seed"]),
    }
    return ordered, exclusions, summary


def write_source_census(config: Mapping[str, Any], *, repo_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    rows, exclusions, summary = build_source_census(config, repo_root=repo_root)
    root.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(root / "eligible_source_manifest.jsonl", rows)
    atomic_jsonl(root / "source_exclusions.jsonl", exclusions)
    manifest_sha = file_sha256(root / "eligible_source_manifest.jsonl")
    metadata = {
        "schema_version": 1,
        "status": "FROZEN_SOURCE_CENSUS",
        "manifest_sha256": manifest_sha,
        "manifest_rows": len(rows),
        **summary,
        "capacity_assessment": capacity_assessment(summary["eligible_counts"]),
    }
    atomic_json(root / "source_manifest_metadata.json", metadata)
    atomic_json(root / "source_census.json", metadata)
    report = [
        "# SafePrefix native failed-trace source census", "",
        "Status: **FROZEN — NO MODEL INFERENCE SUBMITTED**", "",
        "Only `ScaleAI/gsm1k` test and the official `EleutherAI/hendrycks_math` test levels 3 and 4 are included.", "",
        "| Stratum | Raw | Eligible |", "| --- | ---: | ---: |",
    ]
    for stratum in STRATA:
        report.append(f"| {stratum} | {summary['raw_counts'].get(stratum, 0)} | {summary['eligible_counts'].get(stratum, 0)} |")
    report.extend(["", "## Exclusions", ""])
    report.extend(f"- `{reason}`: {count}" for reason, count in sorted(summary["exclusion_counts"].items()))
    report.extend(["", "## Protected-manifest overlap audit", ""])
    for role, provenance in sorted(summary["protected_source_provenance"].items()):
        report.append(
            f"- `{role}`: source rows `{provenance['rows']}`, excluded source matches "
            f"`{summary['overlap_exclusion_counts_by_protected_role'].get(role.split(':')[0], 0)}`, "
            f"SHA-256 `{provenance['sha256']}`."
        )
    report.extend([
        "", "## Capacity", "",
        f"- Assessment: `{metadata['capacity_assessment']['status']}`",
        f"- GSM1K eligible: `{summary['eligible_counts'].get('gsm1k', 0)}` versus planning range `625–1250`.",
        f"- MATH level 3 eligible: `{summary['eligible_counts'].get('math_level_3', 0)}` versus planning range `800–1334`.",
        f"- MATH level 4 eligible: `{summary['eligible_counts'].get('math_level_4', 0)}` versus planning range `750–1200`.",
        (
            "- At the supplied failure-rate endpoints, these reservoirs imply approximately "
            f"`{metadata['capacity_assessment']['expected_failures_from_given_planning_ranges']['lower_rate_endpoint']:.1f}` "
            "to "
            f"`{metadata['capacity_assessment']['expected_failures_from_given_planning_ranges']['upper_rate_endpoint']:.1f}` "
            "failures for a 600-failure target. Source exhaustion is therefore possible, not a launch-time error."
        ),
        "", "The full reservoirs are traversed in immutable hash order. Quotas are not enforced by truncating this source manifest.",
    ])
    (root / "SOURCE_CENSUS_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return metadata


def capacity_assessment(eligible: Mapping[str, int]) -> dict[str, Any]:
    planning = {"gsm1k": (625, 1250), "math_level_3": (800, 1334), "math_level_4": (750, 1200)}
    failure_rate_ranges = {
        "gsm1k": (0.08, 0.16),
        "math_level_3": (0.15, 0.25),
        "math_level_4": (0.25, 0.40),
    }
    details = {
        key: {
            "eligible": int(eligible.get(key, 0)), "planning_low": low,
            "planning_high": high, "covers_low_failure_requirement": int(eligible.get(key, 0)) >= high,
            "covers_high_failure_requirement": int(eligible.get(key, 0)) >= low,
        }
        for key, (low, high) in planning.items()
    }
    expected_low = sum(
        int(eligible.get(key, 0)) * failure_rate_ranges[key][0] for key in STRATA
    )
    expected_high = sum(
        int(eligible.get(key, 0)) * failure_rate_ranges[key][1] for key in STRATA
    )
    return {
        "status": "PLAUSIBLY_SUFFICIENT_WITH_FROZEN_DEFICIT_TRANSFER"
        if all(row["covers_high_failure_requirement"] for row in details.values())
        else "POTENTIAL_SOURCE_EXHAUSTION",
        "by_stratum": details,
        "expected_failures_from_given_planning_ranges": {
            "lower_rate_endpoint": expected_low,
            "upper_rate_endpoint": expected_high,
            "target": 600,
            "interpretation": (
                "upper endpoint reaches target but lower endpoint does not"
                if expected_low < 600 <= expected_high
                else "both endpoints reach target" if expected_low >= 600
                else "neither endpoint reaches target"
            ),
        },
    }


def load_source_manifest(path: str | Path, metadata_path: str | Path) -> list[dict[str, Any]]:
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if file_sha256(path) != str(metadata["manifest_sha256"]):
        raise RuntimeError("eligible source manifest hash mismatch")
    rows = read_jsonl(path)
    if len(rows) != int(metadata["manifest_rows"]):
        raise RuntimeError("eligible source manifest row count mismatch")
    keys = [(row["stratum"], int(row["source_order_rank"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("eligible source order contains duplicate ranks")
    for stratum in STRATA:
        ranks = sorted(int(row["source_order_rank"]) for row in rows if row["stratum"] == stratum)
        if ranks != list(range(len(ranks))):
            raise RuntimeError(f"{stratum}: source ranks are not contiguous")
    return rows


def initial_seed(config: Mapping[str, Any], model_key: str, source_id: str) -> int:
    return stable_seed(int(config["seed"]), "native-initial-v2", model_key, source_id)


def regeneration_seed(config: Mapping[str, Any], model_key: str, trace_id: str, rollout_index: int) -> int:
    if int(rollout_index) not in list(map(int, config["generation"]["rollout_indices"])):
        raise ValueError("regeneration rollout index is outside the frozen four")
    return stable_seed(int(config["seed"]), "native-full-regeneration-v2", model_key, trace_id, int(rollout_index))


def failed_trace_id(model_key: str, attempt_key: str) -> str:
    return stable_hash(["native-failed-trace-v2", model_key, attempt_key])


def build_attempt_packs(
    rows: Iterable[Mapping[str, Any]], *, model_key: str, stratum: str,
    pack_size: int, configuration_hash: str,
) -> list[AttemptPack]:
    values = sorted(
        (dict(row) for row in rows if str(row["stratum"]) == str(stratum)),
        key=lambda row: int(row["source_order_rank"]),
    )
    packs: list[AttemptPack] = []
    for pack_index, start in enumerate(range(0, len(values), int(pack_size))):
        source_ids = tuple(str(row["source_id"]) for row in values[start : start + int(pack_size)])
        packs.append(AttemptPack(
            pack_id=stable_hash(["native-attempt-pack-v2", configuration_hash, model_key, stratum, pack_index, source_ids])[:24],
            model_key=model_key, stratum=stratum, pack_index=pack_index,
            source_ids=source_ids, configuration_hash=configuration_hash,
        ))
    flattened = [source_id for pack in packs for source_id in pack.source_ids]
    if flattened != [str(row["source_id"]) for row in values] or len(flattened) != len(set(flattened)):
        raise AssertionError("attempt packs altered the frozen source order")
    return packs


def build_regeneration_packs(
    cohort: Sequence[Mapping[str, Any]], *, model_key: str, pack_size: int,
    configuration_hash: str,
) -> list[RegenerationPack]:
    values = list(cohort)
    packs = []
    for pack_index, start in enumerate(range(0, len(values), int(pack_size))):
        trace_ids = tuple(str(row["trace_id"]) for row in values[start : start + int(pack_size)])
        packs.append(RegenerationPack(
            pack_id=stable_hash(["native-regeneration-pack-v2", configuration_hash, model_key, pack_index, trace_ids])[:24],
            model_key=model_key, pack_index=pack_index, trace_ids=trace_ids,
            configuration_hash=configuration_hash,
        ))
    if [trace for pack in packs for trace in pack.trace_ids] != [str(row["trace_id"]) for row in values]:
        raise AssertionError("regeneration packs altered cohort order")
    return packs


def valid_initial_failure(row: Mapping[str, Any]) -> bool:
    return str(row.get("initial_status")) == VALID_INITIAL_FAILURE


def select_frozen_cohort(
    attempts: Iterable[Mapping[str, Any]], target_by_stratum: Mapping[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_stratum: dict[str, list[dict[str, Any]]] = {}
    for stratum in STRATA:
        by_stratum[stratum] = sorted(
            (dict(row) for row in attempts if row["stratum"] == stratum and valid_initial_failure(row)),
            key=lambda row: (int(row["source_order_rank"]), str(row["attempt_key"])),
        )
    intended = {key: int(target_by_stratum[key]) for key in STRATA}
    gsm = by_stratum["gsm1k"][: intended["gsm1k"]]
    gsm_deficit = intended["gsm1k"] - len(gsm)
    target_math3 = intended["math_level_3"] + gsm_deficit
    math3 = by_stratum["math_level_3"][:target_math3]
    math3_deficit = target_math3 - len(math3)
    target_math4 = intended["math_level_4"] + math3_deficit
    math4 = by_stratum["math_level_4"][:target_math4]
    selected_attempts = [*gsm, *math3, *math4]
    cohort = []
    for selection_rank, row in enumerate(selected_attempts):
        trace_id = failed_trace_id(str(row["model_key"]), str(row["attempt_key"]))
        cohort.append({
            **row,
            "trace_id": trace_id,
            "selection_rank": selection_rank,
            "initial_completion_token_ids": list(row["completion_token_ids"]),
            "frozen_before_regeneration": True,
            "regeneration_conditioned_selection": False,
        })
    summary = {
        "target_total": sum(intended.values()),
        "selected_total": len(cohort),
        "selected_by_stratum": dict(Counter(row["stratum"] for row in cohort)),
        "available_valid_failures_by_stratum": {key: len(value) for key, value in by_stratum.items()},
        "intended_targets": intended,
        "adjusted_targets": {"gsm1k": intended["gsm1k"], "math_level_3": target_math3, "math_level_4": target_math4},
        "quota_transfers": {"gsm1k_to_math_level_3": gsm_deficit, "math_level_3_to_math_level_4": math3_deficit},
        "source_exhaustion_shortfall": max(0, sum(intended.values()) - len(cohort)),
    }
    return cohort, summary


def validate_logical_rows(rows: Iterable[Mapping[str, Any]], *, required: set[str], key: str) -> None:
    values = list(rows)
    identities = []
    for index, row in enumerate(values):
        missing = required - set(row)
        if missing:
            raise RuntimeError(f"row {index} lacks fields {sorted(missing)}")
        identities.append(str(row[key]))
    if len(identities) != len(set(identities)):
        raise RuntimeError(f"duplicate logical {key}")


def attach_cohort_status(attempts: Iterable[Mapping[str, Any]], cohort: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected = {str(row["attempt_key"]): str(row["trace_id"]) for row in cohort}
    output = []
    for value in attempts:
        row = dict(value)
        if str(row["attempt_key"]) in selected:
            row.update(cohort_inclusion_status="retained", cohort_exclusion_reason=None, trace_id=selected[str(row["attempt_key"])])
        else:
            reason = str(row.get("initial_status", "unknown"))
            if reason == VALID_INITIAL_FAILURE:
                reason = "valid_failure_after_frozen_quota_cutoff"
            row.update(cohort_inclusion_status="excluded", cohort_exclusion_reason=reason, trace_id=None)
        output.append(row)
    return output


def bootstrap_trace_metrics(
    success_counts: Sequence[int], *, replicates: int, seed: int, confidence_level: float,
) -> dict[str, Any]:
    counts = np.asarray(list(map(int, success_counts)), dtype=np.int64)
    if len(counts) == 0:
        return {
            "n": 0, "fr_at_1": None, "fr_at_4": None, "pf_at_4": None,
            "confidence_intervals": {},
            "success_count_counts": {str(value): 0 for value in range(5)},
            "success_count_distribution": {str(value): None for value in range(5)},
        }
    if np.any((counts < 0) | (counts > 4)):
        raise ValueError("success counts must lie in [0,4]")
    point = np.asarray([counts.mean() / 4.0, np.mean(counts > 0), np.mean(counts == 0)])
    rng = np.random.default_rng(int(seed))
    draws = np.empty((int(replicates), 3), dtype=float)
    for index in range(int(replicates)):
        sample = counts[rng.integers(0, len(counts), size=len(counts))]
        draws[index] = [sample.mean() / 4.0, np.mean(sample > 0), np.mean(sample == 0)]
    alpha = (1.0 - float(confidence_level)) / 2.0
    names = ("fr_at_1", "fr_at_4", "pf_at_4")
    return {
        "n": int(len(counts)),
        **{name: float(value) for name, value in zip(names, point)},
        "confidence_intervals": {
            name: [float(np.quantile(draws[:, offset], alpha)), float(np.quantile(draws[:, offset], 1 - alpha))]
            for offset, name in enumerate(names)
        },
        "success_count_counts": {str(value): int(np.sum(counts == value)) for value in range(5)},
        "success_count_distribution": {str(value): float(np.mean(counts == value)) for value in range(5)},
    }


def acquisition_metrics(attempts: Sequence[Mapping[str, Any]], cohort: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = {str(row["attempt_key"]) for row in cohort}
    by_stratum = {}
    for stratum in ("overall", *STRATA):
        values = list(attempts) if stratum == "overall" else [row for row in attempts if row["stratum"] == stratum]
        valid = [row for row in values if row["initial_status"] in {"valid_correct", VALID_INITIAL_FAILURE}]
        failures = [row for row in valid if row["initial_status"] == VALID_INITIAL_FAILURE]
        by_stratum[stratum] = {
            "attempted": len(values), "valid_initial_generations": len(valid),
            "valid_initial_failures": len(failures),
            "initial_failure_yield": len(failures) / max(len(valid), 1),
            "truncated": sum(row["initial_status"] == "truncated" for row in values),
            "answer_extraction_failures": sum(row["initial_status"] == "answer_extraction_failure" for row in values),
            "empty_responses": sum(row["initial_status"] == "empty_response" for row in values),
            "tokenizer_or_context_failures": sum(
                row["initial_status"] == "tokenizer_or_context_failure" for row in values
            ),
            "retained_failures": sum(str(row["attempt_key"]) in selected for row in values),
        }
    return by_stratum


def regeneration_metrics(
    cohort: Sequence[Mapping[str, Any]], rollouts: Sequence[Mapping[str, Any]],
    statistics: Mapping[str, Any],
) -> dict[str, Any]:
    validate_logical_rows(rollouts, required=REGEN_REQUIRED, key="rollout_key")
    by_trace: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rollouts:
        by_trace[str(row["trace_id"])].append(row)
    records = []
    for trace in cohort:
        values = sorted(by_trace.get(str(trace["trace_id"]), []), key=lambda row: int(row["rollout_index"]))
        if [int(row["rollout_index"]) for row in values] != [0, 1, 2, 3]:
            raise RuntimeError(f"trace {trace['trace_id']} does not have exactly four regeneration outcomes")
        records.append({
            "trace_id": trace["trace_id"], "model_key": trace["model_key"], "stratum": trace["stratum"],
            "success_count": sum(int(row["binary_verifier_outcome"]) for row in values),
            "generated_tokens": sum(int(row["generated_token_count"]) for row in values),
            "truncations": sum(bool(row["truncation_flag"]) for row in values),
            "malformed_outputs": sum(row["parser_status"] != "parsed" for row in values),
        })
    output: dict[str, Any] = {"trace_records": records, "metrics": {}}
    for label, selector in [
        ("overall", lambda row: True),
        *((stratum, lambda row, value=stratum: row["stratum"] == value) for stratum in STRATA),
    ]:
        chosen = [row for row in records if selector(row)]
        base = bootstrap_trace_metrics(
            [row["success_count"] for row in chosen],
            replicates=int(statistics["bootstrap_replicates"]),
            seed=stable_seed(2701, "native-regeneration-bootstrap", label),
            confidence_level=float(statistics["confidence_level"]),
        )
        rollout_values = [row for row in rollouts if label == "overall" or row["stratum"] == label]
        token_counts = [int(row["generated_token_count"]) for row in rollout_values]
        base.update(
            mean_generated_tokens=float(np.mean(token_counts)) if token_counts else None,
            median_generated_tokens=float(np.median(token_counts)) if token_counts else None,
            total_generated_tokens=sum(token_counts),
            truncation_rate=sum(bool(row["truncation_flag"]) for row in rollout_values) / max(len(rollout_values), 1),
            malformed_output_rate=sum(row["parser_status"] != "parsed" for row in rollout_values) / max(len(rollout_values), 1),
        )
        output["metrics"][label] = base
    return output


def validate_final_integrity(
    *, config: Mapping[str, Any], source_rows: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]], cohort: Sequence[Mapping[str, Any]],
    rollouts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    validate_logical_rows(attempts, required=INITIAL_REQUIRED, key="attempt_key")
    validate_logical_rows(rollouts, required=REGEN_REQUIRED, key="rollout_key")
    source_ids = {str(row["source_id"]) for row in source_rows}
    if any(str(row["source_id"]) not in source_ids for row in attempts):
        raise RuntimeError("attempt outside frozen source census")
    if any(
        row.get("gold_answer_supplied_to_model")
        or row.get("failed_trace_supplied_to_model")
        or row.get("verifier_feedback_supplied_to_model")
        for row in attempts
    ):
        raise RuntimeError("an initial attempt exposed prohibited information to the model")
    cohort_attempts = [str(row["attempt_key"]) for row in cohort]
    if len(cohort_attempts) != len(set(cohort_attempts)):
        raise RuntimeError("duplicate failed cohort attempt")
    if any(not row.get("frozen_before_regeneration") or row.get("regeneration_conditioned_selection") for row in cohort):
        raise RuntimeError("cohort is not explicitly frozen before regeneration")
    expected_cohort, expected_summary = select_frozen_cohort(
        attempts, config["acquisition"]["target_by_stratum"]
    )
    if [str(row["trace_id"]) for row in cohort] != [str(row["trace_id"]) for row in expected_cohort]:
        raise RuntimeError("frozen cohort is not the quota-ordered initial-failure cohort")
    if any(
        row.get("initial_status") != VALID_INITIAL_FAILURE
        or bool(row.get("truncation_flag"))
        or row.get("parser_status") != "parsed"
        or row.get("verifier_status") != "completed"
        for row in cohort
    ):
        raise RuntimeError("cohort contains an invalid, truncated, unparsed, or unverifiable initial trace")
    cohort_ids = {str(row["trace_id"]) for row in cohort}
    if any(str(row["trace_id"]) not in cohort_ids for row in rollouts):
        raise RuntimeError("regeneration row outside frozen cohort")
    if any(
        row.get("gold_answer_supplied_to_model")
        or row.get("original_failed_trace_supplied_to_model")
        or row.get("verifier_feedback_supplied_to_model")
        for row in rollouts
    ):
        raise RuntimeError("a full regeneration exposed prohibited information to the model")
    counts = Counter(str(row["trace_id"]) for row in rollouts)
    if counts and set(counts.values()) != {4}:
        raise RuntimeError("retained traces do not have exactly four regeneration rows")
    seeds_by_trace: dict[str, set[int]] = defaultdict(set)
    for row in rollouts:
        seeds_by_trace[str(row["trace_id"])].add(int(row["rollout_seed"]))
    if any(len(values) != 4 for values in seeds_by_trace.values()):
        raise RuntimeError("regeneration seed collision within trace")
    indices_by_trace: dict[str, set[int]] = defaultdict(set)
    for row in rollouts:
        trace_id = str(row["trace_id"])
        index = int(row["rollout_index"])
        indices_by_trace[trace_id].add(index)
        expected_seed = regeneration_seed(config, str(row["model_key"]), trace_id, index)
        if int(row["rollout_seed"]) != expected_seed:
            raise RuntimeError("regeneration row uses the wrong identity-bound seed")
    if any(values != {0, 1, 2, 3} for values in indices_by_trace.values()):
        raise RuntimeError("regeneration rollout indices are not exactly 0..3")
    parser_versions = {str(row.get("parser_version")) for row in [*attempts, *rollouts]}
    verifier_versions = {str(row.get("verifier_version")) for row in [*attempts, *rollouts]}
    if parser_versions - {str(config["parser"]["version"])}:
        raise RuntimeError("parser versions differ across logical records")
    if verifier_versions - {str(config["verifier"]["version"])}:
        raise RuntimeError("verifier versions differ across logical records")
    forbidden = {
        "semantic_segmentation_enabled": config["acquisition"]["semantic_segmentation_enabled"],
        "checkpoint_extraction_enabled": config["acquisition"]["checkpoint_extraction_enabled"],
        "hidden_state_extraction_enabled": config["acquisition"]["hidden_state_extraction_enabled"],
        "kv_cache_persistence_enabled": config["acquisition"]["kv_cache_persistence_enabled"],
        "boundary_model_enabled": config["acquisition"]["boundary_model_enabled"],
    }
    if any(bool(value) for value in forbidden.values()):
        raise RuntimeError("an explicitly forbidden native stage was enabled")
    return {
        "status": "PASS",
        "source_manifest_rows": len(source_rows),
        "attempt_rows": len(attempts),
        "cohort_rows": len(cohort),
        "regeneration_rows": len(rollouts),
        "four_rollouts_per_trace": not cohort or len(rollouts) == 4 * len(cohort),
        "duplicate_trace_ids": len(cohort_ids) != len(cohort),
        "seed_collisions": False,
        "regeneration_conditioned_filtering": False,
        "quota_order_recomputed": True,
        "source_exhaustion_shortfall": expected_summary["source_exhaustion_shortfall"],
        "parser_versions": sorted(parser_versions),
        "verifier_versions": sorted(verifier_versions),
        "forbidden_stages": forbidden,
    }


def validate_partial_regeneration_integrity(
    *, config: Mapping[str, Any], source_rows: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]], frozen_cohort: Sequence[Mapping[str, Any]],
    completed_cohort: Sequence[Mapping[str, Any]], rollouts: Sequence[Mapping[str, Any]],
    expected_pack_count: int, completed_pack_count: int,
) -> dict[str, Any]:
    """Validate a user-stopped, immutable-pack prefix without weakening the full gate.

    The ordinary finalizer remains the only path that can emit ``PASS`` for the
    complete frozen cohort.  This helper first applies all ordinary provenance,
    seed, parser/verifier, and forbidden-stage checks to the *full* frozen
    cohort, then verifies that the reported partial cohort is exactly the set of
    traces represented by four complete rollout rows.  The resulting status is
    deliberately ``PARTIAL_PASS`` and is not interchangeable with completion of
    the preregistered 600-trace cohort.
    """

    if expected_pack_count <= 0:
        raise RuntimeError("partial integrity requires a nonempty pack manifest")
    if not 0 <= completed_pack_count < expected_pack_count:
        raise RuntimeError("partial integrity requires an incomplete pack set")

    base = validate_final_integrity(
        config=config,
        source_rows=source_rows,
        attempts=attempts,
        cohort=frozen_cohort,
        rollouts=rollouts,
    )
    frozen_ids = [str(row["trace_id"]) for row in frozen_cohort]
    completed_ids = [str(row["trace_id"]) for row in completed_cohort]
    if len(completed_ids) != len(set(completed_ids)):
        raise RuntimeError("duplicate trace in completed partial cohort")
    completed_set = set(completed_ids)
    if any(trace_id not in set(frozen_ids) for trace_id in completed_ids):
        raise RuntimeError("completed partial cohort is not a subset of the frozen cohort")
    if completed_ids != [trace_id for trace_id in frozen_ids if trace_id in completed_set]:
        raise RuntimeError("completed partial cohort does not preserve frozen cohort order")

    rollout_counts = Counter(str(row["trace_id"]) for row in rollouts)
    if set(rollout_counts) != completed_set:
        raise RuntimeError("partial cohort and rollout trace membership differ")
    if any(count != 4 for count in rollout_counts.values()):
        raise RuntimeError("partial cohort traces do not each have exactly four rollouts")
    if len(rollouts) != 4 * len(completed_cohort):
        raise RuntimeError("partial rollout row total does not equal four per completed trace")

    base.update(
        status="PARTIAL_PASS",
        scientific_status="VALID_FOR_COMPLETED_IMMUTABLE_PACKS_ONLY",
        full_completion_gate_passed=False,
        expected_frozen_cohort_rows=len(frozen_cohort),
        completed_cohort_rows=len(completed_cohort),
        missing_cohort_rows=len(frozen_cohort) - len(completed_cohort),
        expected_regeneration_pack_count=expected_pack_count,
        completed_regeneration_pack_count=completed_pack_count,
        missing_regeneration_pack_count=expected_pack_count - completed_pack_count,
        four_rollouts_per_completed_trace=True,
        four_rollouts_per_frozen_trace=False,
    )
    return base
