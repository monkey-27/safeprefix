"""Outcome-blind, fail-closed preflight for K-densification v1.

This stage deliberately stops before reading any verifier outcome values.  It
proves that the frozen teacher-forced TEST split can supply the requested
per-model problem cohorts, validates the immutable predictor artifacts, and
emits a durable blocker report when any model-domain cohort is impossible.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import pandas as pd

from safeprefix.reproducibility import (
    atomic_json,
    atomic_text,
    git_commit,
    now_iso,
    package_versions,
    stable_hash,
)


MODEL_KEYS = (
    "family_a_small",
    "family_a_large",
    "family_b_small",
    "family_b_large",
)
TEST_SPLIT = "teacher_forced_test"
PROMPT_ROOT_ORDINAL = 0
BLOCKER_CODE = "INSUFFICIENT_PER_MODEL_TEST_PROBLEMS_WITH_FOUR_REASONING_CHECKPOINTS"

TEST_COLUMNS = (
    "base_model",
    "model_id",
    "model_revision",
    "tokenizer_revision",
    "trace_id",
    "common_trace_id",
    "problem_id",
    "problem_group",
    "domain",
    "split",
    "checkpoint_id",
    "checkpoint_ordinal",
    "checkpoint_token_offset",
    "prefix_token_count",
    "total_trace_token_count",
    "total_checkpoint_count",
    "hidden_state_location",
    "hidden_state_source",
    "hidden_state_layer",
    "hidden_state_dimension",
    "num_rollouts",
    "initial_trace_verifier_result",
    "checkpoint_validity_status",
)


class CohortInsufficientError(RuntimeError):
    """Raised after durable blocker artifacts have been written."""

    def __init__(self, message: str, *, summary: Mapping[str, Any]):
        super().__init__(message)
        self.summary = dict(summary)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_non_native(paths: Sequence[Path]) -> None:
    for path in paths:
        lowered = str(path).lower()
        if "native" in lowered:
            raise RuntimeError(f"native artifact path is forbidden: {path}")


def _learning_rate_slug(value: float) -> str:
    return f"lr_{float(value):.0e}"


def _validate_boundary_artifacts(
    *, config: Mapping[str, Any], boundary_root: Path
) -> dict[str, Any]:
    _assert_non_native([boundary_root])
    expected = config["source"]["expected_hashes"]
    manifest_hashes_path = boundary_root / "data/manifest_hashes.json"
    integrity_path = boundary_root / "integrity/final_integrity.json"
    selection_path = boundary_root / "selection/selected_model.json"
    for path in (manifest_hashes_path, integrity_path, selection_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest_hashes = json.loads(manifest_hashes_path.read_text())
    integrity = json.loads(integrity_path.read_text())
    selection = json.loads(selection_path.read_text())
    if integrity.get("status") != "PASS" or integrity.get("failures"):
        raise RuntimeError("boundary-model integrity is not PASS")
    if integrity.get("native_evaluation_used") is not False:
        raise RuntimeError("boundary-model artifact does not prove native isolation")
    if selection.get("status") != "FROZEN":
        raise RuntimeError("boundary-model selection is not frozen")
    if selection.get("selected_architecture") != "linear_probe":
        raise RuntimeError("frozen selected architecture is not linear_probe")
    if float(selection.get("selected_learning_rate")) != 0.001:
        raise RuntimeError("frozen selected learning rate is not 0.001")
    expected_pairs = {
        "canonical_checkpoint_manifest_sha256": expected[
            "canonical_checkpoint_manifest_sha256"
        ],
        "common_trace_manifest_sha256": expected["common_trace_manifest_sha256"],
        "config_sha256": expected["boundary_config_sha256"],
    }
    for key, value in expected_pairs.items():
        if str(manifest_hashes.get(key)) != str(value):
            raise RuntimeError(f"frozen boundary hash differs: {key}")
    split_hash = manifest_hashes.get("split_manifest_sha256", {}).get(TEST_SPLIT)
    if split_hash != expected["teacher_forced_test_split_sha256"]:
        raise RuntimeError("teacher-forced TEST split hash differs")

    feature_hashes: dict[str, str] = {}
    for model_key in MODEL_KEYS:
        path = boundary_root / f"data/features/{model_key}.pt"
        observed = sha256_file(path)
        required = str(expected["feature_store_sha256"][model_key])
        if observed != required:
            raise RuntimeError(f"{model_key}: frozen feature-store hash differs")
        if manifest_hashes["feature_store_sha256"].get(model_key) != required:
            raise RuntimeError(f"{model_key}: recorded feature-store hash differs")
        feature_hashes[model_key] = observed

    architectures = tuple(map(str, config["predictors"]["required_architectures"]))
    seeds = tuple(map(int, config["predictors"]["training_seeds"]))
    rates = {
        str(key): float(value)
        for key, value in config["predictors"]["frozen_learning_rates"].items()
    }
    checkpoint_hashes: dict[str, str] = {}
    for model_key in MODEL_KEYS:
        matrix_path = boundary_root / f"training/{model_key}/matrix_summary.json"
        matrix = json.loads(matrix_path.read_text())
        if matrix.get("status") != "COMPLETE" or int(matrix.get("runs", -1)) != 27:
            raise RuntimeError(f"{model_key}: predictor matrix is incomplete")
        if matrix.get("native_evaluation_used") is not False:
            raise RuntimeError(f"{model_key}: predictor matrix used native data")
        rows = {
            (
                str(row["architecture"]),
                float(row["learning_rate"]),
                int(row["seed"]),
            ): row
            for row in matrix["results"]
        }
        for architecture in architectures:
            rate = rates[architecture]
            for seed in seeds:
                identity = (architecture, rate, seed)
                if identity not in rows:
                    raise RuntimeError(f"{model_key}: frozen predictor absent: {identity}")
                row = rows[identity]
                if row.get("status") != "COMPLETE":
                    raise RuntimeError(f"{model_key}: predictor incomplete: {identity}")
                if row.get("native_evaluation_used") is not False:
                    raise RuntimeError(f"{model_key}: predictor used native data: {identity}")
                checkpoint = (
                    boundary_root
                    / "training"
                    / model_key
                    / architecture
                    / _learning_rate_slug(rate)
                    / f"seed_{seed}"
                    / "best.pt"
                )
                observed = sha256_file(checkpoint)
                if observed != str(row["checkpoint_sha256"]):
                    raise RuntimeError(f"{model_key}: predictor checksum differs: {identity}")
                checkpoint_hashes[
                    f"{model_key}/{architecture}/{_learning_rate_slug(rate)}/seed_{seed}"
                ] = observed
    return {
        "status": "PASS",
        "boundary_integrity": integrity,
        "selection": selection,
        "manifest_hashes": manifest_hashes,
        "feature_store_sha256": feature_hashes,
        "validated_predictor_checkpoint_sha256": checkpoint_hashes,
        "validated_predictor_checkpoint_count": len(checkpoint_hashes),
        "predictor_weights_modified": False,
        "calibrators_modified": False,
        "threshold_modified": False,
        "native_artifacts_loaded": False,
    }


def _load_test_rows(
    *, config: Mapping[str, Any], canonical_manifest_path: Path
) -> pd.DataFrame:
    _assert_non_native([canonical_manifest_path])
    required_hash = config["source"]["expected_hashes"][
        "canonical_checkpoint_manifest_sha256"
    ]
    if sha256_file(canonical_manifest_path) != required_hash:
        raise RuntimeError("canonical checkpoint manifest checksum differs")
    # Only the TEST row group is returned and outcome values are not among the
    # requested columns. This is the evaluation-only and outcome-blind boundary.
    frame = pd.read_parquet(
        canonical_manifest_path,
        columns=list(TEST_COLUMNS),
        filters=[("split", "=", TEST_SPLIT)],
    )
    if set(frame["split"].astype(str)) != {TEST_SPLIT}:
        raise RuntimeError("non-TEST rows crossed the split predicate")
    if len(frame) != int(config["source"]["expected_test_rows"]):
        raise RuntimeError("teacher-forced TEST checkpoint count differs")
    if frame["common_trace_id"].nunique() != int(
        config["source"]["expected_test_common_traces"]
    ):
        raise RuntimeError("teacher-forced TEST common-trace count differs")
    if frame["problem_group"].nunique() != int(
        config["source"]["expected_test_problem_groups"]
    ):
        raise RuntimeError("teacher-forced TEST problem-group count differs")
    domains = set(map(str, config["cohort"]["domains"]))
    if set(frame["domain"].astype(str)) != domains:
        raise RuntimeError("teacher-forced TEST domain definition differs")
    if set(frame["base_model"].astype(str)) != set(MODEL_KEYS):
        raise RuntimeError("teacher-forced TEST model set differs")
    if set(frame["num_rollouts"].astype(int)) != {4}:
        raise RuntimeError("teacher-forced TEST labels are not exact K=4")
    if set(frame["checkpoint_validity_status"].astype(str)) != {
        "included_production"
    }:
        raise RuntimeError("non-production checkpoint entered TEST preflight")
    if frame["initial_trace_verifier_result"].astype(bool).any():
        raise RuntimeError("non-failed trace entered TEST preflight")
    for model_key, expected_model in config["frozen_models"].items():
        part = frame.loc[frame["base_model"].astype(str).eq(str(model_key))]
        for column, expected_key in (
            ("model_id", "model_id"),
            ("model_revision", "model_revision"),
            ("tokenizer_revision", "tokenizer_revision"),
        ):
            if set(part[column].astype(str)) != {str(expected_model[expected_key])}:
                raise RuntimeError(f"{model_key}: frozen {column} differs")
    if frame.duplicated(["base_model", "trace_id", "checkpoint_id"]).any():
        raise RuntimeError("duplicate TEST checkpoint identity")
    return frame


def _load_test_trace_metadata(
    *, test: pd.DataFrame, completion_manifest_root: Path
) -> pd.DataFrame:
    _assert_non_native([completion_manifest_root])
    records: list[dict[str, Any]] = []
    trace_pattern = re.compile(r'"trace_id"\s*:\s*"([^"]+)"')
    for model_key in MODEL_KEYS:
        required_ids = set(
            test.loc[test["base_model"].astype(str).eq(model_key), "trace_id"].astype(str)
        )
        path = completion_manifest_root / f"per_model/{model_key}/trace_manifest.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        observed_ids: set[str] = set()
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                match = trace_pattern.search(line)
                if match is None or match.group(1) not in required_ids:
                    continue
                row = json.loads(line)
                trace_id = str(row["trace_id"])
                if trace_id in observed_ids:
                    raise RuntimeError(f"{model_key}: duplicate TEST trace metadata")
                observed_ids.add(trace_id)
                offsets = list(map(int, row["eligible_checkpoint_offsets"]))
                audits = list(row["eligible_checkpoint_boundary_audits"])
                if not offsets or not audits:
                    raise RuntimeError(f"{model_key}/{trace_id}: empty checkpoint metadata")
                root = audits[0]
                if int(root["checkpoint_index"]) != PROMPT_ROOT_ORDINAL:
                    raise RuntimeError(f"{model_key}/{trace_id}: first checkpoint is not ordinal zero")
                if str(root["checkpoint_kind"]) != "prompt_root":
                    raise RuntimeError(f"{model_key}/{trace_id}: ordinal zero is not prompt_root")
                if offsets[0] != len(row["prompt_token_ids"]):
                    raise RuntimeError(f"{model_key}/{trace_id}: root offset differs from prompt length")
                canonical = test.loc[
                    test["base_model"].astype(str).eq(model_key)
                    & test["trace_id"].astype(str).eq(trace_id)
                ].sort_values("checkpoint_ordinal")
                canonical_offsets = list(map(int, canonical["checkpoint_token_offset"]))
                if offsets != canonical_offsets:
                    raise RuntimeError(f"{model_key}/{trace_id}: checkpoint offsets differ")
                records.append(
                    {
                        "base_model": model_key,
                        "trace_id": trace_id,
                        "common_trace_id": str(row["common_trace_id"]),
                        "problem_id": str(row["problem_id"]),
                        "domain": str(row["source_bucket"]),
                        "prompt_token_count": len(row["prompt_token_ids"]),
                        "all_checkpoint_count": len(offsets),
                        "reasoning_checkpoint_count": len(offsets) - 1,
                        "median_response_length_component": len(row["completion_token_ids"]),
                        "prompt_root_verified": True,
                    }
                )
        if observed_ids != required_ids:
            raise RuntimeError(
                f"{model_key}: TEST trace metadata coverage differs "
                f"({len(observed_ids)} != {len(required_ids)})"
            )
    metadata = pd.DataFrame(records)
    if metadata.duplicated(["base_model", "trace_id"]).any():
        raise RuntimeError("duplicate TEST trace metadata after four-model merge")
    return metadata


def _candidate_census(
    *, config: Mapping[str, Any], metadata: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required_per_trace = int(
        config["cohort"]["minimum_reasoning_checkpoints_per_model_trace"]
    )
    required_columns = (
        "base_model",
        "domain",
        "problem_id",
        "trace_id",
        "common_trace_id",
        "reasoning_checkpoint_count",
        "all_checkpoint_count",
        "median_response_length_component",
    )
    missing = set(required_columns) - set(metadata.columns)
    if missing:
        raise RuntimeError(f"candidate metadata columns are missing: {sorted(missing)}")
    grouped = metadata[list(required_columns)].copy()
    if grouped.duplicated(["base_model", "trace_id"]).any():
        raise RuntimeError("candidate metadata contains duplicate model-trace rows")
    grouped.rename(
        columns={
            "reasoning_checkpoint_count": "minimum_reasoning_checkpoints",
            "all_checkpoint_count": "minimum_all_checkpoints",
            "median_response_length_component": "median_response_length",
        },
        inplace=True,
    )
    grouped["maximum_reasoning_checkpoints"] = grouped[
        "minimum_reasoning_checkpoints"
    ]
    grouped["maximum_all_checkpoints"] = grouped["minimum_all_checkpoints"]
    grouped["stable_trace_hash"] = [
        stable_hash(
            [
                str(config["cohort"]["stable_hash_namespace"]),
                str(row.base_model),
                str(row.domain),
                str(row.problem_id),
                str(row.trace_id),
            ]
        )
        for row in grouped.itertuples(index=False)
    ]
    # If a frozen problem has multiple failed trace variants for one model, the
    # variant is chosen without outcomes by the immutable trace hash.  This
    # makes the experimental unit a model-problem pair, never an opportunistic
    # model-trace variant.
    grouped.sort_values(
        ["base_model", "domain", "problem_id", "stable_trace_hash"],
        kind="stable",
        inplace=True,
    )
    grouped["model_problem_variant_rank"] = grouped.groupby(
        ["base_model", "domain", "problem_id"], sort=False
    ).cumcount()
    grouped["selected_model_problem_variant"] = grouped[
        "model_problem_variant_rank"
    ].eq(0)
    grouped["qualifies_reasoning_only"] = (
        grouped["selected_model_problem_variant"]
        & grouped["minimum_reasoning_checkpoints"].ge(required_per_trace)
    )
    # This counterfactual is diagnostic only. It proves that even incorrectly
    # counting the prompt root would not make the requested cohort possible.
    grouped["qualifies_if_prompt_root_were_incorrectly_counted"] = (
        grouped["selected_model_problem_variant"]
        & grouped["minimum_all_checkpoints"].ge(required_per_trace)
    )
    grouped["stable_problem_hash"] = [
        stable_hash(
            [
                str(config["cohort"]["stable_hash_namespace"]),
                row.base_model,
                row.domain,
                row.problem_id,
            ]
        )
        for row in grouped.itertuples(index=False)
    ]
    grouped["rollout_success_inspected"] = False

    domain_map = {
        str(key): str(value) for key, value in config["cohort"]["domains"].items()
    }
    strict: dict[str, dict[str, int]] = {}
    permissive: dict[str, dict[str, int]] = {}
    for model_key in MODEL_KEYS:
        strict[model_key] = {}
        permissive[model_key] = {}
        for domain, public_name in domain_map.items():
            part = grouped.loc[
                grouped["base_model"].astype(str).eq(model_key)
                & grouped["domain"].astype(str).eq(domain)
            ]
            strict[model_key][public_name] = int(
                part.loc[part["qualifies_reasoning_only"], "problem_id"].nunique()
            )
            permissive[model_key][public_name] = int(
                part.loc[
                    part["qualifies_if_prompt_root_were_incorrectly_counted"],
                    "problem_id",
                ].nunique()
            )
    requested = int(config["cohort"]["problems_per_domain"])
    deficits = {
        model_key: {
            domain: max(0, requested - count)
            for domain, count in model_counts.items()
        }
        for model_key, model_counts in strict.items()
    }
    summary = {
        "requested_problem_count_per_domain": requested,
        "requested_problem_count_per_model": requested * len(domain_map),
        "requested_problem_count_total": requested * len(domain_map) * len(MODEL_KEYS),
        "strict_eligible_problem_count_by_model_domain": strict,
        "permissive_root_inclusive_problem_count_by_model_domain": permissive,
        "strict_deficit_by_model_domain": deficits,
        "strict_total_eligible_model_problems": sum(
            sum(model_counts.values()) for model_counts in strict.values()
        ),
        "cohort_gate_passed": all(
            count >= requested
            for model_counts in strict.values()
            for count in model_counts.values()
        ),
        "cross_model_problem_intersection_required": False,
        "bootstrap_unit": "complete_model_problem_trace_within_model",
        "cross_model_paired_problem_bootstrap": False,
        "aggregate": "equal_weight_four_model_macro_average",
        "prompt_root_counted_as_reasoning_checkpoint": False,
        "rollout_success_values_loaded": False,
        "new_rollout_outcomes_loaded": False,
        "native_outcomes_loaded": False,
    }
    return grouped, summary


def _blocker_markdown(summary: Mapping[str, Any]) -> str:
    strict = summary["strict_eligible_problem_count_by_model_domain"]
    permissive = summary[
        "permissive_root_inclusive_problem_count_by_model_domain"
    ]
    requested = int(summary["requested_problem_count_per_domain"])
    lines = [
        "# K-densification v1 preflight blocker",
        "",
        "Status: **BLOCKED BEFORE PROBLEM MANIFEST AND ROLLOUT GENERATION**",
        "",
        (
            "The frozen teacher-forced TEST split cannot supply one or more of the "
            f"preregistered per-model {requested}-problem domain cohorts while retaining "
            "four eligible reasoning checkpoints for each model-trace. No cross-model "
            "problem intersection was required. The prompt/root checkpoint is not a "
            "reasoning checkpoint and was excluded exactly as required."
        ),
        "",
        "| Model | Domain | Required | Eligible reasoning-only | Deficit | Invalid root-inclusive upper bound |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for model_key in MODEL_KEYS:
        for domain in ("CRV", "Math", "Olympiad", "Omni"):
            count = int(strict[model_key][domain])
            lines.append(
                f"| {model_key} | {domain} | {requested} | {count} | "
                f"{max(0, requested-count)} | {int(permissive[model_key][domain])} |"
            )
    lines.extend(
        [
            "",
            (
                "The root-inclusive column is diagnostic only. Even the invalid choice "
                "to count the prompt root does not change the scientific gate."
            ),
            "",
            "## Integrity boundary",
            "",
            "- Only TEST rows were returned from the canonical Parquet predicate.",
            "- No rollout success value was loaded or inspected.",
            "- No native, calibration, architecture-development, training, or geometry-child outcome was loaded.",
            "- Frozen feature stores and predictor checkpoints were checksum-verified.",
            "- No problem manifest, checkpoint manifest, reuse registry, GPU job, or rollout outcome was created.",
            "- Existing rollout, geometry, calibration, native-evaluation, and predictor artifacts were not modified.",
            "",
            (
                "The experiment remains stopped unless all sixteen model-domain cells "
                "meet the frozen quota under the corrected per-model cohort rule."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run_preflight(
    *,
    config: Mapping[str, Any],
    boundary_root: Path,
    canonical_manifest_path: Path,
    completion_manifest_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Run the real frozen-input gate and fail durably before outcome access."""

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if (output_root / "COMPLETE.json").exists():
        raise RuntimeError("refusing to modify a completed K-densification run")
    source_validation = _validate_boundary_artifacts(
        config=config, boundary_root=Path(boundary_root)
    )
    test = _load_test_rows(
        config=config, canonical_manifest_path=Path(canonical_manifest_path)
    )
    metadata = _load_test_trace_metadata(
        test=test, completion_manifest_root=Path(completion_manifest_root)
    )
    candidates, cohort = _candidate_census(config=config, metadata=metadata)
    atomic_text(output_root / "candidate_problem_census.csv", candidates.to_csv(index=False))
    atomic_json(output_root / "validated_source_hashes.json", source_validation)
    run_manifest = {
        "experiment_id": str(config["experiment"]["id"]),
        "schema_version": int(config["experiment"]["schema_version"]),
        "status": "BLOCKED_BEFORE_ROLLOUT_GENERATION",
        "created_at": now_iso(),
        "git_commit": git_commit(),
        "package_versions": package_versions(),
        "modal_profile": str(config["execution"]["modal_profile"]),
        "expected_workspace": str(config["execution"]["expected_workspace"]),
        "requested_gpu_workers": int(config["execution"]["total_gpu_workers"]),
        "gpu_jobs_submitted": 0,
        "new_rollouts_generated": 0,
        "problem_manifest_written": False,
        "checkpoint_manifest_written": False,
        "registry_written": False,
        "complete_marker_written": False,
        "rollout_success_values_loaded": False,
        "native_outcomes_loaded": False,
        "cohort": cohort,
        "source_validation_status": source_validation["status"],
        "inputs": {
            "boundary_root": str(boundary_root),
            "canonical_manifest": str(canonical_manifest_path),
            "completion_manifest_root": str(completion_manifest_root),
        },
    }
    atomic_json(output_root / "run_manifest.json", run_manifest)
    if cohort["cohort_gate_passed"]:
        return {"status": "COHORT_GATE_PASS", **cohort}

    blocker = {
        "status": "BLOCKED_BEFORE_ROLLOUT_GENERATION",
        "blocker_code": BLOCKER_CODE,
        "blocked_at": "per_model_problem_subset_preflight",
        "created_at": now_iso(),
        "cohort": cohort,
        "gpu_jobs_submitted": 0,
        "new_rollouts_generated": 0,
        "existing_artifacts_modified": False,
        "problem_manifest_written": False,
        "checkpoint_manifest_written": False,
        "rollout_success_values_loaded": False,
        "native_outcomes_loaded": False,
    }
    atomic_json(output_root / "BLOCKED.json", blocker)
    atomic_text(output_root / "PREFLIGHT_BLOCKER_REPORT.md", _blocker_markdown(cohort))
    message = (
        "frozen TEST per-model cohorts are insufficient: "
        + json.dumps(
            cohort["strict_eligible_problem_count_by_model_domain"], sort_keys=True
        )
    )
    raise CohortInsufficientError(message, summary=blocker)
