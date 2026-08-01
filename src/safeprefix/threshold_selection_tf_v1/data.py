"""Frozen calibration cohort, folds, seed identities, and workload manifests."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from safeprefix.reproducibility import (
    atomic_json,
    atomic_jsonl,
    atomic_parquet,
    atomic_text,
    now_iso,
    stable_hash,
    stable_seed,
)


MODEL_KEYS = ("family_a_small", "family_a_large", "family_b_small", "family_b_large")
TRAINING_SEEDS = (0, 1, 2)
SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _learning_rate_slug(value: float) -> str:
    return f"lr_{float(value):.0e}"


def _canonical_artifact_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _canonical_artifact_value(value.item())
    if isinstance(value, np.ndarray):
        return [_canonical_artifact_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_canonical_artifact_value(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_artifact_value(item) for key, item in value.items()
        }
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def row_artifact_hash(row: Mapping[str, Any]) -> str:
    """Hash one outcome row identically before and after Parquet round trips."""
    return stable_hash(
        {
            str(key): _canonical_artifact_value(value)
            for key, value in row.items()
            if key != "artifact_hash"
        }
    )


def select_median_dev_seeds(
    boundary_root: Path,
    *,
    architecture: str,
    learning_rate: float,
) -> tuple[dict[str, int], pd.DataFrame]:
    """Select each model's median dev seed without calibration/test inputs."""
    records: list[dict[str, Any]] = []
    selected: dict[str, int] = {}
    for model_key in MODEL_KEYS:
        matrix_path = Path(boundary_root) / "training" / model_key / "matrix_summary.json"
        matrix = json.loads(matrix_path.read_text())
        candidates = [
            row
            for row in matrix["results"]
            if str(row["architecture"]) == architecture
            and math.isclose(float(row["learning_rate"]), float(learning_rate))
            and int(row["seed"]) in TRAINING_SEEDS
        ]
        if len(candidates) != 3:
            raise RuntimeError(f"{model_key}: expected three {architecture} seed results")
        ordered = sorted(
            candidates,
            key=lambda row: (
                float(row["dev_metrics"]["trace_weighted_binomial_nll"]),
                int(row["seed"]),
            ),
        )
        chosen = ordered[1]
        selected[model_key] = int(chosen["seed"])
        for rank, row in enumerate(ordered):
            checkpoint = (
                Path(boundary_root)
                / "training"
                / model_key
                / architecture
                / _learning_rate_slug(learning_rate)
                / f"seed_{int(row['seed'])}"
                / "best.pt"
            )
            records.append(
                {
                    "base_model": model_key,
                    "architecture": architecture,
                    "learning_rate": float(learning_rate),
                    "seed": int(row["seed"]),
                    "dev_nll": float(
                        row["dev_metrics"]["trace_weighted_binomial_nll"]
                    ),
                    "dev_rank": rank,
                    "operational": int(row["seed"]) == int(chosen["seed"]),
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": sha256_file(checkpoint),
                }
            )
    return selected, pd.DataFrame(records)


def assign_crossfit_folds(
    common: pd.DataFrame,
    *,
    folds: int,
    seed: int,
) -> pd.DataFrame:
    """Hash-greedy, domain-balanced assignment at frozen problem-group level."""
    required = {"common_trace_id", "problem_group", "domain"}
    if required - set(common.columns):
        raise ValueError(f"fold input lacks {sorted(required - set(common.columns))}")
    rows = common[list(required)].drop_duplicates().copy()
    if rows["common_trace_id"].duplicated().any():
        raise RuntimeError("one common trace maps to multiple problem/domain identities")
    groups: list[dict[str, Any]] = []
    for group_id, part in rows.groupby("problem_group", sort=False):
        domains = sorted(set(part["domain"].astype(str)))
        if len(domains) != 1:
            raise RuntimeError(f"problem group {group_id} spans domains: {domains}")
        groups.append(
            {
                "problem_group": str(group_id),
                "domain": domains[0],
                "trace_count": int(part["common_trace_id"].nunique()),
            }
        )
    domain_counts: dict[str, list[int]] = defaultdict(lambda: [0] * int(folds))
    total_counts = [0] * int(folds)
    group_counts = [0] * int(folds)
    assignment: dict[str, int] = {}
    for group in sorted(
        groups,
        key=lambda row: (
            str(row["domain"]),
            -int(row["trace_count"]),
            stable_hash(["threshold-crossfit-order-v1", int(seed), row["domain"], row["problem_group"]]),
        ),
    ):
        domain = str(group["domain"])
        group_id = str(group["problem_group"])
        target = min(
            range(int(folds)),
            key=lambda fold: (
                domain_counts[domain][fold],
                total_counts[fold],
                group_counts[fold],
                stable_hash(["threshold-crossfit-tie-v1", int(seed), domain, group_id, fold]),
            ),
        )
        assignment[group_id] = int(target)
        domain_counts[domain][target] += int(group["trace_count"])
        total_counts[target] += int(group["trace_count"])
        group_counts[target] += 1
    output = rows.sort_values("common_trace_id").copy()
    output["fold"] = output["problem_group"].map(assignment).astype(int)
    if output["fold"].nunique() != int(folds):
        raise RuntimeError("cross-fit assignment did not populate every fold")
    if output.groupby("problem_group")["fold"].nunique().max() != 1:
        raise AssertionError("problem group crossed folds")
    output["assignment_seed"] = int(seed)
    output["assignment_hash"] = [
        stable_hash(["threshold-crossfit-row-v1", int(seed), row.common_trace_id, row.problem_group, int(row.fold)])
        for row in output.itertuples(index=False)
    ]
    return output


def _score_position_only(checkpoint: Path, frame: pd.DataFrame) -> np.ndarray:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    values = np.column_stack(
        [
            frame["checkpoint_ordinal"].to_numpy(float),
            frame["prefix_token_count"].to_numpy(float),
            frame["checkpoint_ordinal"].to_numpy(float)
            / np.maximum(frame["total_checkpoint_count"].to_numpy(float), 1.0),
            frame["prefix_token_count"].to_numpy(float)
            / np.maximum(frame["total_trace_token_count"].to_numpy(float), 1.0),
        ]
    )
    mean = state["position_mean"].detach().cpu().numpy().astype(float)
    std = np.maximum(state["position_std"].detach().cpu().numpy().astype(float), 1e-8)
    weight = state["position_model.weight"].detach().cpu().numpy().astype(float)[0]
    bias = float(state["position_model.bias"].detach().cpu().numpy()[0])
    return ((values - mean) / std) @ weight + bias


def _source_trace_manifest_path(completion_root: Path, model_key: str) -> Path:
    return (
        Path(completion_root)
        / "immutable_manifests"
        / "repairability"
        / "per_model"
        / model_key
        / "trace_manifest.jsonl"
    )


def _load_selected_trace_rows(
    completion_root: Path,
    model_key: str,
    expected_trace_ids: set[str],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    path = _source_trace_manifest_path(completion_root, model_key)
    trace_pattern = re.compile(r'"trace_id"\s*:\s*"([^"]+)"')
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            # JSONL has no row-group predicate. Extract only the identity token
            # first and deserialize a row only when its ID belongs to the
            # hash-frozen calibration cohort. Nonmatching trace objects are
            # never materialized or returned to threshold preparation.
            match = trace_pattern.search(line)
            if match is None or match.group(1) not in expected_trace_ids:
                continue
            row = json.loads(line)
            selected.append(row)
    if {str(row["trace_id"]) for row in selected} != expected_trace_ids:
        raise RuntimeError(f"{model_key}: calibration traces missing from source manifest")
    return sorted(selected, key=lambda row: str(row["trace_id"]))


def _resolve_source_path(path: str, completion_mount: Path) -> Path:
    source = Path(str(path))
    if str(source).startswith("/completion/"):
        relative = Path(str(source)[len("/completion/") :])
        source = Path(completion_mount) / relative
    forbidden = ("native", "teacher_forced_test", "geometry")
    lowered = str(source).lower()
    if any(token in lowered for token in forbidden):
        raise RuntimeError(f"prohibited source path requested: {source}")
    return source


def _original_k4_rows(
    checkpoint_manifest: pd.DataFrame,
    *,
    completion_mount: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    accessed: list[dict[str, Any]] = []
    for source_path in sorted(set(checkpoint_manifest["hidden_state_source"].astype(str))):
        calibration_keys = checkpoint_manifest[
            checkpoint_manifest["hidden_state_source"].astype(str).eq(source_path)
        ][["base_model", "trace_id", "checkpoint_ordinal", "checkpoint_token_offset"]].drop_duplicates()
        permitted_trace_ids = sorted(set(calibration_keys["trace_id"].astype(str)))
        if not permitted_trace_ids:
            raise RuntimeError("source file has no permitted calibration trace IDs")
        feature_path = _resolve_source_path(source_path, completion_mount)
        rollout_path = feature_path.parent / "rollouts.parquet"
        marker_path = feature_path.parent / "complete.json"
        if not rollout_path.is_file():
            raise FileNotFoundError(rollout_path)
        if not marker_path.is_file():
            raise FileNotFoundError(marker_path)
        marker = json.loads(marker_path.read_text())
        if marker.get("status") != "COMPLETE":
            raise RuntimeError(f"original K=4 pack is not complete: {marker_path}")
        recorded_rollout_hash = marker.get("rollouts_sha256")
        if recorded_rollout_hash is not None and recorded_rollout_hash != sha256_file(rollout_path):
            raise RuntimeError(f"original K=4 rollout checksum differs: {rollout_path}")
        if marker.get("integrity", {}).get("passed") is False:
            raise RuntimeError(f"original K=4 pack integrity failed: {marker_path}")
        # This predicate is the outcome-access boundary. Production packs mix
        # split roles, so reading a whole pack and filtering afterward would
        # return prohibited teacher-forced-test outcomes to this process.
        frame = pd.read_parquet(
            rollout_path,
            filters=[("trace_id", "in", permitted_trace_ids)],
        )
        returned_trace_ids = set(frame["trace_id"].astype(str))
        if returned_trace_ids != set(permitted_trace_ids):
            raise RuntimeError(f"calibration-only predicate coverage differs: {rollout_path}")
        frame = frame.rename(columns={"model_key": "base_model", "checkpoint_index": "checkpoint_ordinal"})
        returned_keys = frame[
            ["base_model", "trace_id", "checkpoint_ordinal", "checkpoint_token_offset"]
        ].drop_duplicates()
        if set(map(tuple, returned_keys.to_records(index=False))) != set(
            map(tuple, calibration_keys.to_records(index=False))
        ):
            raise RuntimeError(f"calibration-only checkpoint keys differ: {rollout_path}")
        accessed.append(
            {
                "path": str(rollout_path),
                "predicate": {"trace_id_in": permitted_trace_ids},
                "returned_rows": len(frame),
                "returned_trace_ids": permitted_trace_ids,
                "noncalibration_rows_returned": 0,
            }
        )
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    keys = checkpoint_manifest[
        ["base_model", "trace_id", "checkpoint_ordinal", "checkpoint_token_offset"]
    ].drop_duplicates()
    raw = raw.merge(
        keys,
        on=["base_model", "trace_id", "checkpoint_ordinal", "checkpoint_token_offset"],
        how="inner",
        validate="many_to_one",
    )
    counts = raw.groupby(["base_model", "trace_id", "checkpoint_ordinal"]).size()
    if len(counts) != len(checkpoint_manifest) or set(counts.astype(int)) != {4}:
        raise RuntimeError("original K=4 raw outcome coverage differs")
    if set(raw["rollout_index"].astype(int)) != {0, 1, 2, 3}:
        raise RuntimeError("original rollout indices differ from 0..3")
    if raw[["rollout_seed", "binary_outcome", "generated_token_count"]].isna().any().any():
        raise RuntimeError("original K=4 raw outcomes contain missing required values")
    raw["outcome_origin"] = "original_k4"
    raw["infrastructure_status"] = "executed"
    raw["artifact_hash"] = [row_artifact_hash(row) for row in raw.to_dict("records")]
    return raw.sort_values(
        ["base_model", "trace_id", "checkpoint_ordinal", "rollout_index"]
    ).reset_index(drop=True), accessed


def _build_pack_assignments(
    checkpoint_manifest: pd.DataFrame,
    *,
    traces_per_pack: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    trace_rows = (
        checkpoint_manifest.groupby(["base_model", "trace_id"], as_index=False)
        .agg(checkpoint_count=("checkpoint_id", "size"), domain=("domain", "first"))
    )
    packs: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    for model_key in MODEL_KEYS:
        model_rows = trace_rows[trace_rows["base_model"].eq(model_key)].copy()
        model_rows["work"] = model_rows["checkpoint_count"] * 12 + 16
        ordered = model_rows.sort_values(
            ["work", "domain", "trace_id"], ascending=[False, True, True]
        ).to_dict("records")
        pack_count = int(math.ceil(len(ordered) / int(traces_per_pack)))
        bins: list[list[dict[str, Any]]] = [[] for _ in range(pack_count)]
        work = [0] * pack_count
        for row in ordered:
            available = [i for i, values in enumerate(bins) if len(values) < int(traces_per_pack)]
            target = min(available, key=lambda i: (work[i], len(bins[i]), i))
            bins[target].append(row)
            work[target] += int(row["work"])
        for index, members in enumerate(bins):
            trace_ids = sorted(str(row["trace_id"]) for row in members)
            identity = {
                "schema_version": SCHEMA_VERSION,
                "model_key": model_key,
                "pack_index": index,
                "trace_ids": trace_ids,
                "checkpoint_rollout_indices": list(range(4, 16)),
                "full_regeneration_indices": list(range(16)),
            }
            pack_hash = stable_hash(identity)
            pack_id = f"{model_key}-{index:05d}-{pack_hash[:12]}"
            packs.append(
                {
                    **identity,
                    "pack_id": pack_id,
                    "pack_hash": pack_hash,
                    "trace_count": len(trace_ids),
                    "checkpoint_count": int(sum(int(row["checkpoint_count"]) for row in members)),
                    "added_checkpoint_rollouts": int(sum(int(row["checkpoint_count"]) * 12 for row in members)),
                    "full_regenerations": len(trace_ids) * 16,
                    "estimated_work": work[index],
                }
            )
            assignments.extend(
                {"base_model": model_key, "trace_id": trace_id, "pack_id": pack_id}
                for trace_id in trace_ids
            )
    return pd.DataFrame(assignments), packs


def _seed_manifests(
    checkpoint_manifest: pd.DataFrame,
    assignments: pd.DataFrame,
    original_k4: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assigned = checkpoint_manifest.merge(
        assignments, on=["base_model", "trace_id"], validate="many_to_one"
    )
    checkpoint_rows: list[dict[str, Any]] = []
    for row in assigned.itertuples(index=False):
        for rollout_index in range(4, 16):
            checkpoint_rows.append(
                {
                    "base_model": row.base_model,
                    "trace_id": row.trace_id,
                    "common_trace_id": row.common_trace_id,
                    "problem_id": row.problem_id,
                    "problem_group": row.problem_group,
                    "domain": row.domain,
                    "checkpoint_id": row.checkpoint_id,
                    "checkpoint_ordinal": int(row.checkpoint_ordinal),
                    "checkpoint_token_offset": int(row.checkpoint_token_offset),
                    "rollout_index": rollout_index,
                    "rollout_seed": stable_seed(2701, str(row.trace_id), int(row.checkpoint_ordinal), rollout_index),
                    "pack_id": row.pack_id,
                }
            )
    trace_rows = assigned.drop_duplicates(["base_model", "trace_id"])
    regeneration_rows: list[dict[str, Any]] = []
    for row in trace_rows.itertuples(index=False):
        for rollout_index in range(16):
            regeneration_rows.append(
                {
                    "base_model": row.base_model,
                    "trace_id": row.trace_id,
                    "common_trace_id": row.common_trace_id,
                    "problem_id": row.problem_id,
                    "problem_group": row.problem_group,
                    "domain": row.domain,
                    "rollout_index": rollout_index,
                    "rollout_seed": stable_seed(
                        2701,
                        "safeprefix-threshold-selection-tf-v1",
                        "full-regeneration",
                        str(row.base_model),
                        str(row.trace_id),
                        rollout_index,
                    ),
                    "pack_id": row.pack_id,
                }
            )
    checkpoint_frame = pd.DataFrame(checkpoint_rows)
    regeneration_frame = pd.DataFrame(regeneration_rows)
    expected = set(
        zip(
            original_k4["base_model"].astype(str),
            original_k4["trace_id"].astype(str),
            original_k4["checkpoint_ordinal"].astype(int),
            original_k4["rollout_seed"].astype(int),
        )
    )
    observed = set(
        zip(
            checkpoint_frame["base_model"].astype(str),
            checkpoint_frame["trace_id"].astype(str),
            checkpoint_frame["checkpoint_ordinal"].astype(int),
            checkpoint_frame["rollout_seed"].astype(int),
        )
    )
    if expected & observed:
        raise RuntimeError("new checkpoint seeds collide with original K=4 seeds")
    all_new = pd.concat(
        [
            checkpoint_frame[["base_model", "trace_id", "rollout_seed"]].assign(stage="checkpoint"),
            regeneration_frame[["base_model", "trace_id", "rollout_seed"]].assign(stage="full_regeneration"),
        ],
        ignore_index=True,
    )
    if all_new["rollout_seed"].duplicated().any():
        raise RuntimeError("new scientific rollout seed collision")
    return checkpoint_frame, regeneration_frame


def _prelaunch_report(census: Mapping[str, Any]) -> str:
    lines = [
        "# Prelaunch census and integrity",
        "",
        "Status: **PASS — FROZEN CALIBRATION-ONLY WORKLOAD**",
        "",
        "No native, teacher-forced-test, or geometry-test outcome was opened. The canonical Parquet was read with a calibration-only predicate.",
        "",
        "## Counts",
        "",
        f"- Shared calibration traces: **{census['shared_traces']}**",
        f"- Model-traces: **{census['model_traces']}**",
        f"- Checkpoints: **{census['checkpoints_total']}** ({census['checkpoints_per_model']} per model)",
        f"- Existing K=4 rollouts: **{census['existing_rollouts']}**",
        f"- New checkpoint suffixes: **{census['new_checkpoint_rollouts']}**",
        f"- Full regenerations: **{census['full_regenerations']}**",
        f"- Excluded smoke/validation artifacts: **{census['excluded_artifacts']}**",
        "",
        "## Operational linear-probe seeds",
        "",
        "| Model | Seed | Dev NLL | Checkpoint SHA-256 |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in census["operational_seed_rows"]:
        lines.append(
            f"| {row['base_model']} | {row['seed']} | {row['dev_nll']:.10f} | `{row['checkpoint_sha256']}` |"
        )
    lines += [
        "",
        "## Frozen manifest hashes",
        "",
        f"- Canonical checkpoint manifest: `{census['canonical_manifest_sha256']}`",
        f"- Calibration problem manifest: `{census['calibration_split_sha256']}`",
        "",
        "The five folds were assigned at problem-group level with deterministic domain balancing. Original K=4 rows are immutable and copied into a separate threshold artifact; new seeds use disjoint deterministic namespaces.",
    ]
    return "\n".join(lines) + "\n"


def prepare_threshold_experiment(
    config: Mapping[str, Any],
    *,
    artifact_root: Path,
    boundary_root: Path | None = None,
    completion_root: Path | None = None,
    completion_mount: Path = Path("/completion"),
) -> dict[str, Any]:
    """Freeze calibration-only manifests without accessing native/test outcomes."""
    root = Path(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    existing_protocol = root / "manifests/frozen_protocol.json"
    existing_ready = root / "READY.json"
    if existing_protocol.is_file() and existing_ready.is_file():
        frozen = json.loads(existing_protocol.read_text())
        if frozen.get("configuration_hash") != stable_hash(config):
            raise RuntimeError("existing frozen threshold protocol uses another configuration")
        census_path = root / "manifests/prelaunch_census.json"
        if not census_path.is_file():
            raise RuntimeError("frozen threshold protocol lacks its prelaunch census")
        return json.loads(census_path.read_text())
    source = config["source"]
    boundary = Path(boundary_root or source["boundary_root"])
    completion = Path(completion_root or source["completion_root"])
    expected = source["expected"]
    canonical_path = boundary / "data/canonical_checkpoint_manifest.parquet"
    calibration_split_path = boundary / "data/splits/calibration_problems.jsonl"
    hashes = source["expected_manifest_hashes"]
    canonical_sha = sha256_file(canonical_path)
    split_sha = sha256_file(calibration_split_path)
    if canonical_sha != hashes["canonical_checkpoint_manifest_sha256"]:
        raise RuntimeError("canonical checkpoint manifest hash differs")
    if split_sha != hashes["calibration_split_sha256"]:
        raise RuntimeError("calibration split manifest hash differs")
    boundary_hashes = json.loads((boundary / "data/manifest_hashes.json").read_text())
    expected_boundary_hashes = {
        "common_trace_manifest_sha256": hashes["common_trace_manifest_sha256"],
        "config_sha256": hashes["boundary_config_sha256"],
        "excluded_artifacts_sha256": hashes["exclusion_manifest_sha256"],
    }
    for key, expected_hash in expected_boundary_hashes.items():
        if boundary_hashes.get(key) != expected_hash:
            raise RuntimeError(f"boundary manifest hash differs for {key}")
    selected = json.loads((boundary / "selection/selected_model.json").read_text())
    if (
        selected.get("selected_architecture") != "linear_probe"
        or not math.isclose(float(selected.get("selected_learning_rate")), 1e-3)
        or selected.get("test_used_for_selection") is not False
        or selected.get("native_evaluation_used") is not False
    ):
        raise RuntimeError("frozen boundary selection identity differs")

    # Predicate pushdown is a hard access boundary: no test rows are returned.
    calibration = pd.read_parquet(canonical_path, filters=[("split", "==", "calibration")])
    if set(calibration["split"].astype(str)) != {"calibration"}:
        raise RuntimeError("non-calibration row entered threshold preparation")
    if set(calibration["base_model"].astype(str)) != set(MODEL_KEYS):
        raise RuntimeError("calibration model set differs")
    counts = calibration.groupby("base_model").agg(
        traces=("trace_id", "nunique"), checkpoints=("checkpoint_id", "size")
    )
    if not counts["traces"].eq(int(expected["traces_per_model"])).all():
        raise RuntimeError("calibration trace count differs")
    if not counts["checkpoints"].eq(int(expected["checkpoints_per_model"])).all():
        raise RuntimeError("calibration checkpoint count differs")
    if len(calibration) != int(expected["checkpoints_total"]):
        raise RuntimeError("calibration total checkpoint count differs")
    if calibration["common_trace_id"].nunique() != int(expected["shared_traces"]):
        raise RuntimeError("shared calibration trace count differs")
    if not calibration["checkpoint_validity_status"].eq("included_production").all():
        raise RuntimeError("invalid checkpoint entered threshold cohort")
    if not calibration["num_rollouts"].eq(4).all():
        raise RuntimeError("original K=4 checkpoint aggregate differs")

    linear_seeds, linear_seed_table = select_median_dev_seeds(
        boundary, architecture="linear_probe", learning_rate=1e-3
    )
    position_seeds, position_seed_table = select_median_dev_seeds(
        boundary, architecture="position_only", learning_rate=1e-3
    )
    if linear_seeds != {
        str(key): int(value) for key, value in source["expected_operational_seeds"].items()
    }:
        raise RuntimeError("median-dev operational seed identity differs")
    operational_hashes = {
        str(row["base_model"]): str(row["checkpoint_sha256"])
        for row in linear_seed_table[linear_seed_table["operational"]].to_dict("records")
    }
    if operational_hashes != {
        str(key): str(value)
        for key, value in source["expected_operational_weight_sha256"].items()
    }:
        raise RuntimeError("operational probe weight hash differs")
    seed_table = pd.concat([linear_seed_table, position_seed_table], ignore_index=True)
    atomic_parquet(root / "manifests/model_seed_manifest.parquet", seed_table)
    atomic_json(
        root / "manifests/operational_seeds.json",
        {
            "architecture": "linear_probe",
            "learning_rate": 0.001,
            "selection_split": "architecture_dev",
            "selection_rule": "median_per_model_dev_nll",
            "operational_seeds": linear_seeds,
            "position_only_analysis_seeds": position_seeds,
            "test_used": False,
            "native_used": False,
        },
    )

    prediction_frames: list[pd.DataFrame] = []
    identity_columns = [
        "base_model", "trace_id", "common_trace_id", "problem_id", "problem_group",
        "domain", "split", "checkpoint_id", "checkpoint_ordinal",
        "checkpoint_token_offset", "prefix_token_count", "total_trace_token_count",
        "total_checkpoint_count", "success_count", "num_rollouts", "observed_success_rate",
    ]
    for model_key in MODEL_KEYS:
        model_manifest = calibration[calibration["base_model"].eq(model_key)].sort_values(
            ["trace_id", "checkpoint_ordinal"]
        )
        expected_keys = set(model_manifest["checkpoint_id"].astype(str))
        for seed in TRAINING_SEEDS:
            path = boundary / f"calibration/{model_key}/seed_{seed}/calibration_predictions.parquet"
            frame = pd.read_parquet(path)
            if set(frame["checkpoint_id"].astype(str)) != expected_keys or set(frame["split"].astype(str)) != {"calibration"}:
                raise RuntimeError(f"{model_key}/seed_{seed}: calibration raw-logit identity differs")
            prediction_frames.append(
                frame[identity_columns + ["raw_logit"]].assign(
                    training_seed=seed, architecture="linear_probe", learning_rate=0.001
                )
            )
        for seed in TRAINING_SEEDS:
            checkpoint = boundary / f"training/{model_key}/position_only/lr_1e-03/seed_{seed}/best.pt"
            frame = model_manifest[identity_columns].copy()
            frame["raw_logit"] = _score_position_only(checkpoint, frame)
            frame["training_seed"] = seed
            frame["architecture"] = "position_only"
            frame["learning_rate"] = 0.001
            prediction_frames.append(frame)
    raw_predictions = pd.concat(prediction_frames, ignore_index=True)
    if len(raw_predictions) != len(calibration) * 6:
        raise RuntimeError("frozen raw-logit matrix count differs")
    atomic_parquet(root / "manifests/frozen_raw_logits.parquet", raw_predictions)

    common = calibration[["common_trace_id", "problem_group", "domain"]].drop_duplicates()
    if common["common_trace_id"].nunique() != len(common):
        raise RuntimeError("common trace metadata differs across base models")
    folds = assign_crossfit_folds(
        common,
        folds=int(config["cross_fit"]["folds"]),
        seed=int(config["cross_fit"]["assignment_seed"]),
    )
    atomic_parquet(root / "manifests/five_fold_assignment.parquet", folds)

    manifest_columns = [
        "base_model", "model_id", "model_revision", "tokenizer_revision", "trace_id",
        "common_trace_id", "source_trace_id", "problem_id", "problem_group",
        "canonical_dataset_id", "dataset", "domain", "split", "checkpoint_id",
        "checkpoint_ordinal", "checkpoint_token_offset", "prefix_token_count",
        "total_trace_token_count", "total_checkpoint_count", "hidden_state_location",
        "hidden_state_source", "hidden_state_layer", "hidden_state_dimension",
        "success_count", "num_rollouts", "observed_success_rate",
        "checkpoint_validity_status", "feature_row_index",
    ]
    checkpoint_manifest = calibration[manifest_columns].sort_values(
        ["base_model", "trace_id", "checkpoint_ordinal"]
    ).reset_index(drop=True)
    for model_key in MODEL_KEYS:
        part = checkpoint_manifest[checkpoint_manifest["base_model"].eq(model_key)]
        model_config = config["models"][model_key]
        expected_identity = {
            "model_id": str(model_config["hf_model_id"]),
            "model_revision": str(model_config["revision"]),
            "tokenizer_revision": str(model_config["tokenizer_revision"]),
        }
        for column, expected_value in expected_identity.items():
            if set(part[column].astype(str)) != {expected_value}:
                raise RuntimeError(f"{model_key}: frozen {column} differs")
    atomic_parquet(root / "manifests/checkpoint_manifest.parquet", checkpoint_manifest)
    trace_census = checkpoint_manifest.drop_duplicates(["base_model", "trace_id"])[
        ["base_model", "trace_id", "common_trace_id", "problem_id", "problem_group", "domain"]
    ].merge(folds, on=["common_trace_id", "problem_group", "domain"], validate="many_to_one")
    atomic_parquet(root / "manifests/calibration_trace_manifest.parquet", trace_census)

    for model_key in MODEL_KEYS:
        model_checkpoints = checkpoint_manifest[checkpoint_manifest["base_model"].eq(model_key)]
        trace_ids = set(model_checkpoints["trace_id"].astype(str))
        source_rows = _load_selected_trace_rows(completion, model_key, trace_ids)
        checkpoint_offsets = {
            trace_id: list(part.sort_values("checkpoint_ordinal")["checkpoint_token_offset"].astype(int))
            for trace_id, part in model_checkpoints.groupby("trace_id", sort=False)
        }
        for row in source_rows:
            if list(map(int, row["eligible_checkpoint_offsets"])) != checkpoint_offsets[str(row["trace_id"])]:
                raise RuntimeError(f"{model_key}/{row['trace_id']}: checkpoint offsets drifted")
            if str(row.get("model_revision")) != str(model_checkpoints["model_revision"].iloc[0]):
                raise RuntimeError(f"{model_key}: source model revision differs")
        atomic_jsonl(root / f"manifests/source_traces/{model_key}.jsonl", source_rows)

    original_k4, raw_accessed = _original_k4_rows(
        checkpoint_manifest, completion_mount=completion_mount
    )
    atomic_parquet(root / "raw_outcomes/original_k4_checkpoint_suffixes.parquet", original_k4)
    assignments, packs = _build_pack_assignments(
        checkpoint_manifest, traces_per_pack=int(config["execution"]["traces_per_pack"])
    )
    checkpoint_seed_manifest, regeneration_seed_manifest = _seed_manifests(
        checkpoint_manifest, assignments, original_k4
    )
    atomic_jsonl(root / "manifests/execution_packs.jsonl", packs)
    atomic_parquet(root / "manifests/dense_checkpoint_rollout_manifest.parquet", checkpoint_seed_manifest)
    atomic_parquet(root / "manifests/full_regeneration_manifest.parquet", regeneration_seed_manifest)

    exclusions = pd.read_parquet(boundary / "data/excluded_artifacts.parquet")
    atomic_parquet(root / "manifests/exclusion_manifest.parquet", exclusions)
    exclusion_reasons = (
        exclusions.groupby("exclusion_reason").size().astype(int).to_dict()
        if "exclusion_reason" in exclusions.columns
        else {"excluded_source_artifact": len(exclusions)}
    )
    checkpoint_distribution = (
        checkpoint_manifest[checkpoint_manifest["base_model"].eq(MODEL_KEYS[0])]
        .groupby("trace_id").size().value_counts().sort_index().astype(int).to_dict()
    )
    domain_distribution = (
        trace_census[trace_census["base_model"].eq(MODEL_KEYS[0])]
        .groupby("domain").size().astype(int).to_dict()
    )
    operational_rows = linear_seed_table[linear_seed_table["operational"]].to_dict("records")
    census = {
        "status": "PASS",
        "prepared_at": now_iso(),
        "shared_traces": int(checkpoint_manifest["common_trace_id"].nunique()),
        "model_traces": int(trace_census.shape[0]),
        "traces_per_model": int(trace_census.groupby("base_model").size().iloc[0]),
        "problem_groups": int(checkpoint_manifest["problem_group"].nunique()),
        "checkpoints_total": len(checkpoint_manifest),
        "checkpoints_per_model": int(counts["checkpoints"].iloc[0]),
        "checkpoint_count_distribution_per_model": {str(k): v for k, v in checkpoint_distribution.items()},
        "domain_distribution_per_model": domain_distribution,
        "existing_rollouts": len(original_k4),
        "new_checkpoint_rollouts": len(checkpoint_seed_manifest),
        "full_regenerations": len(regeneration_seed_manifest),
        "excluded_artifacts": len(exclusions),
        "exclusion_reasons": exclusion_reasons,
        "canonical_manifest_sha256": canonical_sha,
        "calibration_split_sha256": split_sha,
        "boundary_manifest_hashes": {
            "canonical_checkpoint_manifest_sha256": boundary_hashes[
                "canonical_checkpoint_manifest_sha256"
            ],
            "common_trace_manifest_sha256": boundary_hashes[
                "common_trace_manifest_sha256"
            ],
            "config_sha256": boundary_hashes["config_sha256"],
            "excluded_artifacts_sha256": boundary_hashes["excluded_artifacts_sha256"],
            "calibration_split_sha256": boundary_hashes["split_manifest_sha256"][
                "calibration"
            ],
            "feature_store_sha256": boundary_hashes["feature_store_sha256"],
        },
        "operational_seeds": linear_seeds,
        "position_only_seeds": position_seeds,
        "operational_seed_rows": operational_rows,
        "model_revisions": checkpoint_manifest.groupby("base_model")["model_revision"].first().to_dict(),
        "tokenizer_revisions": checkpoint_manifest.groupby("base_model")["tokenizer_revision"].first().to_dict(),
        "native_accessed": False,
        "teacher_forced_test_accessed": False,
        "geometry_test_accessed": False,
    }
    if census["new_checkpoint_rollouts"] != int(expected["added_checkpoint_rollouts_total"]):
        raise RuntimeError("added checkpoint workload count differs")
    if census["full_regenerations"] != int(expected["full_regenerations_total"]):
        raise RuntimeError("full-regeneration workload count differs")
    atomic_json(root / "manifests/prelaunch_census.json", census)
    atomic_json(
        root / "manifests/source_access_ledger.json",
        {
            "status": "PASS",
            "opened_roles": [
                "boundary_calibration_predicate_only",
                "boundary_calibration_predictions",
                "teacher_forced_repairability_trace_manifest_filtered_by_calibration_ids",
                "teacher_forced_calibration_k4_raw_rollout_packs",
            ],
            "raw_rollout_predicate_reads": raw_accessed,
            "raw_rollout_files_opened": [row["path"] for row in raw_accessed],
            "noncalibration_outcome_rows_returned": int(
                sum(row["noncalibration_rows_returned"] for row in raw_accessed)
            ),
            "native_paths_opened": [],
            "teacher_forced_test_rows_returned": 0,
            "geometry_paths_opened": [],
        },
    )
    atomic_json(
        root / "manifests/frozen_protocol.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "FROZEN_BEFORE_OUTCOME_GENERATION",
            "configuration_hash": stable_hash(config),
            "threshold_grid": list(config["threshold_selection"]["thresholds"]),
            "full_regeneration_anchor": 1.0,
            "operational_seeds": linear_seeds,
            "fold_assignment_sha256": sha256_file(root / "manifests/five_fold_assignment.parquet"),
            "original_k4_copy_sha256": sha256_file(root / "raw_outcomes/original_k4_checkpoint_suffixes.parquet"),
            "checkpoint_workload_sha256": sha256_file(root / "manifests/dense_checkpoint_rollout_manifest.parquet"),
            "full_regeneration_workload_sha256": sha256_file(root / "manifests/full_regeneration_manifest.parquet"),
            "native_access_allowed": False,
            "teacher_forced_test_access_allowed": False,
            "threshold_refinement_allowed": False,
        },
    )
    atomic_text(root / "reports/PRELAUNCH_CENSUS_AND_INTEGRITY.md", _prelaunch_report(census))
    atomic_json(
        root / "READY.json",
        {
            "status": "READY_FOR_SMOKE",
            "prepared_at": now_iso(),
            "configuration_hash": stable_hash(config),
            "census": census,
        },
    )
    return census
