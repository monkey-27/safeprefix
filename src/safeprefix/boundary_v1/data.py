"""Validate and compact the production teacher-forced recoverability corpus."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
from typing import Any, Iterable, Mapping

import pandas as pd
import torch
import yaml


REQUIRED_CHECKPOINT_COLUMNS = {
    "model_key",
    "trace_id",
    "checkpoint_index",
    "problem_id",
    "source_bucket",
    "checkpoint_token_offset",
    "success_count",
    "trial_count",
    "repairability_discretized",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows))
    temporary.replace(path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(parts: Iterable[Any]) -> str:
    return hashlib.sha256(
        json.dumps(list(parts), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"configuration is not a mapping: {path}")
    return payload


def _normalized_problem(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _assert_teacher_forced_sources(paths: Iterable[Path], source: Mapping[str, Any]) -> None:
    forbidden = {"native_eval", "native-eval", "native_failed", "native-failed"}
    for path in paths:
        lowered = path.as_posix().lower()
        if any(token in lowered for token in forbidden):
            raise RuntimeError(f"native-evaluation path is forbidden: {path}")
    if source.get("completion_run_id") != "safeprefix_teacher_forced_completion_20260727_r5":
        raise RuntimeError("unexpected completion run identity")
    if source.get("original_run_id") != "safeprefix_full_teacher_forced_20260726_r3":
        raise RuntimeError("unexpected original teacher-forced run identity")


def _check_source_access_ledger(manifest_root: Path) -> dict[str, Any]:
    candidates = [
        manifest_root.parent / "source_access_ledger.json",
        manifest_root / "source_access_ledger.json",
    ]
    ledger_path = next((path for path in candidates if path.is_file()), None)
    if ledger_path is None:
        raise FileNotFoundError("teacher-forced source access ledger is missing")
    ledger = json.loads(ledger_path.read_text())
    prohibited = {
        "native_development_outputs_opened": ledger.get("native_development_outputs_opened"),
        "final_test_outputs_opened": ledger.get("final_test_outputs_opened"),
    }
    if any(prohibited.values()):
        raise RuntimeError(f"prohibited source access recorded: {prohibited}")
    return {"path": str(ledger_path), "sha256": sha256_file(ledger_path), **ledger}


def _grouped_split_assignment(
    common_rows: list[dict[str, Any]], *, seed: int
) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in common_rows:
        group = str(row.get("production_problem_group") or row["problem_group_hash"])
        by_group[group].append(row)

    # The production grouping is the frozen near-duplicate unit. Exact normalized
    # duplicates are additionally unioned before the held-out split is divided.
    normalized_to_groups: dict[str, set[str]] = defaultdict(set)
    for group, rows in by_group.items():
        normalized_to_groups[_normalized_problem(rows[0]["problem_text"])].add(group)
    parent = {group: group for group in by_group}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for groups in normalized_to_groups.values():
        ordered = sorted(groups)
        for other in ordered[1:]:
            union(ordered[0], other)

    clusters: dict[str, set[str]] = defaultdict(set)
    for group in by_group:
        clusters[find(group)].add(group)
    cluster_rows: dict[str, list[dict[str, Any]]] = {
        cluster: [row for group in groups for row in by_group[group]]
        for cluster, groups in clusters.items()
    }
    for cluster, rows in cluster_rows.items():
        frozen = {str(row["pipeline_split"]) for row in rows}
        if len(frozen) != 1:
            raise RuntimeError(f"duplicate cluster crosses frozen train/heldout: {cluster}")

    assignments: dict[str, str] = {}
    heldout_by_stratum: dict[str, list[str]] = defaultdict(list)
    for cluster, rows in cluster_rows.items():
        frozen = str(rows[0]["pipeline_split"])
        if frozen == "train":
            for group in clusters[cluster]:
                assignments[group] = "train"
        elif frozen == "dev":
            stratum = "+".join(sorted({str(row["source_bucket"]) for row in rows}))
            heldout_by_stratum[stratum].append(cluster)
        else:
            raise RuntimeError(f"unexpected frozen split {frozen!r}")

    allocation: dict[str, dict[str, int]] = {}
    for stratum, values in sorted(heldout_by_stratum.items()):
        ordered = sorted(values, key=lambda value: stable_hash([seed, stratum, value]))
        count = len(ordered)
        dev_count = int(count * 0.40)
        calibration_count = int(count * 0.30)
        if count >= 3:
            dev_count = max(1, dev_count)
            calibration_count = max(1, calibration_count)
        test_start = min(count, dev_count + calibration_count)
        partitions = {
            "architecture_dev": ordered[:dev_count],
            "calibration": ordered[dev_count:test_start],
            "teacher_forced_test": ordered[test_start:],
        }
        for split, cluster_ids in partitions.items():
            for cluster in cluster_ids:
                for group in clusters[cluster]:
                    assignments[group] = split
        allocation[stratum] = {split: len(cluster_ids) for split, cluster_ids in partitions.items()}

    manifest_rows: list[dict[str, Any]] = []
    for group, rows in sorted(by_group.items()):
        manifest_rows.append(
            {
                "problem_group": group,
                "split": assignments[group],
                "frozen_pipeline_split": rows[0]["pipeline_split"],
                "source_buckets": sorted({str(row["source_bucket"]) for row in rows}),
                "problem_ids": sorted({str(row["problem_id"]) for row in rows}),
                "trace_count": len(rows),
                "normalized_problem_sha256": hashlib.sha256(
                    _normalized_problem(rows[0]["problem_text"]).encode()
                ).hexdigest(),
            }
        )
    summary = {
        "seed": int(seed),
        "precedence_rule": "frozen train plus deterministic 40/30/30 subdivision of frozen heldout",
        "grouping": "frozen production_problem_group plus exact normalized-text union",
        "allocation_by_stratum": allocation,
        "problem_group_counts": dict(Counter(assignments.values())),
    }
    return assignments, manifest_rows, summary


def _validate_pack_marker(
    feature_path: Path, *, expected_revision: str, expected_layers: list[int]
) -> tuple[dict[str, Any], str]:
    marker_path = feature_path.with_name("complete.json")
    rollout_path = feature_path.with_name("rollouts.parquet")
    if not marker_path.is_file() or not rollout_path.is_file():
        raise RuntimeError(f"incomplete production pack beside {feature_path}")
    marker = json.loads(marker_path.read_text())
    if marker.get("model_revision") not in {None, expected_revision}:
        raise RuntimeError(f"model revision mismatch in {marker_path}")
    if marker.get("features_sha256") != sha256_file(feature_path):
        raise RuntimeError(f"feature checksum mismatch: {feature_path}")
    if marker.get("rollouts_sha256") not in {None, sha256_file(rollout_path)}:
        raise RuntimeError(f"rollout checksum mismatch: {rollout_path}")
    if marker.get("feature_integrity", {}).get("passed") is False:
        raise RuntimeError(f"feature integrity failed: {marker_path}")
    if marker.get("integrity", {}).get("passed") is False:
        raise RuntimeError(f"rollout integrity failed: {marker_path}")
    return marker, sha256_file(marker_path)


def _feature_payloads(
    roots: list[Path], *, expected_revision: str, expected_layers: list[int]
) -> tuple[dict[str, tuple[torch.Tensor, list[int], str]], list[dict[str, Any]]]:
    payloads: dict[str, tuple[torch.Tensor, list[int], str]] = {}
    pack_records: list[dict[str, Any]] = []
    for root in roots:
        for path in sorted(root.glob("*/checkpoint_features.pt")):
            marker, marker_hash = _validate_pack_marker(
                path,
                expected_revision=expected_revision,
                expected_layers=expected_layers,
            )
            pack = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(pack, dict):
                raise TypeError(f"feature pack is not keyed by trace: {path}")
            for trace_id, value in pack.items():
                if trace_id in payloads:
                    raise RuntimeError(f"duplicate production feature trace: {trace_id}")
                if list(map(int, value["selected_layers"])) != expected_layers:
                    raise RuntimeError(f"selected layers differ for {trace_id}")
                if str(value["model_revision"]) != expected_revision:
                    raise RuntimeError(f"feature revision differs for {trace_id}")
                payloads[str(trace_id)] = (
                    value["features"],
                    list(map(int, value["checkpoint_offsets"])),
                    str(path),
                )
            pack_records.append(
                {
                    "pack_id": path.parent.name,
                    "path": str(path.parent),
                    "trace_count": len(pack),
                    "marker_sha256": marker_hash,
                    "mode": marker.get("mode", "production"),
                }
            )
    return payloads, pack_records


def _extract_final_layer(features: torch.Tensor, hidden_size: int) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError("checkpoint feature tensor must be rank two")
    auxiliary = features.shape[1] - 9 * hidden_size
    if auxiliary not in {3, 4}:
        raise ValueError(
            f"unexpected feature layout {features.shape[1]} for hidden size {hidden_size}"
        )
    # The frozen source layout is [current(3 layers), delta(3 layers),
    # terminal-summary(3 layers), position, optional NLL]. The requested input
    # is only the final layer within the current checkpoint block.
    return features[:, 2 * hidden_size : 3 * hidden_size].to(torch.float16).contiguous()


def _write_split_manifests(root: Path, rows: list[dict[str, Any]]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for split in ("train", "architecture_dev", "calibration", "teacher_forced_test"):
        path = root / f"{split}_problems.jsonl"
        atomic_jsonl(path, [row for row in rows if row["split"] == split])
        hashes[split] = sha256_file(path)
    return hashes


def prepare_boundary_dataset(
    *,
    config_path: Path,
    old_root: Path,
    completion_root: Path,
    manifest_root: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    config = load_config(config_path)
    ready_path = artifact_root / "data/READY.json"
    if ready_path.is_file():
        return json.loads(ready_path.read_text())
    if artifact_root.exists() and any(artifact_root.iterdir()):
        raise FileExistsError(
            f"partial boundary artifact root exists without READY marker: {artifact_root}"
        )
    _assert_teacher_forced_sources(
        [old_root, completion_root, manifest_root], config["source"]
    )
    access_ledger = _check_source_access_ledger(manifest_root)
    common_path = manifest_root / "repairability/common_trace_manifest.jsonl"
    common_rows = read_jsonl(common_path)
    if len(common_rows) != 2614:
        raise RuntimeError(f"unexpected common teacher-forced trace count: {len(common_rows)}")
    assignments, split_rows, split_summary = _grouped_split_assignment(
        common_rows, seed=int(config["experiment"]["split_seed"])
    )
    split_hashes = _write_split_manifests(artifact_root / "data/splits", split_rows)

    common_by_source = {str(row["source_trace_id"]): row for row in common_rows}
    canonical_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    integrity_models: dict[str, Any] = {}
    census_models: dict[str, Any] = {}
    feature_hashes: dict[str, str] = {}
    model_configs = config["source"]["expected_models"]

    for model_key, expected in model_configs.items():
        trace_manifest_path = (
            manifest_root / f"repairability/per_model/{model_key}/trace_manifest.jsonl"
        )
        trace_rows = read_jsonl(trace_manifest_path)
        trace_by_id = {str(row["trace_id"]): row for row in trace_rows}
        if len(trace_by_id) != 2614:
            raise RuntimeError(f"{model_key}: trace manifest is not the complete 2,614")
        for row in trace_rows:
            common = common_by_source[str(row["source_trace_id"])]
            if str(row["problem_group_hash"]) != str(common["problem_group_hash"]):
                raise RuntimeError(f"{model_key}: common trace identity mismatch")
            if str(row["model_revision"]) != str(expected["model_revision"]):
                raise RuntimeError(f"{model_key}: manifest revision mismatch")

        outcome_path = completion_root / f"repairability/aggregated_checkpoint_outcomes/{model_key}.parquet"
        outcomes = pd.read_parquet(outcome_path)
        missing_columns = REQUIRED_CHECKPOINT_COLUMNS - set(outcomes.columns)
        if missing_columns:
            raise RuntimeError(f"{model_key}: missing outcome columns {sorted(missing_columns)}")
        if outcomes.duplicated(["trace_id", "checkpoint_index"]).any():
            raise RuntimeError(f"{model_key}: duplicated aggregated checkpoints")
        if len(outcomes) != 10988 or set(outcomes["trial_count"].astype(int)) != {4}:
            raise RuntimeError(f"{model_key}: incomplete k=4 checkpoint outcomes")
        if outcomes["repairability_discretized"].astype(bool).any():
            raise RuntimeError(f"{model_key}: hard repairability labels entered production outcomes")
        outcomes_by_trace = {
            str(trace_id): frame.sort_values("checkpoint_index")
            for trace_id, frame in outcomes.groupby("trace_id", sort=False)
        }

        old_feature_root = old_root / f"raw_rollout_shards/{model_key}"
        extension_feature_root = completion_root / f"repairability/raw_rollout_shards/{model_key}"
        features_by_trace, pack_records = _feature_payloads(
            [old_feature_root, extension_feature_root],
            expected_revision=str(expected["model_revision"]),
            expected_layers=list(map(int, expected["selected_layers"])),
        )
        if set(features_by_trace) != set(trace_by_id):
            raise RuntimeError(
                f"{model_key}: feature trace coverage differs; "
                f"missing={len(set(trace_by_id)-set(features_by_trace))} "
                f"extra={len(set(features_by_trace)-set(trace_by_id))}"
            )
        if set(outcomes_by_trace) != set(trace_by_id):
            raise RuntimeError(f"{model_key}: outcome trace coverage differs")

        model_feature_rows: list[torch.Tensor] = []
        row_index = 0
        for trace_id in sorted(trace_by_id):
            trace = trace_by_id[trace_id]
            outcome = outcomes_by_trace[trace_id]
            source_trace_id = str(trace["source_trace_id"])
            common = common_by_source[source_trace_id]
            group = str(common.get("production_problem_group") or common["problem_group_hash"])
            raw_features, offsets, source_path = features_by_trace[trace_id]
            expected_offsets = list(map(int, trace["eligible_checkpoint_offsets"]))
            if offsets != expected_offsets:
                raise RuntimeError(f"{model_key}/{trace_id}: feature offsets differ")
            if list(outcome["checkpoint_token_offset"].astype(int)) != expected_offsets:
                raise RuntimeError(f"{model_key}/{trace_id}: outcome offsets differ")
            if list(outcome["checkpoint_index"].astype(int)) != list(range(len(offsets))):
                raise RuntimeError(f"{model_key}/{trace_id}: checkpoint ordinals differ")
            selected = _extract_final_layer(raw_features, int(expected["hidden_size"]))
            if len(selected) != len(outcome):
                raise RuntimeError(f"{model_key}/{trace_id}: feature/outcome length differs")
            model_feature_rows.append(selected)
            total_tokens = int(trace["full_token_count"])
            total_checkpoints = len(offsets)
            for local_index, (_, label) in enumerate(outcome.iterrows()):
                success_count = int(label["success_count"])
                trial_count = int(label["trial_count"])
                canonical_rows.append(
                    {
                        "base_model": model_key,
                        "model_id": expected["model_id"],
                        "model_revision": expected["model_revision"],
                        "tokenizer_revision": expected["tokenizer_revision"],
                        "trace_id": trace_id,
                        "common_trace_id": trace["common_trace_id"],
                        "source_trace_id": source_trace_id,
                        "problem_id": trace["problem_id"],
                        "problem_group": group,
                        "canonical_dataset_id": f"{trace['source_dataset']}:{trace['source_subset']}:{trace['problem_id']}",
                        "dataset": trace["source_dataset"],
                        "domain": trace["source_bucket"],
                        "split": assignments[group],
                        "frozen_pipeline_split": trace["pipeline_split"],
                        "checkpoint_id": f"{trace_id}:{local_index}",
                        "checkpoint_ordinal": local_index,
                        "checkpoint_token_offset": offsets[local_index],
                        "prefix_token_count": offsets[local_index],
                        "total_trace_token_count": total_tokens,
                        "total_checkpoint_count": total_checkpoints,
                        "hidden_state_location": f"data/features/{model_key}.pt:{row_index + local_index}",
                        "hidden_state_source": source_path,
                        "hidden_state_layer": -1,
                        "hidden_state_dimension": int(expected["hidden_size"]),
                        "success_count": success_count,
                        "num_rollouts": trial_count,
                        "observed_success_rate": success_count / trial_count,
                        "initial_trace_verifier_result": False,
                        "checkpoint_validity_status": "included_production",
                        "exclusion_reason": None,
                        "first_error_zero_based_analysis_only": int(trace["first_error_zero_based"]),
                        "feature_row_index": row_index + local_index,
                    }
                )
            row_index += len(selected)
        compact = torch.cat(model_feature_rows, dim=0)
        if compact.shape != (10988, int(expected["hidden_size"])):
            raise RuntimeError(f"{model_key}: compact feature shape differs: {tuple(compact.shape)}")
        feature_path = artifact_root / f"data/features/{model_key}.pt"
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = feature_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "model_key": model_key,
                "model_id": expected["model_id"],
                "model_revision": expected["model_revision"],
                "tokenizer_revision": expected["tokenizer_revision"],
                "representation": config["representation"],
                "features": compact,
            },
            temporary,
        )
        temporary.replace(feature_path)
        feature_hashes[model_key] = sha256_file(feature_path)

        old_smokes = list((old_root / f"validation/smoke/{model_key}").glob("*/complete.json"))
        completion_base = manifest_root.parent
        completion_smokes = list(
            (completion_base / f"smoke_tests/real_model/{model_key}").glob("**/complete.json")
        ) if completion_base.exists() else []
        for path in [*old_smokes, *completion_smokes]:
            excluded_rows.append(
                {
                    "base_model": model_key,
                    "artifact_path": str(path.parent),
                    "checkpoint_validity_status": "excluded_nonproduction",
                    "exclusion_reason": "smoke_or_validation_artifact",
                }
            )
        success_distribution = {
            str(key): int(value)
            for key, value in outcomes["success_count"].value_counts().sort_index().items()
        }
        census_models[model_key] = {
            "traces": len(trace_rows),
            "unique_problems": len({row["problem_group_hash"] for row in trace_rows}),
            "checkpoints": len(outcomes),
            "rollouts": int(outcomes["trial_count"].sum()),
            "success_count_distribution": success_distribution,
            "domain_distribution_traces": dict(Counter(row["source_bucket"] for row in trace_rows)),
            "domain_distribution_checkpoints": {
                str(key): int(value) for key, value in outcomes["source_bucket"].value_counts().items()
            },
            "checkpoint_count_per_trace": {
                "minimum": min(len(row["eligible_checkpoint_offsets"]) for row in trace_rows),
                "maximum": max(len(row["eligible_checkpoint_offsets"]) for row in trace_rows),
                "mean": sum(len(row["eligible_checkpoint_offsets"]) for row in trace_rows) / len(trace_rows),
            },
            "model_revision": expected["model_revision"],
            "tokenizer_revision": expected["tokenizer_revision"],
            "production_pack_count": len(pack_records),
            "excluded_nonproduction_pack_count": len(old_smokes) + len(completion_smokes),
        }
        integrity_models[model_key] = {
            "status": "PASS",
            "trace_manifest_sha256": sha256_file(trace_manifest_path),
            "outcome_manifest_sha256": sha256_file(outcome_path),
            "feature_store_sha256": feature_hashes[model_key],
            "feature_shape": list(compact.shape),
            "feature_dtype": str(compact.dtype),
            "production_pack_markers": len(pack_records),
            "missing_or_corrupted_artifacts": 0,
            "duplicate_checkpoints": 0,
            "invalid_rollout_counts": 0,
        }

    canonical = pd.DataFrame(canonical_rows).sort_values(
        ["base_model", "trace_id", "checkpoint_ordinal"]
    )
    canonical_path = artifact_root / "data/canonical_checkpoint_manifest.parquet"
    atomic_parquet(canonical_path, canonical)
    exclusions_path = artifact_root / "data/excluded_artifacts.parquet"
    atomic_parquet(exclusions_path, pd.DataFrame(excluded_rows))
    census = {
        "status": "PASS",
        "production_only": True,
        "models": census_models,
        "total_model_traces": sum(row["traces"] for row in census_models.values()),
        "total_checkpoints": len(canonical),
        "total_rollouts": int(canonical["num_rollouts"].sum()),
        "shared_teacher_forced_traces": len(common_rows),
        "split_checkpoint_counts": {
            str(key): int(value) for key, value in canonical["split"].value_counts().items()
        },
        "split_trace_counts_per_model": {
            model: {
                str(key): int(value)
                for key, value in frame.drop_duplicates("trace_id")["split"].value_counts().items()
            }
            for model, frame in canonical.groupby("base_model")
        },
        "excluded_artifacts": len(excluded_rows),
    }
    atomic_json(artifact_root / "data/dataset_census.json", census)
    integrity = {
        "status": "PASS",
        "models": integrity_models,
        "source_access_ledger": access_ledger,
        "split_integrity": split_summary,
        "problem_split_overlap": 0,
        "native_evaluation_rows": 0,
        "first_error_used_as_target": False,
        "success_count_used_as_input": False,
        "future_checkpoint_features_used": False,
    }
    atomic_json(artifact_root / "data/hidden_state_integrity.json", integrity)
    hashes = {
        "config_sha256": sha256_file(config_path),
        "common_trace_manifest_sha256": sha256_file(common_path),
        "canonical_checkpoint_manifest_sha256": sha256_file(canonical_path),
        "excluded_artifacts_sha256": sha256_file(exclusions_path),
        "split_manifest_sha256": split_hashes,
        "feature_store_sha256": feature_hashes,
    }
    atomic_json(artifact_root / "data/manifest_hashes.json", hashes)
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "repository_commit": os.environ.get("SAFEPREFIX_SOURCE_COMMIT")
        or subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip(),
    }
    atomic_json(artifact_root / "data/environment.json", environment)
    ready = {
        "status": "READY",
        "census": census,
        "hashes": hashes,
        "representation": config["representation"],
        "native_evaluation_used": False,
        "final_tau_selected": False,
    }
    atomic_json(ready_path, ready)
    return ready
