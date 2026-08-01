"""Calibration-corpus preparation for the K-densification robustness study.

This module is deliberately split at the outcome-access boundary.  The K=32
confirmation cohort is frozen from checkpoint metadata before verifier success
values or predictor scores are loaded.  The completed K=16 corpus is then
validated and the missing frozen predictor logits are recomputed from saved
hidden states only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch

from safeprefix.boundary_v1.training import ModelCorpus, load_trained_predictor, predict_split
from safeprefix.k_densification_v1.preflight import _validate_boundary_artifacts
from safeprefix.reproducibility import (
    atomic_json,
    atomic_jsonl,
    atomic_parquet,
    now_iso,
    stable_hash,
    stable_seed,
)


MODEL_KEYS = (
    "family_a_small",
    "family_a_large",
    "family_b_small",
    "family_b_large",
)
ARCHITECTURES = (
    "position_only",
    "linear_probe",
    "local_mlp",
    "change_aware_mlp",
    "causal_gru",
)
CHECKPOINT_KEY_COLUMNS = (
    "model_key",
    "trace_id",
    "checkpoint_ordinal",
    "checkpoint_token_offset",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_non_native_paths(paths: Iterable[Path]) -> None:
    for path in paths:
        lowered = path.as_posix().lower()
        if "native_eval" in lowered or "native-eval" in lowered or "native_failed" in lowered:
            raise RuntimeError(f"native artifact path is forbidden: {path}")


def _verify_threshold_bundle(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    _assert_non_native_paths([root])
    expected = config["source"]["threshold_expected_hashes"]
    paths = {
        "complete_sha256": root / "COMPLETE.json",
        "final_integrity_sha256": root / "integrity/final_integrity.json",
        "checkpoint_manifest_sha256": root / "manifests/checkpoint_manifest.parquet",
        "frozen_raw_logits_sha256": root / "manifests/frozen_raw_logits.parquet",
        "merged_k16_outcomes_sha256": root / "raw_outcomes/merged_k16_checkpoint_suffixes.parquet",
    }
    observed: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed[name] = sha256_file(path)
        if observed[name] != str(expected[name]):
            raise RuntimeError(f"threshold source hash differs: {name}")
    for model_key in MODEL_KEYS:
        path = root / f"manifests/source_traces/{model_key}.jsonl"
        observed[f"source_trace_sha256/{model_key}"] = sha256_file(path)
        if observed[f"source_trace_sha256/{model_key}"] != str(
            expected["source_trace_sha256"][model_key]
        ):
            raise RuntimeError(f"{model_key}: threshold source-trace hash differs")

    complete = json.loads(paths["complete_sha256"].read_text())
    integrity = json.loads(paths["final_integrity_sha256"].read_text())
    if complete.get("status") != "COMPLETE" or complete.get("integrity_status") != "PASS":
        raise RuntimeError("threshold experiment is not COMPLETE/PASS")
    if complete.get("run_id") != config["source"]["threshold_run_id"]:
        raise RuntimeError("threshold run identity differs")
    if any(
        complete.get(field) is not False
        for field in ("native_evaluation_used", "teacher_forced_test_used", "geometry_test_used")
    ):
        raise RuntimeError("threshold terminal artifact crossed a prohibited data boundary")
    if integrity.get("status") != "PASS" or integrity.get("failures"):
        raise RuntimeError("threshold final integrity is not PASS")

    artifact_hashes = json.loads((root / "artifact_hashes.json").read_text())
    failures: list[str] = []
    for row in artifact_hashes["files"]:
        path = root / str(row["path"])
        if not path.is_file():
            failures.append(f"missing:{row['path']}")
            continue
        if path.stat().st_size != int(row["size"]) or sha256_file(path) != str(row["sha256"]):
            failures.append(f"mismatch:{row['path']}")
    if failures or int(artifact_hashes["count"]) != len(artifact_hashes["files"]):
        raise RuntimeError(f"threshold artifact manifest failed: {failures[:10]}")
    return {
        "status": "PASS",
        "complete": complete,
        "final_integrity": integrity,
        "verified_artifact_count": len(artifact_hashes["files"]),
        "source_sha256": observed,
        "native_outcomes_loaded": False,
        "teacher_forced_test_loaded": False,
    }


def _load_checkpoint_manifest(config: Mapping[str, Any], root: Path) -> pd.DataFrame:
    frame = pd.read_parquet(root / "manifests/checkpoint_manifest.parquet")
    if len(frame) != int(config["source"]["expected_calibration_rows"]):
        raise RuntimeError("calibration checkpoint count differs")
    if set(frame["split"].astype(str)) != {"calibration"}:
        raise RuntimeError("non-calibration checkpoint entered the robustness cohort")
    if set(frame["base_model"].astype(str)) != set(MODEL_KEYS):
        raise RuntimeError("calibration model set differs")
    frame = frame.rename(columns={"base_model": "model_key"})
    if set(frame["domain"].astype(str)) != set(config["cohort"]["domains"]):
        raise RuntimeError("calibration domain definition differs")
    if set(frame["checkpoint_validity_status"].astype(str)) != {"included_production"}:
        raise RuntimeError("invalid checkpoint entered the calibration cohort")
    if set(frame["num_rollouts"].astype(int)) != {4}:
        raise RuntimeError("checkpoint manifest is not the frozen production K=4 corpus")
    for model_key in MODEL_KEYS:
        part = frame.loc[frame["model_key"].eq(model_key)]
        if len(part) != int(config["source"]["expected_calibration_checkpoints_per_model"]):
            raise RuntimeError(f"{model_key}: calibration checkpoint count differs")
        if part["trace_id"].nunique() != int(
            config["source"]["expected_calibration_traces_per_model"]
        ):
            raise RuntimeError(f"{model_key}: calibration trace count differs")
        frozen = config["frozen_models"][model_key]
        for column, expected_key in (
            ("model_id", "model_id"),
            ("model_revision", "model_revision"),
            ("tokenizer_revision", "tokenizer_revision"),
        ):
            if set(part[column].astype(str)) != {str(frozen[expected_key])}:
                raise RuntimeError(f"{model_key}: frozen {column} differs")
    if frame.duplicated(list(CHECKPOINT_KEY_COLUMNS)).any():
        raise RuntimeError("duplicate calibration checkpoint identity")
    frame["normalized_checkpoint_position"] = (
        frame["prefix_token_count"].to_numpy(float)
        / np.maximum(frame["total_trace_token_count"].to_numpy(float), 1.0)
    )
    return frame.sort_values(list(CHECKPOINT_KEY_COLUMNS), kind="mergesort").reset_index(drop=True)


def _load_source_traces(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for model_key in MODEL_KEYS:
        path = root / f"manifests/source_traces/{model_key}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        keyed = {str(row["trace_id"]): row for row in rows}
        if len(keyed) != len(rows):
            raise RuntimeError(f"{model_key}: duplicate source trace")
        result[model_key] = keyed
    return result


def _token_hash(token_ids: Iterable[int]) -> str:
    return stable_hash([int(value) for value in token_ids])


def _augment_checkpoint_identity(
    frame: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    source_traces: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> pd.DataFrame:
    output = frame.copy()
    continuation_policy_hash = stable_hash(
        {
            "generation": config["generation"],
            "model_context": {
                key: {
                    "max_context_length": config["models"][key]["max_context_length"],
                    "chat_template_kwargs": config["models"][key].get("chat_template_kwargs", {}),
                }
                for key in MODEL_KEYS
            },
        }
    )
    records: list[dict[str, Any]] = []
    for row in output.to_dict("records"):
        model_key = str(row["model_key"])
        trace_id = str(row["trace_id"])
        source = source_traces[model_key].get(trace_id)
        if source is None:
            raise RuntimeError(f"{model_key}/{trace_id}: source trace is missing")
        prompt = list(map(int, source["prompt_token_ids"]))
        complete = prompt + list(map(int, source["completion_token_ids"]))
        token_offset = int(row["checkpoint_token_offset"])
        ordinal = int(row["checkpoint_ordinal"])
        offsets = list(map(int, source["eligible_checkpoint_offsets"]))
        if ordinal >= len(offsets) or offsets[ordinal] != token_offset:
            raise RuntimeError(f"{model_key}/{trace_id}: checkpoint offset differs from source")
        if token_offset <= 0 or token_offset > len(complete):
            raise RuntimeError(f"{model_key}/{trace_id}: checkpoint offset is out of bounds")
        prompt_hash = _token_hash(prompt)
        prefix_hash = _token_hash(complete[:token_offset])
        checkpoint_key = stable_hash(
            [
                "k-densification-checkpoint-v2",
                row["model_id"],
                row["model_revision"],
                row["tokenizer_revision"],
                row["problem_id"],
                trace_id,
                token_offset,
                prefix_hash,
                prompt_hash,
                continuation_policy_hash,
                config["generation"]["verifier_version"],
            ]
        )
        records.append(
            {
                **row,
                "checkpoint_key": checkpoint_key,
                "prompt_token_hash": prompt_hash,
                "prefix_token_hash": prefix_hash,
                "continuation_policy_hash": continuation_policy_hash,
                "verifier_version": str(config["generation"]["verifier_version"]),
            }
        )
    augmented = pd.DataFrame(records)
    if augmented["checkpoint_key"].duplicated().any():
        raise RuntimeError("canonical checkpoint key is not unique")
    return augmented


def _quartiles(frame: pd.DataFrame) -> pd.Series:
    order = frame.sort_values(
        ["normalized_checkpoint_position", "checkpoint_token_offset", "checkpoint_key"],
        kind="mergesort",
    ).index.to_numpy()
    labels = np.empty(len(frame), dtype=np.int64)
    for quartile, indices in enumerate(np.array_split(order, 4), start=1):
        labels[indices] = quartile
    return pd.Series(labels, index=frame.index, dtype="int64")


def select_confirmation_subset(
    frame: pd.DataFrame, *, config: Mapping[str, Any]
) -> pd.DataFrame:
    """Select the outcome-blind model-domain-position cohort."""

    per_slice = int(config["cohort"]["confirmation_per_model_domain"])
    per_quartile = int(config["cohort"]["confirmation_per_position_quartile"])
    namespace = str(config["cohort"]["confirmation_stable_hash_namespace"])
    selected: list[pd.DataFrame] = []
    for model_key in MODEL_KEYS:
        for domain in config["cohort"]["domains"]:
            part = frame.loc[
                frame["model_key"].eq(model_key) & frame["domain"].eq(domain)
            ].copy()
            if part.empty:
                continue
            part.reset_index(drop=True, inplace=True)
            part["position_quartile"] = _quartiles(part)
            part["selection_hash"] = [
                stable_hash([namespace, model_key, domain, key])
                for key in part["checkpoint_key"].astype(str)
            ]
            target = min(per_slice, len(part))
            quota = {quartile: min(per_quartile, int((part["position_quartile"] == quartile).sum())) for quartile in range(1, 5)}
            chosen: list[int] = []
            used_traces: set[str] = set()
            # Unique-trace pass, round-robin across quartiles so an early
            # quartile cannot consume the entire trace pool.
            progress = True
            while progress and len(chosen) < target:
                progress = False
                for quartile in range(1, 5):
                    have = sum(int(part.loc[index, "position_quartile"]) == quartile for index in chosen)
                    if have >= quota[quartile]:
                        continue
                    candidates = part.loc[
                        part["position_quartile"].eq(quartile)
                        & ~part.index.isin(chosen)
                        & ~part["trace_id"].astype(str).isin(used_traces)
                    ].sort_values(["selection_hash", "checkpoint_key"], kind="mergesort")
                    if len(candidates):
                        index = int(candidates.index[0])
                        chosen.append(index)
                        used_traces.add(str(part.loc[index, "trace_id"]))
                        progress = True
            # Only after the unique-trace candidates needed by the quotas are
            # exhausted may a second checkpoint from a trace be selected.
            for quartile in range(1, 5):
                have = sum(int(part.loc[index, "position_quartile"]) == quartile for index in chosen)
                need = max(0, quota[quartile] - have)
                candidates = part.loc[
                    part["position_quartile"].eq(quartile) & ~part.index.isin(chosen)
                ].sort_values(["selection_hash", "checkpoint_key"], kind="mergesort")
                chosen.extend(map(int, candidates.index[:need]))
            if len(chosen) < target:
                remainder = part.loc[~part.index.isin(chosen)].sort_values(
                    ["selection_hash", "checkpoint_key"], kind="mergesort"
                )
                chosen.extend(map(int, remainder.index[: target - len(chosen)]))
            picked = part.loc[chosen].copy()
            if len(picked) != target:
                raise RuntimeError(f"{model_key}/{domain}: confirmation selection count differs")
            selected.append(picked)
    result = pd.concat(selected, ignore_index=True)
    if result["checkpoint_key"].duplicated().any():
        raise RuntimeError("confirmation subset contains a duplicate checkpoint")
    return result.sort_values(
        ["model_key", "domain", "position_quartile", "selection_hash"], kind="mergesort"
    ).reset_index(drop=True)


def _freeze_confirmation_manifest(
    output_root: Path,
    selected: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    source_hashes: Mapping[str, Any],
) -> dict[str, Any]:
    columns = [
        "checkpoint_key",
        "model_key",
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "problem_id",
        "problem_group",
        "domain",
        "trace_id",
        "source_trace_id",
        "checkpoint_id",
        "checkpoint_ordinal",
        "checkpoint_token_offset",
        "prefix_token_count",
        "total_trace_token_count",
        "normalized_checkpoint_position",
        "position_quartile",
        "prefix_token_hash",
        "prompt_token_hash",
        "continuation_policy_hash",
        "verifier_version",
        "selection_hash",
    ]
    rows = selected[columns].to_dict("records")
    realized = (
        selected.groupby(["model_key", "domain"], sort=True)
        .agg(checkpoints=("checkpoint_key", "size"), traces=("trace_id", "nunique"))
        .reset_index()
        .to_dict("records")
    )
    core = {
        "schema_version": 2,
        "status": "FROZEN_BEFORE_K32_GENERATION",
        "selection_namespace": config["cohort"]["confirmation_stable_hash_namespace"],
        "selection_inputs": "checkpoint_metadata_only",
        "rollout_success_inspected": False,
        "probe_scores_inspected": False,
        "calibration_outputs_inspected": False,
        "native_outcomes_inspected": False,
        "normalized_position_convention": config["cohort"]["normalized_position"],
        "unique_trace_first": True,
        "selected_checkpoints": len(rows),
        "realized_by_model_domain": realized,
        "source_sha256": dict(source_hashes),
        "rows": rows,
    }
    core["manifest_sha256"] = stable_hash(core)
    path = output_root / "k32_confirmation_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text())
        comparable = {key: value for key, value in existing.items() if key != "frozen_at"}
        if comparable != core:
            raise RuntimeError("frozen K32 confirmation manifest would change")
        return existing
    payload = {**core, "frozen_at": now_iso()}
    atomic_json(path, payload)
    return payload


def _validate_structural_k16(
    checkpoint: pd.DataFrame, threshold_root: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = [
        "model_key",
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "trace_id",
        "source_trace_id",
        "problem_id",
        "checkpoint_ordinal",
        "checkpoint_token_offset",
        "rollout_index",
        "rollout_seed",
        "infrastructure_status",
        "artifact_hash",
    ]
    structural = pd.read_parquet(
        threshold_root / "raw_outcomes/merged_k16_checkpoint_suffixes.parquet",
        columns=columns,
    )
    key = list(CHECKPOINT_KEY_COLUMNS) + ["rollout_index"]
    if structural.duplicated(key).any():
        raise RuntimeError("K16 corpus contains duplicate checkpoint-slot rows")
    grouped = structural.groupby(list(CHECKPOINT_KEY_COLUMNS), sort=True)
    bad = [
        identity
        for identity, part in grouped
        if sorted(part["rollout_index"].astype(int).tolist()) != list(range(16))
        or part["rollout_seed"].astype(int).nunique() != 16
        or not part["infrastructure_status"].eq("executed").all()
    ]
    if bad or len(grouped) != len(checkpoint):
        raise RuntimeError(f"K16 structural coverage differs: {bad[:5]}")
    expected_keys = set(map(tuple, checkpoint[list(CHECKPOINT_KEY_COLUMNS)].to_numpy()))
    observed_keys = set(map(tuple, structural[list(CHECKPOINT_KEY_COLUMNS)].drop_duplicates().to_numpy()))
    if observed_keys != expected_keys:
        raise RuntimeError("K16 checkpoint identities differ from the calibration manifest")
    return structural, {
        "rows": len(structural),
        "checkpoints": len(grouped),
        "slots": list(range(16)),
        "duplicate_checkpoint_slots": 0,
        "missing_checkpoint_slots": 0,
        "all_infrastructure_executed": True,
        "success_values_loaded_during_structural_validation": False,
    }


def _predict_all_architectures(
    *,
    boundary_root: Path,
    threshold_root: Path,
    checkpoint: pd.DataFrame,
    output_root: Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cached = pd.read_parquet(threshold_root / "manifests/frozen_raw_logits.parquet")
    cached = cached.rename(columns={"base_model": "model_key"})
    cached["raw_probability"] = 1.0 / (1.0 + np.exp(-np.clip(cached["raw_logit"].to_numpy(float), -60, 60)))
    if set(cached["architecture"].astype(str)) != {"position_only", "linear_probe"}:
        raise RuntimeError("cached predictor architecture set differs")
    records = [cached]
    checkpoint_hashes: dict[str, str] = {}
    recomputed_rows = 0
    device = torch.device("cpu")
    for model_key in MODEL_KEYS:
        corpus = ModelCorpus(boundary_root, model_key)
        for architecture in ("local_mlp", "change_aware_mlp", "causal_gru"):
            learning_rate = float(config["predictors"]["frozen_learning_rates"][architecture])
            slug = f"lr_{learning_rate:.0e}"
            for seed in map(int, config["predictors"]["training_seeds"]):
                path = boundary_root / f"training/{model_key}/{architecture}/{slug}/seed_{seed}/best.pt"
                checkpoint_hashes[f"{model_key}/{architecture}/{seed}"] = sha256_file(path)
                model = load_trained_predictor(path, device=device)
                prediction = predict_split(
                    model,
                    corpus,
                    split="calibration",
                    batch_size=64,
                    device=device,
                    seed=seed,
                )
                prediction = prediction.rename(columns={"base_model": "model_key"})
                prediction["training_seed"] = seed
                prediction["architecture"] = architecture
                prediction["learning_rate"] = learning_rate
                records.append(prediction)
                recomputed_rows += len(prediction)
                del model
    combined = pd.concat(records, ignore_index=True, sort=False)
    identity = ["model_key", "trace_id", "checkpoint_id", "architecture", "training_seed"]
    if combined.duplicated(identity).any():
        raise RuntimeError("frozen predictor output identity is duplicated")
    expected = len(checkpoint) * len(ARCHITECTURES) * len(config["predictors"]["training_seeds"])
    if len(combined) != expected:
        raise RuntimeError(f"frozen prediction count {len(combined)} != {expected}")
    if set(combined["architecture"].astype(str)) != set(ARCHITECTURES):
        raise RuntimeError("frozen predictor architecture coverage differs")
    if not np.isfinite(combined["raw_logit"].to_numpy(float)).all():
        raise RuntimeError("frozen predictor logits contain non-finite values")
    combined = combined.merge(
        checkpoint[["model_key", "trace_id", "checkpoint_id", "checkpoint_key"]],
        on=["model_key", "trace_id", "checkpoint_id"],
        how="left",
        validate="many_to_one",
    )
    if combined["checkpoint_key"].isna().any():
        raise RuntimeError("frozen predictor output does not join the calibration checkpoint")
    keep = [
        "checkpoint_key",
        "model_key",
        "trace_id",
        "checkpoint_id",
        "architecture",
        "training_seed",
        "learning_rate",
        "raw_logit",
        "raw_probability",
    ]
    combined = combined[keep].sort_values(identity, kind="mergesort").reset_index(drop=True)
    atomic_parquet(output_root / "frozen_predictor_outputs.parquet", combined)
    return combined, {
        "architectures": list(ARCHITECTURES),
        "training_seeds": list(map(int, config["predictors"]["training_seeds"])),
        "learning_rates": dict(config["predictors"]["frozen_learning_rates"]),
        "cached_rows_reused": len(cached),
        "hidden_state_rows_recomputed": recomputed_rows,
        "base_llm_rerun": False,
        "predictor_retrained": False,
        "calibrator_refit": False,
        "threshold_reselected": False,
        "recomputed_checkpoint_sha256": checkpoint_hashes,
    }


def _write_generation_inputs(
    *,
    output_root: Path,
    selected: pd.DataFrame,
    source_traces: Mapping[str, Mapping[str, Mapping[str, Any]]],
    config: Mapping[str, Any],
) -> None:
    generation_root = output_root / "generation_input"
    columns = [
        "checkpoint_key",
        "model_key",
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "trace_id",
        "source_trace_id",
        "problem_id",
        "domain",
        "checkpoint_id",
        "checkpoint_ordinal",
        "checkpoint_token_offset",
        "prefix_token_hash",
        "prompt_token_hash",
        "continuation_policy_hash",
        "verifier_version",
    ]
    rows: list[dict[str, Any]] = []
    for row in selected[columns].to_dict("records"):
        seeds = []
        for slot in range(16, 32):
            seed = stable_seed(
                config["rollouts"]["seed_namespace"],
                row["model_id"],
                row["problem_id"],
                row["trace_id"],
                int(row["checkpoint_token_offset"]),
                row["prefix_token_hash"],
                slot,
                row["continuation_policy_hash"],
            )
            seeds.append(seed)
            rows.append({**row, "rollout_slot": slot, "rollout_seed": seed})
        if len(seeds) != len(set(seeds)):
            raise RuntimeError(f"{row['checkpoint_key']}: rollout seed collision")
    atomic_parquet(generation_root / "k32_generation_manifest.parquet", pd.DataFrame(rows))
    selected_traces = selected.groupby("model_key")["trace_id"].apply(set).to_dict()
    for model_key in MODEL_KEYS:
        source_rows = [
            source_traces[model_key][trace_id]
            for trace_id in sorted(selected_traces.get(model_key, set()))
        ]
        atomic_jsonl(generation_root / f"source_traces/{model_key}.jsonl", source_rows)


def prepare_calibration_experiment(
    *,
    config: Mapping[str, Any],
    boundary_root: str | Path,
    threshold_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    boundary = Path(boundary_root)
    threshold = Path(threshold_root)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    source_validation = _verify_threshold_bundle(config, threshold)
    boundary_validation = _validate_boundary_artifacts(config=config, boundary_root=boundary)
    checkpoint = _load_checkpoint_manifest(config, threshold)
    source_traces = _load_source_traces(threshold)
    checkpoint = _augment_checkpoint_identity(
        checkpoint, config=config, source_traces=source_traces
    )
    # Structural K16 validation reads identity/slot metadata only.  Selection
    # is frozen before any verifier outcome or predictor score is loaded.
    _, k16_structure = _validate_structural_k16(checkpoint, threshold)
    selected = select_confirmation_subset(checkpoint, config=config)
    confirmation = _freeze_confirmation_manifest(
        output,
        selected,
        config=config,
        source_hashes=source_validation["source_sha256"],
    )
    atomic_parquet(output / "calibration_k16_checkpoints.parquet", checkpoint)
    atomic_parquet(output / "k32_confirmation_checkpoints.parquet", selected)
    _write_generation_inputs(
        output_root=output,
        selected=selected,
        source_traces=source_traces,
        config=config,
    )
    predictions, predictor_summary = _predict_all_architectures(
        boundary_root=boundary,
        threshold_root=threshold,
        checkpoint=checkpoint,
        output_root=output,
        config=config,
    )
    inventory = {
        "schema_version": 2,
        "status": "COMPLETE_K16_REUSE_ONLY",
        "created_at": now_iso(),
        "cohort_role": "post_hoc_calibration_robustness_not_final_test",
        "source_run_id": config["source"]["threshold_run_id"],
        "source_validation": source_validation,
        "boundary_validation": boundary_validation,
        "k16_structure": k16_structure,
        "models": list(MODEL_KEYS),
        "domains": dict(config["cohort"]["domains"]),
        "checkpoint_rows": len(checkpoint),
        "trace_rows_by_model": {
            key: int(checkpoint.loc[checkpoint["model_key"].eq(key), "trace_id"].nunique())
            for key in MODEL_KEYS
        },
        "checkpoints_by_model_domain": (
            checkpoint.groupby(["model_key", "domain"], sort=True).size().rename("checkpoints").reset_index().to_dict("records")
        ),
        "frozen_predictions": predictor_summary,
        "frozen_prediction_rows": len(predictions),
        "new_generation_for_part_a": 0,
        "native_outcomes_loaded": False,
        "teacher_forced_test_loaded": False,
    }
    atomic_json(output / "calibration_k16_inventory.json", inventory)
    return {
        "status": "READY_FOR_K32_REUSE_SCAN",
        "k16_checkpoints": len(checkpoint),
        "confirmation_checkpoints": int(confirmation["selected_checkpoints"]),
        "frozen_prediction_rows": len(predictions),
        "output_root": str(output),
    }
