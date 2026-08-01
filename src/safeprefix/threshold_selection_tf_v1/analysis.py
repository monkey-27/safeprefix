"""Cross-fitted calibration, outcome-blind policies, and frozen tau selection."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from safeprefix.boundary_v1.evaluation import (
    apply_calibration,
    fit_positive_affine_calibrator,
    metric_suite,
    reliability_table,
)
from safeprefix.reproducibility import atomic_json, atomic_parquet, now_iso, stable_hash

from .data import MODEL_KEYS, read_jsonl, row_artifact_hash, sha256_file


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    """Write a CSV with the same replace-on-completion contract as Parquet."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=destination.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _assert_frozen_protocol(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    path = Path(root) / "manifests/frozen_protocol.json"
    if not path.is_file():
        raise RuntimeError("threshold analysis requires a frozen pre-outcome protocol")
    frozen = json.loads(path.read_text())
    if frozen.get("configuration_hash") != stable_hash(config):
        raise RuntimeError("live analysis configuration differs from the pre-outcome freeze")
    if list(map(float, frozen.get("threshold_grid", []))) != list(
        map(float, config["threshold_selection"]["thresholds"])
    ):
        raise RuntimeError("live threshold grid differs from the pre-outcome freeze")
    if float(frozen.get("full_regeneration_anchor", -1)) != 1.0:
        raise RuntimeError("full-regeneration anchor differs from the pre-outcome freeze")
    if frozen.get("threshold_refinement_allowed") is not False:
        raise RuntimeError("frozen protocol does not forbid threshold refinement")
    return frozen


def _read_completed_packs(artifact_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    root = Path(artifact_root)
    packs = read_jsonl(root / "manifests/execution_packs.jsonl")
    added_frames: list[pd.DataFrame] = []
    full_frames: list[pd.DataFrame] = []
    markers: list[dict[str, Any]] = []
    for pack in packs:
        pack_root = root / "raw_outcomes/production_packs" / pack["model_key"] / pack["pack_id"]
        marker_path = pack_root / "complete.json"
        added_path = pack_root / "added_checkpoint_suffixes.parquet"
        full_path = pack_root / "full_regenerations.parquet"
        if not all(path.is_file() for path in (marker_path, added_path, full_path)):
            raise RuntimeError(f"incomplete production pack: {pack['pack_id']}")
        marker = json.loads(marker_path.read_text())
        if (
            marker.get("status") != "COMPLETE"
            or marker.get("scientific") is not True
            or marker.get("pack_hash") != pack["pack_hash"]
            or marker.get("dense_sha256") != sha256_file(added_path)
            or marker.get("full_regeneration_sha256") != sha256_file(full_path)
        ):
            raise RuntimeError(f"invalid production pack marker: {pack['pack_id']}")
        added_frames.append(pd.read_parquet(added_path))
        full_frames.append(pd.read_parquet(full_path))
        markers.append(marker)
    return pd.concat(added_frames, ignore_index=True), pd.concat(full_frames, ignore_index=True), markers


def aggregate_dense_outcomes(
    config: Mapping[str, Any], *, artifact_root: Path
) -> dict[str, Any]:
    """Merge immutable original K=4 rows with exactly twelve new rows."""
    root = Path(artifact_root)
    expected = config["source"]["expected"]
    original_path = root / "raw_outcomes/original_k4_checkpoint_suffixes.parquet"
    frozen = json.loads((root / "manifests/frozen_protocol.json").read_text())
    if sha256_file(original_path) != frozen["original_k4_copy_sha256"]:
        raise RuntimeError("immutable original K=4 copy changed after protocol freeze")
    original = pd.read_parquet(original_path)
    if any(
        row_artifact_hash(row) != str(row["artifact_hash"])
        for row in original.to_dict("records")
    ):
        raise RuntimeError("original K=4 row artifact hash differs")
    added, full, markers = _read_completed_packs(root)
    if len(added) != int(expected["added_checkpoint_rollouts_total"]):
        raise RuntimeError("added checkpoint suffix count differs")
    if len(full) != int(expected["full_regenerations_total"]):
        raise RuntimeError("full-regeneration count differs")
    if not added["infrastructure_status"].eq("executed").all() or not full["infrastructure_status"].eq("executed").all():
        raise RuntimeError("unresolved infrastructure row entered dense aggregation")
    checkpoint_seed_manifest = pd.read_parquet(
        root / "manifests/dense_checkpoint_rollout_manifest.parquet"
    )
    regeneration_seed_manifest = pd.read_parquet(
        root / "manifests/full_regeneration_manifest.parquet"
    )
    expected_added_keys = set(
        zip(
            checkpoint_seed_manifest["base_model"].astype(str),
            checkpoint_seed_manifest["trace_id"].astype(str),
            checkpoint_seed_manifest["checkpoint_id"].astype(str),
            checkpoint_seed_manifest["checkpoint_ordinal"].astype(int),
            checkpoint_seed_manifest["rollout_index"].astype(int),
            checkpoint_seed_manifest["rollout_seed"].astype(int),
        )
    )
    observed_added_keys = set(
        zip(
            added["model_key"].astype(str), added["trace_id"].astype(str),
            added["checkpoint_id"].astype(str), added["checkpoint_ordinal"].astype(int),
            added["rollout_index"].astype(int), added["rollout_seed"].astype(int),
        )
    )
    expected_full_keys = set(
        zip(
            regeneration_seed_manifest["base_model"].astype(str),
            regeneration_seed_manifest["trace_id"].astype(str),
            regeneration_seed_manifest["rollout_index"].astype(int),
            regeneration_seed_manifest["rollout_seed"].astype(int),
        )
    )
    observed_full_keys = set(
        zip(
            full["model_key"].astype(str), full["trace_id"].astype(str),
            full["rollout_index"].astype(int), full["rollout_seed"].astype(int),
        )
    )
    if observed_added_keys != expected_added_keys or observed_full_keys != expected_full_keys:
        raise RuntimeError("generated logical seed keys differ from frozen manifests")
    for name, frame in (("added", added), ("full", full)):
        if any(
            row_artifact_hash(row) != str(row["artifact_hash"])
            for row in frame.to_dict("records")
        ):
            raise RuntimeError(f"{name} row artifact hash differs")
    original = original.rename(
        columns={
            "generated_text": "raw_suffix",
            "parsed_answer": "normalized_final_answer",
            "truncation_flag": "truncation_status",
            "verifier_pass": "verifier_outcome",
        }
    )
    if "verifier_outcome" not in original:
        original["verifier_outcome"] = original["binary_outcome"].astype(bool)
    if "binary_outcome" not in original:
        original["binary_outcome"] = original["verifier_outcome"].astype(bool)
    original["model_key"] = original["base_model"]
    keep = sorted(set(original.columns) & set(added.columns))
    required = {
        "model_key", "trace_id", "checkpoint_ordinal", "checkpoint_token_offset",
        "rollout_index", "rollout_seed", "generated_token_count", "binary_outcome",
        "raw_suffix", "artifact_hash", "infrastructure_status",
    }
    if required - set(keep):
        raise RuntimeError(f"merged K=16 schema lacks {sorted(required - set(keep))}")
    merged = pd.concat([original[keep], added[keep]], ignore_index=True)
    logical = merged[["model_key", "trace_id", "checkpoint_ordinal", "rollout_index"]]
    if logical.duplicated().any():
        raise RuntimeError("duplicate K=16 checkpoint logical row")
    counts = merged.groupby(["model_key", "trace_id", "checkpoint_ordinal"]).size()
    if len(counts) != int(expected["checkpoints_total"]) or set(counts.astype(int)) != {16}:
        raise RuntimeError("not every calibration checkpoint has exactly K=16")
    index_sets = merged.groupby(["model_key", "trace_id", "checkpoint_ordinal"])["rollout_index"].agg(
        lambda values: tuple(sorted(map(int, values)))
    )
    if set(index_sets) != {tuple(range(16))}:
        raise RuntimeError("dense checkpoint rollout indices differ from 0..15")
    manifest = pd.read_parquet(root / "manifests/checkpoint_manifest.parquet")
    dense = (
        merged.groupby(["model_key", "trace_id", "checkpoint_ordinal"], as_index=False)
        .agg(
            total_success_count=("binary_outcome", "sum"),
            mean_suffix_tokens=("generated_token_count", "mean"),
            median_suffix_tokens=("generated_token_count", "median"),
            mean_suffix_latency_seconds=("latency_seconds", "mean"),
            parser_success_rate=("parser_status", lambda values: float(np.mean(pd.Series(values).eq("success")))),
            truncation_rate=("truncation_status", "mean"),
        )
    )
    added_counts = (
        added.groupby(["model_key", "trace_id", "checkpoint_ordinal"])["binary_outcome"]
        .sum().rename("added_success_count").reset_index()
    )
    dense = dense.merge(added_counts, on=["model_key", "trace_id", "checkpoint_ordinal"], validate="one_to_one")
    dense["total_success_count"] = dense["total_success_count"].astype(int)
    dense["original_success_count"] = (
        dense["total_success_count"] - dense["added_success_count"]
    ).astype(int)
    if not dense["original_success_count"].between(0, 4).all():
        raise RuntimeError("derived original K=4 success count is invalid")
    dense["dense_success_rate"] = dense["total_success_count"] / 16.0
    dense["num_rollouts"] = 16
    metadata = manifest.rename(columns={"base_model": "model_key"}).drop(
        columns=["success_count", "num_rollouts", "observed_success_rate"], errors="ignore"
    )
    dense = metadata.merge(
        dense,
        on=["model_key", "trace_id", "checkpoint_ordinal"],
        validate="one_to_one",
    )
    full_counts = full.groupby(["model_key", "trace_id"]).size()
    if len(full_counts) != int(expected["traces_per_model"]) * len(MODEL_KEYS) or set(full_counts.astype(int)) != {16}:
        raise RuntimeError("not every calibration model-trace has 16 full regenerations")
    full_index_sets = full.groupby(["model_key", "trace_id"])["rollout_index"].agg(
        lambda values: tuple(sorted(map(int, values)))
    )
    if set(full_index_sets) != {tuple(range(16))}:
        raise RuntimeError("full-regeneration rollout indices differ from 0..15")
    full_aggregate = (
        full.groupby(["model_key", "trace_id"], as_index=False)
        .agg(
            common_trace_id=("common_trace_id", "first") if "common_trace_id" in full else ("trace_id", "first"),
            problem_id=("problem_id", "first"),
            problem_group=("problem_group", "first"),
            domain=("domain", "first"),
            full_regeneration_success_count=("binary_outcome", "sum"),
            mean_full_response_tokens=("generated_token_count", "mean"),
            median_full_response_tokens=("generated_token_count", "median"),
            mean_full_latency_seconds=("latency_seconds", "mean"),
            prompt_input_token_count=("prompt_input_token_count", "first"),
            parser_success_rate=("parser_status", lambda values: float(np.mean(pd.Series(values).eq("success")))),
            truncation_rate=("truncation_status", "mean"),
        )
    )
    trace_manifest = pd.read_parquet(root / "manifests/calibration_trace_manifest.parquet").rename(
        columns={"base_model": "model_key"}
    )
    full_aggregate = full_aggregate.drop(columns=["common_trace_id"], errors="ignore").merge(
        trace_manifest[["model_key", "trace_id", "common_trace_id", "problem_group", "domain"]],
        on=["model_key", "trace_id", "problem_group", "domain"],
        validate="one_to_one",
    )
    full_aggregate["full_regeneration_success_count"] = full_aggregate[
        "full_regeneration_success_count"
    ].astype(int)
    full_aggregate["full_regeneration_success_rate"] = (
        full_aggregate["full_regeneration_success_count"] / 16.0
    )
    full_aggregate["full_regeneration_fresh_tokens"] = (
        full_aggregate["prompt_input_token_count"] + full_aggregate["mean_full_response_tokens"]
    )
    atomic_parquet(root / "raw_outcomes/all_added_checkpoint_suffixes.parquet", added)
    atomic_parquet(root / "raw_outcomes/all_full_regenerations.parquet", full)
    atomic_parquet(root / "raw_outcomes/merged_k16_checkpoint_suffixes.parquet", merged)
    atomic_parquet(root / "outcomes/dense_checkpoint_outcomes.parquet", dense)
    atomic_parquet(root / "outcomes/full_regeneration_outcomes.parquet", full_aggregate)
    summary = {
        "status": "COMPLETE",
        "model_traces": len(full_aggregate),
        "checkpoints": len(dense),
        "original_checkpoint_rollouts": len(original),
        "added_checkpoint_rollouts": len(added),
        "dense_checkpoint_rollouts": len(merged),
        "full_regenerations": len(full),
        "completed_packs": len(markers),
        "all_infrastructure_statuses_executed": True,
        "original_k4_overwritten": False,
        "native_evaluation_used": False,
    }
    atomic_json(root / "outcomes/dense_outcome_summary.json", summary)
    return summary


def crossfit_calibrators(
    config: Mapping[str, Any], *, artifact_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit every calibrator on dense labels from four folds, score the fifth."""
    root = Path(artifact_root)
    raw = pd.read_parquet(root / "manifests/frozen_raw_logits.parquet")
    folds = pd.read_parquet(root / "manifests/five_fold_assignment.parquet")
    dense = pd.read_parquet(root / "outcomes/dense_checkpoint_outcomes.parquet")
    targets = dense[
        ["model_key", "trace_id", "checkpoint_ordinal", "dense_success_rate", "total_success_count"]
    ].rename(columns={"model_key": "base_model"})
    frame = raw.merge(
        targets,
        on=["base_model", "trace_id", "checkpoint_ordinal"],
        validate="many_to_one",
    ).merge(
        folds[["common_trace_id", "problem_group", "domain", "fold"]],
        on=["common_trace_id", "problem_group", "domain"],
        validate="many_to_one",
    )
    frame["observed_success_rate"] = frame["dense_success_rate"]
    calibrators: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for (architecture, model_key, seed), part in frame.groupby(
        ["architecture", "base_model", "training_seed"], sort=True
    ):
        for fold in range(int(config["cross_fit"]["folds"])):
            train = part[part["fold"].ne(fold)].copy()
            held = part[part["fold"].eq(fold)].copy()
            if not len(train) or not len(held):
                raise RuntimeError("empty cross-fit training or held-out fold")
            train_groups = set(train["problem_group"].astype(str))
            held_groups = set(held["problem_group"].astype(str))
            if train_groups & held_groups:
                raise RuntimeError("held-out problem group entered calibrator fit")
            calibrator = fit_positive_affine_calibrator(
                train,
                max_iterations=int(config["cross_fit"]["max_iterations"]),
            )
            scored = apply_calibration(held, calibrator).rename(
                columns={"calibrated_probability": "oof_probability"}
            )
            predictions.append(scored)
            calibrators.append(
                {
                    **calibrator,
                    "architecture": str(architecture),
                    "base_model": str(model_key),
                    "training_seed": int(seed),
                    "held_out_fold": int(fold),
                    "training_traces": int(train["trace_id"].nunique()),
                    "training_checkpoints": len(train),
                    "held_out_traces": int(held["trace_id"].nunique()),
                    "held_out_checkpoints": len(held),
                    "training_problem_groups_hash": stable_hash(sorted(train_groups)),
                    "held_out_problem_groups_hash": stable_hash(sorted(held_groups)),
                    "fit_target": "dense_k16_training_folds_only",
                    "held_out_fold_labels_used_for_fit": False,
                    "probe_weights_modified": False,
                }
            )
    oof = pd.concat(predictions, ignore_index=True)
    keys = ["architecture", "base_model", "training_seed", "checkpoint_id"]
    if oof.duplicated(keys).any():
        raise RuntimeError("checkpoint received multiple out-of-fold predictions")
    expected_rows = len(raw)
    if len(oof) != expected_rows:
        raise RuntimeError(f"out-of-fold prediction count differs: {len(oof)} != {expected_rows}")
    if not oof["oof_probability"].between(0, 1).all():
        raise RuntimeError("out-of-fold probability outside [0,1]")
    calibrator_frame = pd.DataFrame(calibrators)
    atomic_parquet(root / "calibration/cross_fitted_predictions.parquet", oof)
    atomic_parquet(root / "calibration/cross_fitted_calibrator_manifest.parquet", calibrator_frame)
    return oof, calibrator_frame


def _operational_seed_map(seed_table: pd.DataFrame, architecture: str, rank: int) -> dict[str, int]:
    part = seed_table[
        seed_table["architecture"].eq(architecture) & seed_table["dev_rank"].eq(int(rank))
    ]
    if set(part["base_model"].astype(str)) != set(MODEL_KEYS):
        raise RuntimeError(f"seed-rank map incomplete for {architecture}/rank={rank}")
    return dict(zip(part["base_model"].astype(str), part["seed"].astype(int)))


def compose_seed_predictions(
    oof: pd.DataFrame, *, architecture: str, seed_map: Mapping[str, int]
) -> pd.DataFrame:
    parts = [
        oof[
            oof["architecture"].eq(architecture)
            & oof["base_model"].eq(model_key)
            & oof["training_seed"].eq(int(seed))
        ]
        for model_key, seed in seed_map.items()
    ]
    result = pd.concat(parts, ignore_index=True)
    if len(result) != 1932 or result["checkpoint_id"].nunique() != 1932:
        raise RuntimeError("operational OOF prediction matrix is incomplete")
    return result


def select_threshold_actions(
    predictions: pd.DataFrame, thresholds: Sequence[float]
) -> pd.DataFrame:
    """Select checkpoint IDs without accepting any outcome/full-regeneration column."""
    prohibited = {
        "dense_success_rate", "total_success_count", "full_regeneration_success_rate",
        "binary_outcome", "verifier_outcome", "mean_suffix_tokens",
    }
    if prohibited & set(predictions.columns):
        raise RuntimeError("outcome column was passed to checkpoint selection")
    records: list[dict[str, Any]] = []
    identity = [
        "base_model", "trace_id", "common_trace_id", "problem_id", "problem_group", "domain"
    ]
    for (model_key, trace_id), trace in predictions.groupby(["base_model", "trace_id"], sort=True):
        trace = trace.sort_values("checkpoint_ordinal")
        metadata = {column: trace.iloc[0][column] for column in identity}
        for tau in thresholds:
            tau = float(tau)
            eligible = trace[trace["oof_probability"].ge(tau)] if tau < 1.0 else trace.iloc[0:0]
            if len(eligible):
                selected = eligible.iloc[-1]
                records.append(
                    {
                        **metadata,
                        "threshold": tau,
                        "action": "checkpoint",
                        "selected_checkpoint_id": selected["checkpoint_id"],
                        "selected_checkpoint_probability": float(selected["oof_probability"]),
                        "selected_checkpoint_ordinal": int(selected["checkpoint_ordinal"]),
                        "selected_checkpoint_token_offset": int(selected["checkpoint_token_offset"]),
                        "fallback": False,
                    }
                )
            else:
                records.append(
                    {
                        **metadata,
                        "threshold": tau,
                        "action": "full_regeneration_fallback",
                        "selected_checkpoint_id": None,
                        "selected_checkpoint_probability": np.nan,
                        "selected_checkpoint_ordinal": np.nan,
                        "selected_checkpoint_token_offset": np.nan,
                        "fallback": True,
                    }
                )
    return pd.DataFrame(records)


def evaluate_actions(actions: pd.DataFrame, *, artifact_root: Path) -> pd.DataFrame:
    root = Path(artifact_root)
    dense = pd.read_parquet(root / "outcomes/dense_checkpoint_outcomes.parquet").rename(
        columns={"model_key": "base_model"}
    )
    full = pd.read_parquet(root / "outcomes/full_regeneration_outcomes.parquet").rename(
        columns={"model_key": "base_model"}
    )
    checkpoint_values = dense[
        [
            "base_model", "trace_id", "checkpoint_id", "checkpoint_ordinal",
            "checkpoint_token_offset", "prefix_token_count", "total_trace_token_count",
            "total_checkpoint_count", "total_success_count", "dense_success_rate",
            "mean_suffix_tokens", "median_suffix_tokens",
            "mean_suffix_latency_seconds",
        ]
    ].rename(columns={"checkpoint_ordinal": "outcome_checkpoint_ordinal"})
    result = actions.merge(
        checkpoint_values,
        left_on=["base_model", "trace_id", "selected_checkpoint_id"],
        right_on=["base_model", "trace_id", "checkpoint_id"],
        how="left",
        validate="many_to_one",
    ).merge(
        full[
            [
                "base_model", "trace_id", "full_regeneration_success_count",
                "full_regeneration_success_rate", "mean_full_response_tokens",
                "median_full_response_tokens", "prompt_input_token_count",
                "full_regeneration_fresh_tokens", "mean_full_latency_seconds",
            ]
        ],
        on=["base_model", "trace_id"],
        validate="many_to_one",
    )
    selected = ~result["fallback"].astype(bool)
    if result.loc[selected, "dense_success_rate"].isna().any():
        raise RuntimeError("selected checkpoint lacks dense K=16 outcome")
    result["policy_success"] = np.where(
        selected, result["dense_success_rate"], result["full_regeneration_success_rate"]
    )
    result["fresh_tokens"] = np.where(
        selected, result["mean_suffix_tokens"], result["full_regeneration_fresh_tokens"]
    )
    result["generated_output_tokens"] = np.where(
        selected, result["mean_suffix_tokens"], result["mean_full_response_tokens"]
    )
    result["retained_prefix_tokens"] = np.where(selected, result["prefix_token_count"], 0.0)
    result["retained_prefix_fraction"] = np.where(
        selected,
        result["prefix_token_count"] / np.maximum(result["total_trace_token_count"], 1),
        0.0,
    )
    result["normalized_selected_checkpoint_position"] = np.where(
        selected,
        (result["outcome_checkpoint_ordinal"] + 1)
        / np.maximum(result["total_checkpoint_count"], 1),
        np.nan,
    )
    result["kv_recomputation_avoided"] = result["retained_prefix_tokens"]
    result["measured_latency_seconds"] = np.where(
        selected, result["mean_suffix_latency_seconds"], result["mean_full_latency_seconds"]
    )
    result["success_difference"] = result["policy_success"] - result["full_regeneration_success_rate"]
    return result


def make_bootstrap_plan(policy: pd.DataFrame, *, replicates: int, seed: int) -> tuple[np.ndarray, list[str]]:
    traces = policy[["common_trace_id", "domain"]].drop_duplicates().sort_values("common_trace_id")
    if traces["common_trace_id"].nunique() != len(traces):
        raise RuntimeError("common trace belongs to multiple domains")
    ids = traces["common_trace_id"].astype(str).tolist()
    index = {value: i for i, value in enumerate(ids)}
    counts = np.zeros((int(replicates), len(ids)), dtype=np.int16)
    generator = np.random.default_rng(int(seed))
    rows = np.arange(int(replicates))[:, None]
    for _, domain in traces.groupby("domain", sort=True):
        choices = np.array([index[value] for value in domain["common_trace_id"].astype(str)], dtype=int)
        draw = generator.choice(choices, size=(int(replicates), len(choices)), replace=True)
        np.add.at(counts, (np.broadcast_to(rows, draw.shape), draw), 1)
    if not np.all(counts.sum(axis=1) == len(ids)):
        raise AssertionError("bootstrap did not preserve trace count")
    return counts.astype(np.float64) / len(ids), ids


def bootstrap_policy_intervals(
    policy: pd.DataFrame,
    plan: np.ndarray,
    trace_order: Sequence[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    thresholds = sorted(policy["threshold"].astype(float).unique())
    index = {value: i for i, value in enumerate(trace_order)}
    values = np.empty((len(thresholds), len(MODEL_KEYS), len(trace_order)), dtype=np.float64)
    for t_index, tau in enumerate(thresholds):
        threshold_rows = policy[policy["threshold"].eq(tau)]
        for m_index, model_key in enumerate(MODEL_KEYS):
            part = threshold_rows[threshold_rows["base_model"].eq(model_key)]
            if len(part) != len(trace_order):
                raise RuntimeError("bootstrap model/trace alignment differs")
            values[t_index, m_index] = [
                float(part.loc[part["common_trace_id"].astype(str).eq(trace_id), "success_difference"].iloc[0])
                for trace_id in trace_order
            ]
    samples = np.einsum("rn,tmn->rtm", plan, values, optimize=True)
    macro = samples.mean(axis=2)
    rows: list[dict[str, Any]] = []
    for t_index, tau in enumerate(thresholds):
        row: dict[str, Any] = {
            "threshold": tau,
            "macro_difference_lcb_95": float(np.quantile(macro[:, t_index], 0.025)),
            "macro_difference_ucb_95": float(np.quantile(macro[:, t_index], 0.975)),
        }
        for m_index, model_key in enumerate(MODEL_KEYS):
            row[f"{model_key}_difference_lcb_95"] = float(np.quantile(samples[:, t_index, m_index], 0.025))
            row[f"{model_key}_difference_ucb_95"] = float(np.quantile(samples[:, t_index, m_index], 0.975))
        rows.append(row)
    return pd.DataFrame(rows), samples


def summarize_policy(policy: pd.DataFrame, intervals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    for (tau, model_key), part in policy.groupby(["threshold", "base_model"], sort=True):
        selected = part[~part["fallback"]]
        full_cost = float(part["full_regeneration_fresh_tokens"].mean())
        model_rows.append(
            {
                "threshold": float(tau),
                "base_model": str(model_key),
                "traces": len(part),
                "checkpoint_coverage": float((~part["fallback"]).mean()),
                "fallback_rate": float(part["fallback"].mean()),
                "policy_success": float(part["policy_success"].mean()),
                "full_regeneration_success": float(part["full_regeneration_success_rate"].mean()),
                "success_difference": float(part["success_difference"].mean()),
                "mean_fresh_tokens": float(part["fresh_tokens"].mean()),
                "median_fresh_tokens": float(part["fresh_tokens"].median()),
                "full_regeneration_mean_fresh_tokens": full_cost,
                "fresh_token_savings_fraction": 1.0 - float(part["fresh_tokens"].mean()) / max(full_cost, 1e-12),
                "mean_generated_output_tokens": float(part["generated_output_tokens"].mean()),
                "mean_retained_prefix_fraction": float(part["retained_prefix_fraction"].mean()),
                "median_retained_prefix_fraction": float(part["retained_prefix_fraction"].median()),
                "mean_selected_checkpoint_ordinal": float(selected["selected_checkpoint_ordinal"].mean()) if len(selected) else np.nan,
                "mean_normalized_selected_checkpoint_position": float(selected["normalized_selected_checkpoint_position"].mean()) if len(selected) else np.nan,
                "selected_checkpoint_empirical_success": float(selected["policy_success"].mean()) if len(selected) else np.nan,
                "fallback_empirical_success": float(part.loc[part["fallback"], "policy_success"].mean()) if part["fallback"].any() else np.nan,
                "earliest_checkpoint_frequency": float((selected["outcome_checkpoint_ordinal"] == 0).sum() / len(part)),
                "latest_checkpoint_frequency": float((selected["outcome_checkpoint_ordinal"] == selected["total_checkpoint_count"] - 1).sum() / len(part)),
                "nominal_minus_empirical_selected_success": float(tau - selected["policy_success"].mean()) if len(selected) else np.nan,
            }
        )
        for ordinal, count in selected["outcome_checkpoint_ordinal"].value_counts().sort_index().items():
            distribution_rows.append(
                {
                    "threshold": float(tau),
                    "base_model": str(model_key),
                    "checkpoint_ordinal": int(ordinal),
                    "count": int(count),
                    "fraction_all_traces": float(count / len(part)),
                }
            )
    by_model = pd.DataFrame(model_rows).merge(intervals, on="threshold", validate="many_to_one")
    curve_rows: list[dict[str, Any]] = []
    for tau, part in by_model.groupby("threshold", sort=True):
        raw = policy[policy["threshold"].eq(tau)]
        domain_success = raw.groupby(["base_model", "domain"])["policy_success"].mean()
        interval = intervals[intervals["threshold"].eq(tau)].iloc[0]
        macro_fresh = float(part["mean_fresh_tokens"].mean())
        macro_full_fresh = float(part["full_regeneration_mean_fresh_tokens"].mean())
        # Each model receives total weight 1/4; trace weights within a model
        # sum to that model's share. This remains correct if future frozen
        # calibration manifests have unequal trace counts by model.
        weighted = raw[["base_model", "fresh_tokens", "retained_prefix_fraction"]].copy()
        model_sizes = weighted.groupby("base_model")["base_model"].transform("size")
        weighted["weight"] = 1.0 / len(MODEL_KEYS) / model_sizes
        ordered = weighted.sort_values("fresh_tokens")
        weighted_median = float(
            ordered.loc[ordered["weight"].cumsum().ge(0.5), "fresh_tokens"].iloc[0]
        )
        retained_ordered = weighted.sort_values("retained_prefix_fraction")
        weighted_retained_median = float(
            retained_ordered.loc[
                retained_ordered["weight"].cumsum().ge(0.5),
                "retained_prefix_fraction",
            ].iloc[0]
        )
        row = {
            "threshold": float(tau),
            "macro_checkpoint_coverage": float(part["checkpoint_coverage"].mean()),
            "macro_fallback_rate": float(part["fallback_rate"].mean()),
            "macro_policy_success": float(part["policy_success"].mean()),
            "macro_full_regeneration_success": float(part["full_regeneration_success"].mean()),
            "macro_success_difference": float(part["success_difference"].mean()),
            "macro_difference_lcb_95": float(interval["macro_difference_lcb_95"]),
            "macro_difference_ucb_95": float(interval["macro_difference_ucb_95"]),
            "macro_mean_fresh_tokens": macro_fresh,
            "macro_median_fresh_tokens": weighted_median,
            "macro_full_regeneration_fresh_tokens": macro_full_fresh,
            "macro_fresh_token_savings_fraction": 1.0 - macro_fresh / max(macro_full_fresh, 1e-12),
            "macro_mean_retained_prefix_fraction": float(part["mean_retained_prefix_fraction"].mean()),
            "macro_median_retained_prefix_fraction": weighted_retained_median,
            "pooled_policy_success": float(raw["policy_success"].mean()),
            "pooled_mean_fresh_tokens": float(raw["fresh_tokens"].mean()),
            "domain_macro_policy_success": float(domain_success.mean()),
            "minimum_model_success_difference": float(part["success_difference"].min()),
            "per_model_safeguard_pass": bool((part["success_difference"] >= -0.05 - 1e-12).all()),
        }
        row["noninferiority_pass"] = row["macro_difference_lcb_95"] >= -0.03 - 1e-12
        row["coverage_pass"] = row["macro_checkpoint_coverage"] >= 0.20 - 1e-12
        row["feasible"] = bool(
            float(tau) < 1.0
            and row["noninferiority_pass"]
            and row["per_model_safeguard_pass"]
            and row["coverage_pass"]
        )
        curve_rows.append(row)
    return pd.DataFrame(curve_rows), by_model, pd.DataFrame(distribution_rows)


def select_frozen_threshold(curve: pd.DataFrame) -> dict[str, Any]:
    eligible = curve[curve["feasible"] & curve["threshold"].lt(1.0)].copy()
    if not len(eligible):
        return {
            "status": "NO_VALIDATED_THRESHOLD",
            "selected_tau": None,
            "statement": "No nontrivial SafePrefix threshold met the predefined success and coverage criteria.",
            "full_regeneration_remains_primary": True,
        }
    minimum = float(eligible["macro_mean_fresh_tokens"].min())
    tied = eligible[eligible["macro_mean_fresh_tokens"].le(minimum * 1.01 + 1e-12)].copy()
    selected = tied.sort_values(
        ["threshold", "macro_fallback_rate"], ascending=[False, True]
    ).iloc[0]
    return {
        "status": "THRESHOLD_FROZEN",
        "selected_tau": float(selected["threshold"]),
        "minimum_feasible_macro_mean_fresh_tokens": minimum,
        "selected_macro_mean_fresh_tokens": float(selected["macro_mean_fresh_tokens"]),
        "within_one_percent_cost_tie": bool(
            float(selected["macro_mean_fresh_tokens"]) > minimum + 1e-12
        ),
        "tie_candidates": sorted(tied["threshold"].astype(float).tolist()),
        "selection_rule": "minimum_macro_fresh_tokens_then_highest_tau_within_one_percent",
        "full_regeneration_remains_primary": False,
    }


def _fixed_baseline_actions(predictions: pd.DataFrame, name: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    identity = ["base_model", "trace_id", "common_trace_id", "problem_id", "problem_group", "domain"]
    rewind = {"fixed_rewind_25": 0.25, "fixed_rewind_50": 0.50, "fixed_rewind_75": 0.75}.get(name)
    for _, trace in predictions.groupby(["base_model", "trace_id"], sort=True):
        trace = trace.sort_values("checkpoint_ordinal")
        meta = {column: trace.iloc[0][column] for column in identity}
        selected: pd.Series | None
        if name == "full_regeneration":
            selected = None
        elif name == "always_latest":
            selected = trace.iloc[-1]
        elif name == "always_earliest":
            selected = trace.iloc[0]
        elif rewind is not None:
            target = (1.0 - rewind) * float(trace.iloc[0]["total_trace_token_count"])
            candidates = trace[trace["prefix_token_count"].le(target)]
            selected = candidates.iloc[-1] if len(candidates) else None
        else:
            raise ValueError(name)
        records.append(
            {
                **meta,
                "threshold": np.nan,
                "baseline": name,
                "action": "checkpoint" if selected is not None else "full_regeneration_fallback",
                "selected_checkpoint_id": None if selected is None else selected["checkpoint_id"],
                "selected_checkpoint_probability": np.nan,
                "selected_checkpoint_ordinal": np.nan if selected is None else int(selected["checkpoint_ordinal"]),
                "selected_checkpoint_token_offset": np.nan if selected is None else int(selected["checkpoint_token_offset"]),
                "fallback": selected is None,
            }
        )
    return pd.DataFrame(records)


def _baseline_summary(policy: pd.DataFrame, name: str) -> dict[str, Any]:
    per_model = policy.groupby("base_model").agg(
        success=("policy_success", "mean"),
        fresh_tokens=("fresh_tokens", "mean"),
        coverage=("fallback", lambda values: float((~values).mean())),
        retained=("retained_prefix_fraction", "mean"),
    )
    return {
        "baseline": name,
        "macro_success": float(per_model["success"].mean()),
        "macro_mean_fresh_tokens": float(per_model["fresh_tokens"].mean()),
        "macro_checkpoint_coverage": float(per_model["coverage"].mean()),
        "macro_retained_prefix_fraction": float(per_model["retained"].mean()),
    }


def _calibration_diagnostics(
    operational: pd.DataFrame, *, artifact_root: Path
) -> pd.DataFrame:
    root = Path(artifact_root)
    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", operational)]
    groups += [("base_model", str(key), part) for key, part in operational.groupby("base_model")]
    groups += [("domain", str(key), part) for key, part in operational.groupby("domain")]
    groups += [
        ("model_by_domain", f"{model}:{domain}", part)
        for (model, domain), part in operational.groupby(["base_model", "domain"])
    ]
    for stratum, value, part in groups:
        metric_frame = part.copy()
        metric_frame["observed_success_rate"] = metric_frame["dense_success_rate"]
        rows.append(
            {
                "stratum": stratum,
                "value": value,
                **metric_suite(metric_frame, "oof_probability", ece_bins=10),
            }
        )
    diagnostics = pd.DataFrame(rows)
    atomic_parquet(root / "calibration/calibration_diagnostics.parquet", diagnostics)
    reliability_frames = []
    for model_key, part in operational.groupby("base_model"):
        metric_frame = part.copy()
        metric_frame["observed_success_rate"] = metric_frame["dense_success_rate"]
        table = reliability_table(metric_frame, "oof_probability", bins=10)
        table["base_model"] = model_key
        reliability_frames.append(table)
    atomic_parquet(
        root / "calibration/reliability_tables.parquet",
        pd.concat(reliability_frames, ignore_index=True),
    )
    return diagnostics


def _pareto_frontier(curve: pd.DataFrame) -> pd.DataFrame:
    points = curve.copy()
    nondominated = []
    for row in points.itertuples(index=False):
        dominated = (
            (points["macro_policy_success"] >= float(row.macro_policy_success) - 1e-12)
            & (points["macro_mean_fresh_tokens"] <= float(row.macro_mean_fresh_tokens) + 1e-12)
            & (
                (points["macro_policy_success"] > float(row.macro_policy_success) + 1e-12)
                | (points["macro_mean_fresh_tokens"] < float(row.macro_mean_fresh_tokens) - 1e-12)
            )
        ).any()
        nondominated.append(not bool(dominated))
    points["nondominated"] = nondominated
    return points


def analyze_threshold_experiment(
    config: Mapping[str, Any], *, artifact_root: Path
) -> dict[str, Any]:
    """Run all frozen analyses after exact K=16 generation is complete."""
    root = Path(artifact_root)
    _assert_frozen_protocol(config, root)
    for relative in ("baselines", "calibration", "outcomes", "policy", "policy/seed_variants"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    aggregate_dense_outcomes(config, artifact_root=root)
    oof, calibrators = crossfit_calibrators(config, artifact_root=root)
    seed_table = pd.read_parquet(root / "manifests/model_seed_manifest.parquet")
    thresholds = [*map(float, config["threshold_selection"]["thresholds"]), 1.0]
    seed_configurations = {
        "best_dev": _operational_seed_map(seed_table, "linear_probe", 0),
        "operational_median_dev": _operational_seed_map(seed_table, "linear_probe", 1),
        "worst_dev": _operational_seed_map(seed_table, "linear_probe", 2),
    }
    operational_predictions = compose_seed_predictions(
        oof,
        architecture="linear_probe",
        seed_map=seed_configurations["operational_median_dev"],
    )
    diagnostics = _calibration_diagnostics(operational_predictions, artifact_root=root)
    plan, trace_order = make_bootstrap_plan(
        operational_predictions,
        replicates=int(config["threshold_selection"]["bootstrap_replicates"]),
        seed=int(config["threshold_selection"]["bootstrap_seed"]),
    )
    variant_results: dict[str, dict[str, Any]] = {}
    variant_policies: dict[str, pd.DataFrame] = {}
    primary_curve = primary_by_model = primary_distribution = primary_policy = None
    primary_samples = None
    for variant, seed_map in seed_configurations.items():
        predictions = compose_seed_predictions(oof, architecture="linear_probe", seed_map=seed_map)
        selection_input = predictions[
            [
                "base_model", "trace_id", "common_trace_id", "problem_id", "problem_group",
                "domain", "checkpoint_id", "checkpoint_ordinal", "checkpoint_token_offset",
                "prefix_token_count", "total_trace_token_count", "total_checkpoint_count",
                "oof_probability",
            ]
        ].copy()
        actions = select_threshold_actions(selection_input, thresholds)
        policy = evaluate_actions(actions, artifact_root=root)
        intervals, samples = bootstrap_policy_intervals(policy, plan, trace_order)
        curve, by_model, distribution = summarize_policy(policy, intervals)
        selection = select_frozen_threshold(curve)
        variant_results[variant] = {"seed_map": seed_map, "selection": selection}
        variant_policies[variant] = policy
        atomic_parquet(root / f"policy/seed_variants/{variant}_policy.parquet", policy)
        _atomic_csv(root / f"policy/seed_variants/{variant}_curve.csv", curve)
        if variant == "operational_median_dev":
            primary_curve, primary_by_model, primary_distribution = curve, by_model, distribution
            primary_policy, primary_samples = policy, samples
    assert primary_curve is not None and primary_policy is not None and primary_samples is not None
    selection = variant_results["operational_median_dev"]["selection"]
    selected_tau = selection["selected_tau"]

    # Full-data deployment calibrators retain the same positive affine form.
    deployment_rows = []
    for (model_key, seed), part in operational_predictions.groupby(["base_model", "training_seed"]):
        fit = part.copy()
        fit["observed_success_rate"] = fit["dense_success_rate"]
        calibrator = fit_positive_affine_calibrator(
            fit, max_iterations=int(config["cross_fit"]["max_iterations"])
        )
        deployment_rows.append(
            {
                **calibrator,
                "base_model": model_key,
                "training_seed": int(seed),
                "fit_rows": len(fit),
                "fit_traces": int(fit["trace_id"].nunique()),
                "fit_target": "complete_calibration_dense_k16_after_tau_freeze",
                "threshold_outcomes_changed_calibrator_form": False,
            }
        )
    atomic_parquet(root / "calibration/deployment_calibrator_manifest.parquet", pd.DataFrame(deployment_rows))

    # Per-model analysis-only thresholds use the same shared bootstrap draws.
    per_model_selections = []
    for m_index, model_key in enumerate(MODEL_KEYS):
        curve = primary_by_model[primary_by_model["base_model"].eq(model_key)].copy()
        curve["macro_difference_lcb_95"] = [
            float(np.quantile(primary_samples[:, list(sorted(primary_curve["threshold"])).index(tau), m_index], 0.025))
            for tau in curve["threshold"]
        ]
        curve["macro_mean_fresh_tokens"] = curve["mean_fresh_tokens"]
        curve["macro_fallback_rate"] = curve["fallback_rate"]
        curve["feasible"] = (
            curve["threshold"].lt(1.0)
            & curve["macro_difference_lcb_95"].ge(-0.03)
            & curve["success_difference"].ge(-0.05)
            & curve["checkpoint_coverage"].ge(0.20)
        )
        per_model_selections.append(
            {"base_model": model_key, **select_frozen_threshold(curve)}
        )
    atomic_json(root / "policy/per_model_thresholds.json", {"analysis_only": True, "models": per_model_selections})

    # Cross-seed agreement and primary-tau sensitivity.
    robustness_rows: list[dict[str, Any]] = []
    agreement_rows: list[dict[str, Any]] = []
    reference_tau = selected_tau
    for variant, result in variant_results.items():
        curve_path = root / f"policy/seed_variants/{variant}_curve.csv"
        curve = pd.read_csv(curve_path)
        row = curve[curve["threshold"].eq(reference_tau)].iloc[0] if reference_tau is not None else None
        robustness_rows.append(
            {
                "variant": variant,
                "seed_map": json.dumps(result["seed_map"], sort_keys=True),
                "own_selected_tau": result["selection"]["selected_tau"],
                "primary_tau": reference_tau,
                "primary_tau_feasible": None if row is None else bool(row["feasible"]),
                "primary_tau_success": None if row is None else float(row["macro_policy_success"]),
                "primary_tau_fresh_tokens": None if row is None else float(row["macro_mean_fresh_tokens"]),
            }
        )
    variants = list(seed_configurations)
    for tau in thresholds:
        for left_index, left in enumerate(variants):
            a = variant_policies[left][variant_policies[left]["threshold"].eq(tau)].copy()
            for right in variants[left_index + 1 :]:
                b = variant_policies[right][variant_policies[right]["threshold"].eq(tau)].copy()
                merged = a.merge(
                    b,
                    on=["base_model", "trace_id"],
                    suffixes=("_left", "_right"),
                    validate="one_to_one",
                )
                fallback_agreement = merged["fallback_left"].eq(merged["fallback_right"])
                exact = fallback_agreement & (
                    merged["fallback_left"]
                    | merged["selected_checkpoint_id_left"].eq(merged["selected_checkpoint_id_right"])
                )
                agreement_rows.append(
                    {
                        "left_variant": left,
                        "right_variant": right,
                        "threshold": float(tau),
                        "is_primary_tau": reference_tau is not None and math.isclose(float(tau), float(reference_tau)),
                        "fallback_agreement": float(fallback_agreement.mean()),
                        "exact_action_checkpoint_agreement": float(exact.mean()),
                    }
                )
    atomic_parquet(root / "policy/seed_robustness.parquet", pd.DataFrame(robustness_rows))
    atomic_parquet(root / "policy/seed_agreement.parquet", pd.DataFrame(agreement_rows))
    selection_presence = [row["selection"]["selected_tau"] is not None for row in variant_results.values()]
    primary_feasibility = [
        row["primary_tau_feasible"]
        for row in robustness_rows
        if row["primary_tau_feasible"] is not None
    ]
    qualitative_stable = bool(
        len(set(selection_presence)) == 1
        and (
            reference_tau is None
            or (len(primary_feasibility) == len(variants) and all(primary_feasibility))
        )
    )
    atomic_json(
        root / "policy/seed_robustness_summary.json",
        {
            "qualitative_tradeoff_stable": qualitative_stable,
            "criterion": "all seed-rank configurations agree on threshold availability and, when primary tau exists, keep it feasible",
            "threshold_availability_by_variant": {
                key: value["selection"]["selected_tau"] is not None
                for key, value in variant_results.items()
            },
            "primary_tau": reference_tau,
        },
    )

    # Outcome-blind fixed baselines.
    baseline_rows = []
    baseline_policy_frames = []
    baseline_prediction_input = operational_predictions[
        [
            "base_model", "trace_id", "common_trace_id", "problem_id", "problem_group", "domain",
            "checkpoint_id", "checkpoint_ordinal", "checkpoint_token_offset", "prefix_token_count",
            "total_trace_token_count", "total_checkpoint_count", "oof_probability",
        ]
    ]
    for name in ("full_regeneration", "always_latest", "always_earliest", "fixed_rewind_25", "fixed_rewind_50", "fixed_rewind_75"):
        actions = _fixed_baseline_actions(baseline_prediction_input, name)
        policy = evaluate_actions(actions, artifact_root=root)
        baseline_rows.append(_baseline_summary(policy, name))
        baseline_policy_frames.append(policy)
    if not np.allclose(
        baseline_rows[1]["macro_mean_fresh_tokens"],
        float(primary_curve.loc[primary_curve["threshold"].eq(0.0), "macro_mean_fresh_tokens"].iloc[0]),
    ):
        raise RuntimeError("always-latest baseline does not equal tau=0 policy")

    position_seed_map = _operational_seed_map(seed_table, "position_only", 1)
    position_predictions = compose_seed_predictions(oof, architecture="position_only", seed_map=position_seed_map)
    position_actions = select_threshold_actions(
        position_predictions[
            [
                "base_model", "trace_id", "common_trace_id", "problem_id", "problem_group", "domain",
                "checkpoint_id", "checkpoint_ordinal", "checkpoint_token_offset", "prefix_token_count",
                "total_trace_token_count", "total_checkpoint_count", "oof_probability",
            ]
        ],
        thresholds,
    )
    position_policy = evaluate_actions(position_actions, artifact_root=root)
    position_intervals, _ = bootstrap_policy_intervals(position_policy, plan, trace_order)
    position_curve, _, _ = summarize_policy(position_policy, position_intervals)
    position_selection = select_frozen_threshold(position_curve)
    position_reference_tau = position_selection["selected_tau"]
    if position_reference_tau is not None:
        position_selected = position_policy[position_policy["threshold"].eq(position_reference_tau)]
        baseline_rows.append(_baseline_summary(position_selected, "position_only_at_own_tau"))
    else:
        position_anchor = position_policy[position_policy["threshold"].eq(1.0)]
        baseline_rows.append(
            _baseline_summary(position_anchor, "position_only_no_validated_threshold_full_regeneration")
        )
    _atomic_csv(root / "baselines/position_only_threshold_curve.csv", position_curve)
    atomic_json(root / "baselines/position_only_selection.json", position_selection)
    atomic_parquet(root / "baselines/baseline_policy_rows.parquet", pd.concat(baseline_policy_frames, ignore_index=True))
    _atomic_csv(root / "baselines/baseline_summary.csv", pd.DataFrame(baseline_rows))

    frontier = _pareto_frontier(primary_curve)
    atomic_parquet(root / "policy/threshold_policy_manifest.parquet", primary_policy)
    _atomic_csv(root / "policy/threshold_curve.csv", primary_curve)
    _atomic_csv(root / "policy/threshold_metrics_by_model.csv", primary_by_model)
    _atomic_csv(root / "policy/selection_distribution.csv", primary_distribution)
    _atomic_csv(root / "policy/success_compute_frontier.csv", frontier)
    atomic_json(root / "policy/threshold_selection.json", selection)
    atomic_json(root / "policy/seed_configuration_results.json", variant_results)

    summary = {
        "status": "COMPLETE",
        "completed_at": now_iso(),
        "selected_tau": selected_tau,
        "threshold_selection": selection,
        "operational_seeds": seed_configurations["operational_median_dev"],
        "position_only_seeds": position_seed_map,
        "oof_predictions": len(oof),
        "cross_fitted_calibrators": len(calibrators),
        "bootstrap_replicates": int(config["threshold_selection"]["bootstrap_replicates"]),
        "threshold_count_including_anchor": len(thresholds),
        "calibration_diagnostic_rows": len(diagnostics),
        "native_evaluation_used": False,
        "teacher_forced_test_used": False,
        "geometry_test_used": False,
        "threshold_grid_refined": False,
    }
    atomic_json(root / "analysis_summary.json", summary)
    return summary
