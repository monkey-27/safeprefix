"""Executable post-download orchestration for recoverability geometry.

The runner performs no generation and accepts only teacher-forced inputs.  It
rejects paths whose components indicate native evaluation before reading them.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch

from .analysis import (
    AffineAxis,
    classify_trajectory,
    compute_branch_metrics,
    compute_h1_metrics,
    compute_h2_metrics,
    compute_margin,
    cross_domain_transfer_summary,
    fit_binomial_glm,
    margin_probability_equivalence,
    select_canonical_seed,
    select_local_branch_parents,
    summarize_seed_stability,
    summarize_trajectory_prevalence,
    trajectory_sensitivity,
)
from .diagnostics import predict_diagnostic, train_diagnostic
from .reporting import write_geometry_outputs


IDENTITY_COLUMNS = ["base_model", "trace_id", "checkpoint_id"]
FORBIDDEN_PATH_TOKENS = {"native", "native_data", "native_eval", "native_evaluation"}


def _guard_non_native(path: Path) -> None:
    lowered = [part.lower() for part in path.parts]
    if any(token in part for part in lowered for token in FORBIDDEN_PATH_TOKENS):
        raise RuntimeError(f"native artifact path is prohibited: {path}")


def _read_table(path: Path) -> pd.DataFrame:
    _guard_non_native(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported table format: {path}")


def _normalised_alias_values(series: pd.Series, *, kind: str) -> pd.Series:
    """Return comparable values for a production/analysis alias pair."""

    if kind == "text":
        return series.astype("string")
    if kind == "integer":
        return pd.to_numeric(series, errors="raise").astype("Int64")
    if kind == "binary":
        values = pd.to_numeric(series, errors="raise").astype("Int64")
        if not values.dropna().isin((0, 1)).all():
            raise RuntimeError("inference outcome alias is not binary")
        return values
    raise ValueError(f"unknown alias kind: {kind}")


def _normalise_alias(
    frame: pd.DataFrame,
    *,
    analysis_name: str,
    inference_name: str,
    kind: str,
) -> None:
    """Create one analysis alias and reject contradictory duplicate fields."""

    if analysis_name not in frame and inference_name in frame:
        frame[analysis_name] = frame[inference_name]
        return
    if analysis_name in frame and inference_name in frame:
        analysis = _normalised_alias_values(frame[analysis_name], kind=kind)
        inference = _normalised_alias_values(frame[inference_name], kind=kind)
        if not analysis.equals(inference):
            raise RuntimeError(
                f"inference aliases disagree: {analysis_name} != {inference_name}"
            )


def _normalize_inference_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    """Bridge the production inference schema to analysis names losslessly.

    Production inference uses ``model_key``, ``binary_outcome`` and
    ``rollout_seed``.  Phase-2 analysis historically uses ``base_model``,
    ``verifier_outcome`` and (for prompt generations) ``generation_seed``.
    Both spellings may be present in newly written artifacts.  In that case we
    verify equality rather than silently preferring either copy.
    """

    output = frame.copy()
    _normalise_alias(
        output,
        analysis_name="base_model",
        inference_name="model_key",
        kind="text",
    )
    _normalise_alias(
        output,
        analysis_name="verifier_outcome",
        inference_name="binary_outcome",
        kind="binary",
    )
    _normalise_alias(
        output,
        analysis_name="generation_seed",
        inference_name="rollout_seed",
        kind="integer",
    )
    return output


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")
    temporary.replace(path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_dense_k32(
    canonical_manifest: pd.DataFrame,
    added_outcomes: pd.DataFrame,
) -> pd.DataFrame:
    """Merge immutable K=4 counts with exactly 28 new executed outcomes."""
    added_outcomes = _normalize_inference_outcomes(added_outcomes)
    test = canonical_manifest.loc[canonical_manifest["split"] == "teacher_forced_test"].copy()
    if test.duplicated(IDENTITY_COLUMNS).any():
        raise RuntimeError("canonical test manifest contains duplicate checkpoints")
    if set(test["num_rollouts"].astype(int)) != {4}:
        raise RuntimeError("canonical labels are not exactly K=4")
    required = {*IDENTITY_COLUMNS, "rollout_seed", "verifier_outcome", "infrastructure_status"}
    if missing := required - set(added_outcomes):
        raise KeyError(f"dense outcomes missing {sorted(missing)}")
    added = added_outcomes.copy()
    if added.duplicated([*IDENTITY_COLUMNS, "rollout_seed"]).any():
        raise RuntimeError("duplicate dense rollout identity")
    if set(added["infrastructure_status"].astype(str)) != {"complete"}:
        raise RuntimeError("unresolved dense infrastructure status")
    counts = added.groupby(IDENTITY_COLUMNS, sort=True).agg(
        new_rollout_count=("rollout_seed", "size"),
        new_success_count=("verifier_outcome", "sum"),
    ).reset_index()
    expected = set(map(tuple, test[IDENTITY_COLUMNS].astype(str).to_numpy()))
    observed = set(map(tuple, counts[IDENTITY_COLUMNS].astype(str).to_numpy()))
    if expected != observed:
        raise RuntimeError(
            f"dense checkpoint set differs: missing={len(expected-observed)}, extra={len(observed-expected)}"
        )
    if set(counts["new_rollout_count"].astype(int)) != {28}:
        raise RuntimeError("every checkpoint must have exactly 28 added outcomes")
    merged = test.merge(counts, on=IDENTITY_COLUMNS, how="left", validate="one_to_one")
    merged.rename(columns={"success_count": "original_k4_success_count"}, inplace=True)
    merged["original_k4_rollout_count"] = 4
    merged["total_success_count"] = merged["original_k4_success_count"] + merged["new_success_count"]
    merged["num_rollouts"] = 32
    merged["dense_recoverability"] = merged["total_success_count"] / 32.0
    return merged


def _build_seed_stability_table(
    scored: pd.DataFrame,
    dense: pd.DataFrame,
    *,
    model: str,
) -> pd.DataFrame:
    """Join K=32 outcomes without shadowing the immutable K=4 trial count.

    ``scored`` inherits ``num_rollouts=4`` from the canonical checkpoint
    manifest.  The dense table independently carries ``num_rollouts=32``.
    Pandas would otherwise suffix both columns during the merge, leaving no
    column literally named ``num_rollouts`` and obscuring which experiment a
    trial count refers to.  Keep both counts, with the dense count explicitly
    named for the seed-stability analysis.
    """

    scored_test = scored.loc[scored["split"] == "teacher_forced_test"].copy()
    if "num_rollouts" not in scored_test:
        raise KeyError("canonical K=4 trial count is absent")
    if set(scored_test["num_rollouts"].astype(int)) != {4}:
        raise RuntimeError("seed-stability source rows are not exactly K=4")
    dense_model = dense.loc[
        dense["base_model"].astype(str) == str(model),
        [*IDENTITY_COLUMNS, "total_success_count", "num_rollouts"],
    ].rename(columns={"num_rollouts": "dense_num_rollouts"})
    if dense_model.duplicated(IDENTITY_COLUMNS).any():
        raise RuntimeError("dense seed-stability table contains duplicate checkpoints")
    if set(dense_model["dense_num_rollouts"].astype(int)) != {32}:
        raise RuntimeError("dense seed-stability trial count is not exactly K=32")
    result = scored_test.merge(
        dense_model,
        on=IDENTITY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    if len(result) != len(scored_test) or len(result) != len(dense_model):
        raise RuntimeError("dense seed-stability checkpoint set differs")
    return result


def aggregate_prompt_solvability(prompt_outcomes: pd.DataFrame) -> pd.DataFrame:
    prompt_outcomes = _normalize_inference_outcomes(prompt_outcomes)
    if "problem_group" not in prompt_outcomes:
        # Unit fixtures and legacy prelaunch tables may contain one row per
        # unique problem ID. Production artifacts always carry problem_group.
        prompt_outcomes["problem_group"] = prompt_outcomes["problem_id"].astype(str)
    required = {"base_model", "problem_id", "problem_group", "generation_seed", "verifier_outcome", "infrastructure_status"}
    if missing := required - set(prompt_outcomes):
        raise KeyError(f"prompt outcomes missing {sorted(missing)}")
    if prompt_outcomes.duplicated(["base_model", "problem_group", "generation_seed"]).any():
        raise RuntimeError("duplicate prompt-solvability outcome")
    if set(prompt_outcomes["infrastructure_status"].astype(str)) != {"complete"}:
        raise RuntimeError("unresolved prompt-solvability infrastructure status")
    result = prompt_outcomes.groupby(["base_model", "problem_group"], sort=True).agg(
        representative_problem_id=("problem_id", "first"),
        prompt_generation_count=("generation_seed", "size"),
        prompt_success_count=("verifier_outcome", "sum"),
    ).reset_index()
    if set(result["prompt_generation_count"].astype(int)) != {16}:
        raise RuntimeError("every problem/model must have exactly 16 prompt generations")
    result["prompt_solvability_raw"] = result["prompt_success_count"] / 16.0
    result["prompt_solvability_smoothed"] = (result["prompt_success_count"] + 0.5) / 17.0
    return result


def _training_path(
    boundary_root: Path,
    model: str,
    seed: int,
    architecture: str = "linear_probe",
    *,
    learning_rate: float | None = None,
) -> Path:
    rate = learning_rate
    if rate is None:
        rate = 3e-4 if architecture == "local_mlp" else 1e-3
    slug = f"lr_{rate:.0e}".replace("+", "")
    return boundary_root / f"training/{model}/{architecture}/{slug}/seed_{seed}/best.pt"


def load_frozen_axes(boundary_root: Path, model: str) -> tuple[dict[int, AffineAxis], int, dict[int, float]]:
    axes: dict[int, AffineAxis] = {}
    dev_nll: dict[int, float] = {}
    for seed in (0, 1, 2):
        path = _training_path(boundary_root, model, seed)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["predictor_config"]["architecture"] != "linear_probe":
            raise RuntimeError(f"{path}: not the frozen linear probe")
        if float(payload["resolved_training_config"]["learning_rate"]) != 1e-3:
            raise RuntimeError(f"{path}: wrong frozen learning rate")
        axes[seed] = AffineAxis.from_state_dict(
            payload["state_dict"], layernorm_epsilon=1e-5
        )
        dev_nll[seed] = float(payload["best_dev_metrics"]["trace_weighted_binomial_nll"])
    return axes, select_canonical_seed(dev_nll), dev_nll


def _load_model_features(boundary_root: Path, model: str) -> np.ndarray:
    payload = torch.load(
        boundary_root / f"data/features/{model}.pt", map_location="cpu", weights_only=False
    )
    features = payload["features"].to(torch.float32).numpy().astype(float)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise RuntimeError(f"{model}: invalid frozen feature store")
    return features


def _score_all_splits(
    model_manifest: pd.DataFrame,
    raw_features: np.ndarray,
    axes: Mapping[int, AffineAxis],
    canonical_seed: int,
) -> tuple[pd.DataFrame, dict[int, np.ndarray], np.ndarray, np.ndarray]:
    indices = model_manifest["feature_row_index"].to_numpy(int)
    if set(indices) != set(range(len(raw_features))):
        raise RuntimeError("feature index is not a bijection")
    raw = raw_features[indices]
    transformed = {seed: axis.transform_raw(raw) for seed, axis in axes.items()}
    scores = {seed: axes[seed].logit_from_feature(transformed[seed]) for seed in axes}
    canonical_feature = transformed[canonical_seed]
    decomposition = axes[canonical_seed].decompose(canonical_feature)
    output = model_manifest.copy()
    for seed, values in scores.items():
        output[f"seed_{seed}_raw_logit"] = values
    output["canonical_seed"] = canonical_seed
    output["canonical_raw_logit"] = scores[canonical_seed]
    return output, scores, canonical_feature, decomposition["orthogonal_component"]


def _save_fit(path: Path, fit: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": fit.kind,
            "seed": fit.seed,
            "best_epoch": fit.best_epoch,
            "best_dev_nll": fit.best_dev_nll,
            "train_trace_count": fit.train_trace_count,
            "dev_trace_count": fit.dev_trace_count,
            "state_dict": fit.state_dict,
            "history": list(fit.history),
            "resolved_protocol": {
                "objective": "trace_weighted_binomial_nll_on_individual_k4_counts",
                "max_epochs": 50,
                "patience": 7,
                "batch_size_traces": 64,
                "learning_rate": 3e-4 if fit.kind in {"orthogonal_mlp", "axis_plus_residual"} else 1e-3,
                "weight_decay": 1e-3 if fit.kind in {"orthogonal_mlp", "axis_plus_residual"} else 1e-4,
                "test_labels_used": False,
            },
        },
        path,
    )


def _train_h1_diagnostics(
    scored: pd.DataFrame,
    canonical_feature: np.ndarray,
    orthogonal: np.ndarray,
    output_root: Path,
) -> pd.DataFrame:
    split_arrays: dict[str, dict[str, Any]] = {}
    for split in ("train", "architecture_dev", "teacher_forced_test"):
        mask = scored["split"].astype(str).to_numpy() == split
        split_arrays[split] = {
            "mask": mask,
            "axis": scored.loc[mask, "canonical_raw_logit"].to_numpy(float),
            "orthogonal": orthogonal[mask],
            "success": scored.loc[mask, "success_count"].to_numpy(float),
            "trials": scored.loc[mask, "num_rollouts"].to_numpy(float),
            "traces": scored.loc[mask, "trace_id"].astype(str).to_numpy(),
        }
    predictions = scored.loc[split_arrays["teacher_forced_test"]["mask"], IDENTITY_COLUMNS].copy()
    train, dev, test = (split_arrays[name] for name in ("train", "architecture_dev", "teacher_forced_test"))
    for kind in ("axis_only", "orthogonal_linear", "orthogonal_mlp", "axis_plus_residual"):
        seeds = (0,) if kind == "axis_only" else (0, 1, 2)
        values = []
        for seed in seeds:
            fit = train_diagnostic(
                kind=kind,
                train_axis_score=train["axis"], train_features=train["orthogonal"],
                train_successes=train["success"], train_trials=train["trials"], train_trace_ids=train["traces"],
                dev_axis_score=dev["axis"], dev_features=dev["orthogonal"],
                dev_successes=dev["success"], dev_trials=dev["trials"], dev_trace_ids=dev["traces"],
                seed=seed,
            )
            _save_fit(output_root / f"diagnostics/{kind}/seed_{seed}.pt", fit)
            seed_values = predict_diagnostic(
                fit, axis_score=test["axis"], features=test["orthogonal"]
            )
            predictions[f"{kind}_seed_{seed}_logit"] = seed_values
            values.append(seed_values)
        predictions[f"{kind}_logit"] = np.mean(values, axis=0)
    return predictions


def _crossfit_position_prompt_control(frame: pd.DataFrame, folds: int = 5) -> np.ndarray:
    """Transparent test cross-fit control grouped by related problem.

    Prompt solvability exists only for the frozen test problems.  This control
    is consequently an explicitly reported cross-fit descriptive baseline: no
    row, trace, or related problem-group is predicted from its own K=32 label.
    It is never used to train the primary probe or select a hypothesis.
    """
    if "problem_group" not in frame:
        raise KeyError("position-plus-prompt crossfit requires problem_group")
    groups = sorted(frame["problem_group"].astype(str).unique())
    fold_for_group = {
        group: int(hashlib.sha256(group.encode()).hexdigest(), 16) % folds for group in groups
    }
    prompt = np.clip(frame["prompt_solvability_smoothed"].to_numpy(float), 1e-6, 1 - 1e-6)
    design = np.column_stack([
        np.ones(len(frame)),
        frame["checkpoint_ordinal"].to_numpy(float) / np.maximum(frame["total_checkpoint_count"].to_numpy(float) - 1, 1),
        frame["prefix_token_count"].to_numpy(float) / np.maximum(frame["total_trace_token_count"].to_numpy(float), 1),
        np.log(prompt / (1 - prompt)),
    ])
    fold_ids = frame["problem_group"].astype(str).map(fold_for_group).to_numpy(int)
    output = np.full(len(frame), np.nan, dtype=float)
    for fold in range(folds):
        train, test = fold_ids != fold, fold_ids == fold
        if not test.any() or not train.any():
            continue
        fit = fit_binomial_glm(
            design[train],
            frame.loc[train, "total_success_count"],
            np.full(train.sum(), 32),
            cluster_ids=frame.loc[train, "trace_id"],
        )
        output[test] = design[test] @ fit.coefficients
    if not np.isfinite(output).all():
        raise RuntimeError("position-plus-prompt crossfit did not cover every trace")
    return output


def _train_cross_domain(
    scored: pd.DataFrame,
    raw_features: np.ndarray,
    global_axis: AffineAxis,
    global_calibrator: Mapping[str, float],
    output_root: Path,
) -> pd.DataFrame:
    domains = sorted(scored.loc[scored["split"] == "train", "domain"].astype(str).unique())
    rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    for held_or_train in [*domains, *(f"leave_out:{domain}" for domain in domains)]:
        train_domain = held_or_train.removeprefix("leave_out:")
        if held_or_train.startswith("leave_out:"):
            train_mask = (scored["split"] == "train") & (scored["domain"].astype(str) != train_domain)
            dev_mask = (scored["split"] == "architecture_dev") & (scored["domain"].astype(str) != train_domain)
        else:
            train_mask = (scored["split"] == "train") & (scored["domain"].astype(str) == train_domain)
            dev_mask = (scored["split"] == "architecture_dev") & (scored["domain"].astype(str) == train_domain)
        if scored.loc[train_mask, "trace_id"].nunique() < 2 or scored.loc[dev_mask, "trace_id"].nunique() < 1:
            continue
        train_index = scored.loc[train_mask, "feature_row_index"].to_numpy(int)
        dev_index = scored.loc[dev_mask, "feature_row_index"].to_numpy(int)
        fitted_axes: dict[int, AffineAxis] = {}
        held_rows: list[dict[str, Any]] = []
        for seed in (0, 1, 2):
            fit = train_diagnostic(
                kind="layernorm_linear",
                train_axis_score=np.zeros(len(train_index)), train_features=raw_features[train_index],
                train_successes=scored.loc[train_mask, "success_count"].to_numpy(float),
                train_trials=scored.loc[train_mask, "num_rollouts"].to_numpy(float),
                train_trace_ids=scored.loc[train_mask, "trace_id"].astype(str).to_numpy(),
                dev_axis_score=np.zeros(len(dev_index)), dev_features=raw_features[dev_index],
                dev_successes=scored.loc[dev_mask, "success_count"].to_numpy(float),
                dev_trials=scored.loc[dev_mask, "num_rollouts"].to_numpy(float),
                dev_trace_ids=scored.loc[dev_mask, "trace_id"].astype(str).to_numpy(),
                seed=seed,
            )
            _save_fit(output_root / f"cross_domain/{held_or_train}/seed_{seed}.pt", fit)
            fitted_axis = AffineAxis.from_state_dict(
                fit.state_dict, layernorm_epsilon=1e-5, prefix="model"
            )
            fitted_axes[seed] = fitted_axis
            direction_cosine = float(
                np.dot(fitted_axis.weight, global_axis.weight)
                / (fitted_axis.norm * global_axis.norm)
            )
            for test_domain in domains:
                test_mask = (scored["split"] == "teacher_forced_test") & (scored["domain"].astype(str) == test_domain)
                test_index = scored.loc[test_mask, "feature_row_index"].to_numpy(int)
                if not len(test_index):
                    continue
                logits = predict_diagnostic(fit, axis_score=np.zeros(len(test_index)), features=raw_features[test_index])
                temporary = scored.loc[test_mask, ["trace_id", "success_count", "num_rollouts"]].copy()
                temporary["logit"] = logits
                global_logits = scored.loc[test_mask, "canonical_raw_logit"].to_numpy(float)
                prediction_frame = scored.loc[
                    test_mask,
                    [*IDENTITY_COLUMNS, "problem_group", "domain", "success_count", "num_rollouts"],
                ].copy()
                prediction_frame["train_domain"] = held_or_train
                prediction_frame["test_domain"] = test_domain
                prediction_frame["training_seed"] = seed
                prediction_frame["domain_probe_raw_logit"] = logits
                prediction_frame["global_canonical_raw_logit"] = global_logits
                prediction_frame["test_calibrator_fit"] = False
                prediction_rows.append(prediction_frame)
                probability = 1 / (1 + np.exp(-np.clip(logits, -60, 60)))
                target = temporary["success_count"] / temporary["num_rollouts"]
                temporary["loss"] = -(target*np.log(probability)+(1-target)*np.log(1-probability))
                nll = float(temporary.groupby("trace_id")["loss"].mean().mean())
                global_calibrated = (
                    float(global_calibrator["a"]) * global_logits
                    + float(global_calibrator["b"])
                )
                global_probability = 1 / (
                    1 + np.exp(-np.clip(global_calibrated, -60, 60))
                )
                temporary["global_loss"] = -(
                    target * np.log(global_probability)
                    + (1 - target) * np.log(1 - global_probability)
                )
                global_nll = float(
                    temporary.groupby("trace_id")["global_loss"].mean().mean()
                )
                ranking = float(temporary.assign(target=target).groupby("trace_id").apply(
                    lambda part: part["logit"].corr(part["target"], method="spearman"), include_groups=False
                ).dropna().mean())
                held_rows.append({
                    "train_domain": held_or_train,
                    "test_domain": test_domain,
                    "seed": seed,
                    "dense_nll": nll,
                    "within_trace_ranking": ranking,
                    "direction_cosine_to_global": direction_cosine,
                    "logit_correlation_with_global": float(
                        np.corrcoef(logits, global_logits)[0, 1]
                    ) if len(logits) > 1 and np.std(logits) > 0 and np.std(global_logits) > 0 else float("nan"),
                    "test_calibrator_fit": False,
                    "shared_global_frozen_calibrated_nll": global_nll,
                })
        pair_cosines = []
        for left in (0, 1, 2):
            for right in range(left + 1, 3):
                pair_cosines.append(
                    float(
                        np.dot(fitted_axes[left].weight, fitted_axes[right].weight)
                        / (fitted_axes[left].norm * fitted_axes[right].norm)
                    )
                )
        median_pair_cosine = float(np.median(pair_cosines))
        for row in held_rows:
            row["direction_cosine_across_seeds"] = median_pair_cosine
        rows.extend(held_rows)
    if prediction_rows:
        _atomic_parquet(
            output_root / "cross_domain/per_checkpoint_logits.parquet",
            pd.concat(prediction_rows, ignore_index=True),
        )
    return pd.DataFrame(rows)


def _train_split_half_reproducibility(
    scored: pd.DataFrame,
    raw_features: np.ndarray,
    output_root: Path,
) -> dict[str, Any]:
    """Train-only deterministic split-half direction reproducibility audit."""
    train = scored.loc[scored["split"] == "train"].copy()
    dev = scored.loc[scored["split"] == "architecture_dev"].copy()
    groups = train["problem_group"].astype(str)
    half = groups.map(
        lambda value: int(hashlib.sha256(f"split-half||{value}".encode()).hexdigest(), 16) % 2
    ).to_numpy(int)
    if set(half) != {0, 1}:
        raise RuntimeError("split-half partition is degenerate")
    fitted: dict[int, dict[int, AffineAxis]] = {0: {}, 1: {}}
    for half_id in (0, 1):
        train_part = train.loc[half == half_id]
        train_index = train_part["feature_row_index"].to_numpy(int)
        dev_index = dev["feature_row_index"].to_numpy(int)
        for seed in (0, 1, 2):
            fit = train_diagnostic(
                kind="layernorm_linear",
                train_axis_score=np.zeros(len(train_part)),
                train_features=raw_features[train_index],
                train_successes=train_part["success_count"].to_numpy(float),
                train_trials=train_part["num_rollouts"].to_numpy(float),
                train_trace_ids=train_part["trace_id"].astype(str).to_numpy(),
                dev_axis_score=np.zeros(len(dev)),
                dev_features=raw_features[dev_index],
                dev_successes=dev["success_count"].to_numpy(float),
                dev_trials=dev["num_rollouts"].to_numpy(float),
                dev_trace_ids=dev["trace_id"].astype(str).to_numpy(),
                seed=seed,
            )
            _save_fit(output_root / f"split_half/half_{half_id}/seed_{seed}.pt", fit)
            fitted[half_id][seed] = AffineAxis.from_state_dict(
                fit.state_dict, layernorm_epsilon=1e-5, prefix="model"
            )
    rows = []
    for left_seed in (0, 1, 2):
        for right_seed in (0, 1, 2):
            left, right = fitted[0][left_seed], fitted[1][right_seed]
            rows.append(
                {
                    "left_half_seed": left_seed,
                    "right_half_seed": right_seed,
                    "effective_direction_cosine": float(
                        np.dot(left.weight, right.weight) / (left.norm * right.norm)
                    ),
                }
            )
    values = np.asarray([row["effective_direction_cosine"] for row in rows])
    return {
        "partition_unit": "problem_group",
        "partition_hash": "sha256(split-half||problem_group) mod 2",
        "half_trace_counts": {
            str(value): int(train.loc[half == value, "trace_id"].nunique())
            for value in (0, 1)
        },
        "pairwise": rows,
        "median_effective_direction_cosine": float(np.median(values)),
        "minimum_effective_direction_cosine": float(np.min(values)),
    }


def _score_existing_control(
    boundary_root: Path,
    scored_test: pd.DataFrame,
    raw_features: np.ndarray,
    *,
    architecture: str,
) -> pd.DataFrame:
    from safeprefix.boundary_v1.models import build_predictor
    from safeprefix.boundary_v1.training import position_features

    feature_index = scored_test["feature_row_index"].to_numpy(int)
    hidden = torch.tensor(raw_features[feature_index], dtype=torch.float32)[None, :, :]
    positions = position_features(scored_test)[None, :, :]
    mask = torch.ones((1, len(scored_test)), dtype=torch.bool)
    values = []
    model = str(scored_test["base_model"].iloc[0])
    for seed in (0, 1, 2):
        payload = torch.load(
            _training_path(boundary_root, model, seed, architecture),
            map_location="cpu",
            weights_only=False,
        )
        predictor = build_predictor(
            architecture,
            int(payload["predictor_config"]["input_dim"]),
            hidden_width=int(payload["predictor_config"]["hidden_width"]),
            dropout=float(payload["predictor_config"]["dropout"]),
        )
        predictor.load_state_dict(payload["state_dict"])
        predictor.eval()
        with torch.inference_mode():
            values.append(predictor(hidden, positions, mask).squeeze(0).numpy())
    output = scored_test[IDENTITY_COLUMNS].copy().reset_index(drop=True)
    for seed, seed_values in enumerate(values):
        output[f"existing_{architecture}_seed_{seed}_logit"] = seed_values
    output[f"existing_{architecture}_logit"] = np.mean(values, axis=0)
    return output


def run_post_download_analysis(
    *,
    boundary_root: Path,
    dense_outcomes_path: Path,
    prompt_outcomes_path: Path,
    output_root: Path,
    published_root: Path | None = None,
) -> dict[str, Any]:
    for path in (boundary_root, dense_outcomes_path, prompt_outcomes_path, output_root):
        _guard_non_native(path)
    canonical_path = boundary_root / "data/canonical_checkpoint_manifest.parquet"
    canonical = pd.read_parquet(canonical_path)
    dense = merge_dense_k32(canonical, _read_table(dense_outcomes_path))
    prompt = aggregate_prompt_solvability(_read_table(prompt_outcomes_path))
    dense = dense.merge(
        prompt,
        on=["base_model", "problem_group"],
        how="left",
        validate="many_to_one",
    )
    if dense["prompt_solvability_smoothed"].isna().any():
        raise RuntimeError("prompt solvability is missing for a dense checkpoint")
    _atomic_parquet(output_root / "tables/merged_k32_checkpoint_table.parquet", dense)
    _atomic_parquet(output_root / "tables/prompt_solvability.parquet", prompt)

    model_summaries: dict[str, Any] = {}
    all_scored: list[pd.DataFrame] = []
    all_diagnostic: list[pd.DataFrame] = []
    all_transfer: list[pd.DataFrame] = []
    trajectory_rows: list[dict[str, Any]] = []
    for model in sorted(canonical["base_model"].astype(str).unique()):
        model_manifest = canonical.loc[canonical["base_model"].astype(str) == model].copy().sort_values("feature_row_index")
        features = _load_model_features(boundary_root, model)
        axes, canonical_seed, dev_nll = load_frozen_axes(boundary_root, model)
        scored, _, canonical_feature, orthogonal = _score_all_splits(model_manifest, features, axes, canonical_seed)
        stability_test = _build_seed_stability_table(
            scored,
            dense,
            model=model,
        )
        for seed in (0, 1, 2):
            frozen_calibrator = json.loads(
                (
                    boundary_root
                    / f"calibration/{model}/seed_{seed}/calibrator.json"
                ).read_text()
            )
            stability_test[f"seed_{seed}_frozen_calibrated_logit"] = (
                float(frozen_calibrator["a"]) * stability_test[f"seed_{seed}_raw_logit"]
                + float(frozen_calibrator["b"])
            )
        stability = summarize_seed_stability(
            stability_test,
            score_columns={seed: f"seed_{seed}_raw_logit" for seed in (0, 1, 2)},
            axes=axes,
            success_column="total_success_count",
            trials_column="dense_num_rollouts",
            dense_score_columns={
                seed: f"seed_{seed}_frozen_calibrated_logit" for seed in (0, 1, 2)
            },
        )
        stability["split_half_train_direction_reproducibility"] = (
            _train_split_half_reproducibility(scored, features, output_root / model)
        )
        diagnostics = _train_h1_diagnostics(scored, canonical_feature, orthogonal, output_root / model)
        diagnostic_test = dense.loc[dense["base_model"].astype(str) == model].merge(diagnostics, on=IDENTITY_COLUMNS, validate="one_to_one")
        scored_test = scored.loc[scored["split"] == "teacher_forced_test", [*IDENTITY_COLUMNS, "canonical_raw_logit", "seed_0_raw_logit", "seed_1_raw_logit", "seed_2_raw_logit"]]
        diagnostic_test = diagnostic_test.merge(scored_test, on=IDENTITY_COLUMNS, validate="one_to_one")
        diagnostic_test["success_count"] = diagnostic_test["total_success_count"]
        diagnostic_test["num_rollouts"] = 32
        canonical_frozen_calibrator = json.loads(
            (
                boundary_root
                / f"calibration/{model}/seed_{canonical_seed}/calibrator.json"
            ).read_text()
        )
        diagnostic_test["canonical_frozen_calibrated_logit"] = (
            float(canonical_frozen_calibrator["a"])
            * diagnostic_test["canonical_raw_logit"]
            + float(canonical_frozen_calibrator["b"])
        )
        diagnostic_test["canonical_frozen_calibrated_probability"] = 1 / (
            1 + np.exp(-np.clip(diagnostic_test["canonical_frozen_calibrated_logit"], -60, 60))
        )
        diagnostic_test["canonical_signed_margin"] = (
            diagnostic_test["canonical_raw_logit"]
            - (-float(canonical_frozen_calibrator["b"]) / float(canonical_frozen_calibrator["a"]))
        ) / axes[canonical_seed].norm
        diagnostic_test["position_prompt_control_logit"] = _crossfit_position_prompt_control(diagnostic_test)
        scored_test_rows = scored.loc[scored["split"] == "teacher_forced_test"].copy()
        for architecture in ("local_mlp", "position_only"):
            control = _score_existing_control(
                boundary_root, scored_test_rows, features, architecture=architecture
            )
            diagnostic_test = diagnostic_test.merge(
                control, on=IDENTITY_COLUMNS, validate="one_to_one"
            )
        h1 = compute_h1_metrics(
            diagnostic_test,
            model_logits={
                "axis_only": "axis_only_logit",
                "orthogonal_linear": "orthogonal_linear_logit",
                "orthogonal_mlp": "orthogonal_mlp_logit",
                "axis_plus_residual": "axis_plus_residual_logit",
                "full_state": "canonical_frozen_calibrated_logit",
                "existing_local_mlp": "existing_local_mlp_logit",
                "existing_position_only": "existing_position_only_logit",
                "position_prompt_control": "position_prompt_control_logit",
            },
            best_candidates=("axis_plus_residual", "full_state"),
        )
        h2_frame = diagnostic_test.copy()
        h2_frame["normalized_checkpoint_ordinal"] = h2_frame["checkpoint_ordinal"] / np.maximum(h2_frame["total_checkpoint_count"] - 1, 1)
        h2_frame["normalized_checkpoint_token_position"] = h2_frame["prefix_token_count"] / np.maximum(h2_frame["total_trace_token_count"], 1)
        h2 = compute_h2_metrics(h2_frame)
        calibrator = json.loads(
            (
                boundary_root
                / f"calibration/{model}/seed_{canonical_seed}/calibrator.json"
            ).read_text()
        )
        calibrator_a = float(calibrator["a"])
        calibrator_b = float(calibrator["b"])
        for trace_id, part in diagnostic_test.sort_values("checkpoint_ordinal").groupby("trace_id", sort=True):
            result = classify_trajectory(part["total_success_count"], np.full(len(part), 32))
            one_change = result["models"].get("one_change", {})
            change_points = one_change.get("change_points", ())
            collapse_checkpoint = int(change_points[0]) if result["category"] == "single_collapse" and change_points else None
            first_error = int(part["first_error_zero_based_analysis_only"].iloc[0])
            calibrated_logit = calibrator_a * part["canonical_raw_logit"].to_numpy(float) + calibrator_b
            downward = np.flatnonzero(
                (calibrated_logit[:-1] >= 0) & (calibrated_logit[1:] < 0)
            )
            axis_crossing = int(downward[0] + 1) if len(downward) else None
            collapse_token_position = (
                float(part.iloc[collapse_checkpoint]["prefix_token_count"])
                / max(float(part.iloc[collapse_checkpoint]["total_trace_token_count"]), 1.0)
                if collapse_checkpoint is not None and collapse_checkpoint < len(part)
                else None
            )
            trajectory_rows.append({
                "base_model": model,
                "trace_id": trace_id,
                "common_trace_id": str(part["common_trace_id"].iloc[0]),
                "problem_group": str(part["problem_group"].iloc[0]),
                "domain": str(part["domain"].iloc[0]),
                "checkpoint_count": int(len(part)),
                **{key: value for key, value in result.items() if key != "models"},
                "total_change": float(result["model_free"]["total_change"]),
                "largest_adjacent_drop": float(result["model_free"]["largest_adjacent_drop"]),
                "largest_adjacent_recovery": float(result["model_free"]["largest_adjacent_recovery"]),
                "position_spearman": float(result["model_free"]["position_spearman"]),
                "models": json.dumps(result["models"], default=_json_default),
                "sensitivity": json.dumps(
                    trajectory_sensitivity(
                        part["total_success_count"], np.full(len(part), 32)
                    )
                ),
                "collapse_checkpoint": collapse_checkpoint,
                "collapse_normalized_token_position": collapse_token_position,
                "axis_downward_crossing_checkpoint": axis_crossing,
                "first_error_checkpoint": first_error,
                "collapse_minus_first_error": (
                    collapse_checkpoint-first_error if collapse_checkpoint is not None else None
                ),
                "axis_crossing_minus_first_error": (
                    axis_crossing-first_error if axis_crossing is not None else None
                ),
                "first_error_normalized_token_position": None,
                "first_error_token_position_available": False,
            })
        transfer = _train_cross_domain(
            scored,
            features,
            axes[canonical_seed],
            calibrator,
            output_root / model,
        )
        if not transfer.empty:
            transfer["base_model"] = model
            all_transfer.append(transfer)
        _atomic_json(output_root / f"analyses/{model}/axis_manifest.json", {"canonical_seed": canonical_seed, "dev_nll": dev_nll, "audit": axes[canonical_seed].audit(), "stability": stability})
        _atomic_json(output_root / f"analyses/{model}/h1.json", h1)
        _atomic_json(output_root / f"analyses/{model}/h2.json", h2)
        _atomic_parquet(output_root / f"tables/{model}_checkpoint_predictions.parquet", diagnostic_test)
        all_scored.append(scored)
        all_diagnostic.append(diagnostic_test)
        model_summaries[model] = {"canonical_seed": canonical_seed, "stability": stability, "H1": h1, "H2": h2}

    checkpoints = pd.concat(all_diagnostic, ignore_index=True)
    trajectories = pd.DataFrame(trajectory_rows)
    transfers = pd.concat(all_transfer, ignore_index=True) if all_transfer else pd.DataFrame()
    _atomic_parquet(output_root / "tables/checkpoint_predictions.parquet", checkpoints)
    _atomic_parquet(output_root / "tables/trajectory_classifications.parquet", trajectories)
    h3_summary = summarize_trajectory_prevalence(trajectories)
    _atomic_json(output_root / "analyses/h3_trajectory_prevalence.json", h3_summary)
    if not transfers.empty:
        _atomic_parquet(output_root / "tables/cross_domain_transfer.parquet", transfers)
    parents, parent_audit = select_local_branch_parents(checkpoints)
    _atomic_parquet(output_root / "manifests/local_parent_manifest.parquet", parents)
    _atomic_parquet(output_root / "manifests/local_parent_selection_audit.parquet", parent_audit)
    summary = {
        "status": "PHASE2_COMPLETE_H4_PARENT_BRIDGE_READY",
        "native_artifacts_accessed": False,
        "new_calibrator_fit": False,
        "operational_tau_selected": False,
        "unique_checkpoint_model_pairs": int(len(dense)),
        "seed_expanded_prediction_rows": int(len(dense) * 3),
        "dense_new_rollouts": int(len(dense) * 28),
        "prompt_generations": int(len(prompt) * 16),
        "models": model_summaries,
        "cross_domain": cross_domain_transfer_summary(transfers) if not transfers.empty else None,
        "H3": h3_summary,
        "local_parents_selected": int(len(parents)),
        "input_hashes": {"canonical_manifest": _hash_file(canonical_path), "dense_outcomes": _hash_file(dense_outcomes_path), "prompt_outcomes": _hash_file(prompt_outcomes_path)},
    }
    _atomic_json(output_root / "summary.json", summary)
    write_geometry_outputs(output_root, checkpoints=checkpoints, trajectories=trajectories, transfers=transfers, parents=parents, summary=summary)
    if published_root is not None:
        _guard_non_native(published_root)
        if published_root.exists():
            shutil.rmtree(published_root)
        published_root.mkdir(parents=True)
        for relative in ("summary.json", "reports", "figures", "manifests/local_parent_manifest.parquet"):
            source = output_root / relative
            destination = published_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, destination)
            elif source.is_file():
                shutil.copy2(source, destination)
    return summary


def run_phase4_analysis(
    *,
    boundary_root: Path,
    child_states_path: Path,
    output_root: Path,
    published_root: Path | None = None,
) -> dict[str, Any]:
    """Score frozen child states and finish H4/margin reports without retraining."""
    for path in (boundary_root, child_states_path, output_root):
        _guard_non_native(path)
    children = _read_table(child_states_path)
    required = {
        "base_model", "parent_id", "branch_id", "horizon", "raw_hidden",
        "parent_raw_hidden", "child_success_count", "child_num_rollouts",
        "horizon_available", "parent_recoverability",
    }
    if missing := required - set(children):
        raise KeyError(f"child-state table missing {sorted(missing)}")
    if children.duplicated(["base_model", "parent_id", "branch_id", "horizon"]).any():
        raise RuntimeError("duplicate local child state")
    scored_parts = []
    for model, part in children.groupby("base_model", sort=True):
        axes, canonical_seed, _ = load_frozen_axes(boundary_root, str(model))
        axis = axes[canonical_seed]
        output = part.copy()
        available = (
            output["horizon_available"].astype(bool)
            & output["raw_hidden"].notna()
            & output["parent_raw_hidden"].notna()
            & (output["child_num_rollouts"].astype(int) > 0)
        )
        output["canonical_seed"] = canonical_seed
        for column in (
            "child_score", "parent_score", "delta_r",
            "orthogonal_displacement_norm", "child_recoverability", "signed_margin",
            "child_frozen_calibrated_probability",
        ):
            output[column] = np.nan
        output["orthogonal_displacement"] = None
        if not available.any():
            scored_parts.append(output)
            continue
        child_raw = np.stack(output.loc[available, "raw_hidden"].map(np.asarray))
        parent_raw = np.stack(output.loc[available, "parent_raw_hidden"].map(np.asarray))
        child_feature = axis.transform_raw(child_raw)
        parent_feature = axis.transform_raw(parent_raw)
        child_score = axis.logit_from_feature(child_feature)
        parent_score = axis.logit_from_feature(parent_feature)
        displacement = child_feature - parent_feature
        decomposition = axis.decompose(displacement)
        calibrator = json.loads((boundary_root / f"calibration/{model}/seed_{canonical_seed}/calibrator.json").read_text())
        output.loc[available, "child_score"] = child_score
        output.loc[available, "parent_score"] = parent_score
        output.loc[available, "delta_r"] = child_score - parent_score
        output.loc[available, "orthogonal_displacement_norm"] = np.linalg.norm(
            decomposition["orthogonal_component"], axis=1
        )
        output.loc[available, "orthogonal_displacement"] = pd.Series(
            [value.tolist() for value in decomposition["orthogonal_component"]],
            index=output.index[available],
            dtype=object,
        )
        output.loc[available, "child_recoverability"] = (
            output.loc[available, "child_success_count"].to_numpy(float)
            / output.loc[available, "child_num_rollouts"].to_numpy(float)
        )
        output.loc[available, "signed_margin"] = compute_margin(
            child_feature,
            axis,
            calibrator_a=float(calibrator["a"]),
            calibrator_b=float(calibrator["b"]),
        )
        calibrated_child_logit = (
            float(calibrator["a"]) * child_score + float(calibrator["b"])
        )
        output.loc[available, "child_frozen_calibrated_probability"] = np.clip(
            1 / (1 + np.exp(-np.clip(calibrated_child_logit, -60, 60))),
            1e-12,
            1 - 1e-12,
        )
        for seed in (0, 1, 2):
            output[f"seed_{seed}_child_score"] = np.nan
            output.loc[available, f"seed_{seed}_child_score"] = axes[seed].logit_from_raw(
                child_raw
            )
        scored_parts.append(output)
    scored = pd.concat(scored_parts, ignore_index=True)
    branch = {
        "pooled": compute_branch_metrics(scored),
        "by_model": {
            str(model): compute_branch_metrics(part)
            for model, part in scored.groupby("base_model", sort=True)
        },
        "available_child_state_rows": int(scored["horizon_available"].astype(bool).sum()),
        "unavailable_child_state_rows": int((~scored["horizon_available"].astype(bool)).sum()),
    }
    # Parent is the independent unit for branch inference.  Bootstrap the
    # within-parent axis separation at each horizon; unavailable horizons are
    # retained as attrition, not imputed.
    from .statistics import clustered_bootstrap, holm_correction
    branch_bootstrap: dict[str, Any] = {}
    for horizon, part in scored.loc[scored["horizon_available"].astype(bool)].groupby(
        "horizon", sort=True
    ):
        rates = part["child_success_count"] / part["child_num_rollouts"]
        temporary = part.assign(_rate=rates)
        gaps = []
        parent_ids = []
        for parent_id, parent in temporary.groupby("parent_id", sort=True):
            high = parent.loc[parent["_rate"] >= 0.75, "delta_r"]
            low = parent.loc[parent["_rate"] <= 0.25, "delta_r"]
            if len(high) and len(low):
                parent_ids.append(str(parent_id))
                gaps.append(float(high.mean() - low.mean()))
        if gaps:
            branch_bootstrap[str(horizon)] = clustered_bootstrap(
                parent_ids, gaps, replicates=10_000, seed=20260728 + int(horizon)
            )
    branch["parent_clustered_bootstrap"] = branch_bootstrap
    available_scored = scored.loc[scored["horizon_available"].astype(bool)].copy()
    margin_equivalence = margin_probability_equivalence(
        available_scored["signed_margin"],
        available_scored["child_frozen_calibrated_probability"],
    )
    _atomic_parquet(output_root / "tables/local_child_states.parquet", scored)
    _atomic_json(output_root / "analyses/h4_branch_metrics.json", branch)
    summary = json.loads((output_root / "summary.json").read_text())
    summary["status"] = "FINALIZING_INTEGRITY_CHECK"
    summary["H4"] = branch
    summary["margin_and_robustness"] = {
        "child_margin_probability_equivalence": margin_equivalence,
        "incremental_margin_model_fit": False,
        "reason": (
            "signed margin and frozen calibrated probability are one-to-one transforms "
            "of the same canonical score; independent incremental information is not identifiable"
        ),
    }
    summary["local_child_states"] = int(len(scored))
    summary["child_terminal_completions"] = int(scored["child_num_rollouts"].sum())
    primary_p: dict[str, float] = {}
    for family in ("H1", "H2"):
        values = []
        for model_summary in summary.get("models", {}).values():
            if family == "H1":
                item = model_summary.get("H1", {}).get("axis_plus_residual_improvement")
                if item is not None:
                    values.append(float(item["two_sided_sign_p"]))
            else:
                coefficient = model_summary.get("H2", {}).get("strict_within_trace", {})
                if coefficient.get("standard_error", 0) > 0:
                    from scipy.stats import norm
                    z = abs(float(coefficient["coefficient"]) / float(coefficient["standard_error"]))
                    values.append(float(2 * norm.sf(z)))
        if values:
            # Conservative across-model consistency summary; detailed per-model
            # intervals remain primary.
            primary_p[family] = max(values)
    h4_p = [
        float(value["two_sided_sign_p"])
        for value in branch_bootstrap.values()
        if "two_sided_sign_p" in value
    ]
    if h4_p:
        primary_p["H4"] = max(h4_p)
    summary["holm_correction"] = {
        "reported_primary_families": sorted(primary_p),
        "H3_note": "H3 is a pre-registered descriptive BIC classification with no null significance test",
        "results": holm_correction(primary_p) if primary_p else {},
    }
    summary["child_state_input_hash"] = _hash_file(child_states_path)
    _atomic_json(output_root / "summary.json", summary)
    checkpoints = pd.read_parquet(output_root / "tables/checkpoint_predictions.parquet")
    trajectories = pd.read_parquet(output_root / "tables/trajectory_classifications.parquet")
    transfer_path = output_root / "tables/cross_domain_transfer.parquet"
    transfers = pd.read_parquet(transfer_path) if transfer_path.is_file() else pd.DataFrame()
    parents = pd.read_parquet(output_root / "manifests/local_parent_manifest.parquet")
    output_status = write_geometry_outputs(
        output_root,
        checkpoints=checkpoints,
        trajectories=trajectories,
        transfers=transfers,
        parents=parents,
        summary=summary,
        children=scored,
    )
    integrity_failures: list[str] = []
    if int(summary.get("unique_checkpoint_model_pairs", -1)) != 1908:
        integrity_failures.append("teacher-forced test checkpoint/model count is not 1,908")
    if int(summary.get("dense_new_rollouts", -1)) != 53_424:
        integrity_failures.append("dense K=28 added rollout count is not 53,424")
    if int(summary.get("prompt_generations", -1)) != 7_552:
        integrity_failures.append("prompt generation count is not 7,552")
    if len(parents) != 160:
        integrity_failures.append(f"local parent count is {len(parents)}, expected 160")
    if not output_status["required_output_validation"]["passed"]:
        integrity_failures.append("required reports or figures are missing")
    summary["final_integrity"] = {
        "passed": not integrity_failures,
        "failures": integrity_failures,
        "required_outputs": output_status["required_output_validation"],
    }
    summary["status"] = "COMPLETE" if not integrity_failures else "INCOMPLETE_INTEGRITY_FAILURE"
    _atomic_json(output_root / "summary.json", summary)
    # Rewrite reports once, after final status is known, so Markdown never
    # presents a stale pre-integrity state as complete.
    write_geometry_outputs(
        output_root,
        checkpoints=checkpoints,
        trajectories=trajectories,
        transfers=transfers,
        parents=parents,
        summary=summary,
        children=scored,
    )
    if published_root is not None:
        if published_root.exists(): shutil.rmtree(published_root)
        published_root.mkdir(parents=True)
        for relative in ("summary.json", "reports", "figures", "manifests/local_parent_manifest.parquet"):
            source, destination = output_root / relative, published_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, destination) if source.is_dir() else shutil.copy2(source, destination)
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary-root", type=Path, required=True)
    parser.add_argument("--dense-outcomes", type=Path)
    parser.add_argument("--prompt-outcomes", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--published-root", type=Path)
    parser.add_argument("--child-states", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.child_states is not None:
        summary = run_phase4_analysis(boundary_root=args.boundary_root, child_states_path=args.child_states, output_root=args.output_root, published_root=args.published_root)
    else:
        if args.dense_outcomes is None or args.prompt_outcomes is None:
            parser.error("Phase 2 requires --dense-outcomes and --prompt-outcomes")
        summary = run_post_download_analysis(boundary_root=args.boundary_root, dense_outcomes_path=args.dense_outcomes, prompt_outcomes_path=args.prompt_outcomes, output_root=args.output_root, published_root=args.published_root)
    print(json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
