"""Nested-label reliability and frozen-predictor analysis.

All architecture comparisons average the three already-frozen training seeds;
no new seed is selected from the calibration outcomes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from safeprefix.recoverability_geometry.analysis import within_trace_concordance
from safeprefix.reproducibility import atomic_json, atomic_parquet, now_iso, stable_seed

from .calibration import ARCHITECTURES, MODEL_KEYS, sha256_file


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _safe_spearman(left: Iterable[float], right: Iterable[float]) -> float:
    lhs = np.asarray(list(left), dtype=float)
    rhs = np.asarray(list(right), dtype=float)
    if len(lhs) < 2 or np.unique(lhs).size < 2 or np.unique(rhs).size < 2:
        return float("nan")
    return float(spearmanr(lhs, rhs).statistic)


def _majority(values: np.ndarray) -> np.ndarray:
    # The repository's K=4 majority convention treats an exact tie as the
    # positive/recoverable class.
    return np.asarray(values, dtype=float) >= 0.5


def _trace_weights(trace_ids: Sequence[Any]) -> np.ndarray:
    series = pd.Series(list(map(str, trace_ids)))
    counts = series.map(series.value_counts()).to_numpy(float)
    return 1.0 / counts / float(series.nunique())


def _fractional_nll(logits: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, logits) - q[:, None, None] * logits


def _metric_suite(frame: pd.DataFrame) -> dict[str, float | int]:
    q = frame["q_k"].to_numpy(float)
    probability = frame["raw_probability"].to_numpy(float)
    probability = np.clip(probability, 1e-8, 1 - 1e-8)
    loss = -(q * np.log(probability) + (1 - q) * np.log(1 - probability))
    weights = _trace_weights(frame["trace_id"].astype(str).tolist())
    label = _majority(q).astype(int)
    both = np.unique(label).size == 2
    return {
        "trace_weighted_binomial_nll": float(np.sum(loss * weights)),
        "brier_score": float(np.mean((probability - q) ** 2)),
        "roc_auc": float(roc_auc_score(label, probability)) if both else float("nan"),
        "average_precision": float(average_precision_score(label, probability)) if label.any() else float("nan"),
        "within_trace_concordance": within_trace_concordance(
            frame,
            score_column="raw_probability",
            outcome_column="q_k",
            trace_column="trace_id",
        ),
        "traces": int(frame["trace_id"].nunique()),
        "checkpoints": int(len(frame)),
    }


def load_k16_outcomes(
    threshold_root: str | Path, checkpoint: pd.DataFrame
) -> pd.DataFrame:
    path = Path(threshold_root) / "raw_outcomes/merged_k16_checkpoint_suffixes.parquet"
    columns = [
        "model_key",
        "trace_id",
        "source_trace_id",
        "problem_id",
        "checkpoint_ordinal",
        "checkpoint_token_offset",
        "rollout_index",
        "rollout_seed",
        "verifier_outcome",
        "binary_outcome",
        "artifact_hash",
        "generated_token_count",
        "infrastructure_status",
        "model_id",
        "model_revision",
        "tokenizer_revision",
    ]
    outcomes = pd.read_parquet(path, columns=columns)
    if not outcomes["verifier_outcome"].astype(bool).eq(outcomes["binary_outcome"].astype(bool)).all():
        raise RuntimeError("verifier and binary outcome fields differ")
    if not outcomes["infrastructure_status"].eq("executed").all():
        raise RuntimeError("non-executed row entered K16 analysis")
    join = ["model_key", "trace_id", "checkpoint_ordinal", "checkpoint_token_offset"]
    outcomes = outcomes.merge(
        checkpoint[
            join
            + [
                "checkpoint_key",
                "checkpoint_id",
                "domain",
                "problem_group",
                "normalized_checkpoint_position",
                "prefix_token_hash",
                "prompt_token_hash",
                "continuation_policy_hash",
                "verifier_version",
            ]
        ],
        on=join,
        how="inner",
        validate="many_to_one",
    )
    if len(outcomes) != len(checkpoint) * 16 or outcomes["checkpoint_key"].isna().any():
        raise RuntimeError("K16 outcome does not join the frozen checkpoint inventory")
    if outcomes.duplicated(["checkpoint_key", "rollout_index"]).any():
        raise RuntimeError("duplicate K16 checkpoint slot")
    grouped = outcomes.groupby("checkpoint_key", sort=True)
    if len(grouped) != len(checkpoint) or any(
        sorted(part["rollout_index"].astype(int)) != list(range(16))
        for _, part in grouped
    ):
        raise RuntimeError("K16 outcomes are not exact ordered pools")
    return outcomes.sort_values(["checkpoint_key", "rollout_index"], kind="mergesort").reset_index(drop=True)


def build_nested_labels(
    outcomes: pd.DataFrame,
    checkpoint: pd.DataFrame,
    *,
    ks: Sequence[int],
    dense_k: int,
) -> pd.DataFrame:
    matrix = outcomes.pivot(index="checkpoint_key", columns="rollout_index", values="verifier_outcome")
    matrix = matrix.reindex(checkpoint["checkpoint_key"].astype(str))
    required = list(range(dense_k))
    if list(matrix.columns) != required or matrix.isna().any().any():
        raise RuntimeError("nested outcome matrix does not have the exact dense prefix")
    values = matrix.to_numpy(float)
    cumulative = np.cumsum(values, axis=1)
    records: list[pd.DataFrame] = []
    for k in ks:
        if int(k) > dense_k:
            raise ValueError("nested K exceeds the available pool")
        part = checkpoint.copy()
        part["K"] = int(k)
        part["success_count_k"] = cumulative[:, int(k) - 1].astype(int)
        part["q_k"] = cumulative[:, int(k) - 1] / float(k)
        part["q_dense"] = cumulative[:, dense_k - 1] / float(dense_k)
        records.append(part)
    return pd.concat(records, ignore_index=True)


def compute_label_reliability(
    nested: pd.DataFrame,
    *,
    analysis_name: str,
    dense_k: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for k, k_frame in nested.groupby("K", sort=True):
        for model_key, part in k_frame.groupby("model_key", sort=True):
            error = part["q_k"].to_numpy(float) - part["q_dense"].to_numpy(float)
            rows.append(
                {
                    "analysis": analysis_name,
                    "K": int(k),
                    "dense_reference_K": dense_k,
                    "model_key": model_key,
                    "aggregation": "model",
                    "mae": float(np.mean(np.abs(error))),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "spearman": _safe_spearman(part["q_k"], part["q_dense"]),
                    "majority_label_agreement": float(np.mean(_majority(part["q_k"].to_numpy()) == _majority(part["q_dense"].to_numpy()))),
                    "checkpoints": len(part),
                }
            )
        model_rows = pd.DataFrame(rows).loc[
            lambda value: value["analysis"].eq(analysis_name)
            & value["K"].eq(int(k))
            & value["aggregation"].eq("model")
        ]
        macro = {
            column: float(model_rows[column].mean())
            for column in ("mae", "rmse", "spearman", "majority_label_agreement")
        }
        rows.append(
            {
                "analysis": analysis_name,
                "K": int(k),
                "dense_reference_K": dense_k,
                "model_key": "equal_model_macro",
                "aggregation": "equal_model_macro",
                **macro,
                "checkpoints": int(len(k_frame)),
            }
        )
    return pd.DataFrame(rows)


def compute_predictor_metrics(
    nested: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    analysis_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for k, labels in nested.groupby("K", sort=True):
        joined = predictions.merge(
            labels[["checkpoint_key", "trace_id", "q_k"]],
            on=["checkpoint_key", "trace_id"],
            how="inner",
            validate="many_to_one",
        )
        if len(joined) != len(predictions):
            raise RuntimeError("predictor/label join lost rows")
        for (model_key, architecture, seed), part in joined.groupby(
            ["model_key", "architecture", "training_seed"], sort=True
        ):
            rows.append(
                {
                    "analysis": analysis_name,
                    "K": int(k),
                    "model_key": model_key,
                    "architecture": architecture,
                    "aggregation": "seed",
                    "training_seed": int(seed),
                    **_metric_suite(part),
                }
            )
        seed = pd.DataFrame(rows).loc[
            lambda value: value["analysis"].eq(analysis_name)
            & value["K"].eq(int(k))
            & value["aggregation"].eq("seed")
        ]
        metric_columns = [
            "trace_weighted_binomial_nll",
            "brier_score",
            "roc_auc",
            "average_precision",
            "within_trace_concordance",
        ]
        for (model_key, architecture), part in seed.groupby(["model_key", "architecture"], sort=True):
            rows.append(
                {
                    "analysis": analysis_name,
                    "K": int(k),
                    "model_key": model_key,
                    "architecture": architecture,
                    "aggregation": "seed_macro",
                    "training_seed": "all_0_1_2",
                    **{column: float(part[column].mean()) for column in metric_columns},
                    "traces": int(part["traces"].max()),
                    "checkpoints": int(part["checkpoints"].max()),
                }
            )
        seed_macro = pd.DataFrame(rows).loc[
            lambda value: value["analysis"].eq(analysis_name)
            & value["K"].eq(int(k))
            & value["aggregation"].eq("seed_macro")
        ]
        for architecture, part in seed_macro.groupby("architecture", sort=True):
            rows.append(
                {
                    "analysis": analysis_name,
                    "K": int(k),
                    "model_key": "equal_model_macro",
                    "architecture": architecture,
                    "aggregation": "equal_model_macro",
                    "training_seed": "all_0_1_2",
                    **{column: float(part[column].mean()) for column in metric_columns},
                    "traces": int(part["traces"].sum()),
                    "checkpoints": int(part["checkpoints"].sum()),
                }
            )
    return pd.DataFrame(rows)


def _analysis_arrays(
    checkpoint: pd.DataFrame,
    outcomes: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    dense_k: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[str], list[int]]:
    ordered = checkpoint.reset_index(drop=True)
    keys = ordered["checkpoint_key"].astype(str).tolist()
    outcome = outcomes.pivot(index="checkpoint_key", columns="rollout_index", values="verifier_outcome").reindex(keys)
    if outcome.shape != (len(keys), dense_k) or outcome.isna().any().any():
        raise RuntimeError("analysis outcome array shape differs")
    architectures = list(ARCHITECTURES)
    seeds = sorted(predictions["training_seed"].astype(int).unique())
    logits = np.empty((len(keys), len(architectures), len(seeds)), dtype=float)
    for a_index, architecture in enumerate(architectures):
        for s_index, seed in enumerate(seeds):
            part = predictions.loc[
                predictions["architecture"].eq(architecture)
                & predictions["training_seed"].astype(int).eq(seed)
            ].set_index("checkpoint_key")["raw_logit"].reindex(keys)
            if part.isna().any():
                raise RuntimeError("analysis predictor array is incomplete")
            logits[:, a_index, s_index] = part.to_numpy(float)
    model_indices = {
        model: np.flatnonzero(ordered["model_key"].astype(str).to_numpy() == model)
        for model in MODEL_KEYS
    }
    trace_weights = {
        model: _trace_weights(ordered.iloc[indices]["trace_id"].astype(str).tolist())
        for model, indices in model_indices.items()
    }
    return outcome.to_numpy(float), logits, {
        "indices": model_indices,
        "weights": trace_weights,
    }, architectures, seeds


def _macro_nll(
    q: np.ndarray,
    logits: np.ndarray,
    model_arrays: Mapping[str, Mapping[str, np.ndarray]],
) -> np.ndarray:
    losses = _fractional_nll(logits, q)
    model_values = []
    for model in MODEL_KEYS:
        indices = model_arrays["indices"][model]
        weights = model_arrays["weights"][model]
        model_values.append(np.sum(losses[indices] * weights[:, None, None], axis=0))
    return np.mean(np.stack(model_values), axis=0).mean(axis=1)


def compute_permutation_stability(
    checkpoint: pd.DataFrame,
    outcomes: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    analysis_name: str,
    ks: Sequence[int],
    dense_k: int,
    replicates: int,
    seed_namespace: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcome, logits, model_arrays, architectures, _ = _analysis_arrays(
        checkpoint, outcomes, predictions, dense_k=dense_k
    )
    keys = checkpoint["checkpoint_key"].astype(str).tolist()
    full_nll = _macro_nll(outcome.mean(axis=1), logits, model_arrays)
    full_order = np.argsort(full_nll)
    full_rank = np.empty(len(architectures), dtype=int)
    full_rank[full_order] = np.arange(1, len(architectures) + 1)
    linear_index = architectures.index("linear_probe")
    nonlinear = [architectures.index(name) for name in ("local_mlp", "change_aware_mlp", "causal_gru")]
    full_delta = float(np.min(full_nll[nonlinear]) - full_nll[linear_index])
    replicate_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for replicate in range(replicates):
        permuted = np.empty_like(outcome)
        for index, key in enumerate(keys):
            rng = np.random.default_rng(stable_seed(seed_namespace, replicate, key))
            permuted[index] = outcome[index, rng.permutation(dense_k)]
        cumulative = np.cumsum(permuted, axis=1)
        for k in ks:
            if int(k) == dense_k:
                continue
            q = cumulative[:, int(k) - 1] / float(k)
            nll = _macro_nll(q, logits, model_arrays)
            order = np.argsort(nll)
            rank = np.empty(len(architectures), dtype=int)
            rank[order] = np.arange(1, len(architectures) + 1)
            delta = float(np.min(nll[nonlinear]) - nll[linear_index])
            for index, architecture in enumerate(architectures):
                replicate_rows.append(
                    {
                        "analysis": analysis_name,
                        "replicate": replicate,
                        "replicate_kind": "permutation",
                        "K": int(k),
                        "architecture": architecture,
                        "macro_nll": float(nll[index]),
                        "rank": int(rank[index]),
                        "win": bool(rank[index] == 1),
                        "rank_correlation_with_dense": _safe_spearman(rank, full_rank),
                        "best_nonlinear_minus_linear_nll": delta,
                        "dense_best_nonlinear_minus_linear_nll": full_delta,
                        "delta_sign_changed": bool(np.sign(delta) != np.sign(full_delta)),
                    }
                )
            if 2 * int(k) <= dense_k:
                left = permuted[:, : int(k)].mean(axis=1)
                right = permuted[:, int(k) : 2 * int(k)].mean(axis=1)
                for model in MODEL_KEYS:
                    indices = model_arrays["indices"][model]
                    split_rows.append(
                        {
                            "analysis": analysis_name,
                            "replicate": replicate,
                            "K": int(k),
                            "model_key": model,
                            "mae": float(np.mean(np.abs(left[indices] - right[indices]))),
                            "rmse": float(np.sqrt(np.mean((left[indices] - right[indices]) ** 2))),
                            "spearman": _safe_spearman(left[indices], right[indices]),
                            "majority_label_agreement": float(np.mean(_majority(left[indices]) == _majority(right[indices]))),
                        }
                    )
    # The densest pool is invariant to permutation and is reported once.
    for index, architecture in enumerate(architectures):
        replicate_rows.append(
            {
                "analysis": analysis_name,
                "replicate": -1,
                "replicate_kind": "full_pool",
                "K": dense_k,
                "architecture": architecture,
                "macro_nll": float(full_nll[index]),
                "rank": int(full_rank[index]),
                "win": bool(full_rank[index] == 1),
                "rank_correlation_with_dense": 1.0,
                "best_nonlinear_minus_linear_nll": full_delta,
                "dense_best_nonlinear_minus_linear_nll": full_delta,
                "delta_sign_changed": False,
            }
        )
    return pd.DataFrame(replicate_rows), pd.DataFrame(split_rows)


def summarize_stability(replicates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (analysis_name, k), part in replicates.groupby(["analysis", "K"], sort=True):
        replicate_count = part["replicate"].nunique()
        architecture_count = part["architecture"].nunique()
        delta = part.drop_duplicates("replicate")["best_nonlinear_minus_linear_nll"].to_numpy(float)
        for architecture, architecture_part in part.groupby("architecture", sort=True):
            rows.append(
                {
                    "analysis": analysis_name,
                    "K": int(k),
                    "architecture": architecture,
                    "replicates": int(replicate_count),
                    "architecture_count": int(architecture_count),
                    "win_fraction": float(architecture_part["win"].astype(float).mean()),
                    "mean_rank": float(architecture_part["rank"].mean()),
                    "mean_rank_correlation_with_dense": float(architecture_part["rank_correlation_with_dense"].mean()),
                    "mean_best_nonlinear_minus_linear_nll": float(np.mean(delta)),
                    "q025_best_nonlinear_minus_linear_nll": float(np.quantile(delta, 0.025)),
                    "q975_best_nonlinear_minus_linear_nll": float(np.quantile(delta, 0.975)),
                    "delta_sign_change_fraction": float(architecture_part["delta_sign_changed"].astype(float).mean()),
                }
            )
    return pd.DataFrame(rows)


def summarize_split_sample(replicates: pd.DataFrame) -> pd.DataFrame:
    if replicates.empty:
        return replicates
    rows: list[dict[str, Any]] = []
    metrics = ("mae", "rmse", "spearman", "majority_label_agreement")
    for (analysis_name, k, model), part in replicates.groupby(
        ["analysis", "K", "model_key"], sort=True
    ):
        rows.append(
            {
                "analysis": analysis_name,
                "K": int(k),
                "model_key": model,
                "aggregation": "model_permutation_mean",
                "replicates": int(part["replicate"].nunique()),
                **{metric: float(part[metric].mean()) for metric in metrics},
            }
        )
    model_rows = pd.DataFrame(rows)
    for (analysis_name, k), part in model_rows.groupby(["analysis", "K"], sort=True):
        rows.append(
            {
                "analysis": analysis_name,
                "K": int(k),
                "model_key": "equal_model_macro",
                "aggregation": "equal_model_macro",
                "replicates": int(part["replicates"].max()),
                **{metric: float(part[metric].mean()) for metric in metrics},
            }
        )
    return pd.DataFrame(rows)


def bootstrap_primary_differences(
    checkpoint: pd.DataFrame,
    outcomes: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    analysis_name: str,
    ks: Sequence[int],
    dense_k: int,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    outcome, logits, _, architectures, _ = _analysis_arrays(
        checkpoint, outcomes, predictions, dense_k=dense_k
    )
    q_by_k = {int(k): outcome[:, : int(k)].mean(axis=1) for k in ks}
    linear = architectures.index("linear_probe")
    nonlinear = [architectures.index(name) for name in ("local_mlp", "change_aware_mlp", "causal_gru")]
    # Freeze the best nonlinear identity per K from the observed equal-model
    # macro before resampling.  Bootstrap draws never reselect the comparator.
    observed_best: dict[int, int] = {}
    per_trace: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for k in ks:
        k = int(k)
        model_arch = []
        per_trace[k] = {}
        loss = _fractional_nll(logits, q_by_k[k]).mean(axis=2)
        for model in MODEL_KEYS:
            indices = np.flatnonzero(checkpoint["model_key"].astype(str).to_numpy() == model)
            traces = checkpoint.iloc[indices]["trace_id"].astype(str).to_numpy()
            frame = pd.DataFrame(loss[indices], columns=architectures)
            frame["trace_id"] = traces
            collapsed = frame.groupby("trace_id", sort=True)[architectures].mean()
            per_trace[k][model] = {
                "trace_ids": collapsed.index.to_numpy(str),
                "loss": collapsed.to_numpy(float),
            }
            model_arch.append(collapsed.to_numpy(float).mean(axis=0))
        macro = np.mean(np.stack(model_arch), axis=0)
        observed_best[k] = nonlinear[int(np.argmin(macro[nonlinear]))]

    rng = np.random.default_rng(seed)
    results: dict[str, Any] = {}
    for k in map(int, ks):
        sampled = np.empty(replicates, dtype=float)
        for replicate in range(replicates):
            model_differences = []
            for model in MODEL_KEYS:
                values = per_trace[k][model]["loss"]
                draw = rng.integers(0, len(values), size=len(values))
                means = values[draw].mean(axis=0)
                model_differences.append(means[observed_best[k]] - means[linear])
            sampled[replicate] = float(np.mean(model_differences))
        observed_model = []
        for model in MODEL_KEYS:
            means = per_trace[k][model]["loss"].mean(axis=0)
            observed_model.append(means[observed_best[k]] - means[linear])
        results[f"K{k}"] = {
            "comparison": f"{architectures[observed_best[k]]}_minus_linear_probe_nll",
            "estimate": float(np.mean(observed_model)),
            "ci_low": float(np.quantile(sampled, 0.025)),
            "ci_high": float(np.quantile(sampled, 0.975)),
            "replicates": int(replicates),
            "bootstrap_unit": "complete_trace_within_model",
            "models_resampled_independently": True,
            "aggregate": "equal_weight_four_model_macro",
        }
    return {
        "analysis": analysis_name,
        "dense_reference_K": dense_k,
        "results": results,
    }


def run_analysis(
    *,
    config: Mapping[str, Any],
    threshold_root: str | Path,
    output_root: str | Path,
    analysis_name: str,
    checkpoint_path: str | Path,
    outcomes_path: str | Path | None,
    ks: Sequence[int],
    dense_k: int,
    required_metrics_filename: str,
) -> dict[str, Any]:
    output = Path(output_root)
    checkpoint = pd.read_parquet(checkpoint_path)
    predictions = pd.read_parquet(output / "frozen_predictor_outputs.parquet")
    predictions = predictions.loc[
        predictions["checkpoint_key"].astype(str).isin(set(checkpoint["checkpoint_key"].astype(str)))
    ].copy()
    if outcomes_path is None:
        outcomes = load_k16_outcomes(threshold_root, checkpoint)
    else:
        outcomes = pd.read_parquet(outcomes_path)
        required = {"checkpoint_key", "rollout_index", "verifier_outcome"}
        if not required <= set(outcomes):
            raise RuntimeError("confirmation outcome file lacks analysis columns")
        outcomes = outcomes.loc[outcomes["rollout_index"].astype(int).lt(dense_k)].copy()
    nested = build_nested_labels(outcomes, checkpoint, ks=ks, dense_k=dense_k)
    metrics = compute_predictor_metrics(nested, predictions, analysis_name=analysis_name)
    reliability = compute_label_reliability(nested, analysis_name=analysis_name, dense_k=dense_k)
    replicate, split_replicate = compute_permutation_stability(
        checkpoint,
        outcomes,
        predictions,
        analysis_name=analysis_name,
        ks=ks,
        dense_k=dense_k,
        replicates=int(config["analysis"]["permutation_replicates"]),
        seed_namespace=str(config["rollouts"]["permutation_seed_namespace"]),
    )
    stability = summarize_stability(replicate)
    split = summarize_split_sample(split_replicate)
    bootstrap = bootstrap_primary_differences(
        checkpoint,
        outcomes,
        predictions,
        analysis_name=analysis_name,
        ks=ks,
        dense_k=dense_k,
        replicates=int(config["analysis"]["bootstrap_replicates"]),
        seed=int(config["seed"]) + dense_k,
    )
    _atomic_csv(output / required_metrics_filename, metrics)
    _atomic_csv(output / f"internal/{analysis_name}_label_reliability.csv", reliability)
    _atomic_csv(output / f"internal/{analysis_name}_architecture_stability.csv", stability)
    _atomic_csv(output / f"internal/{analysis_name}_split_sample.csv", split)
    atomic_parquet(output / f"internal/{analysis_name}_permutation_rows.parquet", replicate)
    atomic_json(output / f"internal/{analysis_name}_bootstrap.json", bootstrap)
    atomic_parquet(output / f"internal/{analysis_name}_nested_labels.parquet", nested)
    return {
        "status": "COMPLETE",
        "analysis": analysis_name,
        "created_at": now_iso(),
        "checkpoint_count": len(checkpoint),
        "trace_count": int(checkpoint[["model_key", "trace_id"]].drop_duplicates().shape[0]),
        "outcome_rows": len(outcomes),
        "nested_K": list(map(int, ks)),
        "dense_reference_K": dense_k,
        "metric_rows": len(metrics),
        "permutation_replicates": int(config["analysis"]["permutation_replicates"]),
        "bootstrap_replicates": int(config["analysis"]["bootstrap_replicates"]),
        "source_outcomes_sha256": (
            sha256_file(Path(threshold_root) / "raw_outcomes/merged_k16_checkpoint_suffixes.parquet")
            if outcomes_path is None
            else sha256_file(outcomes_path)
        ),
        "native_outcomes_loaded": False,
        "predictor_retrained": False,
        "calibrator_refit": False,
        "threshold_reselected": False,
    }
