"""Dev-only selection, held-out calibration, and teacher-forced evaluation."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch

from .data import atomic_json, atomic_parquet, load_config
from .models import ARCHITECTURE_COMPLEXITY
from .training import (
    ModelCorpus,
    learning_rate_slug,
    load_trained_predictor,
    predict_split,
    trace_weighted_nll,
)


def _run_root(
    artifact_root: Path,
    model_key: str,
    architecture: str,
    learning_rate: float,
    seed: int,
) -> Path:
    return (
        artifact_root
        / "training"
        / model_key
        / architecture
        / learning_rate_slug(learning_rate)
        / f"seed_{seed}"
    )


def select_architecture(*, config_path: Path, artifact_root: Path) -> dict[str, Any]:
    """Apply the common one-standard-error rule using uncalibrated dev NLL only."""
    config = load_config(config_path)
    models = list(config["source"]["expected_models"])
    seeds = list(map(int, config["experiment"]["training_seeds"]))
    rows: list[dict[str, Any]] = []
    expected_hidden = {
        (architecture, learning_rate, seed)
        for architecture in map(str, config["training"]["architectures"])
        for learning_rate in map(float, config["training"]["learning_rates"])
        for seed in seeds
    }
    for model_key in models:
        matrix_path = artifact_root / f"training/{model_key}/matrix_summary.json"
        if not matrix_path.is_file():
            raise FileNotFoundError(f"required training matrix is incomplete: {matrix_path}")
        matrix = json.loads(matrix_path.read_text())
        if matrix.get("runs") != matrix.get("expected_runs") or matrix.get("runs") != 27:
            raise RuntimeError(f"{model_key}: training matrix is incomplete")
        observed_hidden: set[tuple[str, float, int]] = set()
        for complete in matrix["results"]:
            architecture = str(complete["architecture"])
            if architecture == "position_only":
                continue
            identity = (
                architecture,
                float(complete["learning_rate"]),
                int(complete["seed"]),
            )
            observed_hidden.add(identity)
            if complete.get("native_evaluation_used") is not False:
                raise RuntimeError(f"native input guard failed in {matrix_path}")
            metric = complete["dev_metrics"]
            rows.append(
                {
                    "architecture": architecture,
                    "learning_rate": float(complete["learning_rate"]),
                    "base_model": model_key,
                    "seed": int(complete["seed"]),
                    "dev_trace_weighted_nll": float(
                        metric["trace_weighted_binomial_nll"]
                    ),
                    "dev_domain_macro_nll": float(
                        metric["domain_macro_trace_weighted_binomial_nll"]
                    ),
                    "complexity_rank": ARCHITECTURE_COMPLEXITY[architecture],
                }
            )
        if observed_hidden != expected_hidden:
            raise RuntimeError(f"{model_key}: hidden-state matrix identities differ")
    comparison = pd.DataFrame(rows)
    comparison_path = artifact_root / "selection/architecture_comparison.csv"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_path, index=False)

    seed_macro = (
        comparison.groupby(["architecture", "learning_rate", "seed"], as_index=False)
        .agg(
            macro_model_dev_nll=("dev_trace_weighted_nll", "mean"),
            macro_model_domain_macro_nll=("dev_domain_macro_nll", "mean"),
        )
        .sort_values(["architecture", "learning_rate", "seed"])
    )
    seed_macro.to_csv(artifact_root / "selection/seed_macro_dev_metrics.csv", index=False)
    summary = (
        seed_macro.groupby(["architecture", "learning_rate"], as_index=False)
        .agg(
            mean_macro_dev_nll=("macro_model_dev_nll", "mean"),
            std_macro_dev_nll=("macro_model_dev_nll", "std"),
            mean_macro_domain_macro_nll=("macro_model_domain_macro_nll", "mean"),
            seed_count=("seed", "nunique"),
        )
    )
    summary["standard_error"] = summary["std_macro_dev_nll"] / np.sqrt(
        summary["seed_count"]
    )
    summary["complexity_rank"] = summary["architecture"].map(ARCHITECTURE_COMPLEXITY)
    best = summary.loc[summary["mean_macro_dev_nll"].idxmin()]
    threshold = float(best["mean_macro_dev_nll"] + best["standard_error"])
    summary["within_one_standard_error"] = summary["mean_macro_dev_nll"] <= threshold
    eligible = summary.loc[summary["within_one_standard_error"]].sort_values(
        ["complexity_rank", "mean_macro_dev_nll", "learning_rate"]
    )
    selected = eligible.iloc[0]
    summary.to_csv(artifact_root / "selection/one_se_summary.csv", index=False)
    result = {
        "status": "FROZEN",
        "selection_split": "architecture_dev",
        "selection_metric": "uncalibrated_trace_weighted_binomial_nll",
        "calibration_used_for_selection": False,
        "test_used_for_selection": False,
        "native_evaluation_used": False,
        "best_numerical_architecture": str(best["architecture"]),
        "best_numerical_learning_rate": float(best["learning_rate"]),
        "best_mean_macro_dev_nll": float(best["mean_macro_dev_nll"]),
        "best_standard_error": float(best["standard_error"]),
        "one_se_threshold": threshold,
        "selected_architecture": str(selected["architecture"]),
        "selected_learning_rate": float(selected["learning_rate"]),
        "selected_mean_macro_dev_nll": float(selected["mean_macro_dev_nll"]),
        "complexity_order": list(ARCHITECTURE_COMPLEXITY),
        "justification": (
            "Selected the simplest hidden-state architecture and its common learning rate "
            "within one standard error of the best four-model, three-seed macro dev NLL."
        ),
    }
    atomic_json(artifact_root / "selection/selected_model.json", result)
    return result


def _trace_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("trace_id")["trace_id"].transform("size").to_numpy(float)
    return 1.0 / counts / float(frame["trace_id"].nunique())


def fit_positive_affine_calibrator(frame: pd.DataFrame, max_iterations: int) -> dict[str, Any]:
    if set(frame["split"].astype(str)) != {"calibration"}:
        raise RuntimeError("calibration may only be fit on the calibration split")
    logits = torch.tensor(frame["raw_logit"].to_numpy(float), dtype=torch.float64)
    target = torch.tensor(frame["observed_success_rate"].to_numpy(float), dtype=torch.float64)
    weights = torch.tensor(_trace_weights(frame), dtype=torch.float64)
    # softplus(theta)=1 at this initialization, so optimization starts at identity.
    theta = torch.nn.Parameter(torch.tensor(math.log(math.e - 1.0), dtype=torch.float64))
    intercept = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [theta, intercept],
        lr=0.5,
        max_iter=int(max_iterations),
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        slope = torch.nn.functional.softplus(theta) + 1e-8
        calibrated_logits = slope * logits + intercept
        loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                calibrated_logits, target, reduction="none"
            )
            * weights
        ).sum()
        loss.backward()
        return loss

    initial_nll = float(closure().detach())
    optimizer.step(closure)
    with torch.no_grad():
        slope = float(torch.nn.functional.softplus(theta) + 1e-8)
        bias = float(intercept)
        final_nll = float(
            (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    slope * logits + bias, target, reduction="none"
                )
                * weights
            ).sum()
        )
    if not math.isfinite(slope) or not math.isfinite(bias) or slope <= 0:
        raise RuntimeError("positive affine logistic calibration did not converge")
    return {
        "kind": "positive_affine_logistic",
        "a": slope,
        "b": bias,
        "fit_split": "calibration",
        "objective": "trace_weighted_normalized_binomial_nll",
        "initial_nll": initial_nll,
        "fitted_nll": final_nll,
        "ranking_preserved": True,
        "native_evaluation_used": False,
    }


def apply_calibration(frame: pd.DataFrame, calibrator: Mapping[str, Any]) -> pd.DataFrame:
    output = frame.copy()
    calibrated_logit = (
        float(calibrator["a"]) * output["raw_logit"].to_numpy(float)
        + float(calibrator["b"])
    )
    calibrated_logit = np.clip(calibrated_logit, -60.0, 60.0)
    output["calibrated_probability"] = 1.0 / (1.0 + np.exp(-calibrated_logit))
    return output


def reliability_table(
    frame: pd.DataFrame, probability_column: str, *, bins: int = 10
) -> pd.DataFrame:
    ordered = frame.sort_values(probability_column).reset_index(drop=True)
    records: list[dict[str, Any]] = []
    for index, indices in enumerate(np.array_split(np.arange(len(ordered)), bins), start=1):
        if not len(indices):
            continue
        part = ordered.iloc[indices]
        records.append(
            {
                "bin": index,
                "count": len(part),
                "mean_prediction": float(part[probability_column].mean()),
                "empirical_success_rate": float(part["observed_success_rate"].mean()),
                "minimum_prediction": float(part[probability_column].min()),
                "maximum_prediction": float(part[probability_column].max()),
            }
        )
    return pd.DataFrame(records)


def _calibration_line(frame: pd.DataFrame, probability_column: str) -> tuple[float, float]:
    if frame["observed_success_rate"].nunique() < 2:
        return float("nan"), float("nan")
    probability = np.clip(frame[probability_column].to_numpy(float), 1e-6, 1 - 1e-6)
    source_logit = torch.tensor(np.log(probability / (1 - probability)), dtype=torch.float64)
    target = torch.tensor(frame["observed_success_rate"].to_numpy(float), dtype=torch.float64)
    weights = torch.tensor(_trace_weights(frame), dtype=torch.float64)
    slope = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
    intercept = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [slope, intercept], max_iter=100, tolerance_grad=1e-9, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                slope * source_logit + intercept, target, reduction="none"
            )
            * weights
        ).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(slope.detach()), float(intercept.detach())


def _within_trace_concordance(frame: pd.DataFrame, probability_column: str) -> float:
    per_trace: list[float] = []
    for _, trace in frame.groupby("trace_id", sort=False):
        observed = trace["observed_success_rate"].to_numpy(float)
        predicted = trace[probability_column].to_numpy(float)
        scores: list[float] = []
        for left in range(len(trace)):
            for right in range(left + 1, len(trace)):
                observed_difference = observed[left] - observed[right]
                if observed_difference == 0:
                    continue
                predicted_difference = predicted[left] - predicted[right]
                if predicted_difference == 0:
                    scores.append(0.5)
                else:
                    scores.append(float(np.sign(observed_difference) == np.sign(predicted_difference)))
        if scores:
            per_trace.append(float(np.mean(scores)))
    return float(np.mean(per_trace)) if per_trace else float("nan")


def metric_suite(
    frame: pd.DataFrame, probability_column: str, *, ece_bins: int = 10
) -> dict[str, float | int]:
    probability = frame[probability_column].to_numpy(float)
    observed = frame["observed_success_rate"].to_numpy(float)
    reliability = reliability_table(frame, probability_column, bins=ece_bins)
    ece = float(
        (
            reliability["count"]
            * (reliability["mean_prediction"] - reliability["empirical_success_rate"]).abs()
        ).sum()
        / reliability["count"].sum()
    )
    slope, intercept = _calibration_line(frame, probability_column)
    return {
        "trace_weighted_binomial_nll": trace_weighted_nll(frame, probability_column),
        "brier_score": float(np.mean((probability - observed) ** 2)),
        "ece_equal_count_10": ece,
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "mean_predicted_probability": float(np.mean(probability)),
        "empirical_rollout_success_rate": float(np.mean(observed)),
        "spearman": float(pd.Series(probability).corr(pd.Series(observed), method="spearman")),
        "within_trace_concordance": _within_trace_concordance(frame, probability_column),
        "traces": int(frame["trace_id"].nunique()),
        "checkpoints": len(frame),
    }


def _plot_calibration(
    frame: pd.DataFrame,
    path: Path,
    *,
    probability_columns: Iterable[str],
    bins: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for column in probability_columns:
        table = reliability_table(frame, column, bins=bins)
        axes[0].plot(
            table["mean_prediction"],
            table["empirical_success_rate"],
            marker="o",
            label=column,
        )
        axes[1].hist(frame[column], bins=20, alpha=0.45, label=column)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    axes[0].set(xlabel="Predicted probability", ylabel="Empirical success rate", xlim=(0, 1), ylim=(0, 1))
    axes[1].set(xlabel="Predicted probability", ylabel="Checkpoint count", xlim=(0, 1))
    axes[0].legend()
    axes[1].legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _add_strata(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    checkpoint_fraction = output["checkpoint_ordinal"] / np.maximum(
        output["total_checkpoint_count"] - 1, 1
    )
    output["checkpoint_position_quartile"] = pd.cut(
        checkpoint_fraction,
        bins=[-np.inf, 0.25, 0.5, 0.75, np.inf],
        labels=["Q1", "Q2", "Q3", "Q4"],
    ).astype(str)
    trace_lengths = output.drop_duplicates("trace_id")[["trace_id", "total_checkpoint_count"]].copy()
    trace_lengths["trace_length_quartile"] = pd.qcut(
        trace_lengths["total_checkpoint_count"].rank(method="first"),
        4,
        labels=["Q1", "Q2", "Q3", "Q4"],
    ).astype(str)
    output = output.merge(trace_lengths[["trace_id", "trace_length_quartile"]], on="trace_id")
    output["observed_success_count_stratum"] = output["success_count"].astype(str)
    return output


def stratified_metrics(
    frame: pd.DataFrame, probability_column: str, *, bins: int
) -> pd.DataFrame:
    prepared = _add_strata(frame)
    records: list[dict[str, Any]] = []
    strata = {
        "overall": None,
        "domain": "domain",
        "checkpoint_position_quartile": "checkpoint_position_quartile",
        "trace_length_quartile": "trace_length_quartile",
        "observed_success_count": "observed_success_count_stratum",
    }
    for stratum, column in strata.items():
        groups = [("all", prepared)] if column is None else prepared.groupby(column, observed=True)
        for value, part in groups:
            records.append(
                {
                    "stratum": stratum,
                    "value": str(value),
                    **metric_suite(part, probability_column, ece_bins=bins),
                }
            )
    return pd.DataFrame(records)


def trace_bootstrap(
    frame: pd.DataFrame,
    probability_column: str,
    *,
    bins: int,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    grouped = [part for _, part in frame.groupby("trace_id", sort=True)]
    generator = np.random.default_rng(seed)
    keys = [
        "trace_weighted_binomial_nll",
        "brier_score",
        "ece_equal_count_10",
        "spearman",
        "within_trace_concordance",
    ]
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        draw = generator.integers(0, len(grouped), size=len(grouped))
        pieces: list[pd.DataFrame] = []
        for draw_index, source_index in enumerate(draw):
            part = grouped[int(source_index)].copy()
            part["trace_id"] = part["trace_id"].astype(str) + f"#bootstrap_{draw_index}"
            pieces.append(part)
        sampled = pd.concat(pieces, ignore_index=True)
        probability = sampled[probability_column].to_numpy(float)
        observed = sampled["observed_success_rate"].to_numpy(float)
        reliability = reliability_table(sampled, probability_column, bins=bins)
        metrics = {
            "trace_weighted_binomial_nll": trace_weighted_nll(sampled, probability_column),
            "brier_score": float(np.mean((probability - observed) ** 2)),
            "ece_equal_count_10": float(
                (
                    reliability["count"]
                    * (
                        reliability["mean_prediction"]
                        - reliability["empirical_success_rate"]
                    ).abs()
                ).sum()
                / reliability["count"].sum()
            ),
            "spearman": float(
                pd.Series(probability).corr(pd.Series(observed), method="spearman")
            ),
            "within_trace_concordance": _within_trace_concordance(
                sampled, probability_column
            ),
        }
        for key in keys:
            samples[key].append(float(metrics[key]))
    return {
        key: {
            "mean": float(np.nanmean(values)),
            "lower_95": float(np.nanquantile(values, 0.025)),
            "upper_95": float(np.nanquantile(values, 0.975)),
        }
        for key, values in samples.items()
    }


def threshold_sweep(frame: pd.DataFrame, thresholds: Iterable[float]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    trace_groups = list(frame.groupby("trace_id", sort=True))
    for threshold in thresholds:
        selected: list[dict[str, Any]] = []
        fallback = 0
        for trace_id, trace in trace_groups:
            ordered = trace.sort_values("checkpoint_ordinal")
            eligible = ordered.loc[ordered["calibrated_probability"] >= float(threshold)]
            successful = ordered.loc[ordered["success_count"] > 0]
            if eligible.empty:
                fallback += 1
                continue
            choice = eligible.iloc[-1]
            oracle = successful.iloc[-1] if not successful.empty else None
            dangerous = oracle is None or int(choice["checkpoint_ordinal"]) > int(
                oracle["checkpoint_ordinal"]
            )
            unnecessary_rewind = (
                max(
                    0.0,
                    float(oracle["prefix_token_count"] - choice["prefix_token_count"])
                    / max(float(choice["total_trace_token_count"]), 1.0),
                )
                if oracle is not None
                else 0.0
            )
            selected.append(
                {
                    "trace_id": trace_id,
                    "checkpoint_fraction": float(choice["checkpoint_ordinal"] + 1)
                    / max(float(choice["total_checkpoint_count"]), 1.0),
                    "retained_prefix_token_fraction": float(choice["prefix_token_count"])
                    / max(float(choice["total_trace_token_count"]), 1.0),
                    "observed_success_rate": float(choice["observed_success_rate"]),
                    "dangerous_late_selection": float(dangerous),
                    "unnecessary_rewind_fraction": unnecessary_rewind,
                }
            )
        chosen = pd.DataFrame(selected)
        total = len(trace_groups)
        records.append(
            {
                "tau": float(threshold),
                "trace_count": total,
                "selection_fraction": len(chosen) / total,
                "full_regeneration_fallback_fraction": fallback / total,
                "mean_selected_checkpoint_position_fraction": float(
                    chosen["checkpoint_fraction"].mean()
                ) if len(chosen) else float("nan"),
                "mean_retained_prefix_token_fraction": float(
                    chosen["retained_prefix_token_fraction"].mean()
                ) if len(chosen) else float("nan"),
                "observed_selected_checkpoint_success_rate": float(
                    chosen["observed_success_rate"].mean()
                ) if len(chosen) else float("nan"),
                "dangerous_late_selection_frequency": float(
                    chosen["dangerous_late_selection"].mean()
                ) if len(chosen) else float("nan"),
                "mean_unnecessary_rewind_fraction": float(
                    chosen["unnecessary_rewind_fraction"].mean()
                ) if len(chosen) else float("nan"),
                "diagnostic_only": True,
                "final_tau_selected": False,
            }
        )
    return pd.DataFrame(records)


def finalize_selected_model(
    *,
    config_path: Path,
    artifact_root: Path,
    device_name: str,
) -> dict[str, Any]:
    config = load_config(config_path)
    selected_path = artifact_root / "selection/selected_model.json"
    if not selected_path.is_file():
        raise FileNotFoundError("architecture must be frozen before calibration or test evaluation")
    selected = json.loads(selected_path.read_text())
    if selected.get("test_used_for_selection") is not False:
        raise RuntimeError("selection guard indicates test leakage")
    architecture = str(selected["selected_architecture"])
    learning_rate = float(selected["selected_learning_rate"])
    seeds = list(map(int, config["experiment"]["training_seeds"]))
    models = list(config["source"]["expected_models"])
    bins = int(config["evaluation"]["ece_bins"])
    device = torch.device(device_name)
    all_predictions: list[pd.DataFrame] = []
    metric_records: list[dict[str, Any]] = []
    calibration_records: list[dict[str, Any]] = []
    bootstrap_records: dict[str, Any] = {}
    sweep_records: list[pd.DataFrame] = []
    diagnostic_records: list[dict[str, Any]] = []
    strata_records: list[pd.DataFrame] = []

    for model_key in models:
        corpus = ModelCorpus(artifact_root, model_key)
        for seed in seeds:
            checkpoint = (
                _run_root(artifact_root, model_key, architecture, learning_rate, seed)
                / "best.pt"
            )
            model = load_trained_predictor(checkpoint, device=device)
            calibration_predictions = predict_split(
                model,
                corpus,
                split="calibration",
                batch_size=int(config["training"]["batch_size_traces"]),
                device=device,
                seed=seed,
            )
            calibrator = fit_positive_affine_calibrator(
                calibration_predictions, int(config["calibration"]["max_iterations"])
            )
            calibrator.update(
                {
                    "base_model": model_key,
                    "training_seed": seed,
                    "architecture": architecture,
                    "learning_rate": learning_rate,
                }
            )
            calibration_predictions = apply_calibration(calibration_predictions, calibrator)
            calibration_predictions["training_seed"] = seed
            calibration_predictions["architecture"] = architecture
            calibration_predictions["learning_rate"] = learning_rate
            calibration_root = artifact_root / f"calibration/{model_key}/seed_{seed}"
            atomic_json(calibration_root / "calibrator.json", calibrator)
            atomic_parquet(calibration_root / "calibration_predictions.parquet", calibration_predictions)
            reliability = pd.concat(
                [
                    reliability_table(calibration_predictions, "raw_probability", bins=bins).assign(stage="pre"),
                    reliability_table(calibration_predictions, "calibrated_probability", bins=bins).assign(stage="post"),
                ],
                ignore_index=True,
            )
            reliability.to_csv(calibration_root / "reliability.csv", index=False)
            pre_cal = metric_suite(calibration_predictions, "raw_probability", ece_bins=bins)
            post_cal = metric_suite(calibration_predictions, "calibrated_probability", ece_bins=bins)
            calibration_metrics = {"pre": pre_cal, "post": post_cal}
            atomic_json(calibration_root / "metrics.json", calibration_metrics)
            _plot_calibration(
                calibration_predictions,
                calibration_root / "reliability_and_histogram.png",
                probability_columns=["raw_probability", "calibrated_probability"],
                bins=bins,
            )
            calibration_records.append(
                {
                    "base_model": model_key,
                    "seed": seed,
                    "a": calibrator["a"],
                    "b": calibrator["b"],
                    "pre_nll": pre_cal["trace_weighted_binomial_nll"],
                    "post_nll": post_cal["trace_weighted_binomial_nll"],
                    "pre_brier": pre_cal["brier_score"],
                    "post_brier": post_cal["brier_score"],
                    "pre_ece": pre_cal["ece_equal_count_10"],
                    "post_ece": post_cal["ece_equal_count_10"],
                }
            )

            test = predict_split(
                model,
                corpus,
                split="teacher_forced_test",
                batch_size=int(config["training"]["batch_size_traces"]),
                device=device,
                seed=seed,
            )
            test = apply_calibration(test, calibrator)
            test["training_seed"] = seed
            test["selected_architecture"] = architecture
            test["selected_learning_rate"] = learning_rate
            all_predictions.append(test)
            for stage, column in (
                ("pre_calibration", "raw_probability"),
                ("post_calibration", "calibrated_probability"),
            ):
                metrics = metric_suite(test, column, ece_bins=bins)
                metric_records.append(
                    {"base_model": model_key, "seed": seed, "stage": stage, **metrics}
                )
                strata = stratified_metrics(test, column, bins=bins)
                strata.insert(0, "stage", stage)
                strata.insert(0, "seed", seed)
                strata.insert(0, "base_model", model_key)
                strata_records.append(strata)
            bootstrap_records[f"{model_key}/seed_{seed}"] = trace_bootstrap(
                test,
                "calibrated_probability",
                bins=bins,
                replicates=int(config["evaluation"]["bootstrap_replicates"]),
                seed=int(config["evaluation"]["bootstrap_seed"]) + seed,
            )
            sweep = threshold_sweep(test, config["evaluation"]["threshold_grid"])
            sweep.insert(0, "seed", seed)
            sweep.insert(0, "base_model", model_key)
            sweep_records.append(sweep)
            _plot_calibration(
                test,
                artifact_root / f"test/plots/{model_key}_seed_{seed}.png",
                probability_columns=["raw_probability", "calibrated_probability"],
                bins=bins,
            )

            # Frozen test diagnostics. These are evaluated only after the common
            # architecture and LR have been selected and cannot change selection.
            diagnostic_candidates = ["position_only", *config["training"]["architectures"]]
            for candidate in map(str, diagnostic_candidates):
                candidate_lr = (
                    float(config["training"]["position_only_learning_rate"])
                    if candidate == "position_only"
                    else learning_rate
                )
                candidate_checkpoint = (
                    _run_root(artifact_root, model_key, candidate, candidate_lr, seed) / "best.pt"
                )
                candidate_model = load_trained_predictor(candidate_checkpoint, device=device)
                candidate_test = predict_split(
                    candidate_model,
                    corpus,
                    split="teacher_forced_test",
                    batch_size=int(config["training"]["batch_size_traces"]),
                    device=device,
                    seed=seed,
                )
                candidate_metrics = metric_suite(candidate_test, "raw_probability", ece_bins=bins)
                diagnostic_records.append(
                    {
                        "base_model": model_key,
                        "seed": seed,
                        "architecture": candidate,
                        "learning_rate": candidate_lr,
                        "test_uncalibrated_nll": candidate_metrics[
                            "trace_weighted_binomial_nll"
                        ],
                        "test_brier": candidate_metrics["brier_score"],
                        "test_spearman": candidate_metrics["spearman"],
                        "selection_already_frozen": True,
                    }
                )

    predictions = pd.concat(all_predictions, ignore_index=True)
    atomic_parquet(artifact_root / "test/teacher_forced_test_predictions.parquet", predictions)
    metrics_frame = pd.DataFrame(metric_records)
    metrics_frame.to_csv(artifact_root / "test/metrics_by_model_seed.csv", index=False)
    pd.concat(strata_records, ignore_index=True).to_csv(
        artifact_root / "test/stratified_metrics.csv", index=False
    )
    seed_summary = (
        metrics_frame.groupby(["base_model", "stage"], as_index=False)
        .agg(
            nll_mean=("trace_weighted_binomial_nll", "mean"),
            nll_std=("trace_weighted_binomial_nll", "std"),
            brier_mean=("brier_score", "mean"),
            brier_std=("brier_score", "std"),
            ece_mean=("ece_equal_count_10", "mean"),
            ece_std=("ece_equal_count_10", "std"),
            spearman_mean=("spearman", "mean"),
            concordance_mean=("within_trace_concordance", "mean"),
        )
    )
    seed_summary.to_csv(artifact_root / "test/seed_summary.csv", index=False)
    pd.DataFrame(calibration_records).to_csv(
        artifact_root / "calibration/calibration_summary.csv", index=False
    )
    pd.concat(sweep_records, ignore_index=True).to_csv(
        artifact_root / "test/threshold_diagnostics.csv", index=False
    )
    pd.DataFrame(diagnostic_records).to_csv(
        artifact_root / "test/architecture_diagnostics.csv", index=False
    )
    atomic_json(artifact_root / "test/bootstrap_confidence_intervals.json", bootstrap_records)
    result = {
        "status": "COMPLETE",
        "selected_architecture": architecture,
        "selected_learning_rate": learning_rate,
        "models": models,
        "seeds": seeds,
        "checkpoint_predictions": len(predictions),
        "calibrators": len(calibration_records),
        "bootstrap_replicates": int(config["evaluation"]["bootstrap_replicates"]),
        "native_evaluation_used": False,
        "final_tau_selected": False,
    }
    atomic_json(artifact_root / "test/complete.json", result)
    return result
