"""Publication tables, figures, reports, and terminal integrity audit."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping
import uuid

import numpy as np
import pandas as pd

from safeprefix.reproducibility import atomic_json, atomic_text, now_iso, stable_hash

from .data import MODEL_KEYS, row_artifact_hash, sha256_file


REPORT_NAMES = (
    "PRELAUNCH_CENSUS_AND_INTEGRITY.md",
    "CROSS_FITTED_CALIBRATION_REPORT.md",
    "DENSE_CHECKPOINT_OUTCOMES_REPORT.md",
    "FULL_REGENERATION_REPORT.md",
    "THRESHOLD_CURVE_REPORT.md",
    "NONINFERIORITY_REPORT.md",
    "THRESHOLD_SELECTION_REPORT.md",
    "SEED_ROBUSTNESS_REPORT.md",
    "BASELINE_POLICY_REPORT.md",
    "FINAL_POLICY_SUMMARY.md",
    "INTEGRITY_REPORT.md",
    "README.md",
)


def _markdown_table(frame: pd.DataFrame, columns: list[str], *, digits: int = 6) -> str:
    if not len(frame):
        return "_No rows._"
    headers = columns
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame[columns].itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append("NA" if not math.isfinite(float(value)) else f"{float(value):.{digits}f}")
            elif isinstance(value, (bool, np.bool_)):
                values.append("PASS" if bool(value) else "FAIL")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write(path: Path, title: str, body: str) -> None:
    atomic_text(path, f"# {title}\n\n{body.strip()}\n")


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
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


def _policy_slices(root: Path, policy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_domain_rows: list[dict[str, Any]] = []
    for (tau, model_key, domain_name), part in policy.groupby(
        ["threshold", "base_model", "domain"], sort=True
    ):
        selected = part[~part["fallback"]]
        fallback = part[part["fallback"]]
        full_cost = float(part["full_regeneration_fresh_tokens"].mean())
        model_domain_rows.append(
            {
                "threshold": float(tau),
                "base_model": str(model_key),
                "domain": str(domain_name),
                "traces": len(part),
                "checkpoint_coverage": float((~part["fallback"]).mean()),
                "fallback_rate": float(part["fallback"].mean()),
                "policy_success": float(part["policy_success"].mean()),
                "full_regeneration_success": float(part["full_regeneration_success_rate"].mean()),
                "success_difference": float(part["success_difference"].mean()),
                "mean_fresh_tokens": float(part["fresh_tokens"].mean()),
                "median_fresh_tokens": float(part["fresh_tokens"].median()),
                "fresh_token_savings_fraction": 1.0 - float(part["fresh_tokens"].mean()) / max(full_cost, 1e-12),
                "mean_generated_output_tokens": float(part["generated_output_tokens"].mean()),
                "mean_retained_prefix_tokens": float(part["retained_prefix_tokens"].mean()),
                "mean_retained_prefix_fraction": float(part["retained_prefix_fraction"].mean()),
                "median_retained_prefix_fraction": float(part["retained_prefix_fraction"].median()),
                "mean_selected_checkpoint_ordinal": float(selected["selected_checkpoint_ordinal"].mean()) if len(selected) else np.nan,
                "mean_normalized_selected_checkpoint_position": float(selected["normalized_selected_checkpoint_position"].mean()) if len(selected) else np.nan,
                "selected_checkpoint_empirical_success": float(selected["policy_success"].mean()) if len(selected) else np.nan,
                "fallback_empirical_success": float(fallback["policy_success"].mean()) if len(fallback) else np.nan,
                "earliest_checkpoint_frequency": float((selected["outcome_checkpoint_ordinal"] == 0).sum() / len(part)),
                "latest_checkpoint_frequency": float((selected["outcome_checkpoint_ordinal"] == selected["total_checkpoint_count"] - 1).sum() / len(part)),
                "nominal_minus_empirical_selected_success": float(tau - selected["policy_success"].mean()) if len(selected) else np.nan,
                "mean_measured_latency_seconds": float(part["measured_latency_seconds"].mean()),
            }
        )
    model_domain = pd.DataFrame(model_domain_rows)
    domain_rows: list[dict[str, Any]] = []
    for (tau, domain_name), part in model_domain.groupby(["threshold", "domain"], sort=True):
        raw = policy[policy["threshold"].eq(tau) & policy["domain"].eq(domain_name)]
        full_cost = float(part["mean_fresh_tokens"].mean())
        full_reference = float(
            raw.groupby("base_model")["full_regeneration_fresh_tokens"].mean().mean()
        )
        domain_rows.append(
            {
                "threshold": float(tau),
                "domain": str(domain_name),
                "model_count": int(part["base_model"].nunique()),
                "macro_checkpoint_coverage": float(part["checkpoint_coverage"].mean()),
                "macro_fallback_rate": float(part["fallback_rate"].mean()),
                "macro_policy_success": float(part["policy_success"].mean()),
                "macro_full_regeneration_success": float(part["full_regeneration_success"].mean()),
                "macro_success_difference": float(part["success_difference"].mean()),
                "macro_mean_fresh_tokens": full_cost,
                "macro_median_fresh_tokens": float(raw["fresh_tokens"].median()),
                "macro_fresh_token_savings_fraction": 1.0 - full_cost / max(full_reference, 1e-12),
                "macro_mean_retained_prefix_fraction": float(part["mean_retained_prefix_fraction"].mean()),
                "macro_median_retained_prefix_fraction": float(raw["retained_prefix_fraction"].median()),
                "macro_selected_checkpoint_empirical_success": float(part["selected_checkpoint_empirical_success"].mean()),
                "macro_fallback_empirical_success": float(part["fallback_empirical_success"].mean()),
                "macro_earliest_checkpoint_frequency": float(part["earliest_checkpoint_frequency"].mean()),
                "macro_latest_checkpoint_frequency": float(part["latest_checkpoint_frequency"].mean()),
                "macro_nominal_minus_empirical_selected_success": float(part["nominal_minus_empirical_selected_success"].mean()),
            }
        )
    domain = pd.DataFrame(domain_rows)
    _atomic_csv(root / "policy/threshold_metrics_by_model_domain.csv", model_domain)
    _atomic_csv(root / "policy/threshold_metrics_by_domain.csv", domain)
    return domain, model_domain


def _plot_figures(
    root: Path,
    curve: pd.DataFrame,
    by_model: pd.DataFrame,
    domain: pd.DataFrame,
    policy: pd.DataFrame,
    selection: Mapping[str, Any],
    seed_agreement: pd.DataFrame,
    baselines: pd.DataFrame,
    reliability: pd.DataFrame,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    selected_tau = selection.get("selected_tau")

    def finish(fig: Any, path: str) -> None:
        fig.tight_layout()
        fig.savefig(figures / path, dpi=220, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.plot(curve["macro_mean_fresh_tokens"], curve["macro_policy_success"], "o-", label="threshold curve")
    anchor = curve[curve["threshold"].eq(1.0)]
    ax.scatter(anchor["macro_mean_fresh_tokens"], anchor["macro_policy_success"], marker="s", s=70, label="full regeneration")
    if selected_tau is not None:
        chosen = curve[curve["threshold"].eq(float(selected_tau))]
        ax.scatter(chosen["macro_mean_fresh_tokens"], chosen["macro_policy_success"], marker="*", s=180, label=f"tau*={selected_tau:.2f}")
    ax.set(xlabel="Mean fresh-token cost", ylabel="One-rollout repair success", title="Success-compute threshold curve")
    ax.grid(alpha=0.25); ax.legend()
    finish(fig, "01_success_vs_fresh_tokens.png")

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.errorbar(
        curve["threshold"], curve["macro_success_difference"],
        yerr=[curve["macro_success_difference"] - curve["macro_difference_lcb_95"], curve["macro_difference_ucb_95"] - curve["macro_success_difference"]],
        fmt="o-", capsize=2,
    )
    ax.axhline(-0.03, color="red", linestyle="--", label="noninferiority margin")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set(xlabel="tau", ylabel="SafePrefix minus full-regeneration success", title="Paired noninferiority")
    ax.grid(alpha=0.25); ax.legend()
    finish(fig, "02_success_difference_ci.png")

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.plot(curve["threshold"], curve["macro_checkpoint_coverage"], "o-", label="checkpoint coverage")
    ax.plot(curve["threshold"], curve["macro_fallback_rate"], "s-", label="fallback rate")
    ax.axhline(0.20, color="red", linestyle="--", label="minimum coverage")
    ax.set(xlabel="tau", ylabel="Fraction", ylim=(-0.02, 1.02), title="Coverage and fallback")
    ax.grid(alpha=0.25); ax.legend()
    finish(fig, "03_coverage_and_fallback.png")

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.plot(curve["threshold"], 100 * curve["macro_fresh_token_savings_fraction"], "o-")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xlabel="tau", ylabel="Fresh-token savings (%)", title="Compute savings versus full regeneration")
    ax.grid(alpha=0.25)
    finish(fig, "04_fresh_token_savings.png")

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.plot(curve["threshold"], curve["macro_mean_retained_prefix_fraction"], "o-")
    ax.set(xlabel="tau", ylabel="Mean retained-prefix fraction", ylim=(-0.02, 1.02), title="Retained computation")
    ax.grid(alpha=0.25)
    finish(fig, "05_retained_prefix_fraction.png")

    selected_empirical = (
        policy[~policy["fallback"]].groupby("threshold")["policy_success"].mean().reset_index()
    )
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.plot(selected_empirical["threshold"], selected_empirical["policy_success"], "o-", label="empirical selected success")
    ax.plot([0, 1], [0, 1], "--", color="black", label="nominal=empirical")
    ax.set(xlabel="Nominal tau", ylabel="Empirical selected-checkpoint success", xlim=(0, 1), ylim=(0, 1), title="Threshold calibration")
    ax.grid(alpha=0.25); ax.legend()
    finish(fig, "06_empirical_success_vs_nominal_tau.png")

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for model_key, part in by_model.groupby("base_model"):
        ax.plot(part["threshold"], part["policy_success"], marker="o", label=model_key)
    ax.set(xlabel="tau", ylabel="One-rollout success", title="Model-specific threshold curves")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    finish(fig, "07_model_specific_curves.png")

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for domain_name, part in domain.groupby("domain"):
        ax.plot(part["threshold"], part["macro_policy_success"], marker="o", label=domain_name)
    ax.set(xlabel="tau", ylabel="Macro one-rollout success", title="Domain-specific threshold curves")
    ax.grid(alpha=0.25); ax.legend(fontsize=7)
    finish(fig, "08_domain_specific_curves.png")

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    if len(seed_agreement):
        if selected_tau is not None:
            agreement_plot = seed_agreement[seed_agreement["is_primary_tau"]].copy()
        else:
            agreement_plot = (
                seed_agreement.groupby(["left_variant", "right_variant"], as_index=False)
                .agg(
                    fallback_agreement=("fallback_agreement", "mean"),
                    exact_action_checkpoint_agreement=("exact_action_checkpoint_agreement", "mean"),
                )
            )
        labels = [f"{a}\nvs\n{b}" for a, b in zip(agreement_plot["left_variant"], agreement_plot["right_variant"])]
        x = np.arange(len(labels))
        ax.bar(x - 0.18, agreement_plot["fallback_agreement"], width=0.36, label="fallback agreement")
        ax.bar(x + 0.18, agreement_plot["exact_action_checkpoint_agreement"], width=0.36, label="exact checkpoint agreement")
        ax.set_xticks(x, labels)
    ax.set(ylabel="Agreement", ylim=(0, 1), title="Cross-seed checkpoint-selection agreement")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)
    finish(fig, "09_cross_seed_agreement.png")

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    x = np.arange(len(baselines))
    scatter = ax.scatter(baselines["macro_mean_fresh_tokens"], baselines["macro_success"], s=70)
    for _, row in baselines.iterrows():
        ax.annotate(str(row["baseline"]), (row["macro_mean_fresh_tokens"], row["macro_success"]), fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.set(xlabel="Mean fresh-token cost", ylabel="Macro one-rollout success", title="Teacher-forced policy baselines")
    ax.grid(alpha=0.25)
    finish(fig, "10_baseline_policy_comparison.png")

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for model_key, part in reliability.groupby("base_model"):
        ax.plot(part["mean_prediction"], part["empirical_success_rate"], marker="o", label=model_key)
    ax.plot([0, 1], [0, 1], "--", color="black")
    ax.set(xlabel="OOF calibrated probability", ylabel="Dense K=16 empirical success", xlim=(0, 1), ylim=(0, 1), title="Cross-fitted reliability")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    finish(fig, "11_cross_fitted_reliability.png")


def _terminal_integrity(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    expected = config["source"]["expected"]
    failures: list[str] = []
    original = pd.read_parquet(root / "raw_outcomes/original_k4_checkpoint_suffixes.parquet")
    added = pd.read_parquet(root / "raw_outcomes/all_added_checkpoint_suffixes.parquet")
    full = pd.read_parquet(root / "raw_outcomes/all_full_regenerations.parquet")
    dense = pd.read_parquet(root / "outcomes/dense_checkpoint_outcomes.parquet")
    oof = pd.read_parquet(root / "calibration/cross_fitted_predictions.parquet")
    calibrators = pd.read_parquet(root / "calibration/cross_fitted_calibrator_manifest.parquet")
    policy = pd.read_parquet(root / "policy/threshold_policy_manifest.parquet")
    frozen = json.loads((root / "manifests/frozen_protocol.json").read_text())
    access = json.loads((root / "manifests/source_access_ledger.json").read_text())
    checkpoint_manifest = pd.read_parquet(root / "manifests/dense_checkpoint_rollout_manifest.parquet")
    regeneration_manifest = pd.read_parquet(root / "manifests/full_regeneration_manifest.parquet")
    if len(original) != int(expected["checkpoints_total"]) * 4:
        failures.append("original_k4_count")
    if sha256_file(root / "raw_outcomes/original_k4_checkpoint_suffixes.parquet") != frozen["original_k4_copy_sha256"]:
        failures.append("original_k4_copy_changed")
    if len(added) != int(expected["added_checkpoint_rollouts_total"]):
        failures.append("added_checkpoint_count")
    if len(full) != int(expected["full_regenerations_total"]):
        failures.append("full_regeneration_count")
    expected_added_keys = set(
        zip(
            checkpoint_manifest["base_model"].astype(str),
            checkpoint_manifest["trace_id"].astype(str),
            checkpoint_manifest["checkpoint_id"].astype(str),
            checkpoint_manifest["checkpoint_ordinal"].astype(int),
            checkpoint_manifest["rollout_index"].astype(int),
            checkpoint_manifest["rollout_seed"].astype(int),
        )
    )
    observed_added_keys = set(
        zip(
            added["model_key"].astype(str),
            added["trace_id"].astype(str),
            added["checkpoint_id"].astype(str),
            added["checkpoint_ordinal"].astype(int),
            added["rollout_index"].astype(int),
            added["rollout_seed"].astype(int),
        )
    )
    if observed_added_keys != expected_added_keys:
        failures.append("added_logical_seed_keys")
    expected_full_keys = set(
        zip(
            regeneration_manifest["base_model"].astype(str),
            regeneration_manifest["trace_id"].astype(str),
            regeneration_manifest["rollout_index"].astype(int),
            regeneration_manifest["rollout_seed"].astype(int),
        )
    )
    observed_full_keys = set(
        zip(
            full["model_key"].astype(str),
            full["trace_id"].astype(str),
            full["rollout_index"].astype(int),
            full["rollout_seed"].astype(int),
        )
    )
    if observed_full_keys != expected_full_keys:
        failures.append("full_logical_seed_keys")
    full_index_sets = full.groupby(["model_key", "trace_id"])["rollout_index"].agg(
        lambda values: tuple(sorted(map(int, values)))
    )
    if set(full_index_sets) != {tuple(range(16))}:
        failures.append("full_regeneration_index_set")
    if len(dense) != int(expected["checkpoints_total"]) or not dense["num_rollouts"].eq(16).all():
        failures.append("dense_k16_coverage")
    if len(oof) != int(expected["checkpoints_total"]) * 3 * 2:
        failures.append("oof_prediction_count")
    if len(calibrators) != 4 * 3 * 2 * 5:
        failures.append("crossfit_calibrator_count")
    if len(policy) != int(expected["traces_per_model"]) * 4 * 21:
        failures.append("policy_row_count")
    if sorted(set(policy["threshold"].astype(float))) != sorted([*map(float, config["threshold_selection"]["thresholds"]), 1.0]):
        failures.append("threshold_grid")
    if (
        access.get("native_paths_opened")
        or access.get("teacher_forced_test_rows_returned")
        or access.get("noncalibration_outcome_rows_returned")
        or access.get("geometry_paths_opened")
    ):
        failures.append("prohibited_source_access")
    if not added["infrastructure_status"].eq("executed").all() or not full["infrastructure_status"].eq("executed").all():
        failures.append("infrastructure_status")
    if not calibrators["held_out_fold_labels_used_for_fit"].eq(False).all():
        failures.append("crossfit_label_leakage")
    if not calibrators["probe_weights_modified"].eq(False).all():
        failures.append("probe_modified")
    if original["artifact_hash"].duplicated().any() or added["artifact_hash"].duplicated().any() or full["artifact_hash"].duplicated().any():
        failures.append("duplicate_artifact_hash")
    for name, frame in (("original", original), ("added", added), ("full", full)):
        if any(
            row_artifact_hash(row) != str(row["artifact_hash"])
            for row in frame.to_dict("records")
        ):
            failures.append(f"{name}_artifact_hash")
    if "scientific" in added and not added["scientific"].eq(True).all():
        failures.append("nonscientific_added_row")
    if "scientific" in full and not full["scientific"].eq(True).all():
        failures.append("nonscientific_full_row")
    result = {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "checked_at": now_iso(),
        "counts": {
            "original_k4": len(original),
            "added_checkpoint_rollouts": len(added),
            "full_regenerations": len(full),
            "dense_checkpoints": len(dense),
            "oof_predictions": len(oof),
            "crossfit_calibrators": len(calibrators),
            "policy_rows": len(policy),
        },
        "native_evaluation_used": False,
        "teacher_forced_test_used": False,
        "geometry_test_used": False,
        "threshold_grid_refined": False,
    }
    if failures:
        raise RuntimeError(f"terminal threshold integrity failed: {failures}")
    return result


def report_threshold_experiment(
    config: Mapping[str, Any],
    *,
    artifact_root: Path,
    run_id: str,
) -> dict[str, Any]:
    root = Path(artifact_root)
    frozen = json.loads((root / "manifests/frozen_protocol.json").read_text())
    if frozen.get("configuration_hash") != stable_hash(config):
        raise RuntimeError("live reporting configuration differs from the pre-outcome freeze")
    if list(map(float, frozen.get("threshold_grid", []))) != list(
        map(float, config["threshold_selection"]["thresholds"])
    ):
        raise RuntimeError("live reporting threshold grid differs from the pre-outcome freeze")
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    census = json.loads((root / "manifests/prelaunch_census.json").read_text())
    outcome_summary = json.loads((root / "outcomes/dense_outcome_summary.json").read_text())
    selection = json.loads((root / "policy/threshold_selection.json").read_text())
    curve = pd.read_csv(root / "policy/threshold_curve.csv")
    by_model = pd.read_csv(root / "policy/threshold_metrics_by_model.csv")
    policy = pd.read_parquet(root / "policy/threshold_policy_manifest.parquet")
    diagnostics = pd.read_parquet(root / "calibration/calibration_diagnostics.parquet")
    calibrators = pd.read_parquet(root / "calibration/cross_fitted_calibrator_manifest.parquet")
    reliability = pd.read_parquet(root / "calibration/reliability_tables.parquet")
    full = pd.read_parquet(root / "outcomes/full_regeneration_outcomes.parquet")
    robustness = pd.read_parquet(root / "policy/seed_robustness.parquet")
    agreement = pd.read_parquet(root / "policy/seed_agreement.parquet")
    seed_summary = json.loads((root / "policy/seed_robustness_summary.json").read_text())
    baselines = pd.read_csv(root / "baselines/baseline_summary.csv")
    domain, _ = _policy_slices(root, policy)
    _plot_figures(root, curve, by_model, domain, policy, selection, agreement, baselines, reliability)

    overall_cal = diagnostics[(diagnostics["stratum"].eq("overall"))]
    model_cal = diagnostics[diagnostics["stratum"].eq("base_model")]
    worst_domain = diagnostics[diagnostics["stratum"].eq("model_by_domain")].sort_values(
        "ece_equal_count_10", ascending=False
    ).head(10)
    _write(
        reports / "CROSS_FITTED_CALIBRATION_REPORT.md",
        "Cross-fitted calibration report",
        f"""The frozen linear probes were never retrained. Each positive-affine calibrator was fit on dense K=16 outcomes from four problem-group folds and scored only the held-out fifth fold. The held-out fold's outcomes were not fit inputs. There are {len(calibrators[calibrators['architecture'].eq('linear_probe')])} linear-probe fold calibrators plus {len(calibrators[calibrators['architecture'].eq('position_only')])} position-only baseline calibrators.

## Operational OOF diagnostics

{_markdown_table(overall_cal, ['value','trace_weighted_binomial_nll','brier_score','ece_equal_count_10','calibration_slope','calibration_intercept','spearman'])}

## By model

{_markdown_table(model_cal, ['value','trace_weighted_binomial_nll','brier_score','ece_equal_count_10','calibration_slope','calibration_intercept','spearman'])}

## Highest-ECE model-domain slices

{_markdown_table(worst_domain, ['value','trace_weighted_binomial_nll','brier_score','ece_equal_count_10','traces','checkpoints'])}

No domain-specific calibrator was fit.""",
    )

    dense_by_model = pd.read_parquet(root / "outcomes/dense_checkpoint_outcomes.parquet").groupby("model_key").agg(
        checkpoints=("checkpoint_id", "size"),
        original_successes=("original_success_count", "sum"),
        added_successes=("added_success_count", "sum"),
        dense_success_rate=("dense_success_rate", "mean"),
        mean_suffix_tokens=("mean_suffix_tokens", "mean"),
    ).reset_index()
    _write(
        reports / "DENSE_CHECKPOINT_OUTCOMES_REPORT.md",
        "Dense checkpoint outcomes report",
        f"""Every eligible calibration checkpoint has exactly 16 validly executed outcomes: the immutable original four plus twelve new independent suffixes. Incorrect, truncated, and unparsable normally executed continuations remain valid failures under the frozen verifier.

{_markdown_table(dense_by_model, ['model_key','checkpoints','original_successes','added_successes','dense_success_rate','mean_suffix_tokens'])}

Total added checkpoint suffixes: **{outcome_summary['added_checkpoint_rollouts']}**. Original K=4 rows were copied separately and not overwritten.""",
    )

    full_by_model = full.groupby("model_key").agg(
        traces=("trace_id", "size"),
        success=("full_regeneration_success_rate", "mean"),
        mean_response_tokens=("mean_full_response_tokens", "mean"),
        mean_fresh_tokens=("full_regeneration_fresh_tokens", "mean"),
    ).reset_index()
    _write(
        reports / "FULL_REGENERATION_REPORT.md",
        "Full regeneration report",
        f"""Each calibration model-trace has exactly 16 prompt-root regenerations. Requests contained the original prompt only; failed trace text, wrong answers, verifier feedback, and gold answers were not generation inputs.

{_markdown_table(full_by_model, ['model_key','traces','success','mean_response_tokens','mean_fresh_tokens'])}

Total full regenerations: **{outcome_summary['full_regenerations']}**.""",
    )

    curve_columns = [
        "threshold", "macro_checkpoint_coverage", "macro_fallback_rate", "macro_policy_success",
        "macro_full_regeneration_success", "macro_success_difference", "macro_difference_lcb_95",
        "macro_difference_ucb_95", "macro_mean_fresh_tokens", "macro_fresh_token_savings_fraction",
        "macro_mean_retained_prefix_fraction", "feasible",
    ]
    _write(
        reports / "THRESHOLD_CURVE_REPORT.md",
        "Threshold curve report",
        f"""The raw curve includes the frozen grid from 0.00 through 0.95 and the forced full-regeneration anchor at 1.00. Selection used out-of-fold probabilities only; selected checkpoint outcomes were joined afterward.

{_markdown_table(curve, curve_columns)}

The complete raw and nondominated tables are `policy/threshold_curve.csv` and `policy/success_compute_frontier.csv`.""",
    )

    noninferiority = curve[
        [
            "threshold", "macro_success_difference", "macro_difference_lcb_95", "macro_difference_ucb_95",
            "minimum_model_success_difference", "macro_checkpoint_coverage", "noninferiority_pass",
            "per_model_safeguard_pass", "coverage_pass", "feasible",
        ]
    ]
    _write(
        reports / "NONINFERIORITY_REPORT.md",
        "Noninferiority report",
        f"""Paired 95% intervals use 10,000 domain-stratified bootstrap draws of complete shared trace IDs, preserving all four aligned model rows. The margin is -0.03, every model must have point deficit at least -0.05, and macro checkpoint coverage must reach 20%.

{_markdown_table(noninferiority, list(noninferiority.columns))}""",
    )

    if selection["selected_tau"] is None:
        selection_body = selection["statement"] + " Full regeneration remains the primary operational policy."
        selected_model_rows = pd.DataFrame()
    else:
        tau = float(selection["selected_tau"])
        selected_row = curve[curve["threshold"].eq(tau)]
        selected_model_rows = by_model[by_model["threshold"].eq(tau)]
        selection_body = f"""Frozen global threshold: **tau* = {tau:.2f}**.

{_markdown_table(selected_row, curve_columns)}

## Per-model safeguards

{_markdown_table(selected_model_rows, ['base_model','policy_success','full_regeneration_success','success_difference','checkpoint_coverage','mean_fresh_tokens','fresh_token_savings_fraction'])}"""
    _write(
        reports / "THRESHOLD_SELECTION_REPORT.md",
        "Threshold selection report",
        selection_body
        + "\n\nThe choice was the minimum unrounded equal-model macro fresh-token cost among feasible thresholds, with the frozen 1% conservative higher-tau tie rule.",
    )

    if selection["selected_tau"] is not None:
        agreement_display = agreement[agreement["is_primary_tau"]].copy()
        agreement_heading = "Pairwise action agreement at primary tau"
    else:
        agreement_heading = "Pairwise action agreement across the frozen grid"
        agreement_display = (
            agreement.groupby(["left_variant", "right_variant"], as_index=False)
            .agg(
                mean_fallback_agreement=("fallback_agreement", "mean"),
                minimum_fallback_agreement=("fallback_agreement", "min"),
                mean_exact_action_checkpoint_agreement=("exact_action_checkpoint_agreement", "mean"),
                minimum_exact_action_checkpoint_agreement=("exact_action_checkpoint_agreement", "min"),
            )
        )
    _write(
        reports / "SEED_ROBUSTNESS_REPORT.md",
        "Seed robustness report",
        f"""The operational per-model seeds were frozen as the median architecture-dev NLL seeds before dense outcomes were generated. Best- and worst-dev rank configurations are secondary only.

{_markdown_table(robustness, ['variant','own_selected_tau','primary_tau','primary_tau_feasible','primary_tau_success','primary_tau_fresh_tokens'])}

## {agreement_heading}

{_markdown_table(agreement_display, list(agreement_display.columns) if len(agreement_display) else [])}

Qualitative tradeoff stable under the frozen criterion: **{seed_summary['qualitative_tradeoff_stable']}**. The complete 21-threshold pairwise agreement table is stored in `policy/seed_agreement.parquet`.""",
    )

    _write(
        reports / "BASELINE_POLICY_REPORT.md",
        "Baseline policy report",
        f"""All baselines reuse the same dense K=16 checkpoint and full-regeneration outcomes. Fixed rewinds target retained token fractions 0.75, 0.50, and 0.25 and choose the nearest earlier eligible checkpoint; if none exists they fall back.

{_markdown_table(baselines, list(baselines.columns))}""",
    )

    per_model_thresholds = json.loads((root / "policy/per_model_thresholds.json").read_text())
    final_row = None if selection["selected_tau"] is None else curve[curve["threshold"].eq(float(selection["selected_tau"]))].iloc[0].to_dict()
    final_summary = {
        "status": "COMPLETE",
        "run_id": run_id,
        "selected_tau": selection["selected_tau"],
        "selection_status": selection["status"],
        "calibration_shared_traces": census["shared_traces"],
        "checkpoints": census["checkpoints_total"],
        "new_checkpoint_rollouts": outcome_summary["added_checkpoint_rollouts"],
        "full_regenerations": outcome_summary["full_regenerations"],
        "operational_seeds": census["operational_seeds"],
        "selected_metrics": final_row,
        "per_model_thresholds_analysis_only": per_model_thresholds,
        "seed_robustness": seed_summary,
        "worst_model_domain_calibration_ece": (
            None if not len(worst_domain) else float(worst_domain.iloc[0]["ece_equal_count_10"])
        ),
        "worst_model_domain_calibration_slice": (
            None if not len(worst_domain) else str(worst_domain.iloc[0]["value"])
        ),
        "native_evaluation_used": False,
        "teacher_forced_test_used": False,
        "geometry_test_used": False,
    }
    atomic_json(root / "final_policy_summary.json", final_summary)
    final_text = selection_body if selection["selected_tau"] is not None else selection["statement"]
    _write(
        reports / "FINAL_POLICY_SUMMARY.md",
        "Final policy summary",
        f"""{final_text}

- Calibration cohort: **{census['shared_traces']} shared traces**, **{census['checkpoints_total']} checkpoints**.
- New suffix continuations: **{outcome_summary['added_checkpoint_rollouts']}**.
- Full regenerations: **{outcome_summary['full_regenerations']}**.
- Native outcomes accessed: **no**.
- Teacher-forced test accessed: **no**.
- Geometry-test data accessed: **no**.

The principal calibration limitation remains nonuniform transfer across model-domain slices; no domain-specific correction was added.""",
    )

    _write(
        reports / "README.md",
        "SafePrefix teacher-forced threshold selection v1",
        f"""Run ID: `{run_id}`

This artifact freezes the operational checkpoint threshold using only the completed boundary-model calibration split, five-fold problem-level cross-fitting, dense K=16 checkpoint outcomes, and paired K=16 full regenerations. It contains no native or teacher-forced-test result.

Primary result: **{selection['status']}**, selected tau: **{selection['selected_tau']}**.

See `FINAL_POLICY_SUMMARY.md`, `THRESHOLD_SELECTION_REPORT.md`, and the machine-readable tables under `policy/`, `calibration/`, `outcomes/`, and `baselines/`.""",
    )

    integrity = _terminal_integrity(config, root)
    atomic_json(root / "integrity/final_integrity.json", integrity)
    _write(
        reports / "INTEGRITY_REPORT.md",
        "Integrity report",
        f"""Status: **{integrity['status']}**

Failures: `{integrity['failures']}`

{json.dumps(integrity['counts'], indent=2)}

All scientific pack markers, exact K=16 coverage, OOF fold guards, frozen grid, original K=4 copy hash, and prohibited-access ledgers passed.""",
    )

    # Hash completed scientific artifacts after report generation. The hash
    # manifest and terminal state intentionally exclude themselves.
    hash_rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.endswith(".tmp"):
            continue
        relative = path.relative_to(root).as_posix()
        if relative in {"artifact_hashes.json", "COMPLETE.json"}:
            continue
        hash_rows.append({"path": relative, "size": path.stat().st_size, "sha256": sha256_file(path)})
    atomic_json(root / "artifact_hashes.json", {"files": hash_rows, "count": len(hash_rows)})
    terminal = {
        "status": "COMPLETE",
        "run_id": run_id,
        "selected_tau": selection["selected_tau"],
        "selection_status": selection["status"],
        "integrity_status": integrity["status"],
        "artifact_hash_count": len(hash_rows),
        "native_evaluation_used": False,
        "teacher_forced_test_used": False,
        "geometry_test_used": False,
        "completed_at": now_iso(),
    }
    missing_reports = [name for name in REPORT_NAMES if not (reports / name).is_file()]
    required_figures = [f"{index:02d}_" for index in range(1, 11)]
    figure_names = [path.name for path in (root / "figures").glob("*.png")]
    missing_figures = [prefix for prefix in required_figures if not any(name.startswith(prefix) for name in figure_names)]
    if missing_reports or missing_figures:
        raise RuntimeError(
            f"required publication products missing: reports={missing_reports}, figures={missing_figures}"
        )
    atomic_json(root / "COMPLETE.json", terminal)
    return terminal


def publish_compact_export(
    *, artifact_root: Path, destination: Path, run_id: str
) -> dict[str, Any]:
    """Publish reports and compact machine-readable support, excluding raw generations."""
    source = Path(artifact_root)
    complete = json.loads((source / "COMPLETE.json").read_text())
    integrity = json.loads((source / "integrity/final_integrity.json").read_text())
    if complete.get("status") != "COMPLETE" or integrity.get("status") != "PASS":
        raise RuntimeError("only a complete, integrity-passing threshold run may be published")
    if str(complete.get("run_id")) != str(run_id):
        raise RuntimeError("published run ID differs from the completed artifact")
    target = Path(destination)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite an existing published export: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    relative_files = [
        "final_policy_summary.json",
        "artifact_hashes.json",
        "analysis_summary.json",
        "integrity/final_integrity.json",
        "manifests/prelaunch_census.json",
        "manifests/operational_seeds.json",
        "manifests/frozen_protocol.json",
        "manifests/source_access_ledger.json",
        "manifests/five_fold_assignment.parquet",
        "manifests/model_seed_manifest.parquet",
        "manifests/exclusion_manifest.parquet",
        "calibration/calibration_diagnostics.parquet",
        "calibration/reliability_tables.parquet",
        "calibration/cross_fitted_calibrator_manifest.parquet",
        "calibration/deployment_calibrator_manifest.parquet",
        "policy/threshold_curve.csv",
        "policy/threshold_metrics_by_model.csv",
        "policy/threshold_metrics_by_domain.csv",
        "policy/threshold_metrics_by_model_domain.csv",
        "policy/selection_distribution.csv",
        "policy/success_compute_frontier.csv",
        "policy/threshold_selection.json",
        "policy/per_model_thresholds.json",
        "policy/seed_configuration_results.json",
        "policy/seed_robustness_summary.json",
        "policy/seed_robustness.parquet",
        "policy/seed_agreement.parquet",
        "baselines/baseline_summary.csv",
        "baselines/position_only_threshold_curve.csv",
        "baselines/position_only_selection.json",
    ]
    relative_files += [str(path.relative_to(source)) for path in sorted((source / "reports").glob("*.md"))]
    relative_files += [str(path.relative_to(source)) for path in sorted((source / "figures").glob("*.png"))]
    copied: list[dict[str, Any]] = []
    try:
        for relative in dict.fromkeys(relative_files):
            source_path = source / relative
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            destination_path = staging / relative
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, destination_path)
            copied.append(
                {
                    "path": relative,
                    "size": destination_path.stat().st_size,
                    "sha256": sha256_file(destination_path),
                }
            )
        atomic_text(
            staging / "README.md",
            (source / "reports/README.md").read_text()
            + "\nThis compact export intentionally excludes raw suffix and full-regeneration text. "
            + "See `EXPORT_MANIFEST.json` for exact hashes.\n",
        )
        copied.append(
            {
                "path": "README.md",
                "size": (staging / "README.md").stat().st_size,
                "sha256": sha256_file(staging / "README.md"),
            }
        )
        copied.append(
            {
                "path": "COMPLETE.json",
                "size": (source / "COMPLETE.json").stat().st_size,
                "sha256": sha256_file(source / "COMPLETE.json"),
            }
        )
        atomic_json(
            staging / "EXPORT_MANIFEST.json",
            {
                "status": "COMPLETE",
                "run_id": run_id,
                "raw_generations_included": False,
                "file_count": len(copied),
                "files": copied,
            },
        )
        # The source terminal marker is copied last, then the complete staging
        # directory is exposed with one same-filesystem rename.
        shutil.copyfile(source / "COMPLETE.json", staging / "COMPLETE.json")
        staging.replace(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "status": "COMPLETE",
        "run_id": run_id,
        "destination": str(target),
        "file_count": len(copied),
        "raw_generations_included": False,
    }
