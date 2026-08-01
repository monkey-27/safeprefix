"""Publication tables and figures for prefix validity and gate effects."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from safeprefix.reproducibility import atomic_json, atomic_parquet

from .evaluation import boundary_records


def compact_localization_table(
    metrics: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> pd.DataFrame:
    rows = []
    for model, architectures in sorted(metrics.items()):
        for architecture, value in sorted(architectures.items()):
            boundary = value["overall"]["boundary"]
            checkpoint = value["overall"]["checkpoint"]
            rows.append({
                "model_key": model, "probe": architecture,
                "checkpoint_trace_weighted_nll": checkpoint.get("trace_weighted_nll"),
                "checkpoint_trace_weighted_brier": checkpoint.get("trace_weighted_brier"),
                "checkpoint_trace_weighted_ece": checkpoint.get("trace_weighted_ece"),
                "checkpoint_roc_auc": checkpoint.get("roc_auc"),
                "checkpoint_average_precision": checkpoint.get("average_precision"),
                "checkpoint_accuracy": checkpoint.get("accuracy"),
                "checkpoint_balanced_accuracy": checkpoint.get("balanced_accuracy"),
                "exact_boundary_accuracy": boundary["exact_last_valid_checkpoint_accuracy"],
                "within_one_accuracy": boundary["within_one_checkpoint_accuracy"],
                "mean_absolute_checkpoint_error": boundary["mean_absolute_checkpoint_error"],
                "median_absolute_checkpoint_error": boundary.get("median_absolute_checkpoint_error"),
                "normalized_token_position_error": boundary.get("normalized_token_position_error"),
                "late_boundary_rate": boundary["late_boundary_rate"],
                "early_boundary_rate": boundary["early_boundary_rate"],
                "retained_valid_prefix_fraction": boundary["mean_retained_valid_prefix_fraction"],
                "non_root_coverage": boundary["non_root_predicted_boundary_coverage"],
            })
    return pd.DataFrame(rows)


def write_publication_artifacts(
    *, output_root: str | Path,
    processbench_metrics: Mapping[str, Mapping[str, Mapping[str, Any]]],
    hidden_test_predictions: pd.DataFrame,
    gate_cutoffs: Mapping[str, float],
) -> dict[str, str]:
    """Write requested compact outputs after all underlying rows are final."""

    import matplotlib.pyplot as plt

    root = Path(output_root) / "publication"
    root.mkdir(parents=True, exist_ok=True)
    localization = compact_localization_table(processbench_metrics)
    localization.to_csv(root / "first_error_localization.csv", index=False)
    numeric = [
        column for column in localization.columns
        if column not in {"model_key", "probe"}
    ]
    macro = localization.groupby("probe", as_index=False)[numeric].mean()
    macro.insert(0, "aggregation", "equal_model_macro")
    macro.to_csv(root / "first_error_localization_macro.csv", index=False)
    domain_rows = []
    for model, architectures in sorted(processbench_metrics.items()):
        for architecture, values in sorted(architectures.items()):
            for domain, metrics in sorted(values.get("by_domain", {}).items()):
                checkpoint = metrics["checkpoint"]
                boundary = metrics["boundary"]
                domain_rows.append({
                    "model_key": model,
                    "probe": architecture,
                    "domain": domain,
                    "checkpoint_trace_weighted_nll": checkpoint["trace_weighted_nll"],
                    "checkpoint_trace_weighted_brier": checkpoint["trace_weighted_brier"],
                    "checkpoint_trace_weighted_ece": checkpoint["trace_weighted_ece"],
                    "checkpoint_roc_auc": checkpoint["roc_auc"],
                    "checkpoint_average_precision": checkpoint["average_precision"],
                    "checkpoint_accuracy": checkpoint["accuracy"],
                    "checkpoint_balanced_accuracy": checkpoint["balanced_accuracy"],
                    **{
                        f"boundary_{key}": value
                        for key, value in boundary.items()
                        if key != "traces"
                    },
                })
    domain_table = pd.DataFrame(domain_rows)
    if len(domain_table):
        domain_table.to_csv(root / "first_error_localization_by_domain.csv", index=False)
        domain_numeric = [
            column for column in domain_table.columns
            if column not in {"model_key", "probe", "domain"}
        ]
        domain_macro = domain_table.groupby(
            ["probe", "domain"], as_index=False
        )[domain_numeric].mean()
        domain_macro.insert(0, "aggregation", "equal_model_macro")
        domain_macro.to_csv(
            root / "first_error_localization_by_domain_macro.csv", index=False
        )
    boundary_parts = []
    for model, part in hidden_test_predictions.groupby("model_key", sort=True):
        boundary_parts.append(boundary_records(part, gamma=float(gate_cutoffs[str(model)])))
    boundaries = pd.concat(boundary_parts, ignore_index=True)
    atomic_parquet(root / "predicted_true_boundaries.parquet", boundaries)
    figure, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=False, sharey=False)
    for axis, (model, part) in zip(axes.ravel(), boundaries.groupby("base_model", sort=True)):
        axis.scatter(part["true_boundary"], part["predicted_boundary"], s=16, alpha=0.6)
        maximum = max(int(part["true_boundary"].max()), int(part["predicted_boundary"].max()), 1)
        axis.plot([0, maximum], [0, maximum], linestyle="--", color="black", linewidth=1)
        axis.set_title(str(model)); axis.set_xlabel("True last-valid checkpoint"); axis.set_ylabel("Predicted")
    figure.tight_layout(); figure.savefig(root / "predicted_vs_true_first_error_boundary.png", dpi=220)
    plt.close(figure)

    manifest = {
        "status": "COMPLETE", "localization_rows": len(localization),
        "localization_macro_rows": len(macro),
        "domain_localization_rows": len(domain_table),
        "boundary_trace_rows": len(boundaries),
        "native_application_outputs_written": False,
        "manuscript_modified": False,
    }
    atomic_json(root / "publication_artifact_manifest.json", manifest)
    return {"root": str(root), **{key: str(root / value) for key, value in {
        "localization_table": "first_error_localization.csv",
        "boundary_figure": "predicted_vs_true_first_error_boundary.png",
    }.items()}}
