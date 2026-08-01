"""Required report/figure contracts for geometry orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import json
import shutil

import numpy as np
import pandas as pd


REQUIRED_REPORTS = (
    "PRELAUNCH_CENSUS_AND_INTEGRITY.md",
    "DENSE_ROLLOUT_REPORT.md",
    "AXIS_STABILITY_REPORT.md",
    "ONE_DIMENSIONALITY_REPORT.md",
    "TRAJECTORY_SPECIFICITY_REPORT.md",
    "TRAJECTORY_GEOMETRY_REPORT.md",
    "LOCAL_BRANCH_GEOMETRY_REPORT.md",
    "CROSS_DOMAIN_GEOMETRY_REPORT.md",
    "MARGIN_AND_ROBUSTNESS_REPORT.md",
    "FINAL_GEOMETRY_SUMMARY.md",
    "INTEGRITY_REPORT.md",
    "README.md",
)

REQUIRED_FIGURES = (
    "dense_recoverability_vs_axis.png",
    "predictor_comparison.png",
    "within_trace_centered.png",
    "trajectory_shape_prevalence.png",
    "representative_trajectories.png",
    "collapse_vs_first_error.png",
    "child_branch_separation.png",
    "parent_vs_high_child_probability.png",
    "cross_domain_transfer_matrix.png",
    "seed_stability.png",
)


def required_output_paths(root: Path) -> dict[str, list[Path]]:
    return {
        "reports": [root / "reports" / name for name in REQUIRED_REPORTS],
        "figures": [root / "figures" / name for name in REQUIRED_FIGURES],
    }


def validate_required_outputs(root: Path) -> dict[str, Any]:
    paths = required_output_paths(root)
    missing = {
        kind: [str(path) for path in values if not path.is_file()]
        for kind, values in paths.items()
    }
    missing = {kind: values for kind, values in missing.items() if values}
    return {
        "passed": not missing,
        "missing": missing,
        "report_count": len(paths["reports"]),
        "figure_count": len(paths["figures"]),
    }


def report_context(
    *,
    hypothesis_results: Mapping[str, Any],
    native_artifacts_accessed: bool,
    new_calibrator_fit: bool,
    operational_tau_selected: bool,
) -> dict[str, Any]:
    """Machine-readable guard included by every final report writer."""
    if native_artifacts_accessed:
        raise RuntimeError("native artifacts are prohibited in teacher-forced geometry")
    if new_calibrator_fit:
        raise RuntimeError("new calibration is prohibited")
    if operational_tau_selected:
        raise RuntimeError("the geometry study must not select an operational tau")
    required = {"H1", "H2", "H3", "H4"}
    if set(hypothesis_results) != required:
        raise ValueError("final geometry summary requires H1--H4")
    return {
        "hypotheses": dict(hypothesis_results),
        "native_artifacts_accessed": False,
        "new_calibrator_fit": False,
        "operational_tau_selected": False,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def generate_geometry_figures(
    output_dir: Path,
    *,
    checkpoint_frame: pd.DataFrame,
    shape_records: list[Mapping[str, Any]],
    branch_frame: pd.DataFrame,
    transfer_frame: pd.DataFrame,
) -> dict[str, str]:
    """Create compact diagnostic hooks; publication orchestration may restyle them."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    figure, axis = plt.subplots(figsize=(5, 4))
    outcome = checkpoint_frame.get("success_count", pd.Series(np.zeros(len(checkpoint_frame))))
    trial = checkpoint_frame.get("trial_count", pd.Series(np.ones(len(checkpoint_frame))))
    axis.scatter(checkpoint_frame.get("axis_score", np.arange(len(checkpoint_frame))), outcome / trial, s=12, alpha=0.6)
    axis.set(xlabel="Recoverability-axis score", ylabel="Observed recoverability")
    path = output_dir / "h1_h2_axis_geometry.png"
    figure.tight_layout(); figure.savefig(path, dpi=180); plt.close(figure)
    paths["h1_h2_axis_geometry"] = str(path)

    figure, axis = plt.subplots(figsize=(5, 4))
    counts = pd.Series([str(row["shape"]) for row in shape_records]).value_counts().sort_index()
    counts.plot.bar(ax=axis); axis.set(xlabel="Trajectory shape", ylabel="Count")
    path = output_dir / "h3_shape_counts.png"
    figure.tight_layout(); figure.savefig(path, dpi=180); plt.close(figure)
    paths["h3_shape_counts"] = str(path)

    figure, axis = plt.subplots(figsize=(5, 4))
    axis.scatter(branch_frame["child_axis_score"], branch_frame["binary_outcome"], s=14, alpha=0.6)
    axis.set(xlabel="Child axis score", ylabel="Verified outcome")
    path = output_dir / "h4_branch_separation.png"
    figure.tight_layout(); figure.savefig(path, dpi=180); plt.close(figure)
    paths["h4_branch_separation"] = str(path)

    figure, axis = plt.subplots(figsize=(5, 4))
    matrix = transfer_frame.pivot_table(index="train_domain", columns="eval_domain", values="metric", aggfunc="mean")
    image = axis.imshow(matrix.to_numpy(float), aspect="auto")
    axis.set_xticks(range(len(matrix.columns)), matrix.columns)
    axis.set_yticks(range(len(matrix.index)), matrix.index)
    axis.set(xlabel="Evaluation domain", ylabel="Training domain")
    figure.colorbar(image, ax=axis)
    path = output_dir / "cross_domain_transfer.png"
    figure.tight_layout(); figure.savefig(path, dpi=180); plt.close(figure)
    paths["cross_domain_transfer"] = str(path)
    return paths


def render_geometry_report(
    path: Path,
    results: Mapping[str, Any],
    *,
    figure_paths: Mapping[str, str] | None = None,
) -> Path:
    """Render supplied results only; this hook never discovers or reads inputs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Recoverability Geometry Diagnostics",
        "",
        "This CPU report hook does not run inference or access native artifacts.",
        "",
    ]
    for key in sorted(results):
        lines.extend(
            [
                f"## {key}",
                "",
                "```json",
                json.dumps(results[key], indent=2, sort_keys=True, default=_json_default),
                "```",
                "",
            ]
        )
    if figure_paths:
        lines.extend(["## Figures", ""])
        for name, figure_path in sorted(figure_paths.items()):
            lines.append(f"- {name}: `{figure_path}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _save_figure(path: Path, draw: Any) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.2, 4.4))
    draw(axis)
    figure.tight_layout()
    figure.savefig(path, dpi=240)
    plt.close(figure)


def write_geometry_outputs(
    root: Path,
    *,
    checkpoints: pd.DataFrame,
    trajectories: pd.DataFrame,
    transfers: pd.DataFrame,
    parents: pd.DataFrame,
    summary: Mapping[str, Any],
    children: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Write data-backed Phase-2 reports and, when supplied, Phase-4 outputs."""
    reports = root / "reports"
    figures = root / "figures"
    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    report_payloads: dict[str, Any] = {
        "PRELAUNCH_CENSUS_AND_INTEGRITY.md": {"census": {"checkpoints": len(checkpoints), "traces": checkpoints["trace_id"].nunique(), "models": checkpoints["base_model"].nunique()}, "native_artifacts_accessed": False},
        "DENSE_ROLLOUT_REPORT.md": {"dense": {"rows": len(checkpoints), "new_rollouts": int(summary["dense_new_rollouts"]), "total_k": 32}},
        "AXIS_STABILITY_REPORT.md": {model: value["stability"] for model, value in summary["models"].items()},
        "ONE_DIMENSIONALITY_REPORT.md": {model: value["H1"] for model, value in summary["models"].items()},
        "TRAJECTORY_SPECIFICITY_REPORT.md": {model: value["H2"] for model, value in summary["models"].items()},
        "TRAJECTORY_GEOMETRY_REPORT.md": summary.get("H3", {
            "category_counts": trajectories["category"].value_counts(dropna=False).to_dict(),
            "formally_eligible": int(trajectories["formal_classification_eligible"].sum()),
        }),
        "LOCAL_BRANCH_GEOMETRY_REPORT.md": {
            "parents_selected": len(parents),
            "child_analysis_status": "complete" if children is not None else "awaiting_pre_registered_branch_inference",
            "H4": summary.get("H4"),
        },
        "CROSS_DOMAIN_GEOMETRY_REPORT.md": summary.get("cross_domain"),
        "MARGIN_AND_ROBUSTNESS_REPORT.md": {
            "status": "complete" if children is not None else "requires_child_outcomes",
            "analysis_surface": "frozen calibrated probability 0.5; not operational tau",
            "results": summary.get("margin_and_robustness"),
        },
        "FINAL_GEOMETRY_SUMMARY.md": dict(summary),
        "INTEGRITY_REPORT.md": {
            "native_artifacts_accessed": False,
            "new_calibrator_fit": False,
            "operational_tau_selected": False,
            "phase4_complete": children is not None,
            "unique_checkpoint_model_pairs": int(summary.get("unique_checkpoint_model_pairs", 0)),
            "prompt_generations": int(summary.get("prompt_generations", 0)),
            "parents_selected": int(len(parents)),
        },
        "README.md": {"artifact_root": str(root), "status": summary["status"], "reports": list(REQUIRED_REPORTS), "figures_required_after_phase4": list(REQUIRED_FIGURES)},
    }
    for name, payload in report_payloads.items():
        canonical_prelaunch = root / name
        if name == "PRELAUNCH_CENSUS_AND_INTEGRITY.md" and canonical_prelaunch.is_file():
            shutil.copy2(canonical_prelaunch, reports / name)
        else:
            render_geometry_report(reports / name, {"Results": payload})
        json_path = reports / name.replace(".md", ".json")
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")

    _save_figure(figures / REQUIRED_FIGURES[0], lambda ax: (
        ax.scatter(checkpoints["canonical_raw_logit"], checkpoints["dense_recoverability"], s=8, alpha=.3),
        ax.set(xlabel="Canonical raw logit", ylabel="Dense recoverability")
    ))
    h1_rows = []
    for model, value in summary["models"].items():
        for predictor, metric in value["H1"]["models"].items():
            h1_rows.append({"model": model, "predictor": predictor, "nll": metric["trace_weighted_nll"]})
    h1 = pd.DataFrame(h1_rows)
    h1.to_csv(root / "tables/predictor_comparison.csv", index=False)
    _save_figure(figures / REQUIRED_FIGURES[1], lambda ax: (
        h1.groupby("predictor")["nll"].mean().sort_values().plot.bar(ax=ax),
        ax.set(ylabel="Trace-weighted dense NLL", xlabel="Predictor")
    ))
    centered = checkpoints.copy()
    centered["score_centered"] = centered["canonical_raw_logit"] - centered.groupby(["base_model", "trace_id"])["canonical_raw_logit"].transform("mean")
    centered["outcome_centered"] = centered["dense_recoverability"] - centered.groupby(["base_model", "trace_id"])["dense_recoverability"].transform("mean")
    _save_figure(figures / REQUIRED_FIGURES[2], lambda ax: (
        ax.scatter(centered["score_centered"], centered["outcome_centered"], s=8, alpha=.3),
        ax.set(xlabel="Within-trace centered score", ylabel="Within-trace centered recoverability")
    ))
    _save_figure(figures / REQUIRED_FIGURES[3], lambda ax: (
        trajectories["category"].value_counts().plot.bar(ax=ax),
        ax.set(xlabel="Pre-registered category", ylabel="Traces")
    ))
    # Deterministic multivariate category medoids; never hand-pick attractive
    # curves after inspecting outcomes.
    from .analysis import select_representative_trajectory_medoid
    medoid_input = trajectories.loc[
        trajectories["formal_classification_eligible"].astype(bool)
    ].rename(columns={"category": "trajectory_category"})
    representatives = select_representative_trajectory_medoid(medoid_input)
    representatives.to_csv(root / "tables/representative_trajectory_medoids.csv", index=False)
    representative_ids = {
        (row["base_model"], row["trace_id"]): row["trajectory_category"]
        for _, row in representatives.iterrows()
    }
    def draw_representatives(ax: Any) -> None:
        for identity, label in representative_ids.items():
            part = checkpoints.loc[(checkpoints["base_model"] == identity[0]) & (checkpoints["trace_id"] == identity[1])].sort_values("checkpoint_ordinal")
            ax.plot(part["checkpoint_ordinal"], part["dense_recoverability"], marker="o", label=label)
        ax.set(xlabel="Checkpoint", ylabel="Dense recoverability"); ax.legend(fontsize=7)
    _save_figure(figures / REQUIRED_FIGURES[4], draw_representatives)
    collapse = trajectories.loc[trajectories["category"] == "single_collapse"]
    collapse.to_csv(root / "tables/collapse_vs_first_error.csv", index=False)
    _save_figure(figures / REQUIRED_FIGURES[5], lambda ax: (
        ax.scatter(collapse["first_error_checkpoint"], collapse["collapse_checkpoint"], s=12, alpha=.5),
        ax.set(xlabel="First annotated error checkpoint", ylabel="Dense collapse checkpoint")
    ))
    if not transfers.empty:
        matrix = transfers.pivot_table(index="train_domain", columns="test_domain", values="dense_nll", aggfunc="mean")
        def draw_transfer(ax: Any) -> None:
            image = ax.imshow(matrix.to_numpy(float), aspect="auto")
            ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(matrix.index)), matrix.index)
            ax.figure.colorbar(image, ax=ax); ax.set(xlabel="Test domain", ylabel="Train domain")
        _save_figure(figures / REQUIRED_FIGURES[8], draw_transfer)
    stability = pd.DataFrame([
        {"model": model, "logit_correlation": value["stability"]["median_logit_correlation"], "direction_cosine": value["stability"]["median_direction_cosine"]}
        for model, value in summary["models"].items()
    ])
    stability.to_csv(root / "tables/seed_stability.csv", index=False)
    _save_figure(figures / REQUIRED_FIGURES[9], lambda ax: (
        stability.set_index("model")[["logit_correlation", "direction_cosine"]].plot.bar(ax=ax),
        ax.set(ylabel="Similarity", ylim=(0, 1.05))
    ))
    if children is not None and not children.empty:
        available_children = children.loc[
            children.get("horizon_available", pd.Series(True, index=children.index)).astype(bool)
        ].copy()
        available_children["outcome_group"] = np.where(
            available_children["child_recoverability"] >= 0.75,
            "R>=.75",
            np.where(available_children["child_recoverability"] <= 0.25, "R<=.25", "middle"),
        )
        def draw_children(ax: Any) -> None:
            horizons = sorted(available_children["horizon"].unique())
            offsets = {"R<=.25": -0.18, "middle": 0.0, "R>=.75": 0.18}
            colors = {"R<=.25": "#d55e00", "middle": "#999999", "R>=.75": "#0072b2"}
            for label in ("R<=.25", "middle", "R>=.75"):
                subset = available_children.loc[available_children["outcome_group"] == label]
                for index, horizon in enumerate(horizons):
                    values = subset.loc[subset["horizon"] == horizon, "child_score"].dropna()
                    if len(values):
                        ax.scatter(
                            np.full(len(values), index + offsets[label]), values,
                            s=8, alpha=.28, color=colors[label], label=label if index == 0 else None,
                        )
            ax.set_xticks(range(len(horizons)), [str(value) for value in horizons])
            ax.set(xlabel="Child horizon", ylabel="Frozen-axis score")
            handles, labels = ax.get_legend_handles_labels()
            if handles: ax.legend(handles, labels, fontsize=8)
        _save_figure(figures / REQUIRED_FIGURES[6], draw_children)
        parent_discovery = children.groupby("parent_id").agg(parent_recoverability=("parent_recoverability", "first"), any_high=("child_recoverability", lambda value: (value >= .75).any())).reset_index()
        _save_figure(figures / REQUIRED_FIGURES[7], lambda ax: (
            ax.scatter(parent_discovery["parent_recoverability"], parent_discovery["any_high"].astype(float), s=15, alpha=.5),
            ax.set(xlabel="Parent recoverability", ylabel="Any child R >= .75")
        ))
    validation = validate_required_outputs(root)
    return {
        "reports": report_payloads,
        "figures_present": sorted(path.name for path in figures.glob("*.png")),
        "required_output_validation": validation,
    }
