#!/usr/bin/env python3
"""Entry point for the evaluation-only K-densification v1 protocol."""

from __future__ import annotations

from pathlib import Path

import typer

from safeprefix.config import dump_resolved, load_config
from safeprefix.k_densification_v1.analysis import run_analysis
from safeprefix.k_densification_v1.calibration import prepare_calibration_experiment
from safeprefix.k_densification_v1.preflight import CohortInsufficientError, run_preflight
from safeprefix.k_densification_v1.reporting import finalize_k_densification_reporting
from safeprefix.k_densification_v1.registry import prepare_registry_and_reuse_report
from safeprefix.reproducibility import atomic_text


app = typer.Typer(no_args_is_help=True)


@app.command("finalize-report")
def finalize_report(
    config: Path = typer.Option(Path("configs/k_densification_v1.yaml")),
    threshold_root: Path = typer.Option(..., exists=True, file_okay=False),
    output_root: Path | None = typer.Option(None),
    generated_outcomes: Path | None = typer.Option(None, exists=True, dir_okay=False),
    run_part_b: bool = typer.Option(True, "--run-part-b/--use-existing-part-b"),
) -> None:
    """Validate the nested K32 pool and publish terminal calibration reports."""

    resolved = load_config(config)
    destination = output_root or Path(resolved.data["artifacts_root"])
    result = finalize_k_densification_reporting(
        config=resolved.data,
        threshold_root=threshold_root,
        output_root=destination,
        generated_outcomes_path=generated_outcomes,
        run_part_b=run_part_b,
    )
    typer.echo(result)


@app.command("prepare-registry")
def prepare_registry(
    threshold_root: Path = typer.Option(..., exists=True, file_okay=False),
    local_search_root: Path = typer.Option(..., exists=True, file_okay=False),
    marketing_volume: list[str] = typer.Option([], "--marketing-volume"),
    output_root: Path = typer.Option(Path("outputs/k_densification_v1")),
) -> None:
    """Register reused K16 slots and all missing four-slot K32 jobs."""

    result = prepare_registry_and_reuse_report(
        output_root=output_root,
        threshold_root=threshold_root,
        local_search_root=local_search_root,
        marketing_volume_inventory=[{"name": name} for name in marketing_volume],
    )
    typer.echo(
        {
            "status": result["status"],
            "reused_slots_0_15": result["reused_slots_0_15"],
            "reused_slots_16_31": result["reused_slots_16_31"],
            "projected_new_rollouts": result["projected_new_rollouts"],
        }
    )


@app.command("analyze-k16")
def analyze_k16(
    config: Path = typer.Option(Path("configs/k_densification_v1.yaml")),
    threshold_root: Path = typer.Option(..., exists=True, file_okay=False),
    output_root: Path | None = typer.Option(None),
) -> None:
    """Run the no-generation full calibration K<=16 analysis."""

    resolved = load_config(config)
    destination = output_root or Path(resolved.data["artifacts_root"])
    result = run_analysis(
        config=resolved.data,
        threshold_root=threshold_root,
        output_root=destination,
        analysis_name="full_calibration_k16",
        checkpoint_path=destination / "calibration_k16_checkpoints.parquet",
        outcomes_path=None,
        ks=(1, 2, 4, 8, 16),
        dense_k=16,
        required_metrics_filename="k16_full_corpus_metrics.csv",
    )
    atomic_text(
        destination / "internal/full_calibration_k16_complete.json",
        __import__("json").dumps(result, indent=2, sort_keys=True) + "\n",
    )
    typer.echo(result)


@app.command("prepare-calibration")
def prepare_calibration(
    config: Path = typer.Option(Path("configs/k_densification_v1.yaml")),
    boundary_root: Path = typer.Option(..., exists=True, file_okay=False),
    threshold_root: Path = typer.Option(..., exists=True, file_okay=False),
    output_root: Path | None = typer.Option(None),
) -> None:
    """Freeze the calibration K16 inventory and outcome-blind K32 subset."""

    resolved = load_config(config)
    destination = output_root or Path(resolved.data["artifacts_root"])
    destination.mkdir(parents=True, exist_ok=True)
    atomic_text(
        destination / "resolved_config_calibration_v2.yaml",
        dump_resolved(resolved.data),
    )
    result = prepare_calibration_experiment(
        config=resolved.data,
        boundary_root=boundary_root,
        threshold_root=threshold_root,
        output_root=destination,
    )
    typer.echo(result)


@app.command("preflight")
def preflight(
    config: Path = typer.Option(Path("configs/k_densification_v1.yaml")),
    boundary_root: Path = typer.Option(..., exists=True, file_okay=False),
    canonical_manifest: Path = typer.Option(..., exists=True, dir_okay=False),
    completion_manifest_root: Path = typer.Option(..., exists=True, file_okay=False),
    output_root: Path | None = typer.Option(None),
) -> None:
    """Validate frozen inputs and freeze or reject the 48-problem cohort."""

    resolved = load_config(config)
    destination = output_root or Path(resolved.data["artifacts_root"])
    destination.mkdir(parents=True, exist_ok=True)
    atomic_text(destination / "resolved_config.yaml", dump_resolved(resolved.data))
    try:
        result = run_preflight(
            config=resolved.data,
            boundary_root=boundary_root,
            canonical_manifest_path=canonical_manifest,
            completion_manifest_root=completion_manifest_root,
            output_root=destination,
        )
    except CohortInsufficientError as exc:
        typer.echo(exc.summary)
        raise typer.Exit(code=2) from exc
    typer.echo(result)


if __name__ == "__main__":
    app()
