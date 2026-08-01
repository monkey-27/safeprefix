"""Small command map for the curated SafePrefix repository."""

from __future__ import annotations

import json
from pathlib import Path

import typer


app = typer.Typer(add_completion=False, no_args_is_help=True)


ROOT = Path(__file__).resolve().parents[2]


COMMANDS = {
    "data": [
        "python3 scripts/00_audit_datasets.py --config configs/datasets.yaml",
        "python3 scripts/33_prepare_native_data_compilation.py --config configs/native_data_compilation.yaml",
        "python3 scripts/35_prepare_native_failed_trace_acquisition.py --config configs/native_failed_trace_acquisition.yaml",
    ],
    "teacher_forced": [
        "python3 scripts/25_run_full_teacher_forced_rollouts.py --config configs/full_teacher_forced_suite.yaml --help",
        "python3 scripts/26_reconcile_safeprefix_counts.py --help",
        "python3 scripts/27_complete_teacher_forced_corpora.py --help",
        "python3 scripts/28_freeze_teacher_forced_completion_manifest.py --help",
    ],
    "training": [
        "python3 scripts/run_boundary_model_v1.py prepare --help",
        "python3 scripts/run_boundary_model_v1.py train-model --help",
        "python3 scripts/run_boundary_model_v1.py select --help",
        "python3 scripts/run_boundary_model_v1.py finalize --help",
        "python3 scripts/run_boundary_model_v1.py report --help",
    ],
    "evaluation": [
        "python3 scripts/run_safeprefix_prefix_validity_v1.py --config configs/prefix_validity_v1.yaml",
        "python3 scripts/run_safeprefix_threshold_selection_tf_v1.py --help",
        "python3 scripts/run_recoverability_geometry_tf.py --help",
        "python3 scripts/run_k_densification_v1.py --help",
    ],
}


@app.command()
def commands() -> None:
    """Print the maintained local command map."""

    typer.echo(json.dumps(COMMANDS, indent=2))


@app.command()
def doctor() -> None:
    """Check that the curated repository has the expected top-level folders."""

    required = ["src/safeprefix", "scripts", "configs", "docs", "tests"]
    missing = [path for path in required if not (ROOT / path).exists()]
    status = "PASS" if not missing else "FAIL"
    typer.echo(json.dumps({"status": status, "missing": missing}, indent=2))
    if missing:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
