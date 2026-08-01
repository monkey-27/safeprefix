"""Small command map for the curated SafePrefix repository."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess

import typer


app = typer.Typer(add_completion=False, no_args_is_help=True)


ROOT = Path(__file__).resolve().parents[2]


COMMANDS = {
    "navigation": [
        "docs/NAVIGATION.md",
        "docs/WORKFLOWS.md",
        "PILOT_PROTOCOL.md",
        "DATA_SCHEMA.md",
    ],
    "data": [
        "python3 scripts/00_audit_datasets.py --config configs/dataset_audit.yaml",
        "python3 scripts/33_prepare_native_data_compilation.py --config configs/native_data_compilation.yaml",
        "python3 scripts/35_prepare_native_failed_trace_acquisition.py --config configs/native_failed_trace_acquisition.yaml",
    ],
    "teacher_forced": [
        "python3 scripts/25_run_full_teacher_forced_rollouts.py --help",
        "python3 scripts/25_run_full_teacher_forced_rollouts.py prepare --config configs/full_teacher_forced_suite.yaml --run-id <run-id>",
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
    "cloud": [
        "python3 -m pip install -e \".[dev,cloud]\"",
        "MODAL_PROFILE=<profile> python3 -m modal run scripts/modal_safeprefix_full_teacher_forced.py --action submit --run-id <run-id>",
        "MODAL_PROFILE=<profile> python3 -m modal run scripts/modal_safeprefix_boundary_model_v1.py --action launch --run-id <run-id>",
        "MODAL_PROFILE=<profile> python3 -m modal run scripts/modal_safeprefix_k_densification_v1.py --action launch --run-id <run-id>",
    ],
}


@app.command()
def commands() -> None:
    """Print the maintained local command map."""

    typer.echo(json.dumps(COMMANDS, indent=2))


@app.command()
def doctor() -> None:
    """Check the curated repository shape and tracked-file policy."""

    required = [
        "src/safeprefix",
        "scripts",
        "configs",
        "docs/NAVIGATION.md",
        "docs/WORKFLOWS.md",
        "tests",
    ]
    missing = [path for path in required if not (ROOT / path).exists()]
    tracked_policy_violations = _tracked_policy_violations()
    failures = {"missing": missing, "tracked_policy_violations": tracked_policy_violations}
    status = "PASS" if not missing and not tracked_policy_violations else "FAIL"
    typer.echo(json.dumps({"status": status, **failures}, indent=2))
    if status != "PASS":
        raise typer.Exit(1)


def _tracked_policy_violations() -> list[str]:
    try:
        output = subprocess.check_output(
            ["git", "ls-files"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    root_pattern = re.compile(
        r"^(artifacts|outputs|published_artifacts|results|data|weights|checkpoints|models)/"
    )
    extension_pattern = re.compile(
        r"\.(parquet|jsonl|pt|pth|safetensors|bin|ckpt|onnx|npy|npz|h5|hdf5|pkl|"
        r"pickle|tar|tgz|gz|zip|sqlite|db|csv|tsv)$",
        re.IGNORECASE,
    )
    violations = []
    for path in output.splitlines():
        if root_pattern.search(path) or extension_pattern.search(path):
            violations.append(path)
    return violations


if __name__ == "__main__":
    app()
