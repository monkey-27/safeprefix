#!/usr/bin/env python3
"""Freeze reconciled membership and group-level split decisions without models."""

from pathlib import Path
import sys

import typer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safeprefix.config import load_config  # noqa: E402
from safeprefix.teacher_forced_completion import freeze_membership_artifacts  # noqa: E402


def main(
    config: Path = typer.Option(ROOT / "configs/teacher_forced_completion.yaml"),
    frozen_source_root: Path = typer.Option(Path("/private/tmp/safeprefix_frozen_manifests")),
    completed_root: Path = typer.Option(
        Path("/private/tmp/safeprefix_final_download/safeprefix_full_teacher_forced_20260726_r3/artifacts/full_teacher_forced_suite")
    ),
    output_root: Path = typer.Option(ROOT / "artifacts/teacher_forced_completion_protocol"),
) -> None:
    resolved = load_config(config).data
    result = freeze_membership_artifacts(
        resolved,
        output_root=output_root,
        frozen_source_root=frozen_source_root,
        completed_root=completed_root,
    )
    typer.echo(result)


if __name__ == "__main__":
    typer.run(main)
