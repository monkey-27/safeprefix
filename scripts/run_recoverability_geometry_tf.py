#!/usr/bin/env python3
"""CPU orchestration and reporting entry point for teacher-forced geometry."""

from __future__ import annotations

from pathlib import Path
import sys

import typer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safeprefix.config import dump_resolved, load_config
from safeprefix.recoverability_geometry_protocol import prepare_geometry_manifests
from safeprefix.recoverability_geometry_orchestration import (
    build_phase4_child_state_table,
    combine_model_stage_outcomes,
    freeze_local_parent_bridge,
    run_phase2_bridge,
    run_phase4_bridge,
)
from safeprefix.reproducibility import atomic_text


app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Run isolated teacher-forced recoverability-geometry stages."""


@app.command("prepare")
def prepare(
    config: Path = typer.Option(Path("configs/recoverability_geometry_tf_v1.yaml")),
    canonical_manifest: Path = typer.Option(
        Path("artifacts/boundary_model_v1/data/canonical_checkpoint_manifest.parquet")
    ),
    source_manifest: Path = typer.Option(
        Path(
            "artifacts/recoverability_geometry_tf_v1/prelaunch/source/"
            "common_trace_manifest.jsonl"
        )
    ),
    boundary_root: Path = typer.Option(
        Path("artifacts/boundary_model_v1_remote/boundary_model_v1")
    ),
    output_root: Path | None = typer.Option(None),
) -> None:
    resolved = load_config(config)
    destination = output_root or Path(resolved.data["artifacts_root"])
    destination.mkdir(parents=True, exist_ok=True)
    atomic_text(destination / "resolved_config.yaml", dump_resolved(resolved.data))
    summary = prepare_geometry_manifests(
        config=resolved.data,
        canonical_manifest_path=canonical_manifest,
        source_manifest_path=source_manifest,
        boundary_root=boundary_root,
        output_root=destination,
    )
    typer.echo(summary)


@app.command("combine-stage")
def combine_stage(
    stage: str = typer.Argument(..., help="dense_checkpoint_k32 or prompt_solvability_k16"),
    output_root: Path = typer.Option(Path("artifacts/recoverability_geometry_tf_v1")),
) -> None:
    """Combine four completed per-model inference aggregates exactly once."""

    path, summary = combine_model_stage_outcomes(
        output_root=output_root, stage=stage
    )
    typer.echo({"path": str(path), **summary})


@app.command("phase2")
def phase2(
    boundary_root: Path = typer.Option(
        Path("artifacts/boundary_model_v1_remote/boundary_model_v1")
    ),
    output_root: Path = typer.Option(Path("artifacts/recoverability_geometry_tf_v1")),
    published_root: Path | None = typer.Option(None),
) -> None:
    """Run H1--H3 and freeze the exact local-parent JSONL handoff."""

    typer.echo(
        run_phase2_bridge(
            boundary_root=boundary_root,
            output_root=output_root,
            published_root=published_root,
        )
    )


@app.command("freeze-local-parents")
def freeze_local_parents(
    output_root: Path = typer.Option(Path("artifacts/recoverability_geometry_tf_v1")),
) -> None:
    """Regenerate the deterministic Phase-2-to-Phase-3 parent handoff."""

    path, summary = freeze_local_parent_bridge(output_root=output_root)
    typer.echo({"path": str(path), **summary})


@app.command("build-phase4-input")
def build_phase4_input(
    boundary_root: Path = typer.Option(
        Path("artifacts/boundary_model_v1_remote/boundary_model_v1")
    ),
    output_root: Path = typer.Option(Path("artifacts/recoverability_geometry_tf_v1")),
) -> None:
    """Join parent features, child features, and local terminal outcomes."""

    path, summary = build_phase4_child_state_table(
        boundary_root=boundary_root, output_root=output_root
    )
    typer.echo({"path": str(path), **summary})


@app.command("phase4")
def phase4(
    boundary_root: Path = typer.Option(
        Path("artifacts/boundary_model_v1_remote/boundary_model_v1")
    ),
    output_root: Path = typer.Option(Path("artifacts/recoverability_geometry_tf_v1")),
    published_root: Path | None = typer.Option(None),
) -> None:
    """Build local-child inputs and finish H4 plus final reports."""

    typer.echo(
        run_phase4_bridge(
            boundary_root=boundary_root,
            output_root=output_root,
            published_root=published_root,
        )
    )


if __name__ == "__main__":
    app()
