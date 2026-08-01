#!/usr/bin/env python3
"""Freeze the full GSM1K/MATH-3/MATH-4 source census without model inference."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safeprefix.config import dump_resolved, load_config, parse_overrides  # noqa: E402
from safeprefix.native_failed_trace_acquisition import write_source_census  # noqa: E402
from safeprefix.reproducibility import atomic_text  # noqa: E402

app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Path = typer.Option(ROOT / "configs/native_failed_trace_acquisition.yaml"),
    output_dir: Path | None = typer.Option(None),
    override: list[str] = typer.Option([], "--set"),
) -> None:
    cfg = load_config(config, parse_overrides(override)).data
    target = output_dir or ROOT / str(cfg["artifacts_root"]) / "prelaunch"
    summary = write_source_census(cfg, repo_root=ROOT, output_dir=target)
    atomic_text(target / "resolved_config.yaml", dump_resolved(cfg))
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
