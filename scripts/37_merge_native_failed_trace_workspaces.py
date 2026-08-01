#!/usr/bin/env python3
"""Merge four downloaded per-model summaries without recomputing inference."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safeprefix.native_failed_trace_pipeline import merge_workspace_summaries  # noqa: E402

app = typer.Typer(add_completion=False)


@app.command()
def main(
    model_summary: list[Path] = typer.Option(..., "--model-summary"),
    output_dir: Path = typer.Option(..., "--output-dir"),
) -> None:
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in model_summary]
    result = merge_workspace_summaries(summaries, output_dir=output_dir)
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
