#!/usr/bin/env python3
"""Local analysis/validation CLI for monotonic prefix validity."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import typer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safeprefix.prefix_validity_v1.runner import load_prefix_validity_config  # noqa: E402

app = typer.Typer(add_completion=False)


@app.command("validate-config")
def validate_config(config: Path = typer.Option(...)) -> None:
    payload = load_prefix_validity_config(config)
    print(json.dumps({
        "status": "PASS", "experiment": payload["experiment"]["id"],
        "selected_models": payload["selected_models"],
        "recoverability_probe_modification": False,
        "native_application_enabled": False,
        "suffix_rollouts_generated": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
