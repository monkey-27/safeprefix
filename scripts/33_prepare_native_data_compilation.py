#!/usr/bin/env python3
"""Freeze and validate native-development compilation inputs without a model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safeprefix.config import dump_resolved, load_config, parse_overrides  # noqa: E402
from safeprefix.native_data_compilation import build_protected_group_manifest, load_frozen_native_manifest, validate_protected_groups, write_cpu_preflight  # noqa: E402
from safeprefix.reproducibility import atomic_text  # noqa: E402

app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Path = typer.Option(ROOT / "configs" / "native_data_compilation.yaml"),
    override: list[str] = typer.Option([], "--set"),
    output_dir: Path | None = typer.Option(None),
    frozen_manifest_root: Path | None = typer.Option(None, help="Build the protected identity manifest from this frozen root."),
) -> None:
    resolved = load_config(config, parse_overrides(override))
    cfg = resolved.data
    target = output_dir or ROOT / cfg["artifacts_root"] / "prelaunch"
    spec = cfg["frozen_native_manifest"]
    if frozen_manifest_root is not None:
        protected_path = target / "protected_problem_groups.json"
        build_protected_group_manifest(frozen_manifest_root, protected_path)
        spec = {**spec, "protected_identity_file": str(protected_path)}
    rows = load_frozen_native_manifest(spec["file"], spec)
    protected = validate_protected_groups(rows, spec["protected_identity_file"])
    summary = write_cpu_preflight(rows=rows, config=cfg, output_dir=target, protected_report=protected)
    atomic_text(target / "resolved_config.yaml", dump_resolved(cfg))
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
