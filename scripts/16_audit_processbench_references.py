#!/usr/bin/env python3
"""Audit exact gold-answer recovery for ProcessBench source problems."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import typer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safeprefix.config import load_config, parse_overrides  # noqa: E402
from safeprefix.data.reference_join import build_reference_index, join_processbench_references, source_required_columns  # noqa: E402
from safeprefix.reproducibility import atomic_json, atomic_jsonl, atomic_parquet, stage_manifest  # noqa: E402

app = typer.Typer(add_completion=False)


def _load_source_rows(source_key: str, entry: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset
    from huggingface_hub import HfApi

    repo_id = str(entry["hf_dataset_id"])
    requested_revision = str(entry.get("revision", "main"))
    resolved_revision = str(HfApi().dataset_info(repo_id=repo_id, revision=requested_revision).sha)
    rows: list[dict[str, Any]] = []
    configurations = get_dataset_config_names(repo_id, revision=resolved_revision)
    for configuration in configurations:
        for split in get_dataset_split_names(repo_id, configuration, revision=resolved_revision):
            dataset = load_dataset(repo_id, configuration, split=split, revision=resolved_revision)
            retained_columns = [column for column in source_required_columns(source_key) if column in dataset.column_names]
            if not retained_columns:
                continue
            # Drop image/audio columns before iteration so datasets never tries
            # to decode irrelevant multimodal fields for this text-only join.
            dataset = dataset.select_columns(retained_columns)
            rows.extend(dict(row) for row in dataset)
    return rows, {
        "source_key": source_key,
        "hf_dataset_id": repo_id,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "configurations": configurations,
        "row_count": len(rows),
    }


@app.command()
def main(
    config: Path = typer.Option(ROOT / "configs" / "configuration_pilot.yaml"),
    override: list[str] = typer.Option([], "--set"),
) -> None:
    resolved = load_config(config, parse_overrides(override)); cfg = resolved.data
    root = ROOT / cfg["artifacts_root"]
    audit_path = root / "audit" / "examples.parquet"
    if not audit_path.exists():
        raise FileNotFoundError("run dataset audit before ProcessBench reference coverage")
    frame = pd.read_parquet(audit_path)
    indices = {}; sources = []; conflicts = {}
    with stage_manifest(root / "reference_join", "16_audit_processbench_references", cfg, " ".join(sys.argv)):
        minimum_coverage = float(cfg["reference_sources"].get("minimum_exact_coverage_per_source", 1.0))
        source_entries = {
            key: value for key, value in cfg["reference_sources"].items()
            if isinstance(value, dict) and "hf_dataset_id" in value
        }
        for source_key, entry in source_entries.items():
            rows, source_report = _load_source_rows(str(source_key), dict(entry))
            index, source_conflicts = build_reference_index(str(source_key), rows)
            indices[str(source_key)] = index
            source_report.update(indexed_problem_count=len(index), conflicting_problem_count=len(source_conflicts))
            sources.append(source_report)
            conflicts[str(source_key)] = source_conflicts
        enriched, coverage = join_processbench_references(frame.to_dict("records"), indices)
        source_rates = {
            key: values["matched"] / values["total"] if values["total"] else 0.0
            for key, values in coverage["by_source"].items()
        }
        coverage_passes = bool(source_rates) and all(rate >= minimum_coverage for rate in source_rates.values())
        payload = {
            "status": (
                "PASS" if coverage["matched"] == coverage["total"]
                else "PASS_WITH_DOCUMENTED_EXCLUSIONS" if coverage_passes
                else "INCOMPLETE"
            ),
            "configuration_hash": resolved.digest,
            "coverage": coverage,
            "minimum_exact_coverage_per_source": minimum_coverage,
            "exact_coverage_rates": source_rates,
            "sources": sources,
            "conflict_counts": {key: len(value) for key, value in conflicts.items()},
        }
        atomic_parquet(root / "reference_join" / "enriched_examples.parquet", pd.DataFrame(enriched))
        atomic_json(root / "reference_join" / "coverage.json", payload)
        atomic_jsonl(root / "reference_join" / "unmatched.jsonl", coverage["unmatched"])
        if not str(payload["status"]).startswith("PASS"):
            raise RuntimeError(
                f"ProcessBench exact reference coverage is incomplete: "
                f"{coverage['matched']}/{coverage['total']} matched"
            )
    typer.echo(json.dumps(payload, indent=2))


if __name__ == "__main__":
    app()
