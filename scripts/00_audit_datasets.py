#!/usr/bin/env python3
"""Normalize, split, deduplicate, tokenize, and audit ProcessBench and CRV."""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import typer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safeprefix.config import dump_resolved, load_config, parse_overrides  # noqa: E402
from safeprefix.data.audit import audit_rows, render_report  # noqa: E402
from safeprefix.data.common import load_hf_rows, read_local_rows, source_trace_identity  # noqa: E402
from safeprefix.data.crv import materialize_annotated_files, normalize_rows as normalize_crv  # noqa: E402
from safeprefix.data.dedup import latex_whitespace_hash  # noqa: E402
from safeprefix.data.processbench import normalize_rows as normalize_processbench  # noqa: E402
from safeprefix.data.splits import assign_problem_splits, assert_problem_disjoint, problem_group_key  # noqa: E402
from safeprefix.manifests import NormalizedTrace, encode_reference_answer  # noqa: E402
from safeprefix.models.loader import load_model  # noqa: E402
from safeprefix.reproducibility import atomic_json, atomic_jsonl, atomic_parquet, atomic_text, provenance, refuse_overwrite, stage_manifest  # noqa: E402
from safeprefix.testing import CharacterTokenizer, mock_traces  # noqa: E402

app = typer.Typer(add_completion=False)


def _evaluation_hashes(config: dict[str, Any], exclusions: list[dict[str, Any]]) -> set[str]:
    hashes: set[str] = set()
    for value in config.get("evaluation_pool_manifests", []):
        path = ROOT / value
        if not path.exists():
            exclusions.append({"source_dataset": "evaluation_pool", "source_subset": str(value), "row_index": None, "reason": "configured evaluation manifest is missing"})
            continue
        if path.suffix == ".parquet": rows = pd.read_parquet(path).to_dict("records")
        else: rows = read_local_rows(path)
        for row in rows:
            text = row.get("problem_text") or row.get("problem") or row.get("question") or row.get("prompt")
            if text: hashes.add(latex_whitespace_hash(str(text)))
    return hashes


def _load_rows(config: dict[str, Any], mock: bool) -> tuple[list[NormalizedTrace], list[dict[str, Any]], dict[str, str | None]]:
    if mock:
        return [NormalizedTrace(**row) for row in mock_traces()], [], {"mock/safeprefix": "mock-v1"}
    accepted: list[NormalizedTrace] = []
    excluded: list[dict[str, Any]] = []
    revisions: dict[str, str | None] = {}
    process = config["datasets"]["processbench"]
    for subset in process["subsets"]:
        try:
            raw, fingerprint = load_hf_rows(process["hf_dataset_id"], None, subset, process.get("revision"))
            rows, rejected = normalize_processbench(raw, subset)
            accepted.extend(rows)
            excluded.extend(rejected)
            revisions[f"{process['hf_dataset_id']}:{subset}"] = fingerprint or process.get("revision")
        except Exception as exc:
            excluded.append({"source_dataset": process["hf_dataset_id"], "source_subset": subset, "row_index": None, "reason": f"dataset_load_failed: {type(exc).__name__}: {exc}"})
    crv = config["datasets"]["crv"]
    matched = []
    for pattern in crv.get("local_globs", []):
        matched.extend(glob.glob(str(ROOT / pattern)))
    if not matched and crv.get("download_if_missing", False):
        try:
            matched, resolved_crv = materialize_annotated_files(crv["hf_dataset_id"], revision=crv.get("revision"), local_root=str(ROOT / crv.get("local_root", "data/crv")))
            revisions[crv["hf_dataset_id"]] = resolved_crv
        except Exception as exc:
            excluded.append({"source_dataset": crv["hf_dataset_id"], "source_subset": "all", "row_index": None, "reason": f"annotated_download_failed: {type(exc).__name__}: {exc}"})
    if not matched:
        excluded.append({"source_dataset": crv["hf_dataset_id"], "source_subset": "all", "row_index": None, "reason": "no configured local annotated files; CRV is a file-backed Hub repository"})
    for raw_path in sorted(set(matched)):
        path = Path(raw_path)
        rows, rejected = normalize_crv(read_local_rows(path), path.stem)
        accepted.extend(rows)
        excluded.extend(rejected)
        revisions[f"{crv['hf_dataset_id']}:{path.name}"] = revisions.get(crv["hf_dataset_id"], crv.get("revision"))
    if not accepted:
        raise RuntimeError("no dataset rows were normalized; inspect the exclusion manifest")
    return accepted, excluded, revisions


def _token_lengths(
    config: dict[str, Any],
    rows: list[NormalizedTrace],
    mock: bool,
    tokenizer_models: list[str] | None = None,
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]], dict[str, dict[str, float]], dict[str, str | None]]:
    result: dict[str, dict[str, int]] = {}
    positions: dict[str, dict[str, int]] = {}
    fractions: dict[str, dict[str, float]] = {}
    revisions: dict[str, str | None] = {}
    def measure(name: str, tokenizer: Any) -> None:
        result[name] = {}; positions[name] = {}; fractions[name] = {}
        for row in rows:
            trace_id = source_trace_identity(row)
            prefix = row.problem_text + "\n\n"
            completion = "\n\n".join(row.reasoning_steps)
            whole = prefix + completion
            total = len(tokenizer(whole, add_special_tokens=False)["input_ids"])
            result[name][trace_id] = total
            error = row.first_error_zero_based
            if error is not None:
                before = prefix + "\n\n".join(row.reasoning_steps[:error])
                if error:
                    before += "\n\n"
                position = len(tokenizer(before, add_special_tokens=False)["input_ids"])
                positions[name][trace_id] = position
                fractions[name][trace_id] = position / max(total, 1)
    if mock:
        tokenizer = CharacterTokenizer()
        measure("mock_character", tokenizer)
        revisions["mock_character"] = "mock-v1"
        return result, positions, fractions, revisions
    from transformers import AutoTokenizer

    for name in tokenizer_models or config["selected_models"]:
        entry = config["models"][name]
        tokenizer = AutoTokenizer.from_pretrained(entry["tokenizer_id"], revision=entry.get("tokenizer_revision"), trust_remote_code=entry.get("trust_remote_code", False), use_fast=True)
        measure(name, tokenizer)
        revisions[name] = tokenizer.init_kwargs.get("_commit_hash") or entry.get("tokenizer_revision")
    return result, positions, fractions, revisions


@app.command()
def main(
    config: Path = typer.Option(ROOT / "configs" / "prompt_pilot.yaml"),
    override: list[str] = typer.Option([], "--set"),
    mock: bool = typer.Option(False),
    resume: bool = typer.Option(True),
    overwrite: bool = typer.Option(False),
    tokenizer_model: list[str] = typer.Option(
        [],
        "--tokenizer-model",
        help="Restrict audit token-length statistics without changing the configured experiment model matrix.",
    ),
) -> None:
    resolved = load_config(config, parse_overrides(override))
    cfg = resolved.data
    out = ROOT / cfg["artifacts_root"] / "audit"
    target = out / "summary.json"
    if refuse_overwrite(target, resume=resume, overwrite=overwrite):
        typer.echo(f"skip valid artifact: {target}")
        return
    with stage_manifest(out, "00_audit_datasets", cfg, " ".join(sys.argv)):
        rows, exclusions, dataset_revisions = _load_rows(cfg, mock)
        assignments = assign_problem_splits([row.to_dict() for row in rows], cfg["split_fractions"], cfg["seed"])
        mock_splits = (["prompt_pilot"] * 2 + ["train"] * 8 + ["dev"] * 4 + ["dense_audit"] * 4 + ["post_error_audit"] * 2 + ["native_eval"] * 4) if mock and len(rows) == 24 else None
        serialized = []
        for index, row in enumerate(rows):
            value = row.to_dict()
            # The source datasets legitimately mix numeric, Boolean, and text
            # answers. A tagged JSON string preserves type in one Arrow column.
            value["reference_answer"] = encode_reference_answer(value.get("reference_answer"))
            value["source_trace_id"] = source_trace_identity(row)
            value["problem_group_hash"] = problem_group_key(value)
            value["split"] = mock_splits[index] if mock_splits else assignments[value["problem_group_hash"]]
            serialized.append(value)
        assert_problem_disjoint(serialized)
        unknown_tokenizers = sorted(set(tokenizer_model) - set(cfg["selected_models"]))
        if unknown_tokenizers:
            raise ValueError(f"unknown --tokenizer-model entries: {unknown_tokenizers}")
        lengths, error_positions, error_fractions, tokenizer_revisions = _token_lengths(
            cfg,
            rows,
            mock,
            tokenizer_models=tokenizer_model or None,
        )
        summary = audit_rows(
            rows,
            lengths,
            error_positions,
            error_fractions,
            _evaluation_hashes(cfg, exclusions),
            fuzzy_similarity_threshold=float(cfg["fuzzy_similarity_threshold"]),
        )
        summary.update({"dataset_revisions": dataset_revisions, "tokenizer_revisions": tokenizer_revisions, "mock": mock, "mock_fixed_split_fixture": bool(mock_splits), "split_counts": dict(pd.Series([row["split"] for row in serialized]).value_counts())})
        atomic_parquet(out / "examples.parquet", pd.DataFrame(serialized))
        atomic_json(out / "summary.json", summary)
        atomic_jsonl(out / "exclusions.jsonl", exclusions)
        atomic_text(out / "report.md", render_report(summary, exclusions))
        atomic_text(out / "resolved_config.yaml", dump_resolved(cfg))
        atomic_json(out / "provenance.json", provenance(cfg, " ".join(sys.argv)))
    typer.echo(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    app()
