#!/usr/bin/env python3
"""Prepare, validate, execute, and aggregate teacher-forced corpus completion."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch
import typer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safeprefix.config import load_config  # noqa: E402
from safeprefix.full_teacher_forced import build_execution_packs  # noqa: E402
from safeprefix.models.loader import load_model  # noqa: E402
from safeprefix.production_suite import execute_pack, read_jsonl  # noqa: E402
from safeprefix.reproducibility import atomic_json, stable_hash  # noqa: E402
from safeprefix.teacher_forced_completion import (  # noqa: E402
    _old_root,
    _safety_pack_root,
    aggregate_completion,
    assert_completion_scope,
    completion_root,
    execute_extension_pack,
    execute_safety_pack,
    feature_reuse_equivalence,
    load_completion_context,
    prepare_completion,
    sha256_file,
    valid_safety_pack,
    validate_prepared_completion,
)


app = typer.Typer(add_completion=False, no_args_is_help=True)


def _load(path: Path) -> dict[str, Any]:
    config = load_config(path).data
    assert_completion_scope(config)
    return config


@app.command("prepare")
def prepare(config: Path = typer.Option(...), run_id: str = typer.Option(...)) -> None:
    cfg = _load(config)
    root = completion_root(cfg, run_id)
    immutable = root / "immutable_manifests/immutable_protocol_manifest.json"
    if immutable.exists():
        validate_prepared_completion(cfg, run_id=run_id)
        typer.echo(immutable.read_text())
        return
    result = prepare_completion(cfg, run_id=run_id, config_path=config)
    validate_prepared_completion(cfg, run_id=run_id)
    typer.echo(json.dumps(result, indent=2))


@app.command("cpu-smoke")
def cpu_smoke(config: Path = typer.Option(...), run_id: str = typer.Option(...)) -> None:
    cfg = _load(config)
    root = completion_root(cfg, run_id)
    validation = validate_prepared_completion(cfg, run_id=run_id)
    protocol = json.loads(
        (root / "immutable_manifests/immutable_protocol_manifest.json").read_text()
    )
    access = json.loads((root / "source_access_ledger.json").read_text())
    reuse = json.loads((root / "reuse_ledger/completed_942.json").read_text())
    checks = {
        "prepared": validation["passed"],
        "repairability_2614": protocol["repairability"]["traces"] == 2614,
        "repairability_10988": protocol["repairability"]["checkpoints_per_model"] == 10988,
        "extension_1672": protocol["repairability"]["extension_traces"] == 1672,
        "extension_7093": protocol["repairability"]["extension_checkpoints_per_model"] == 7093,
        "safety_5900": protocol["safety"]["traces"] == 5900,
        "reuse_valid": reuse["status"] == "PASS",
        "no_protected_content": access["protected_group_content_opened"] is False,
        "no_native_or_final": not access["native_development_outputs_opened"]
        and not access["final_test_outputs_opened"],
        "training_disabled": protocol["scope"]["boundary_training"] is False,
        "unsafe_derivation_disabled": protocol["scope"]["unsafe_label_derivation"] is False,
    }
    for model_key in cfg["selected_models"]:
        context = load_completion_context(cfg, run_id=run_id, model_key=model_key)
        logical = [
            (
                key["trace_id"],
                int(key["checkpoint_index"]),
                int(key["rollout_index"]),
                int(key["rollout_seed"]),
            )
            for pack in context["repair_packs"].values()
            for key in pack["logical_rollout_keys"]
        ]
        checks[f"{model_key}:extension_traces"] = len(context["repair_rows"]) == 1672
        checks[f"{model_key}:extension_rollouts"] = len(logical) == 28372
        checks[f"{model_key}:logical_unique"] = len(logical) == len(set(logical))
        checks[f"{model_key}:safety_traces"] = len(context["safety_rows"]) == 5900
        checks[f"{model_key}:safety_no_rollout_keys"] = all(
            "logical_rollout_keys" not in pack for pack in context["safety_packs"].values()
        )
    if not all(checks.values()):
        raise AssertionError({key: value for key, value in checks.items() if not value})
    result = {"status": "PASS", "checks": checks, "models_loaded": 0}
    atomic_json(root / "smoke_tests/cpu_non_model.json", result)
    typer.echo(json.dumps(result, indent=2))


def _old_feature_for_source(cfg: dict[str, Any], model_key: str, source_trace_id: str) -> dict[str, Any]:
    root = _old_root(cfg)
    trace_rows = read_jsonl(
        root / "immutable_manifests/per_model" / model_key / "trace_manifest.jsonl"
    )
    old_trace = next(row for row in trace_rows if row["source_trace_id"] == source_trace_id)
    packs = read_jsonl(
        root / "immutable_manifests/per_model" / model_key / "execution_packs.jsonl"
    )
    pack = next(pack for pack in packs if old_trace["trace_id"] in pack["trace_ids"])
    payload = torch.load(
        root / "raw_rollout_shards" / model_key / pack["pack_id"] / "checkpoint_features.pt",
        map_location="cpu",
        weights_only=False,
    )
    return {"trace": old_trace, "features": payload[old_trace["trace_id"]]}


@app.command("gpu-smoke")
def gpu_smoke(
    config: Path = typer.Option(...),
    run_id: str = typer.Option(...),
    model_key: str = typer.Option(...),
) -> None:
    cfg = _load(config)
    context = load_completion_context(cfg, run_id=run_id, model_key=model_key)
    loaded = load_model(cfg["models"][model_key])
    root = completion_root(cfg, run_id)
    settings = context["settings"]

    extension_rows = sorted(context["repair_rows"].values(), key=lambda row: row["trace_id"])[:2]
    smoke_pack = build_execution_packs(
        extension_rows,
        model_key=model_key,
        base_seed=int(cfg["rollout"]["base_seed"]),
        configuration_hash=stable_hash(cfg),
        engine_revision="completion-real-model-smoke-v1",
        traces_per_pack=2,
    )[0]
    smoke_generation = {**context["generation"], "max_new_tokens": 128}
    smoke_root = root / "smoke_tests/real_model" / model_key / "repairability"
    first = execute_pack(
        config=cfg,
        model_key=model_key,
        loaded=loaded,
        pack=smoke_pack,
        trace_rows={row["trace_id"]: row for row in extension_rows},
        pack_root=smoke_root,
        generation=smoke_generation,
        batch_size=4,
        compaction_quantum=8,
        prefill_chunk_size=int(settings["prefill_chunk_size"]),
        traces_per_decode_group=2,
        maximum_decode_kv_bytes=int(settings["maximum_decode_kv_bytes"]),
        mode="smoke",
        execution_freeze_digest="completion-real-model-smoke-v1",
        model_execution_settings_hash="completion-real-model-smoke-v1",
    )
    second = execute_pack(
        config=cfg,
        model_key=model_key,
        loaded=loaded,
        pack=smoke_pack,
        trace_rows={row["trace_id"]: row for row in extension_rows},
        pack_root=smoke_root,
        generation=smoke_generation,
        batch_size=4,
        compaction_quantum=8,
        prefill_chunk_size=int(settings["prefill_chunk_size"]),
        traces_per_decode_group=2,
        maximum_decode_kv_bytes=int(settings["maximum_decode_kv_bytes"]),
        mode="smoke",
        execution_freeze_digest="completion-real-model-smoke-v1",
        model_execution_settings_hash="completion-real-model-smoke-v1",
    )
    if second.get("status") != "SKIPPED_VALID":
        raise RuntimeError("repairability smoke did not exercise exact pack resume")

    old_manifest = read_jsonl(
        _old_root(cfg) / "immutable_manifests/per_model" / model_key / "trace_manifest.jsonl"
    )
    source_trace_id = str(old_manifest[0]["source_trace_id"])
    failed_safety = next(
        row for row in context["safety_rows"].values() if row["source_trace_id"] == source_trace_id
    )
    correct_safety = next(
        row for row in context["safety_rows"].values() if row["terminal_correct"]
    )
    safety_pack = {
        "pack_id": f"{model_key}-safety-smoke",
        "pack_hash": stable_hash(["safety-smoke-v1", model_key, failed_safety["trace_id"], correct_safety["trace_id"]]),
        "trace_ids": [failed_safety["trace_id"], correct_safety["trace_id"]],
        "trace_count": 2,
        "checkpoint_count": len(failed_safety["checkpoint_offsets"]) + len(correct_safety["checkpoint_offsets"]),
    }
    safety_smoke_root = root / "smoke_tests/real_model" / model_key / "safety"
    smoke_context = {
        **context,
        "root": safety_smoke_root,
        "safety_packs": {safety_pack["pack_id"]: safety_pack},
    }
    safety_first = execute_safety_pack(
        cfg,
        run_id=run_id,
        model_key=model_key,
        pack_id=safety_pack["pack_id"],
        loaded=loaded,
        context=smoke_context,
    )
    safety_second = execute_safety_pack(
        cfg,
        run_id=run_id,
        model_key=model_key,
        pack_id=safety_pack["pack_id"],
        loaded=loaded,
        context=smoke_context,
    )
    if safety_second.get("status") != "SKIPPED_VALID":
        raise RuntimeError("safety smoke did not exercise exact pack resume")
    payload = torch.load(
        _safety_pack_root(safety_smoke_root, model_key, safety_pack["pack_id"])
        / "features.pt",
        map_location="cpu",
        weights_only=False,
    )[failed_safety["trace_id"]]
    old = _old_feature_for_source(cfg, model_key, source_trace_id)
    eligible_count = len(old["trace"]["eligible_checkpoint_offsets"])
    new_prefix = payload["features"][:eligible_count]
    old_features = old["features"]["features"]
    equivalence = feature_reuse_equivalence(new_prefix, old_features)
    if not equivalence["shape_equal"]:
        raise RuntimeError(
            f"safety feature representation shape differs: {equivalence}"
        )
    result = {
        "status": "PASS",
        "model_key": model_key,
        "repairability_first": first,
        "repairability_resume": second["status"],
        "safety_first": safety_first,
        "safety_resume": safety_second["status"],
        "existing_pre_error_feature_equivalence": equivalence,
        "existing_pre_error_features_reused": bool(equivalence["reuse_permitted"]),
        "four_real_rollouts_per_checkpoint": True,
        "boundary_training_occurred": False,
        "native_or_final_access": False,
    }
    atomic_json(root / "smoke_tests/real_model" / model_key / "summary.json", result)
    typer.echo(json.dumps(result, indent=2))


@app.command("run-repair-pack")
def run_repair_pack(
    config: Path = typer.Option(...), run_id: str = typer.Option(...),
    model_key: str = typer.Option(...), pack_id: str = typer.Option(...),
) -> None:
    cfg = _load(config)
    loaded = load_model(cfg["models"][model_key])
    result = execute_extension_pack(cfg, run_id=run_id, model_key=model_key, pack_id=pack_id, loaded=loaded)
    typer.echo(json.dumps(result, indent=2))


@app.command("run-safety-pack")
def run_safety_pack(
    config: Path = typer.Option(...), run_id: str = typer.Option(...),
    model_key: str = typer.Option(...), pack_id: str = typer.Option(...),
) -> None:
    cfg = _load(config)
    loaded = load_model(cfg["models"][model_key])
    result = execute_safety_pack(cfg, run_id=run_id, model_key=model_key, pack_id=pack_id, loaded=loaded)
    typer.echo(json.dumps(result, indent=2))


@app.command("aggregate")
def aggregate(config: Path = typer.Option(...), run_id: str = typer.Option(...)) -> None:
    cfg = _load(config)
    result = aggregate_completion(cfg, run_id=run_id)
    typer.echo(json.dumps(result, indent=2))


if __name__ == "__main__":
    app()
