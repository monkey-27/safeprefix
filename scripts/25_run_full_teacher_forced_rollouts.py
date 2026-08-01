#!/usr/bin/env python3
"""Prepare, execute, resume, and aggregate the rollout-only production suite."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd
import torch
import typer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safeprefix.config import load_config  # noqa: E402
from safeprefix.full_teacher_forced import (  # noqa: E402
    aggregate_checkpoint_outcomes,
    logical_rollout_keys,
    validate_pack_rows,
)
from safeprefix.models.loader import load_model  # noqa: E402
from safeprefix.production_suite import (  # noqa: E402
    aggregate_production,
    assert_rollout_only,
    engine_revision,
    execute_pack,
    execute_frozen_production_pack,
    load_frozen_production_context,
    manifest_paths,
    parser_verifier_hashes,
    prepare_manifests,
    read_jsonl,
    render_final_report,
    run_root,
    select_validation_pack,
    validate_prepared_manifests,
    valid_pack_artifact,
)
from safeprefix.reproducibility import atomic_json, atomic_text, now_iso, stable_hash  # noqa: E402
from safeprefix.rollout.verifier import resolve_verifier  # noqa: E402


app = typer.Typer(add_completion=False, no_args_is_help=True)


def _load(config: Path) -> tuple[dict[str, Any], str]:
    resolved = load_config(config)
    assert_rollout_only(resolved.data)
    return resolved.data, resolved.digest


def _event(root: Path, event: str, **details: Any) -> None:
    atomic_json(
        root / "event_log" / f"{time.time_ns()}-{event}.json",
        {"event": event, "created_at": now_iso(), **details},
    )


def _model_rows_and_packs(
    cfg: dict[str, Any], run_id: str, model_key: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    trace_path, pack_path, assignment_path = manifest_paths(cfg, run_id, model_key)
    if not all(path.exists() for path in (trace_path, pack_path, assignment_path)):
        raise FileNotFoundError("run prepare before GPU execution")
    return (
        read_jsonl(trace_path),
        read_jsonl(pack_path),
        json.loads(assignment_path.read_text(encoding="utf-8")),
    )


@app.command("prepare")
def prepare(
    config: Path = typer.Option(...),
    run_id: str = typer.Option(...),
) -> None:
    cfg, _ = _load(config)
    root = run_root(cfg, run_id)
    immutable = root / "immutable_manifests/immutable_protocol_manifest.json"
    if immutable.exists():
        existing = json.loads(immutable.read_text(encoding="utf-8"))
        if existing.get("configuration_hash") != stable_hash(cfg):
            raise RuntimeError("existing immutable manifest has a different configuration")
        if existing.get("engine_revision") != engine_revision():
            raise RuntimeError("existing immutable manifest has a different engine revision")
        validate_prepared_manifests(cfg, run_id=run_id)
        typer.echo(json.dumps(existing, indent=2))
        return
    payload = prepare_manifests(cfg, config_path=config, run_id=run_id)
    validate_prepared_manifests(cfg, run_id=run_id)
    _event(root, "PREPARE_COMPLETE", summary=payload)
    typer.echo(json.dumps(payload, indent=2))


@app.command("cpu-smoke")
def cpu_smoke(
    config: Path = typer.Option(...),
    run_id: str = typer.Option(...),
) -> None:
    cfg, digest = _load(config)
    root = run_root(cfg, run_id)
    protocol = json.loads(
        (root / "immutable_manifests/immutable_protocol_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    prepared_validation = validate_prepared_manifests(cfg, run_id=run_id)
    source_access = json.loads(
        (root / "source_manifest_access_ledger.json").read_text(encoding="utf-8")
    )
    checks: dict[str, bool] = {
        "configuration_hash": protocol["configuration_hash"] == digest,
        "engine_revision": protocol["engine_revision"] == engine_revision(),
        "rollout_only": protocol["rollout_only"] is True,
        "boundary_training_disabled": protocol["boundary_training_enabled"] is False,
        "native_evaluation_disabled": protocol["native_evaluation_enabled"] is False,
        "parser_loads": parser_verifier_hashes()["parser_sha256"]
        == protocol["parser_sha256"],
        "verifier_loads": parser_verifier_hashes()["verifier_sha256"]
        == protocol["verifier_sha256"],
        "verifier_constructs": resolve_verifier("exact_answer") is not None,
        "prepared_manifest_integrity": bool(prepared_validation["passed"]),
        "frozen_teacher_forced_roles": protocol["cohort"]["cohort_source"]
        == "authoritative_frozen_teacher_forced_manifests",
        "no_new_split": protocol["cohort"]["new_split_created"] is False,
        "no_native_or_final_manifest_access": not any(
            bool(source_access.get(field))
            for field in (
                "native_configuration_development_opened",
                "prompt_pilot_manifest_opened",
                "native_final_test_manifest_opened",
                "native_or_final_outputs_opened",
            )
        ),
    }
    expected_total = 0
    for model_key in cfg["selected_models"]:
        rows, packs, assignments = _model_rows_and_packs(cfg, run_id, model_key)
        expected = sum(int(pack["rollout_count"]) for pack in packs)
        observed = sum(
            len(
                logical_rollout_keys(
                    row,
                    model_key=model_key,
                    base_seed=int(cfg["rollout"]["base_seed"]),
                )
            )
            for row in rows
        )
        checks[f"{model_key}_request_construction"] = expected == observed
        checks[f"{model_key}_all_checkpoints"] = all(
            len(row["eligible_checkpoint_offsets"])
            == int(row["first_error_zero_based"]) + 1
            for row in rows
        )
        checks[f"{model_key}_pack_partition"] = (
            sorted(pack_id for values in assignments.values() for pack_id in values)
            == sorted(pack["pack_id"] for pack in packs)
        )
        expected_total += expected
    # Exact aggregation and duplicate/missing guards on one synthetic row set.
    row = {
        "model_key": "m",
        "trace_id": "t",
        "problem_id": "p",
        "source_bucket": "s",
        "checkpoint_index": 0,
        "checkpoint_token_offset": 1,
        "first_visible_error_zero_based": 0,
    }
    synthetic = [
        {**row, "rollout_index": index, "rollout_seed": index, "binary_outcome": index == 0}
        for index in range(4)
    ]
    aggregate = aggregate_checkpoint_outcomes(synthetic)[0]
    checks["raw_aggregation"] = (
        aggregate["binary_outcomes"] == [True, False, False, False]
        and aggregate["success_count"] == 1
        and aggregate["repairability_discretized"] is False
    )
    if not all(checks.values()):
        raise AssertionError({key: value for key, value in checks.items() if not value})
    result = {
        "status": "PASS",
        "checks": checks,
        "expected_total_rollouts": expected_total,
        "models_loaded": 0,
    }
    atomic_json(root / "smoke_tests/cpu_non_model.json", result)
    typer.echo(json.dumps(result, indent=2))


def _hardware_record() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("GPU worker requires CUDA")
    names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    expected = os.environ.get("SAFEPREFIX_EXPECTED_GPU")
    if expected and any(expected.upper() not in value.upper() for value in names):
        raise RuntimeError(f"required {expected}, resolved {names}")
    return {
        "device_names": names,
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
    }


def _benchmark_candidate_settings(
    cfg: dict[str, Any], model_key: str
) -> list[dict[str, int]]:
    """Resolve and validate the finite performance-only benchmark grid."""

    execution = cfg["production_execution"]
    tunable = execution["benchmark_tunable"]
    raw = tunable.get("candidate_settings", {}).get(model_key)
    if not isinstance(raw, list) or not raw:
        raw = [
            {
                "branch_batch_size": tunable["branch_batch_sizes"][model_key],
                "compaction_quantum": tunable["compaction_quantum"][model_key],
            }
        ]
    common = {
        "prefill_chunk_size": int(execution.get("prefill_chunk_size", 256)),
        "traces_per_decode_group": int(tunable["traces_per_decode_group"][model_key]),
        "maximum_decode_kv_bytes": int(tunable["maximum_decode_kv_bytes"][model_key]),
    }
    candidates: list[dict[str, int]] = []
    seen: set[tuple[int, int]] = set()
    for item in raw:
        if set(item) != {"branch_batch_size", "compaction_quantum"}:
            raise ValueError(
                "benchmark candidates may change only branch_batch_size and "
                "compaction_quantum"
            )
        batch = int(item["branch_batch_size"])
        quantum = int(item["compaction_quantum"])
        if batch < 1 or quantum < 1:
            raise ValueError("benchmark batch size and quantum must be positive")
        identity = (batch, quantum)
        if identity in seen:
            raise ValueError(f"duplicate benchmark candidate for {model_key}: {identity}")
        seen.add(identity)
        candidates.append(
            {
                "branch_batch_size": batch,
                "compaction_quantum": quantum,
                **common,
            }
        )
    return candidates


def _evaluate_benchmark_candidate(
    candidate: dict[str, Any],
    *,
    expected_rollouts: int,
    worker_count: int,
    benchmark_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Compute launch-gate metrics for one measured production-shaped setting."""

    packs = [item for item in candidate.get("packs", []) if item.get("decode_metrics")]
    failures: list[str] = []
    if candidate.get("status") != "COMPLETE" or not packs:
        failures.append("candidate_incomplete")
    useful = sum(int(item["decode_metrics"]["useful_output_tokens"]) for item in packs)
    decode_seconds = sum(float(item["decode_metrics"]["decode_wall_seconds"]) for item in packs)
    total_wall_seconds = sum(float(item["total_wall_seconds"]) for item in packs)
    benchmark_rollouts = sum(int(item["row_count"]) for item in packs)
    forwarded = sum(int(item["decode_metrics"]["forwarded_row_steps"]) for item in packs)
    utilization_samples = sum(int(item.get("gpu_utilization_sample_count", 0)) for item in packs)
    utilization_weighted = sum(
        float(item.get("gpu_utilization_mean_percent") or 0.0)
        * int(item.get("gpu_utilization_sample_count", 0))
        for item in packs
    )
    mean_utilization = (
        utilization_weighted / utilization_samples if utilization_samples else None
    )
    maximum_reserved = max(
        (int(item["decode_metrics"].get("maximum_reserved_bytes", 0)) for item in packs),
        default=0,
    )
    total_memory = min(
        (int(item.get("gpu_total_memory_bytes") or 0) for item in packs),
        default=0,
    )
    headroom = 1.0 - maximum_reserved / max(total_memory, 1)
    rate = useful / max(decode_seconds, 1e-12)
    forwarded_ratio = forwarded / max(useful, 1)
    projected_seconds = (
        expected_rollouts / max(benchmark_rollouts, 1)
        * total_wall_seconds
        / max(int(worker_count), 1)
    )
    if useful <= 0 or decode_seconds <= 0 or rate <= 0:
        failures.append("nonpositive_throughput")
    if benchmark_rollouts <= 0 or total_wall_seconds <= 0:
        failures.append("invalid_wall_projection")
    if any(
        int(item["decode_metrics"].get("preallocated_cache_waves", 0)) <= 0
        or int(item["decode_metrics"].get("heterogeneous_prefix_waves", 0)) <= 0
        for item in packs
    ):
        failures.append("production_path_not_exercised")
    minimum_headroom = float(
        benchmark_cfg["minimum_gpu_memory_headroom_fraction"]
    )
    if total_memory <= 0 or headroom < minimum_headroom:
        failures.append("gpu_memory_headroom")
    if utilization_samples < int(
        benchmark_cfg.get("minimum_gpu_utilization_samples", 1)
    ):
        failures.append("gpu_utilization_samples")
    if mean_utilization is None or mean_utilization < float(
        benchmark_cfg.get("minimum_mean_gpu_utilization_percent", 0.0)
    ):
        failures.append("gpu_utilization")
    if forwarded_ratio > float(
        benchmark_cfg.get("maximum_forwarded_to_useful_ratio", float("inf"))
    ):
        failures.append("forwarded_work_waste")
    return {
        "candidate_id": candidate.get("candidate_id"),
        "settings": candidate.get("settings"),
        "eligible": not failures,
        "gate_failures": failures,
        "useful_output_tokens": useful,
        "decode_seconds": decode_seconds,
        "total_wall_seconds": total_wall_seconds,
        "useful_output_tokens_per_second": rate,
        "benchmark_rollouts": benchmark_rollouts,
        "mean_generated_tokens": useful / max(benchmark_rollouts, 1),
        "forwarded_row_steps": forwarded,
        "forwarded_to_useful_ratio": forwarded_ratio,
        "gpu_utilization_sample_count": utilization_samples,
        "gpu_utilization_mean_percent": mean_utilization,
        "maximum_reserved_bytes": maximum_reserved,
        "gpu_total_memory_bytes": total_memory,
        "gpu_memory_headroom_fraction": headroom,
        "truncation_count": int(candidate.get("truncation_count", 0)),
        "truncation_rate": int(candidate.get("truncation_count", 0))
        / max(benchmark_rollouts, 1),
        "projected_eight_h100_seconds": projected_seconds,
    }


def _select_benchmark_candidate(
    candidates: list[dict[str, Any]],
    *,
    expected_rollouts: int,
    worker_count: int,
    benchmark_cfg: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select the measured fastest candidate after all fixed gates pass."""

    evaluated = [
        _evaluate_benchmark_candidate(
            candidate,
            expected_rollouts=expected_rollouts,
            worker_count=worker_count,
            benchmark_cfg=benchmark_cfg,
        )
        for candidate in candidates
    ]
    eligible = [item for item in evaluated if item["eligible"]]
    if not eligible:
        raise RuntimeError(
            "no benchmark candidate passed the fixed launch gates: "
            + json.dumps(evaluated, sort_keys=True)
        )
    selected = min(
        eligible,
        key=lambda item: (
            -float(item["useful_output_tokens_per_second"]),
            float(item["projected_eight_h100_seconds"]),
            int(item["settings"]["branch_batch_size"]),
            int(item["settings"]["compaction_quantum"]),
            str(item["candidate_id"]),
        ),
    )
    return selected, evaluated


@app.command("gpu-worker")
def gpu_worker(
    config: Path = typer.Option(...),
    run_id: str = typer.Option(...),
    model_key: str = typer.Option(...),
    worker_index: int = typer.Option(0),
    mode: str = typer.Option(...),
) -> None:
    cfg, digest = _load(config)
    if model_key not in cfg["selected_models"]:
        raise ValueError(f"unknown model key: {model_key}")
    if mode not in {"smoke", "benchmark", "production"}:
        raise ValueError("mode must be smoke, benchmark, or production")
    root = run_root(cfg, run_id)
    hardware = _hardware_record()
    rows, packs, assignments = _model_rows_and_packs(cfg, run_id, model_key)
    rows_by_id = {str(row["trace_id"]): row for row in rows}
    if len(rows_by_id) != len(rows):
        raise AssertionError("duplicate model trace IDs")
    if mode == "production":
        context = load_frozen_production_context(
            cfg, run_id=run_id, model_key=model_key
        )
        frozen = context["frozen"]
        wanted = set(assignments.get(str(worker_index), []))
        selected_packs = [pack for pack in packs if pack["pack_id"] in wanted]
        settings = frozen["model_settings"][model_key]
        generation = dict(frozen["generation"])
        execution_freeze_digest = str(frozen["freeze_digest"])
        settings_hash = str(settings["settings_hash"])
        settings_candidates = [settings]
    else:
        validation_cfg = cfg["pre_submit_validation"][
            "real_model_smoke" if mode == "smoke" else "production_benchmark"
        ]
        selected_packs = [
            select_validation_pack(
                rows,
                model_key=model_key,
                count=int(validation_cfg["traces_per_model"]),
                mode=mode,
                configuration_hash=digest,
                revision=engine_revision(),
                base_seed=int(cfg["rollout"]["base_seed"]),
            )
        ]
        if mode == "benchmark":
            settings_candidates = _benchmark_candidate_settings(cfg, model_key)
        else:
            settings_candidates = [
                {
                    "branch_batch_size": int(
                        validation_cfg.get(
                            "temporary_branch_batch_size",
                            cfg["production_execution"]["benchmark_tunable"][
                                "branch_batch_sizes"
                            ][model_key],
                        )
                    ),
                    "compaction_quantum": int(
                        validation_cfg.get(
                            "temporary_compaction_quantum",
                            cfg["production_execution"]["benchmark_tunable"][
                                "compaction_quantum"
                            ][model_key],
                        )
                    ),
                    "prefill_chunk_size": int(
                        cfg["production_execution"].get("prefill_chunk_size", 256)
                    ),
                    "traces_per_decode_group": int(
                        validation_cfg.get(
                            "temporary_traces_per_decode_group",
                            cfg["production_execution"]["benchmark_tunable"][
                                "traces_per_decode_group"
                            ][model_key],
                        )
                    ),
                    "maximum_decode_kv_bytes": int(
                        cfg["production_execution"]["benchmark_tunable"][
                            "maximum_decode_kv_bytes"
                        ][model_key]
                    ),
                }
            ]
        settings = settings_candidates[0]
        generation = dict(cfg["rollout"])
        generation["max_new_tokens"] = int(validation_cfg["max_new_tokens"])
        generation["preallocate_kv_cache"] = bool(
            cfg["production_execution"].get("preallocate_kv_cache", True)
        )
        settings_hash = stable_hash(settings)
        execution_freeze_digest = stable_hash(
            {
                "mode": mode,
                "configuration_hash": digest,
                "engine_revision": engine_revision(),
                "pack_hash": selected_packs[0]["pack_hash"],
                "settings": settings,
                "generation": generation,
            }
        )
    loaded = load_model(cfg["models"][model_key])
    summaries: list[dict[str, Any]] = []
    measured_candidates: list[dict[str, Any]] = []
    for candidate_index, candidate_settings in enumerate(settings_candidates):
        settings = dict(candidate_settings)
        stored_settings_hash = settings.pop("settings_hash", None)
        computed_settings_hash = stable_hash(settings)
        if mode == "production" and str(stored_settings_hash) != computed_settings_hash:
            raise RuntimeError("frozen production settings hash differs")
        settings_hash = (
            str(stored_settings_hash)
            if mode == "production"
            else computed_settings_hash
        )
        candidate_id = f"candidate-{candidate_index:02d}-{settings_hash[:12]}"
        if mode == "benchmark":
            execution_freeze_digest = stable_hash(
                {
                    "mode": mode,
                    "configuration_hash": digest,
                    "engine_revision": engine_revision(),
                    "pack_hash": selected_packs[0]["pack_hash"],
                    "settings": settings,
                    "generation": generation,
                }
            )
        candidate_summaries: list[dict[str, Any]] = []
        candidate_truncations = 0
        try:
            for pack in selected_packs:
                pack_root = (
                    root
                    / (
                        "raw_rollout_shards"
                        if mode == "production"
                        else f"validation/{mode}"
                    )
                    / model_key
                )
                if mode == "benchmark":
                    pack_root = pack_root / candidate_id
                pack_root = pack_root / pack["pack_id"]
                if mode == "smoke" and not valid_pack_artifact(
                    pack,
                    pack_root,
                    trace_rows={str(t): rows_by_id[str(t)] for t in pack["trace_ids"]},
                    expected_freeze_digest=execution_freeze_digest,
                    expected_settings_hash=settings_hash,
                    expected_layers=cfg["models"][model_key][
                        "selected_hidden_state_layers"
                    ],
                    expected_model_revision=loaded.model_revision,
                ):
                    pack_root.mkdir(parents=True, exist_ok=True)
                    (pack_root / "interrupted.tmp").write_text(
                        "intentional incomplete pack", encoding="utf-8"
                    )
                    if valid_pack_artifact(pack, pack_root):
                        raise AssertionError(
                            "partial smoke pack was incorrectly accepted"
                        )
                    try:
                        execute_pack(
                            config=cfg,
                            model_key=model_key,
                            loaded=loaded,
                            pack=pack,
                            trace_rows=rows_by_id,
                            pack_root=pack_root,
                            generation=generation,
                            batch_size=int(settings["branch_batch_size"]),
                            compaction_quantum=int(settings["compaction_quantum"]),
                            prefill_chunk_size=int(settings["prefill_chunk_size"]),
                            traces_per_decode_group=int(
                                settings["traces_per_decode_group"]
                            ),
                            maximum_decode_kv_bytes=int(
                                settings["maximum_decode_kv_bytes"]
                            ),
                            mode=mode,
                            execution_freeze_digest=execution_freeze_digest,
                            model_execution_settings_hash=settings_hash,
                            interrupt_after_decode_groups=1,
                        )
                    except RuntimeError as exc:
                        if "INTENTIONAL_SMOKE_INTERRUPTION" not in str(exc):
                            raise
                    else:
                        raise AssertionError(
                            "smoke interruption did not interrupt the pack"
                        )
                    if valid_pack_artifact(pack, pack_root):
                        raise AssertionError(
                            "interrupted smoke pack was incorrectly accepted"
                        )
                summary = execute_pack(
                    config=cfg,
                    model_key=model_key,
                    loaded=loaded,
                    pack=pack,
                    trace_rows=rows_by_id,
                    pack_root=pack_root,
                    generation=generation,
                    batch_size=int(settings["branch_batch_size"]),
                    compaction_quantum=int(settings["compaction_quantum"]),
                    prefill_chunk_size=int(settings["prefill_chunk_size"]),
                    traces_per_decode_group=int(settings["traces_per_decode_group"]),
                    maximum_decode_kv_bytes=int(settings["maximum_decode_kv_bytes"]),
                    mode=mode,
                    execution_freeze_digest=execution_freeze_digest,
                    model_execution_settings_hash=settings_hash,
                )
                if mode == "benchmark":
                    frame = pd.read_parquet(
                        pack_root / "rollouts.parquet", columns=["truncation_flag"]
                    )
                    candidate_truncations += int(frame["truncation_flag"].sum())
                if mode == "smoke":
                    resumed = execute_pack(
                        config=cfg,
                        model_key=model_key,
                        loaded=loaded,
                        pack=pack,
                        trace_rows=rows_by_id,
                        pack_root=pack_root,
                        generation=generation,
                        batch_size=int(settings["branch_batch_size"]),
                        compaction_quantum=int(settings["compaction_quantum"]),
                        prefill_chunk_size=int(settings["prefill_chunk_size"]),
                        traces_per_decode_group=int(
                            settings["traces_per_decode_group"]
                        ),
                        maximum_decode_kv_bytes=int(
                            settings["maximum_decode_kv_bytes"]
                        ),
                        mode=mode,
                        execution_freeze_digest=execution_freeze_digest,
                        model_execution_settings_hash=settings_hash,
                    )
                    if resumed.get("status") != "SKIPPED_VALID":
                        raise AssertionError(
                            "pack-level resume did not skip a valid smoke pack"
                        )
                    metrics = summary["decode_metrics"]
                    if len(
                        {
                            int(offset)
                            for trace_id in pack["trace_ids"]
                            for offset in rows_by_id[str(trace_id)][
                                "eligible_checkpoint_offsets"
                            ]
                        }
                    ) < 2:
                        raise AssertionError(
                            "smoke pack did not contain heterogeneous prefixes"
                        )
                    if int(summary["row_count"]) <= int(
                        settings["branch_batch_size"]
                    ):
                        raise AssertionError("smoke did not require pending-branch refill")
                    required_metric_checks = {
                        "heterogeneous_prefix_waves": int(
                            metrics["heterogeneous_prefix_waves"]
                        )
                        > 0,
                        "refilled_branches": int(metrics["refilled_branches"]) > 0,
                        "completion_removal_events": int(
                            metrics["completion_removal_events"]
                        )
                        > 0,
                        "all_admitted": int(metrics["admitted_branches"])
                        == int(summary["row_count"]),
                        "all_completed": int(metrics["completed_branches"])
                        == int(summary["row_count"]),
                        "preallocated_cache": int(metrics["preallocated_cache_waves"])
                        > 0,
                        "multiple_decode_groups": int(metrics["decode_group_count"])
                        >= 2,
                    }
                    if not all(required_metric_checks.values()):
                        raise AssertionError(
                            {
                                key: value
                                for key, value in required_metric_checks.items()
                                if not value
                            }
                        )
                candidate_summaries.append(summary)
        except torch.OutOfMemoryError as exc:
            if mode != "benchmark":
                raise
            torch.cuda.empty_cache()
            measured_candidates.append(
                {
                    "status": "OOM",
                    "candidate_id": candidate_id,
                    "settings": settings,
                    "settings_hash": settings_hash,
                    "execution_freeze_digest": execution_freeze_digest,
                    "packs": candidate_summaries,
                    "truncation_count": candidate_truncations,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        summaries = candidate_summaries
        if mode == "benchmark":
            measured_candidates.append(
                {
                    "status": "COMPLETE",
                    "candidate_id": candidate_id,
                    "settings": settings,
                    "settings_hash": settings_hash,
                    "execution_freeze_digest": execution_freeze_digest,
                    "packs": candidate_summaries,
                    "truncation_count": candidate_truncations,
                }
            )
    selected_trace_ids = {
        str(trace_id) for pack in selected_packs for trace_id in pack["trace_ids"]
    }
    selection_summary = {
        "traces": len(selected_trace_ids),
        "sources": sorted(
            {str(rows_by_id[trace_id]["source_bucket"]) for trace_id in selected_trace_ids}
        ),
        "checkpoint_offsets": sorted(
            {
                int(offset)
                for trace_id in selected_trace_ids
                for offset in rows_by_id[trace_id]["eligible_checkpoint_offsets"]
            }
        ),
    }
    validation_gate = "NOT_APPLICABLE"
    if mode == "smoke":
        validation_gate = "PASS"
    elif mode == "benchmark":
        required_sources = int(
            cfg["pre_submit_validation"]["production_benchmark"].get(
                "require_source_coverage", 1
            )
        )
        if len(selection_summary["sources"]) < required_sources:
            raise AssertionError(
                f"benchmark source coverage {selection_summary['sources']} < {required_sources}"
            )
        if len(selection_summary["checkpoint_offsets"]) < 2:
            raise AssertionError("benchmark lacks heterogeneous checkpoint lengths")
        validation_gate = "PASS"
    worker_summary = {
        "status": "COMPLETE",
        "validation_gate": validation_gate,
        "mode": mode,
        "model_key": model_key,
        "worker_index": worker_index,
        "hardware": hardware,
        "settings": settings if mode != "benchmark" else None,
        "generation": generation,
        "execution_freeze_digest": (
            execution_freeze_digest if mode != "benchmark" else None
        ),
        "model_execution_settings_hash": settings_hash if mode != "benchmark" else None,
        "packs": summaries if mode != "benchmark" else [],
        "benchmark_candidates": measured_candidates if mode == "benchmark" else [],
        "validation_selection": selection_summary,
    }
    target = root / "worker_summaries" / mode / model_key / f"worker_{worker_index:02d}.json"
    atomic_json(target, worker_summary)
    _event(root, "GPU_WORKER_COMPLETE", **worker_summary)
    typer.echo(json.dumps(worker_summary, indent=2))


@app.command("freeze")
def freeze(
    config: Path = typer.Option(...),
    run_id: str = typer.Option(...),
) -> None:
    cfg, digest = _load(config)
    root = run_root(cfg, run_id)
    frozen_path = root / "frozen_execution_manifest.json"
    if frozen_path.exists():
        existing = json.loads(frozen_path.read_text(encoding="utf-8"))
        expected = stable_hash(
            {key: value for key, value in existing.items() if key != "freeze_digest"}
        )
        if (
            existing.get("status") != "FROZEN_FOR_PRODUCTION"
            or str(existing.get("configuration_hash")) != digest
            or str(existing.get("engine_revision")) != engine_revision()
            or str(existing.get("freeze_digest")) != expected
        ):
            raise RuntimeError("existing production freeze is invalid or incompatible")
        typer.echo(json.dumps(existing, indent=2))
        return
    settings: dict[str, Any] = {}
    benchmarks: dict[str, Any] = {}
    projected_model_seconds: dict[str, float] = {}
    protocol = json.loads(
        (root / "immutable_manifests/immutable_protocol_manifest.json").read_text(encoding="utf-8")
    )
    for model_key in cfg["selected_models"]:
        smoke = root / "worker_summaries/smoke" / model_key / "worker_00.json"
        benchmark = root / "worker_summaries/benchmark" / model_key / "worker_00.json"
        if not smoke.exists() or not benchmark.exists():
            raise RuntimeError(f"smoke/benchmark incomplete for {model_key}")
        smoke_payload = json.loads(smoke.read_text(encoding="utf-8"))
        benchmark_payload = json.loads(benchmark.read_text(encoding="utf-8"))
        if smoke_payload.get("status") != "COMPLETE" or benchmark_payload.get("status") != "COMPLETE":
            raise RuntimeError(f"smoke/benchmark failed for {model_key}")
        if smoke_payload.get("validation_gate") != "PASS":
            raise RuntimeError(f"production-shaped smoke gate failed for {model_key}")
        if benchmark_payload.get("validation_gate") != "PASS":
            raise RuntimeError(f"production benchmark gate failed for {model_key}")
        expected_rollouts = int(protocol["models"][model_key]["rollouts"])
        expected_candidates = _benchmark_candidate_settings(cfg, model_key)
        measured_candidates = list(benchmark_payload.get("benchmark_candidates", []))
        if [item["settings"] for item in measured_candidates] != expected_candidates:
            raise RuntimeError(
                f"benchmark candidate grid was incomplete or reordered for {model_key}"
            )
        benchmark_cfg = cfg["pre_submit_validation"]["production_benchmark"]
        selected, evaluated = _select_benchmark_candidate(
            measured_candidates,
            expected_rollouts=expected_rollouts,
            worker_count=int(cfg["scheduler"]["workers_per_model_wave"]),
            benchmark_cfg=benchmark_cfg,
        )
        selected_settings = dict(selected["settings"])
        selected_settings["settings_hash"] = stable_hash(selected_settings)
        settings[model_key] = selected_settings
        projected_model_seconds[model_key] = float(
            selected["projected_eight_h100_seconds"]
        )
        benchmarks[model_key] = {
            **selected,
            "selected_candidate_id": selected["candidate_id"],
            "candidate_evaluations": evaluated,
        }
    projected_total = sum(projected_model_seconds.values())
    maximum_projected_hours = float(
        cfg["pre_submit_validation"]["production_benchmark"][
            "maximum_projected_total_wall_hours"
        ]
    )
    if projected_total > maximum_projected_hours * 3600:
        raise RuntimeError(
            "selected benchmark settings exceed the fixed projected runtime gate: "
            f"{projected_total / 3600:.2f}h > {maximum_projected_hours:.2f}h"
        )
    generation = dict(cfg["rollout"])
    generation["preallocate_kv_cache"] = bool(
        cfg["production_execution"].get("preallocate_kv_cache", True)
    )
    payload = {
        "status": "FROZEN_FOR_PRODUCTION",
        "frozen_at": now_iso(),
        "configuration_hash": digest,
        "engine_revision": engine_revision(),
        "model_settings": settings,
        "generation": generation,
        "benchmarks": benchmarks,
        "projected_total_wall_seconds_model_major": projected_total,
        "maximum_projected_total_wall_hours": maximum_projected_hours,
        "scheduler": (
            "four model-major waves; dynamically dispatched immutable packs "
            "across eight model-resident literal-H100 workers"
        ),
        "training_enabled": False,
        "native_evaluation_enabled": False,
    }
    payload["freeze_digest"] = stable_hash(payload)
    atomic_json(frozen_path, payload)
    atomic_json(root / "production_benchmark_results.json", benchmarks)
    lines = [
        "# Production launch summary",
        "",
        f"- Eligible traces/model: `{protocol['cohort']['eligible_traces']}`",
        f"- Checkpoints/model: `{protocol['cohort']['anticipated_checkpoints_per_model']}`",
        f"- Expected rollouts/model: `{protocol['cohort']['anticipated_rollouts_per_model']}`",
        f"- Total projected wall: `{payload['projected_total_wall_seconds_model_major'] / 3600:.2f}` hours",
        "- Boundary training: `disabled`",
        "- Native evaluation/final-test access: `disabled`",
        "",
    ]
    for model_key in cfg["selected_models"]:
        value = benchmarks[model_key]
        lines.append(
            f"- `{model_key}`: batch={settings[model_key]['branch_batch_size']}, "
            f"quantum={settings[model_key]['compaction_quantum']}, "
            f"{value['useful_output_tokens_per_second']:.2f} useful tok/s, "
            f"projected {value['projected_eight_h100_seconds'] / 3600:.2f} h"
        )
    atomic_text(root / "PRODUCTION_LAUNCH_SUMMARY.md", "\n".join(lines) + "\n")
    typer.echo(json.dumps(payload, indent=2))


@app.command("aggregate")
def aggregate(
    config: Path = typer.Option(...),
    run_id: str = typer.Option(...),
) -> None:
    cfg, _ = _load(config)
    result = aggregate_production(cfg, run_id=run_id)
    typer.echo(json.dumps(result, indent=2))


@app.command("report")
def report(
    config: Path = typer.Option(...),
    run_id: str = typer.Option(...),
) -> None:
    cfg, _ = _load(config)
    path = render_final_report(cfg, run_id=run_id)
    typer.echo(str(path))


if __name__ == "__main__":
    app()
