#!/usr/bin/env python3
"""Durable Modal orchestration for the rollout-only SafePrefix production suite.

Deployment is deliberately separate from submission::

    MODAL_PROFILE=meskmmy python3 -m modal deploy \
      scripts/modal_safeprefix_full_teacher_forced.py
    MODAL_PROFILE=meskmmy python3 -m modal run \
      scripts/modal_safeprefix_full_teacher_forced.py --action submit \
      --run-id <immutable-run-id>

The local entrypoint submits the *deployed* ``orchestrate`` function by name.
Consequently the orchestration call, its child GPU calls, and their persistent
Volume artifacts survive the short-lived ``modal run`` client process.
"""

import hashlib
import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import modal
import yaml


LOCAL_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/workspace/aaai_monkey")
APP_NAME = "safeprefix-full-teacher-forced-rollouts"
RUN_VOLUME_NAME = "safeprefix-full-teacher-forced-runs-v2"
CACHE_VOLUME_NAME = "safeprefix-hf-cache"
SOURCE_VOLUME_NAME = "safeprefix-runs"
CONFIG_NAME = "full_teacher_forced_suite.yaml"
RUNNER_NAME = "25_run_full_teacher_forced_rollouts.py"
LAUNCHER_NAME = Path(__file__).name
GPU_WORKERS = 8
LAUNCHER_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

app = modal.App(APP_NAME)
# V2 is required because eight workers write disjoint pack directories at once.
run_volume = modal.Volume.from_name(
    RUN_VOLUME_NAME,
    create_if_missing=True,
    version=2,
)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=False)
read_only_cache_volume = cache_volume.with_mount_options(read_only=True)
source_volume = modal.Volume.from_name(SOURCE_VOLUME_NAME, create_if_missing=False)
read_only_source_volume = source_volume.with_mount_options(read_only=True)
hf_secrets = [modal.Secret.from_name("huggingface-token")]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "accelerate==1.10.1",
        "datasets==5.0.0",
        "huggingface-hub==0.34.4",
        "numpy==2.2.6",
        "nvidia-ml-py==12.575.51",
        "pandas==2.3.2",
        "pyarrow==24.0.0",
        "PyYAML==6.0.2",
        "rapidfuzz==3.14.1",
        "scikit-learn==1.7.1",
        "scipy==1.16.1",
        "sentencepiece==0.2.1",
        "torch==2.8.0",
        "transformers==4.55.4",
        "typer==0.17.4",
    )
    .add_local_file(
        LOCAL_ROOT / "configs" / CONFIG_NAME,
        str(REMOTE_ROOT / "configs" / CONFIG_NAME),
        copy=True,
    )
    .add_local_file(
        LOCAL_ROOT / "configs" / "models.yaml",
        str(REMOTE_ROOT / "configs" / "models.yaml"),
        copy=True,
    )
    .add_local_file(
        LOCAL_ROOT / "configs" / "datasets.yaml",
        str(REMOTE_ROOT / "configs" / "datasets.yaml"),
        copy=True,
    )
    .add_local_dir(
        LOCAL_ROOT / "src" / "safeprefix",
        str(REMOTE_ROOT / "src" / "safeprefix"),
        copy=True,
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
    .add_local_file(
        LOCAL_ROOT / "scripts" / "16_audit_processbench_references.py",
        str(REMOTE_ROOT / "scripts" / "16_audit_processbench_references.py"),
        copy=True,
    )
    .add_local_file(
        LOCAL_ROOT / "scripts" / RUNNER_NAME,
        str(REMOTE_ROOT / "scripts" / RUNNER_NAME),
        copy=True,
    )
    .add_local_file(
        LOCAL_ROOT / "scripts" / LAUNCHER_NAME,
        str(REMOTE_ROOT / "scripts" / LAUNCHER_NAME),
        copy=True,
    )
    .add_local_file(
        LOCAL_ROOT / "pyproject.toml",
        str(REMOTE_ROOT / "pyproject.toml"),
        copy=True,
    )
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "HF_HUB_CACHE": "/cache/huggingface/hub",
            "HF_DATASETS_CACHE": "/cache/huggingface/datasets",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(REMOTE_ROOT / "src"),
            "SAFEPREFIX_RUNS_ROOT": "/runs",
            "SAFEPREFIX_CACHE_ROOT": "/cache",
            "SAFEPREFIX_CRV_ROOT": "/cache/datasets/crv",
            "SAFEPREFIX_ROLLOUT_ONLY": "1",
            "SAFEPREFIX_EXPECTED_GPU": "H100",
        }
    )
)


def _safe_run_id(run_id: str) -> None:
    if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
        raise ValueError("run_id must be one non-empty safe path component")


def _config_path() -> Path:
    return REMOTE_ROOT / "configs" / CONFIG_NAME


def _runner_path() -> Path:
    return REMOTE_ROOT / "scripts" / RUNNER_NAME


def _run_root(run_id: str) -> Path:
    _safe_run_id(run_id)
    return Path("/runs") / run_id


def _load_protocol_config() -> dict[str, Any]:
    payload = yaml.safe_load(_config_path().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("production configuration must be a YAML mapping")
    return payload


def _assert_rollout_only_protocol(config: dict[str, Any]) -> tuple[str, ...]:
    """Fail before model work if the immutable rollout-only scope drifted."""

    models = tuple(str(item) for item in config.get("selected_models", []))
    if len(models) != 4 or len(set(models)) != 4:
        raise RuntimeError(f"exactly four distinct configured models are required, got {models}")
    suite = config.get("full_teacher_forced_suite", {})
    rollout = config.get("rollout", {})
    gates = config.get("phase_gates", {})
    if not bool(suite.get("teacher_forced_only", False)):
        raise RuntimeError("teacher_forced_only must be true")
    if int(suite.get("rollouts_per_checkpoint", -1)) != 4:
        raise RuntimeError("production requires exactly four rollouts per checkpoint")
    if int(rollout.get("max_new_tokens", -1)) != 4096:
        raise RuntimeError("production requires max_new_tokens=4096")
    if gates.get("teacher_forced_boundary_training") != "locked":
        raise RuntimeError("boundary training must remain locked for this rollout-only run")
    if gates.get("native_free_form_evaluation") != "locked":
        raise RuntimeError("native free-form evaluation must remain locked")
    if gates.get("native_final_test") != "locked":
        raise RuntimeError("native final-test access must remain locked")
    return models


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _state_path(run_id: str) -> Path:
    return _run_root(run_id) / "orchestration" / "remote_run_state.json"


def _read_state(run_id: str) -> dict[str, Any]:
    path = _state_path(run_id)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_state(run_id: str, **updates: Any) -> dict[str, Any]:
    # Only the orchestrator writes global state. GPU workers write disjoint event
    # files, avoiding concurrent read-modify-write on a shared JSON document.
    payload = {
        **_read_state(run_id),
        **updates,
        "updated_unix": time.time(),
    }
    _atomic_json(_state_path(run_id), payload)
    run_volume.commit()
    return payload


def _event(run_id: str, event: str, **details: Any) -> None:
    path = (
        _run_root(run_id)
        / "orchestration"
        / "events"
        / f"{time.time_ns()}-{uuid.uuid4().hex}.json"
    )
    _atomic_json(
        path,
        {
            "event": event,
            "time_unix": time.time(),
            "modal_task_id": os.environ.get("MODAL_TASK_ID"),
            **details,
        },
    )
    run_volume.commit()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_command(command: list[str], log_path: Path) -> None:
    """Stream one subprocess to a disjoint durable worker log."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["SAFEPREFIX_RUN_ID"] = log_path.parts[2] if len(log_path.parts) > 2 else ""
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("COMMAND: " + " ".join(command) + "\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=REMOTE_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        return_code = process.wait()
    run_volume.commit()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def _runner_command(subcommand: str, run_id: str, *arguments: str) -> list[str]:
    return [
        "python3",
        str(_runner_path()),
        subcommand,
        "--config",
        str(_config_path()),
        "--run-id",
        run_id,
        *arguments,
    ]


def _assert_literal_h100() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in an H100 production function")
    names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    if not names or any("H100" not in name.upper() for name in names):
        raise RuntimeError(f"literal H100 required; resolved CUDA devices were {names}")
    return {
        "device_names": names,
        "device_count": torch.cuda.device_count(),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
    }


def _validate_cached_model_snapshots(config: dict[str, Any]) -> dict[str, Any]:
    """Fail before GPU allocation unless every exact pinned snapshot is complete.

    Production mounts the shared Hugging Face cache read-only and runs offline.
    This check therefore verifies model access without allowing a worker to
    silently resolve ``main`` or download a different revision mid-run.
    """

    from huggingface_hub import snapshot_download

    report: dict[str, Any] = {}
    for model_key in config["selected_models"]:
        entry = config["models"][model_key]
        repositories = {
            "model": (
                str(entry["hf_model_id"]),
                str(entry["revision"]),
            ),
            "tokenizer": (
                str(entry.get("tokenizer_id") or entry["hf_model_id"]),
                str(entry.get("tokenizer_revision") or entry["revision"]),
            ),
        }
        resolved: dict[str, Any] = {}
        for kind, (repo_id, revision) in repositories.items():
            snapshot = Path(
                snapshot_download(
                    repo_id=repo_id,
                    revision=revision,
                    local_files_only=True,
                )
            )
            if snapshot.name != revision:
                raise RuntimeError(
                    f"cached {kind} snapshot for {model_key} resolved to "
                    f"{snapshot.name}, expected {revision}"
                )
            broken = [
                path.relative_to(snapshot).as_posix()
                for path in snapshot.rglob("*")
                if path.is_symlink() and not path.exists()
            ]
            if broken:
                raise RuntimeError(
                    f"cached {kind} snapshot has broken files for {model_key}: "
                    f"{broken[:10]}"
                )
            resolved[kind] = {
                "repo_id": repo_id,
                "revision": revision,
                "snapshot_path": str(snapshot),
            }
        model_snapshot = Path(resolved["model"]["snapshot_path"])
        weight_index = next(
            (
                model_snapshot / name
                for name in (
                    "model.safetensors.index.json",
                    "pytorch_model.bin.index.json",
                )
                if (model_snapshot / name).is_file()
            ),
            None,
        )
        if weight_index is not None:
            index = json.loads(weight_index.read_text(encoding="utf-8"))
            shards = sorted(set(map(str, index.get("weight_map", {}).values())))
            missing_shards = [name for name in shards if not (model_snapshot / name).is_file()]
            if not shards or missing_shards:
                raise RuntimeError(
                    f"cached weight index is incomplete for {model_key}: "
                    f"missing={missing_shards[:10]}"
                )
        else:
            shards = [
                name
                for name in ("model.safetensors", "pytorch_model.bin")
                if (model_snapshot / name).is_file()
            ]
            if not shards:
                raise RuntimeError(f"cached model weights are missing for {model_key}")
        if not (model_snapshot / "config.json").is_file():
            raise RuntimeError(f"cached model config is missing for {model_key}")
        resolved["weight_files"] = shards
        report[str(model_key)] = resolved
    return report


def _freeze_candidates(run_id: str) -> list[Path]:
    """Find already-materialized protocol/pack/benchmark inputs to freeze.

    Only paths present before production are listed. Production outputs cannot
    enlarge or otherwise alter the digest because validation checks this exact
    file list, rather than rescanning a directory that is expected to grow.
    """

    keywords = (
        "manifest",
        "execution_pack",
        "benchmark",
        "resolved_config",
        "protocol",
        "model_setting",
    )
    root = _run_root(run_id)
    candidates: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or "orchestration" in path.parts:
            continue
        lowered = path.name.casefold()
        if any(keyword in lowered for keyword in keywords):
            candidates.append(path)
    return sorted(candidates, key=lambda item: item.relative_to(root).as_posix())


def _write_production_freeze(
    run_id: str,
    source_commit: str,
    source_bundle_sha256: str,
    models: Iterable[str],
    benchmark_results: dict[str, Any],
) -> dict[str, Any]:
    root = _run_root(run_id)
    frozen_files = {
        path.relative_to(root).as_posix(): {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in _freeze_candidates(run_id)
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "frozen_unix": time.time(),
        "source_commit": source_commit,
        "source_bundle_sha256": source_bundle_sha256,
        "config_sha256": _sha256(_config_path()),
        "runner_sha256": _sha256(_runner_path()),
        "models_in_order": list(models),
        "gpu_workers_per_model": GPU_WORKERS,
        "gpu_requirement": "H100!",
        "source_volume": SOURCE_VOLUME_NAME,
        "source_volume_mount": "/source_runs (read-only)",
        "scheduler": "model_major_modal_dynamic_immutable_pack_dispatch",
        "dispatch_priority": "estimated_work_descending_then_pack_id",
        "recovery_unit": "immutable_execution_pack",
        "benchmark_results": benchmark_results,
        "frozen_files": frozen_files,
        "rollout_only": True,
        "boundary_training_enabled": False,
        "native_evaluation_enabled": False,
    }
    payload["freeze_digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    _atomic_json(root / "orchestration" / "production_launch_freeze.json", payload)
    run_volume.commit()
    return payload


def _validate_production_freeze(run_id: str, expected_digest: str) -> None:
    path = _run_root(run_id) / "orchestration" / "production_launch_freeze.json"
    if not path.exists():
        raise RuntimeError("production launch freeze is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("freeze_digest") != expected_digest:
        raise RuntimeError("production launch freeze digest differs from orchestrator input")
    if _sha256(_config_path()) != payload.get("config_sha256"):
        raise RuntimeError("baked production configuration differs from frozen configuration")
    if _sha256(_runner_path()) != payload.get("runner_sha256"):
        raise RuntimeError("baked production runner differs from frozen runner")
    root = _run_root(run_id)
    for relative, expected in payload.get("frozen_files", {}).items():
        candidate = root / relative
        if not candidate.exists():
            raise RuntimeError(f"frozen production input disappeared: {relative}")
        if _sha256(candidate) != expected["sha256"]:
            raise RuntimeError(f"frozen production input changed: {relative}")


def _valid_worker_summary_payload(
    payload: dict[str, Any], *, model_key: str, mode: str
) -> bool:
    """Validate the durable completion record for one validation worker.

    This intentionally checks the scientific identity, not merely file
    existence.  The runner's ``freeze`` command performs the deeper benchmark
    grid validation before production is allowed to start.
    """

    return bool(
        payload.get("status") == "COMPLETE"
        and payload.get("validation_gate") == "PASS"
        and str(payload.get("model_key")) == model_key
        and str(payload.get("mode")) == mode
        and int(payload.get("worker_index", -1)) == 0
        and isinstance(payload.get("validation_selection"), dict)
        and payload.get("generation")
    )


def _partition_pack_resume(
    pack_ids: Iterable[str],
    valid_pack_ids: Iterable[str],
    prior_call_ids: dict[str, Any],
) -> dict[str, list[str]]:
    """Pure deterministic resume plan for immutable logical pack IDs."""

    expected = list(map(str, pack_ids))
    if len(expected) != len(set(expected)):
        raise ValueError("execution-pack IDs must be unique")
    valid = set(map(str, valid_pack_ids))
    unknown = valid - set(expected)
    if unknown:
        raise ValueError(f"completed pack set contains unknown IDs: {sorted(unknown)}")
    reusable = {
        str(pack_id)
        for pack_id, call_id in prior_call_ids.items()
        if str(pack_id) in set(expected) and isinstance(call_id, str) and call_id
    }
    pending = [pack_id for pack_id in expected if pack_id not in valid]
    return {
        "skip_valid": [pack_id for pack_id in expected if pack_id in valid],
        "reuse_calls": [pack_id for pack_id in pending if pack_id in reusable],
        "submit": [pack_id for pack_id in pending if pack_id not in reusable],
    }


def _recovery_metadata(
    previous: dict[str, Any], *, now: float
) -> dict[str, Any]:
    """Advance the orchestrator attempt while preserving terminal error history."""

    attempt = int(previous.get("orchestration_attempt", 0)) + 1
    history = list(previous.get("failure_history", []))
    stale_error = previous.get("current_error")
    if stale_error:
        record = {
            "attempt": int(previous.get("orchestration_attempt", max(attempt - 1, 1))),
            "error": str(stale_error),
            "recorded_unix": now,
        }
        if not history or history[-1].get("error") != record["error"]:
            history.append(record)
    return {
        "orchestration_attempt": attempt,
        "failure_history": history,
        "current_error": None,
        "recovered_after_failure": bool(stale_error or history),
    }


def _prepared_stage_result(
    run_id: str,
    source_commit: str,
    source_bundle_sha256: str,
    models: Iterable[str],
) -> dict[str, Any] | None:
    """Return a validated prepared-stage result, or ``None`` if incomplete."""

    try:
        from safeprefix.config import load_config
        from safeprefix.production_suite import validate_prepared_manifests

        config = load_config(_config_path()).data
        validation = validate_prepared_manifests(config, run_id=run_id)
        root = _run_root(run_id)
        artifact_root = root / str(config["artifacts_root"])
        protocol = json.loads(
            (artifact_root / "immutable_manifests/immutable_protocol_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        cpu_smoke = json.loads(
            (artifact_root / "smoke_tests/cpu_non_model.json").read_text(
                encoding="utf-8"
            )
        )
        cache_access = json.loads(
            (root / "orchestration/model_cache_access.json").read_text(encoding="utf-8")
        )
        if not (
            validation.get("passed") is True
            and protocol.get("status") == "PREPARED"
            and protocol.get("source_commit") == source_commit
            and protocol.get("source_bundle_sha256") == source_bundle_sha256
            and cpu_smoke.get("status") == "PASS"
            and all(cpu_smoke.get("checks", {}).values())
            and cache_access.get("status") == "PASS"
            and set(cache_access.get("models", {})) == set(models)
        ):
            return None
        return {
            "status": "PREPARED",
            "models": tuple(models),
            "source_commit": source_commit,
            "source_bundle_sha256": source_bundle_sha256,
            "cached_model_snapshots": cache_access["models"],
            "resumed_from_valid_artifacts": True,
        }
    except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError, json.JSONDecodeError):
        return None


def _validation_summary(run_id: str, model_key: str, mode: str) -> dict[str, Any] | None:
    artifact_root = str(_load_protocol_config()["artifacts_root"])
    path = (
        _run_root(run_id)
        / artifact_root
        / "worker_summaries"
        / mode
        / model_key
        / "worker_00.json"
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return payload if _valid_worker_summary_payload(
        payload, model_key=model_key, mode=mode
    ) else None


def _resume_validation_workers(
    run_id: str, models: Iterable[str], mode: str
) -> dict[str, Any]:
    """Skip valid summaries and reattach to previously submitted child calls."""

    results: dict[str, Any] = {}
    state_key = f"{mode}_call_ids"
    prior = dict(_read_state(run_id).get(state_key, {}))
    calls: dict[str, Any] = {}
    parameters: dict[str, tuple[str, int, str]] = {}
    for model in models:
        summary = _validation_summary(run_id, model, mode)
        if summary is not None:
            results[model] = {**summary, "resumed_from_valid_artifact": True}
            _event(run_id, "VALIDATION_STAGE_SKIPPED_VALID", mode=mode, model_key=model)
            continue
        call_id = prior.get(model)
        if isinstance(call_id, str) and call_id:
            calls[model] = modal.FunctionCall.from_id(call_id)
            _event(
                run_id,
                "VALIDATION_CALL_REATTACHED",
                mode=mode,
                model_key=model,
                call_id=call_id,
            )
        else:
            call = gpu_worker.spawn(run_id, model, 0, mode)
            calls[model] = call
            prior[model] = call.object_id
            _write_state(run_id, **{state_key: prior})
        parameters[model] = (model, 0, mode)
    if calls:
        results.update(
            _wait_calls_with_one_explicit_resume(run_id, calls, parameters)
        )
    # Trust durable artifacts rather than a potentially lost RPC response.
    for model in models:
        summary = _validation_summary(run_id, model, mode)
        if summary is None:
            raise RuntimeError(f"{mode} artifact is missing or invalid for {model}")
        results[model] = summary
    return results


def _validated_production_pack_ids(
    run_id: str, model_key: str
) -> set[str]:
    """Fully validate existing pack artifacts before excluding them from work."""

    from safeprefix.config import load_config
    from safeprefix.production_suite import (
        load_frozen_production_context,
        valid_pack_artifact,
    )

    config = load_config(_config_path()).data
    context = load_frozen_production_context(
        config, run_id=run_id, model_key=model_key
    )
    layers = list(map(int, config["models"][model_key]["selected_hidden_state_layers"]))
    revision = config["models"][model_key].get("revision")
    complete: set[str] = set()
    for pack_id, pack in context["packs"].items():
        trace_rows = {
            str(trace_id): context["trace_rows"][str(trace_id)]
            for trace_id in pack["trace_ids"]
        }
        pack_root = (
            context["root"] / "raw_rollout_shards" / model_key / pack_id
        )
        if valid_pack_artifact(
            pack,
            pack_root,
            trace_rows=trace_rows,
            expected_freeze_digest=context["freeze_digest"],
            expected_settings_hash=str(context["settings"]["settings_hash"]),
            expected_layers=layers,
            expected_model_revision=revision,
        ):
            complete.add(pack_id)
    return complete


def _validated_launcher_freeze(
    run_id: str,
    source_commit: str,
    source_bundle_sha256: str,
    models: Iterable[str],
) -> dict[str, Any] | None:
    path = _run_root(run_id) / "orchestration/production_launch_freeze.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        _validate_production_freeze(run_id, str(payload["freeze_digest"]))
        if not (
            payload.get("source_commit") == source_commit
            and payload.get("source_bundle_sha256") == source_bundle_sha256
            and payload.get("models_in_order") == list(models)
        ):
            raise RuntimeError(
                "existing production freeze belongs to a different source or model matrix"
            )
        return payload
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("existing production launch freeze is malformed") from exc


def _validated_aggregation(run_id: str, models: Iterable[str]) -> dict[str, Any] | None:
    root = _run_root(run_id) / str(_load_protocol_config()["artifacts_root"])
    try:
        integrity = json.loads(
            (root / "integrity_validation_report.json").read_text(encoding="utf-8")
        )
        final = json.loads((root / "final_summary.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if not (
        integrity == final
        and integrity.get("status") == "INTEGRITY_VALIDATED"
        and integrity.get("models_complete") == len(tuple(models))
        and integrity.get("boundary_training_occurred") is False
        and integrity.get("native_final_test_access_count") == 0
        and integrity.get("raw_outcomes_aggregated") is True
        and integrity.get("repairability_discretized") is False
    ):
        return None
    return integrity


def _validated_final_report(run_id: str, aggregation: dict[str, Any]) -> bool:
    path = (
        _run_root(run_id)
        / str(_load_protocol_config()["artifacts_root"])
        / "FINAL_REPORT.md"
    )
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return False
    return bool(
        "COMPLETE — INTEGRITY VALIDATED" in text
        and f"`{aggregation['total_rollouts']}`" in text
        and "Boundary-model training: `not run`" in text
        and "Native evaluation/final-test access: `not run (0 accesses)`" in text
    )


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=4 * 60 * 60,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=5.0),
    volumes={
        "/runs": run_volume,
        "/cache": read_only_cache_volume,
        "/source_runs": read_only_source_volume,
    },
    secrets=hf_secrets,
)
def prepare_run(run_id: str, source_commit: str, source_bundle_sha256: str) -> dict[str, Any]:
    """Freeze cohort/packs, then run all CPU and mock-engine integrity tests."""

    from safeprefix.config import load_config

    run_volume.reload()
    config = load_config(_config_path()).data
    models = _assert_rollout_only_protocol(config)
    cached_snapshots = _validate_cached_model_snapshots(config)
    _atomic_json(
        _run_root(run_id) / "orchestration" / "model_cache_access.json",
        {"status": "PASS", "models": cached_snapshots},
    )
    run_volume.commit()
    remote_bundle_sha256 = _source_bundle_sha256(REMOTE_ROOT)
    if remote_bundle_sha256 != source_bundle_sha256:
        raise RuntimeError(
            "deployed source bundle differs from the submitted local source bundle: "
            f"remote={remote_bundle_sha256} local={source_bundle_sha256}"
        )
    os.environ["SAFEPREFIX_SOURCE_COMMIT"] = source_commit
    os.environ["SAFEPREFIX_SOURCE_BUNDLE_SHA256"] = source_bundle_sha256
    _event(run_id, "PREPARE_STARTED", source_commit=source_commit)
    _run_command(
        _runner_command("prepare", run_id),
        _run_root(run_id) / "launcher_logs" / "00_prepare.log",
    )
    _run_command(
        _runner_command("cpu-smoke", run_id),
        _run_root(run_id) / "launcher_logs" / "01_cpu_smoke.log",
    )
    result = {
        "status": "PREPARED",
        "models": models,
        "source_commit": source_commit,
        "source_bundle_sha256": source_bundle_sha256,
        "cached_model_snapshots": cached_snapshots,
    }
    _event(run_id, "PREPARE_COMPLETE", **result)
    return result


@app.function(
    image=image,
    gpu="H100!",
    cpu=12,
    memory=98304,
    timeout=24 * 60 * 60,
    max_containers=8,
    scaledown_window=2,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=5.0),
    volumes={
        "/runs": run_volume,
        "/cache": read_only_cache_volume,
        "/source_runs": read_only_source_volume,
    },
    secrets=hf_secrets,
)
def gpu_worker(
    run_id: str,
    model_key: str,
    worker_index: int,
    mode: str,
) -> dict[str, Any]:
    """Execute one smoke or benchmark shard; production is pack-dispatched."""

    if mode not in {"smoke", "benchmark"}:
        raise ValueError(f"unsupported gpu-worker mode: {mode}")
    if worker_index < 0 or worker_index >= GPU_WORKERS:
        raise ValueError(f"worker_index must be in [0,{GPU_WORKERS})")
    run_volume.reload()
    models = _assert_rollout_only_protocol(_load_protocol_config())
    if model_key not in models:
        raise ValueError(f"model {model_key!r} is not in configured matrix {models}")
    hardware = _assert_literal_h100()
    task_id = os.environ.get("MODAL_TASK_ID") or uuid.uuid4().hex
    started = time.time()
    _event(
        run_id,
        "GPU_WORKER_STARTED",
        mode=mode,
        model_key=model_key,
        worker_index=worker_index,
        task_id=task_id,
        hardware=hardware,
    )
    command = _runner_command(
        "gpu-worker",
        run_id,
        "--model-key",
        model_key,
        "--worker-index",
        str(worker_index),
        "--mode",
        mode,
    )
    log_path = (
        _run_root(run_id)
        / "launcher_logs"
        / mode
        / model_key
        / f"worker_{worker_index:02d}_{task_id}.log"
    )
    try:
        _run_command(command, log_path)
    except Exception as exc:
        _event(
            run_id,
            "GPU_WORKER_FAILED",
            mode=mode,
            model_key=model_key,
            worker_index=worker_index,
            task_id=task_id,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    result = {
        "status": "COMPLETE",
        "mode": mode,
        "model_key": model_key,
        "worker_index": worker_index,
        "task_id": task_id,
        "elapsed_seconds": time.time() - started,
        "hardware": hardware,
        "log_path": str(log_path),
    }
    _event(run_id, "GPU_WORKER_COMPLETE", **result)
    return result


@app.cls(
    image=image,
    gpu="H100!",
    cpu=12,
    memory=98304,
    timeout=24 * 60 * 60,
    max_containers=GPU_WORKERS,
    # The model-major queue keeps containers busy within a wave; a one-second
    # idle window releases the previous model before the next wave can scale.
    scaledown_window=2,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=5.0),
    volumes={
        "/runs": run_volume,
        "/cache": read_only_cache_volume,
        "/source_runs": read_only_source_volume,
    },
    secrets=hf_secrets,
)
class ProductionPackWorker:
    """Warm model worker dynamically scheduled at immutable-pack granularity.

    Each Modal method input names exactly one pack from the frozen manifest.
    Modal assigns pending inputs to whichever of the eight warm containers is
    free; pack membership, branch seeds, and artifact paths remain immutable.
    """

    run_id: str = modal.parameter()
    model_key: str = modal.parameter()
    freeze_digest: str = modal.parameter()

    @modal.enter()
    def load_frozen_context(self) -> None:
        from safeprefix.config import load_config
        from safeprefix.models.loader import load_model
        from safeprefix.production_suite import (
            load_frozen_production_context,
        )

        _safe_run_id(self.run_id)
        run_volume.reload()
        _validate_production_freeze(self.run_id, self.freeze_digest)
        resolved = load_config(_config_path())
        self.config = resolved.data
        models = _assert_rollout_only_protocol(self.config)
        if self.model_key not in models:
            raise ValueError(
                f"model {self.model_key!r} is not in configured matrix {models}"
            )
        self.context = load_frozen_production_context(
            self.config, run_id=self.run_id, model_key=self.model_key
        )
        self.packs = self.context["packs"]
        self.hardware = _assert_literal_h100()
        # Model loading happens once per warm Modal container, not once per pack.
        self.loaded = load_model(self.config["models"][self.model_key])

    @modal.method()
    def run_pack(self, pack_id: str) -> dict[str, Any]:
        from safeprefix.production_suite import execute_frozen_production_pack

        if pack_id not in self.packs:
            raise ValueError(f"pack {pack_id!r} is not in the frozen model manifest")
        # A retry can land on another warm container. Reload first so a commit
        # whose response was lost is observed and skipped rather than repeated.
        run_volume.reload()
        _validate_production_freeze(self.run_id, self.freeze_digest)
        pack = self.packs[pack_id]
        task_id = os.environ.get("MODAL_TASK_ID") or uuid.uuid4().hex
        started = time.time()
        summary = execute_frozen_production_pack(
            self.config,
            run_id=self.run_id,
            model_key=self.model_key,
            pack_id=pack_id,
            loaded=self.loaded,
            context=self.context,
        )
        return {
            "status": str(summary.get("status", "COMPLETE")),
            "model_key": self.model_key,
            "pack_id": pack_id,
            "pack_hash": str(pack["pack_hash"]),
            "task_id": task_id,
            "elapsed_seconds": time.time() - started,
            "row_count": int(summary.get("row_count", pack["rollout_count"])),
            "hardware": self.hardware,
        }


@app.function(
    image=image,
    cpu=16,
    memory=65536,
    timeout=6 * 60 * 60,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=5.0),
    volumes={
        "/runs": run_volume,
        "/cache": read_only_cache_volume,
        "/source_runs": read_only_source_volume,
    },
    secrets=hf_secrets,
)
def aggregate_results(run_id: str) -> dict[str, Any]:
    """Validate exact coverage and aggregate only raw checkpoint outcomes."""

    run_volume.reload()
    _assert_rollout_only_protocol(_load_protocol_config())
    _run_command(
        _runner_command("aggregate", run_id),
        _run_root(run_id) / "launcher_logs" / "98_aggregate.log",
    )
    _event(run_id, "AGGREGATION_COMPLETE")
    return {"status": "INTEGRITY_VALIDATED", "run_id": run_id}


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=2 * 60 * 60,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=5.0),
    volumes={
        "/runs": run_volume,
        "/cache": read_only_cache_volume,
        "/source_runs": read_only_source_volume,
    },
    secrets=hf_secrets,
)
def generate_report(run_id: str) -> dict[str, Any]:
    """Write Markdown only after aggregation has established final run status."""

    run_volume.reload()
    _assert_rollout_only_protocol(_load_protocol_config())
    state = _read_state(run_id)
    if state.get("status") != "COMPLETE_REPORT_PENDING":
        raise RuntimeError(
            "report generation requires a terminal integrity-validated run state"
        )
    _run_command(
        _runner_command("report", run_id),
        _run_root(run_id) / "launcher_logs" / "99_report.log",
    )
    _event(run_id, "REPORT_COMPLETE")
    return {"status": "COMPLETE", "run_id": run_id}


def _wait_calls_with_one_explicit_resume(
    run_id: str,
    calls: dict[str, Any],
    parameters: dict[str, tuple[str, int, str]],
) -> dict[str, Any]:
    """Wait for validation workers and explicitly retry the same shard once.

    Production retries are handled separately at exact immutable-pack
    granularity; this helper is retained only for smoke and benchmark workers.
    """

    results: dict[str, Any] = {}
    for name, call in calls.items():
        try:
            results[name] = call.get()
        except Exception as first_error:
            _event(
                run_id,
                "EXPLICIT_VALIDATION_RESUME",
                call_name=name,
                failed_call_id=call.object_id,
                error=f"{type(first_error).__name__}: {first_error}",
            )
            model, worker, mode = parameters[name]
            replacement = gpu_worker.spawn(run_id, model, worker, mode)
            state_key = f"{mode}_call_ids"
            call_ids = dict(_read_state(run_id).get(state_key, {}))
            call_ids[name] = replacement.object_id
            _write_state(run_id, **{state_key: call_ids})
            _event(
                run_id,
                "EXPLICIT_VALIDATION_RESUME_SUBMITTED",
                call_name=name,
                replacement_call_id=replacement.object_id,
            )
            results[name] = replacement.get()
            results[name]["recovered_after_failure"] = True
            results[name]["historical_error"] = f"{type(first_error).__name__}: {first_error}"
    return results


def _production_pack_order(run_id: str, model_key: str) -> list[str]:
    """Return frozen pack IDs in deterministic longest-estimated-work order."""

    config = _load_protocol_config()
    artifact_root = str(config["artifacts_root"])
    path = (
        _run_root(run_id)
        / artifact_root
        / "immutable_manifests"
        / "per_model"
        / model_key
        / "execution_packs.jsonl"
    )
    if not path.exists():
        raise FileNotFoundError(f"frozen execution-pack manifest is missing: {path}")
    packs = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pack_ids = [str(pack["pack_id"]) for pack in packs]
    if not packs or len(pack_ids) != len(set(pack_ids)):
        raise RuntimeError(f"empty or duplicate execution-pack manifest for {model_key}")
    ordered = sorted(
        packs,
        key=lambda pack: (-int(pack["estimated_work"]), str(pack["pack_id"])),
    )
    return [str(pack["pack_id"]) for pack in ordered]


def _wait_pack_calls_with_one_explicit_resume(
    run_id: str,
    model_key: str,
    pack_method: Any,
    calls: dict[str, Any],
) -> dict[str, Any]:
    """Wait for pack inputs and retry only the identical failed pack once."""

    results: dict[str, Any] = {}
    for pack_id, call in calls.items():
        try:
            results[pack_id] = call.get()
        except Exception as first_error:
            _event(
                run_id,
                "EXPLICIT_PACK_RESUME",
                model_key=model_key,
                pack_id=pack_id,
                failed_call_id=call.object_id,
                error=f"{type(first_error).__name__}: {first_error}",
            )
            # The first call is terminal before replacement submission. The
            # valid complete marker makes response-loss recovery idempotent.
            replacement = pack_method.spawn(pack_id)
            call_ids = dict(
                _read_state(run_id).get("production_call_ids", {}).get(model_key, {})
            )
            call_ids[pack_id] = replacement.object_id
            all_call_ids = dict(_read_state(run_id).get("production_call_ids", {}))
            all_call_ids[model_key] = call_ids
            _write_state(run_id, production_call_ids=all_call_ids)
            _event(
                run_id,
                "EXPLICIT_PACK_RESUME_SUBMITTED",
                model_key=model_key,
                pack_id=pack_id,
                replacement_call_id=replacement.object_id,
            )
            results[pack_id] = replacement.get()
            results[pack_id]["recovered_after_failure"] = True
            results[pack_id]["historical_error"] = (
                f"{type(first_error).__name__}: {first_error}"
            )
    if set(results) != set(calls):
        raise RuntimeError(f"pack-call accounting drifted for {model_key}")
    return results


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=24 * 60 * 60,
    retries=modal.Retries(max_retries=3, backoff_coefficient=2.0, initial_delay=10.0),
    volumes={
        "/runs": run_volume,
        "/cache": read_only_cache_volume,
        "/source_runs": read_only_source_volume,
    },
    secrets=hf_secrets,
)
def orchestrate(
    run_id: str,
    source_commit: str,
    source_bundle_sha256: str,
    launcher_source_sha256: str,
) -> dict[str, Any]:
    """Run validation, then four model-major dynamic eight-H100 pack waves."""

    _safe_run_id(run_id)
    run_volume.reload()
    deployed_launcher_sha256 = _sha256(
        REMOTE_ROOT / "scripts" / LAUNCHER_NAME
    )
    if launcher_source_sha256 != deployed_launcher_sha256:
        raise RuntimeError(
            "deployed launcher differs from the submitting launcher; redeploy before submit"
        )
    remote_bundle_sha256 = _source_bundle_sha256(REMOTE_ROOT)
    if source_bundle_sha256 != remote_bundle_sha256:
        raise RuntimeError(
            "deployed runner/config/package bundle differs from submitted bundle; "
            "redeploy before submit"
        )
    models = _assert_rollout_only_protocol(_load_protocol_config())
    previous = _read_state(run_id)
    recovery = _recovery_metadata(previous, now=time.time())
    attempt = int(recovery["orchestration_attempt"])
    started = float(previous.get("started_unix") or time.time())
    _write_state(
        run_id,
        status="PREPARING" if attempt == 1 else "RESUMING",
        current_error=recovery["current_error"],
        failure_history=recovery["failure_history"],
        orchestration_attempt=attempt,
        source_commit=source_commit,
        source_bundle_sha256=source_bundle_sha256,
        launcher_source_sha256=launcher_source_sha256,
        modal_orchestrator_task_id=os.environ.get("MODAL_TASK_ID"),
        modal_app_id=os.environ.get("MODAL_APP_ID"),
        rollout_only=True,
        boundary_training_enabled=False,
        native_evaluation_enabled=False,
        started_unix=started,
        finished_unix=None,
    )
    _event(
        run_id,
        "ORCHESTRATION_ATTEMPT_STARTED",
        attempt=attempt,
        prior_status=previous.get("status"),
        recovering=attempt > 1,
    )
    try:
        preparation = _prepared_stage_result(
            run_id, source_commit, source_bundle_sha256, models
        )
        if preparation is None:
            preparation = prepare_run.remote(
                run_id, source_commit, source_bundle_sha256
            )
        else:
            _event(run_id, "PREPARE_SKIPPED_VALID", attempt=attempt)
        _write_state(run_id, status="RUNNING_MODEL_SMOKES", preparation=preparation)

        smoke_results = _resume_validation_workers(run_id, models, "smoke")
        _write_state(
            run_id,
            status="RUNNING_PRODUCTION_BENCHMARKS",
            smoke_results=smoke_results,
            current_error=None,
        )

        benchmark_results = _resume_validation_workers(
            run_id, models, "benchmark"
        )

        # The runner validates every smoke/benchmark artifact, chooses and
        # materializes concrete model-specific production settings, and writes
        # frozen_execution_manifest.json. No production worker is submitted
        # before this command succeeds.
        run_volume.reload()
        freeze = _validated_launcher_freeze(
            run_id, source_commit, source_bundle_sha256, models
        )
        if freeze is None:
            _run_command(
                _runner_command("freeze", run_id),
                _run_root(run_id) / "launcher_logs" / "02_freeze.log",
            )
            # Hash those exact immutable inputs before any production worker is
            # submitted. This seal supplements frozen_execution_manifest.json.
            run_volume.reload()
            freeze = _write_production_freeze(
                run_id,
                source_commit,
                source_bundle_sha256,
                models,
                benchmark_results,
            )
        else:
            _event(
                run_id,
                "PRODUCTION_FREEZE_SKIPPED_VALID",
                freeze_digest=freeze["freeze_digest"],
            )
        _write_state(
            run_id,
            status="PRODUCTION_FROZEN",
            benchmark_results=benchmark_results,
            production_freeze=freeze,
            current_error=None,
        )

        production_results: dict[str, Any] = dict(
            _read_state(run_id).get("production_results", {})
        )
        for model in models:
            _write_state(
                run_id,
                status=f"RUNNING_PRODUCTION_{model}",
                current_model=model,
                current_error=None,
            )
            pack_ids = _production_pack_order(run_id, model)
            valid_before = _validated_production_pack_ids(run_id, model)
            prior_call_ids = dict(
                _read_state(run_id).get("production_call_ids", {}).get(model, {})
            )
            plan = _partition_pack_resume(pack_ids, valid_before, prior_call_ids)
            _event(
                run_id,
                "PRODUCTION_MODEL_RESUME_PLAN",
                model_key=model,
                skip_valid=len(plan["skip_valid"]),
                reuse_calls=len(plan["reuse_calls"]),
                submit=len(plan["submit"]),
            )
            pack_worker = ProductionPackWorker(
                run_id=run_id,
                model_key=model,
                freeze_digest=str(freeze["freeze_digest"]),
            )
            pack_method = pack_worker.run_pack
            # Pack membership is frozen; only assignment timing is dynamic.
            # Long predicted packs are submitted first to minimize final-tail
            # imbalance while Modal work-steals across eight warm containers.
            calls = {
                pack_id: modal.FunctionCall.from_id(prior_call_ids[pack_id])
                for pack_id in plan["reuse_calls"]
            }
            for pack_id in plan["reuse_calls"]:
                _event(
                    run_id,
                    "PRODUCTION_PACK_CALL_REATTACHED",
                    model_key=model,
                    pack_id=pack_id,
                    call_id=prior_call_ids[pack_id],
                )
            for pack_id in plan["submit"]:
                call = pack_method.spawn(pack_id)
                calls[pack_id] = call
                prior_call_ids[pack_id] = call.object_id
                all_call_ids = dict(
                    _read_state(run_id).get("production_call_ids", {})
                )
                all_call_ids[model] = dict(prior_call_ids)
                # Persist each ID immediately, minimizing the only unavoidable
                # spawn-to-ledger crash window exposed by the Modal API.
                _write_state(run_id, production_call_ids=all_call_ids)
            pack_results = (
                _wait_pack_calls_with_one_explicit_resume(
                    run_id, model, pack_method, calls
                )
                if calls
                else {}
            )
            run_volume.reload()
            valid_after = _validated_production_pack_ids(run_id, model)
            missing = [pack_id for pack_id in pack_ids if pack_id not in valid_after]
            if missing:
                _event(
                    run_id,
                    "PRODUCTION_PACK_ARTIFACT_RECOVERY",
                    model_key=model,
                    pack_ids=missing,
                )
                recovery_calls: dict[str, Any] = {}
                for pack_id in missing:
                    call = pack_method.spawn(pack_id)
                    recovery_calls[pack_id] = call
                    prior_call_ids[pack_id] = call.object_id
                    all_call_ids = dict(
                        _read_state(run_id).get("production_call_ids", {})
                    )
                    all_call_ids[model] = dict(prior_call_ids)
                    _write_state(run_id, production_call_ids=all_call_ids)
                pack_results.update(
                    _wait_pack_calls_with_one_explicit_resume(
                        run_id, model, pack_method, recovery_calls
                    )
                )
                run_volume.reload()
                valid_after = _validated_production_pack_ids(run_id, model)
            if valid_after != set(pack_ids):
                raise RuntimeError(
                    f"production pack recovery incomplete for {model}: "
                    f"missing={sorted(set(pack_ids) - valid_after)}"
                )
            production_results[model] = {
                "status": "COMPLETE",
                "scheduler": "modal_dynamic_immutable_pack_dispatch",
                "pack_count": len(pack_ids),
                "skipped_valid_pack_count": len(plan["skip_valid"]),
                "reattached_call_count": len(plan["reuse_calls"]),
                "recovered_pack_ids": sorted(
                    pack_id
                    for pack_id, result in pack_results.items()
                    if result.get("recovered_after_failure")
                ),
            }
            _write_state(
                run_id,
                completed_models=[*production_results],
                production_results=production_results,
                current_error=None,
            )

        _write_state(run_id, status="AGGREGATING", current_model=None, current_error=None)
        run_volume.reload()
        aggregation = _validated_aggregation(run_id, models)
        if aggregation is None:
            aggregate_results.remote(run_id)
            run_volume.reload()
            aggregation = _validated_aggregation(run_id, models)
            if aggregation is None:
                raise RuntimeError("aggregation returned without valid durable artifacts")
        else:
            _event(run_id, "AGGREGATION_SKIPPED_VALID")
        _write_state(
            run_id,
            status="COMPLETE_REPORT_PENDING",
            current_error=None,
            aggregation=aggregation,
            native_final_test_accessed=False,
            boundary_model_training_occurred=False,
        )
        if _validated_final_report(run_id, aggregation):
            reporting = {
                "status": "COMPLETE",
                "run_id": run_id,
                "resumed_from_valid_artifact": True,
            }
            _event(run_id, "REPORT_SKIPPED_VALID")
        else:
            reporting = generate_report.remote(run_id)
        return _write_state(
            run_id,
            status="COMPLETE",
            current_error=None,
            finished_unix=time.time(),
            wall_seconds=time.time() - started,
            production_results=production_results,
            aggregation=aggregation,
            reporting=reporting,
            native_final_test_accessed=False,
            boundary_model_training_occurred=False,
            recovered_after_failure=bool(recovery["recovered_after_failure"]),
        )
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        _event(run_id, "ORCHESTRATION_FAILED", error=message)
        _write_state(
            run_id,
            status="FAILED",
            current_error=message,
            finished_unix=time.time(),
            wall_seconds=time.time() - started,
        )
        raise


@app.function(
    image=image,
    cpu=1,
    memory=1024,
    timeout=300,
    volumes={"/runs": run_volume},
)
def status_remote(run_id: str) -> dict[str, Any]:
    run_volume.reload()
    state = _read_state(run_id)
    return state or {"run_id": run_id, "status": "NOT_FOUND"}


def _source_bundle_paths(root: Path) -> list[Path]:
    paths = [
        root / "configs" / CONFIG_NAME,
        root / "configs" / "models.yaml",
        root / "configs" / "datasets.yaml",
        root / "scripts" / RUNNER_NAME,
        root / "scripts" / LAUNCHER_NAME,
        root / "scripts" / "16_audit_processbench_references.py",
        root / "pyproject.toml",
        *sorted((root / "src" / "safeprefix").rglob("*.py")),
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"source bundle is incomplete: {missing}")
    return paths


def _source_bundle_sha256(root: Path) -> str:
    paths = _source_bundle_paths(root)
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _assert_clean_source_bundle(root: Path) -> None:
    """Require every deployed scientific input to match the recorded commit."""

    protected = _source_bundle_paths(root)
    relative = [path.relative_to(root).as_posix() for path in protected]
    status = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *relative,
        ],
        cwd=root,
        text=True,
    ).strip()
    if status:
        raise RuntimeError(
            "production source bundle is not clean; commit the exact deployed "
            f"files before submission:\n{status}"
        )


@app.local_entrypoint()
def main(
    action: str = "submit",
    run_id: str = "",
    call_id: str = "",
) -> None:
    """Submit/wait/status against the durable deployed application."""

    if action == "status":
        if not run_id:
            raise ValueError("--run-id is required for --action status")
        deployed_status = modal.Function.from_name(APP_NAME, "status_remote")
        print(json.dumps(deployed_status.remote(run_id), indent=2, sort_keys=True))
        return
    if action == "wait":
        if not call_id:
            if not run_id:
                raise ValueError("--call-id or --run-id is required for --action wait")
            submission_path = (
                LOCAL_ROOT
                / "artifacts"
                / "full_teacher_forced_suite"
                / run_id
                / "modal_submission.json"
            )
            call_id = json.loads(submission_path.read_text(encoding="utf-8"))["call_id"]
        result = modal.FunctionCall.from_id(call_id).get()
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if action != "submit":
        raise ValueError("action must be submit, status, or wait")
    if not run_id:
        run_id = time.strftime(
            "safeprefix_full_teacher_forced_%Y%m%dT%H%M%SZ", time.gmtime()
        )
    _safe_run_id(run_id)
    _assert_clean_source_bundle(LOCAL_ROOT)
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=LOCAL_ROOT, text=True
    ).strip()
    source_bundle_sha256 = _source_bundle_sha256(LOCAL_ROOT)
    deployed_orchestrator = modal.Function.from_name(APP_NAME, "orchestrate")
    call = deployed_orchestrator.spawn(
        run_id,
        source_commit,
        source_bundle_sha256,
        LAUNCHER_SOURCE_SHA256,
    )
    submission = {
        "app_name": APP_NAME,
        "function": "orchestrate",
        "call_id": call.object_id,
        "run_id": run_id,
        "source_commit": source_commit,
        "source_bundle_sha256": source_bundle_sha256,
        "launcher_source_sha256": LAUNCHER_SOURCE_SHA256,
        "submitted_unix": time.time(),
        "run_volume": RUN_VOLUME_NAME,
        "source_volume": SOURCE_VOLUME_NAME,
        "source_volume_mount": "/source_runs (read-only)",
        "gpu_requirement": "H100!",
    }
    local_path = (
        LOCAL_ROOT
        / "artifacts"
        / "full_teacher_forced_suite"
        / run_id
        / "modal_submission.json"
    )
    _atomic_json(local_path, submission)
    print(json.dumps(submission, indent=2, sort_keys=True))
    print(
        "Durable submission created. This client may exit; inspect with:\n"
        f"  MODAL_PROFILE=meskmmy python3 -m modal run {Path(__file__).as_posix()} "
        f"--action status --run-id {run_id}\n"
        "Download after completion with:\n"
        f"  MODAL_PROFILE=meskmmy python3 -m modal volume get {RUN_VOLUME_NAME} "
        f"{run_id} ./artifacts/full_teacher_forced_suite/{run_id}"
    )
