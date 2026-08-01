#!/usr/bin/env python3
"""Durable elastic-H100 orchestration for teacher-forced corpus completion."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
from typing import Any

import modal


LOCAL_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/workspace/safeprefix_code")
APP_NAME = "safeprefix-teacher-forced-completion"
RUN_VOLUME_NAME = "safeprefix-full-teacher-forced-runs-v2"
CACHE_VOLUME_NAME = "safeprefix-hf-cache"
SOURCE_VOLUME_NAME = "safeprefix-runs"
CONFIG_NAME = "teacher_forced_completion.yaml"
RUNNER_NAME = "27_complete_teacher_forced_corpora.py"
GPU_WORKERS = 40
COUNT_RECONCILIATION_ROOT = LOCAL_ROOT / "artifacts/count_reconciliation"
COMPLETION_PROTOCOL_ROOT = LOCAL_ROOT / "artifacts/teacher_forced_completion_protocol"

app = modal.App(APP_NAME)
run_volume = modal.Volume.from_name(RUN_VOLUME_NAME, create_if_missing=False, version=2)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=False)
source_volume = modal.Volume.from_name(SOURCE_VOLUME_NAME, create_if_missing=False)
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
    .add_local_file(LOCAL_ROOT / "configs" / CONFIG_NAME, str(REMOTE_ROOT / "configs" / CONFIG_NAME), copy=True)
    .add_local_file(LOCAL_ROOT / "configs/full_teacher_forced_suite.yaml", str(REMOTE_ROOT / "configs/full_teacher_forced_suite.yaml"), copy=True)
    .add_local_file(LOCAL_ROOT / "configs/models.yaml", str(REMOTE_ROOT / "configs/models.yaml"), copy=True)
    .add_local_file(LOCAL_ROOT / "configs/datasets.yaml", str(REMOTE_ROOT / "configs/datasets.yaml"), copy=True)
    .add_local_dir(LOCAL_ROOT / "src/safeprefix", str(REMOTE_ROOT / "src/safeprefix"), copy=True, ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_file(LOCAL_ROOT / "scripts/16_audit_processbench_references.py", str(REMOTE_ROOT / "scripts/16_audit_processbench_references.py"), copy=True)
    .add_local_file(LOCAL_ROOT / "scripts" / RUNNER_NAME, str(REMOTE_ROOT / "scripts" / RUNNER_NAME), copy=True)
    .add_local_file(LOCAL_ROOT / "scripts" / Path(__file__).name, str(REMOTE_ROOT / "scripts" / Path(__file__).name), copy=True)
    .add_local_dir(COUNT_RECONCILIATION_ROOT, str(REMOTE_ROOT / "artifacts/count_reconciliation"), copy=True)
    .add_local_dir(COMPLETION_PROTOCOL_ROOT, str(REMOTE_ROOT / "artifacts/teacher_forced_completion_protocol"), copy=True)
    .add_local_file(LOCAL_ROOT / "pyproject.toml", str(REMOTE_ROOT / "pyproject.toml"), copy=True)
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
            "SAFEPREFIX_RUN_VOLUME": RUN_VOLUME_NAME,
            "SAFEPREFIX_EXPECTED_GPU": "H100",
            "SAFEPREFIX_ROLLOUT_ONLY": "1",
        }
    )
)


def _safe_run_id(run_id: str) -> None:
    if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
        raise ValueError("run_id must be one safe path component")


def _run_root(run_id: str) -> Path:
    _safe_run_id(run_id)
    return Path("/runs") / run_id


def _artifact_root(run_id: str) -> Path:
    return _run_root(run_id) / "artifacts/teacher_forced_completion"


def _state_path(run_id: str) -> Path:
    return _run_root(run_id) / "orchestration/remote_run_state.json"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)

def _write_state(run_id: str, **updates: Any) -> dict[str, Any]:
    path = _state_path(run_id)
    prior = json.loads(path.read_text()) if path.exists() else {}
    history = list(prior.get("failure_history", []))
    if prior.get("current_error") and updates.get("current_error") is None:
        history.append(
            {
                "error": str(prior["current_error"]),
                "recorded_unix": float(prior.get("updated_unix", time.time())),
            }
        )
    payload = {
        **prior,
        **updates,
        "failure_history": history,
        "updated_unix": time.time(),
    }
    _atomic_json(path, payload)
    run_volume.commit()
    return payload


def _event(run_id: str, event: str, **details: Any) -> None:
    path = _run_root(run_id) / "orchestration/events" / f"{time.time_ns()}-{uuid.uuid4().hex}.json"
    _atomic_json(path, {"event": event, "time_unix": time.time(), **details})
    run_volume.commit()


def _config_path() -> Path:
    return REMOTE_ROOT / "configs" / CONFIG_NAME


def _runner_command(command: str, run_id: str, *args: str) -> list[str]:
    return [
        "python3",
        str(REMOTE_ROOT / "scripts" / RUNNER_NAME),
        command,
        "--config",
        str(_config_path()),
        "--run-id",
        run_id,
        *args,
    ]


def _run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=os.environ.copy())
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    if code:
        raise RuntimeError(f"command failed with exit code {code}: {' '.join(command)}")


def _source_digest() -> str:
    digest = hashlib.sha256()
    roots = [
        REMOTE_ROOT / "src/safeprefix",
        REMOTE_ROOT / "configs" / CONFIG_NAME,
        REMOTE_ROOT / "configs/full_teacher_forced_suite.yaml",
        REMOTE_ROOT / "scripts" / RUNNER_NAME,
        REMOTE_ROOT / "scripts" / Path(__file__).name,
        REMOTE_ROOT / "artifacts/teacher_forced_completion_protocol",
        REMOTE_ROOT / "artifacts/count_reconciliation",
    ]
    files = []
    for root in roots:
        files.extend([root] if root.is_file() else [path for path in root.rglob("*") if path.is_file()])
    for path in sorted(files, key=str):
        digest.update(str(path.relative_to(REMOTE_ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


@app.function(
    image=image,
    cpu=16,
    memory=65536,
    timeout=4 * 60 * 60,
    volumes={"/runs": run_volume, "/cache": cache_volume.with_mount_options(read_only=True), "/source_runs": source_volume.with_mount_options(read_only=True)},
    secrets=hf_secrets,
)
def prepare_run(run_id: str, source_commit: str, expected_source_digest: str) -> dict[str, Any]:
    run_volume.reload()
    actual = _source_digest()
    if actual != expected_source_digest:
        raise RuntimeError(f"deployed source digest differs: {actual} != {expected_source_digest}")
    os.environ["SAFEPREFIX_SOURCE_COMMIT"] = source_commit
    _run_command(_runner_command("prepare", run_id), _run_root(run_id) / "launcher_logs/00_prepare.log")
    _run_command(_runner_command("cpu-smoke", run_id), _run_root(run_id) / "launcher_logs/01_cpu_smoke.log")
    result = {"status": "PASS", "source_commit": source_commit, "source_digest": actual}
    _event(run_id, "PREPARE_AND_CPU_SMOKE_COMPLETE", **result)
    return result


@app.function(
    image=image,
    gpu="H100!",
    cpu=12,
    memory=98304,
    timeout=4 * 60 * 60,
    max_containers=4,
    volumes={"/runs": run_volume, "/cache": cache_volume.with_mount_options(read_only=True), "/source_runs": source_volume.with_mount_options(read_only=True)},
    secrets=hf_secrets,
)
def real_model_smoke(run_id: str, model_key: str) -> dict[str, Any]:
    run_volume.reload()
    _run_command(
        _runner_command("gpu-smoke", run_id, "--model-key", model_key),
        _run_root(run_id) / f"launcher_logs/smoke/{model_key}.log",
    )
    summary = json.loads((_artifact_root(run_id) / f"smoke_tests/real_model/{model_key}/summary.json").read_text())
    if summary.get("status") != "PASS":
        raise RuntimeError(f"{model_key}: real-model smoke failed")
    return summary


@app.cls(
    image=image,
    gpu="H100!",
    cpu=12,
    memory=98304,
    timeout=24 * 60 * 60,
    max_containers=GPU_WORKERS,
    scaledown_window=300,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=5.0),
    volumes={"/runs": run_volume, "/cache": cache_volume.with_mount_options(read_only=True), "/source_runs": source_volume.with_mount_options(read_only=True)},
    secrets=hf_secrets,
)
class CompletionPackWorker:
    run_id: str = modal.parameter()
    model_key: str = modal.parameter()

    @modal.enter()
    def load(self) -> None:
        from safeprefix.config import load_config
        from safeprefix.models.loader import load_model
        from safeprefix.teacher_forced_completion import load_completion_context

        run_volume.reload()
        self.config = load_config(_config_path()).data
        self.context = load_completion_context(self.config, run_id=self.run_id, model_key=self.model_key)
        self.loaded = load_model(self.config["models"][self.model_key])

    @modal.method()
    def run_repair(self, pack_id: str) -> dict[str, Any]:
        from safeprefix.teacher_forced_completion import execute_extension_pack

        run_volume.reload()
        return execute_extension_pack(
            self.config,
            run_id=self.run_id,
            model_key=self.model_key,
            pack_id=pack_id,
            loaded=self.loaded,
            context=self.context,
        )

    @modal.method()
    def run_safety(self, pack_id: str) -> dict[str, Any]:
        from safeprefix.teacher_forced_completion import execute_safety_pack

        run_volume.reload()
        return execute_safety_pack(
            self.config,
            run_id=self.run_id,
            model_key=self.model_key,
            pack_id=pack_id,
            loaded=self.loaded,
            context=self.context,
        )

    @modal.method()
    def run_assigned_workload(self, worker_slot: int) -> dict[str, Any]:
        """Run one frozen LPT shard while keeping exactly one model resident."""
        from safeprefix.teacher_forced_completion import (
            completion_worker_allocation,
            execute_extension_pack,
            execute_safety_pack,
        )

        allocation = completion_worker_allocation(self.config)
        expected_slots = allocation[self.model_key]
        if not 0 <= int(worker_slot) < expected_slots:
            raise RuntimeError(
                f"{self.model_key}: invalid worker slot {worker_slot}/{expected_slots}"
            )
        repair_ids = self.context["repair_assignments"].get(int(worker_slot), [])
        safety_ids = self.context["safety_assignments"].get(int(worker_slot), [])
        if set(self.context["repair_assignments"]) != set(range(expected_slots)):
            raise RuntimeError(f"{self.model_key}: repair assignment slots differ")
        if set(self.context["safety_assignments"]) != set(range(expected_slots)):
            raise RuntimeError(f"{self.model_key}: safety assignment slots differ")

        started = time.time()
        repair_results = []
        for pack_id in repair_ids:
            run_volume.reload()
            repair_results.append(
                execute_extension_pack(
                    self.config,
                    run_id=self.run_id,
                    model_key=self.model_key,
                    pack_id=pack_id,
                    loaded=self.loaded,
                    context=self.context,
                )
            )
        safety_results = []
        for pack_id in safety_ids:
            run_volume.reload()
            safety_results.append(
                execute_safety_pack(
                    self.config,
                    run_id=self.run_id,
                    model_key=self.model_key,
                    pack_id=pack_id,
                    loaded=self.loaded,
                    context=self.context,
                )
            )
        return {
            "status": "COMPLETE",
            "model_key": self.model_key,
            "worker_slot": int(worker_slot),
            "repair_pack_ids": repair_ids,
            "safety_pack_ids": safety_ids,
            "repair_packs": len(repair_results),
            "safety_packs": len(safety_results),
            "elapsed_seconds": time.time() - started,
        }


@app.function(
    image=image,
    cpu=16,
    memory=65536,
    timeout=4 * 60 * 60,
    volumes={"/runs": run_volume, "/cache": cache_volume.with_mount_options(read_only=True), "/source_runs": source_volume.with_mount_options(read_only=True)},
    secrets=hf_secrets,
)
def aggregate_run(run_id: str) -> dict[str, Any]:
    run_volume.reload()
    _write_state(run_id, status="AGGREGATING", current_error=None)
    try:
        _run_command(
            _runner_command("aggregate", run_id),
            _run_root(run_id) / "launcher_logs/99_aggregate.log",
        )
        final = json.loads(
            (_artifact_root(run_id) / "final_summary.json").read_text()
        )
        _write_state(
            run_id,
            status="COMPLETE",
            integrity_status=final["status"],
            final_summary=final,
            current_error=None,
        )
        _event(run_id, "AGGREGATION_COMPLETE", integrity_status=final["status"])
        return final
    except Exception as exc:
        _write_state(
            run_id,
            status="FAILED",
            current_error=f"{type(exc).__name__}: {exc}",
        )
        _event(run_id, "AGGREGATION_FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=24 * 60 * 60,
    volumes={"/runs": run_volume, "/cache": cache_volume.with_mount_options(read_only=True), "/source_runs": source_volume.with_mount_options(read_only=True)},
    secrets=hf_secrets,
)
def orchestrate(run_id: str, source_commit: str, source_digest: str) -> dict[str, Any]:
    from safeprefix.config import load_config
    from safeprefix.teacher_forced_completion import (
        completion_worker_allocation,
        load_completion_context,
    )

    _write_state(
        run_id,
        status="PREPARING",
        source_commit=source_commit,
        source_digest=source_digest,
        boundary_training_enabled=False,
        native_evaluation_enabled=False,
        final_test_access_enabled=False,
        current_error=None,
    )
    try:
        prepared = prepare_run.remote(run_id, source_commit, source_digest)
        config = load_config(_config_path()).data
        models = list(map(str, config["selected_models"]))
        _write_state(run_id, status="REAL_MODEL_SMOKES", prepared=prepared)
        smoke_calls = {model: real_model_smoke.spawn(run_id, model) for model in models}
        smokes = {model: call.get() for model, call in smoke_calls.items()}
        allocation = completion_worker_allocation(config)
        _write_state(
            run_id,
            status="PRODUCTION",
            smokes=smokes,
            gpu_worker_allocation=allocation,
            scheduler_policy="concurrent_model_weighted_lpt_pack_assignment_v1",
        )
        worker_calls: dict[tuple[str, int], Any] = {}
        for model_key in models:
            run_volume.reload()
            context = load_completion_context(config, run_id=run_id, model_key=model_key)
            expected_repair = set(context["repair_packs"])
            assigned_repair = {
                pack_id for values in context["repair_assignments"].values() for pack_id in values
            }
            expected_safety = set(context["safety_packs"])
            assigned_safety = {
                pack_id for values in context["safety_assignments"].values() for pack_id in values
            }
            if expected_repair != assigned_repair or expected_safety != assigned_safety:
                raise RuntimeError(f"{model_key}: frozen worker assignments are not exhaustive")
            worker = CompletionPackWorker(run_id=run_id, model_key=model_key)
            for worker_slot in range(allocation[model_key]):
                worker_calls[(model_key, worker_slot)] = worker.run_assigned_workload.spawn(
                    worker_slot
                )

        slot_results = {
            key: call.get() for key, call in worker_calls.items()
        }
        model_results: dict[str, Any] = {}
        for model_key in models:
            per_slot = [
                slot_results[(model_key, slot)]
                for slot in range(allocation[model_key])
            ]
            model_results[model_key] = {
                "status": "COMPLETE",
                "gpu_workers": allocation[model_key],
                "repair_packs": sum(int(row["repair_packs"]) for row in per_slot),
                "safety_packs": sum(int(row["safety_packs"]) for row in per_slot),
                "maximum_worker_elapsed_seconds": max(
                    float(row["elapsed_seconds"]) for row in per_slot
                ),
                "worker_results": per_slot,
            }
        _write_state(run_id, status="PRODUCTION", models_complete=model_results)
        _write_state(run_id, status="AGGREGATING", models_complete=model_results)
        final = aggregate_run.remote(run_id)
        return _write_state(
            run_id,
            status="COMPLETE",
            integrity_status=final["status"],
            models_complete=model_results,
            final_summary=final,
            current_error=None,
        )
    except Exception as exc:
        _write_state(run_id, status="FAILED", current_error=f"{type(exc).__name__}: {exc}")
        _event(run_id, "ORCHESTRATION_FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


@app.function(image=image, cpu=2, memory=4096, volumes={"/runs": run_volume})
def status_run(run_id: str) -> dict[str, Any]:
    run_volume.reload()
    state = json.loads(_state_path(run_id).read_text()) if _state_path(run_id).exists() else {"status": "NOT_FOUND"}
    root = _artifact_root(run_id)
    state["observed_complete_repair_packs"] = len(list((root / "repairability/raw_rollout_shards").glob("*/*/complete.json")))
    state["observed_complete_safety_packs"] = len(list((root / "safety/hidden_state_feature_shards").glob("*/*/complete.json")))
    return state


def _local_source_digest() -> str:
    digest = hashlib.sha256()
    roots = [
        LOCAL_ROOT / "src/safeprefix",
        LOCAL_ROOT / "configs" / CONFIG_NAME,
        LOCAL_ROOT / "configs/full_teacher_forced_suite.yaml",
        LOCAL_ROOT / "scripts" / RUNNER_NAME,
        LOCAL_ROOT / "scripts" / Path(__file__).name,
        COMPLETION_PROTOCOL_ROOT,
        COUNT_RECONCILIATION_ROOT,
    ]
    files = []
    for root in roots:
        files.extend([root] if root.is_file() else [path for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts])
    for path in sorted(files, key=lambda item: item.relative_to(LOCAL_ROOT).as_posix()):
        digest.update(str(path.relative_to(LOCAL_ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


@app.local_entrypoint()
def main(
    action: str = "submit",
    run_id: str = "safeprefix_teacher_forced_completion_20260727_r1",
) -> None:
    _safe_run_id(run_id)
    if action == "submit":
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=LOCAL_ROOT, text=True).strip()
        digest = _local_source_digest()
        deployed = modal.Function.from_name(APP_NAME, "orchestrate")
        call = deployed.spawn(run_id, commit, digest)
        submission = {
            "status": "SUBMITTED",
            "app_name": APP_NAME,
            "function": "orchestrate",
            "run_id": run_id,
            "function_call_id": call.object_id,
            "source_commit": commit,
            "source_digest": digest,
            "run_volume": RUN_VOLUME_NAME,
            "gpu_pool": "40 x H100 maximum (5/5/10/20 by measured model work)",
        }
        local = LOCAL_ROOT / "artifacts/teacher_forced_completion" / run_id / "modal_submission.json"
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(json.dumps(submission, indent=2, sort_keys=True) + "\n")
        print(json.dumps(submission, indent=2))
    elif action == "status":
        deployed = modal.Function.from_name(APP_NAME, "status_run")
        print(json.dumps(deployed.remote(run_id), indent=2))
    elif action == "aggregate":
        print(json.dumps(aggregate_run.remote(run_id), indent=2))
    else:
        raise ValueError("action must be submit, status, or aggregate")
