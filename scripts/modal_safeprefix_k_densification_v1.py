#!/usr/bin/env python3
"""Ten-worker dynamic Modal runner for calibration K=32 confirmation."""

import gc
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
from typing import Any

import modal


LOCAL_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/workspace/aaai_monkey")
APP_NAME = "safeprefix-k-densification-v1"
OUTPUT_VOLUME_NAME = os.environ.get(
    "SAFEPREFIX_K_DENSIFICATION_OUTPUT_VOLUME", "safeprefix-k-densification-v1"
)
CACHE_VOLUME_NAME = os.environ.get("SAFEPREFIX_HF_CACHE_VOLUME", "safeprefix-hf-cache")
HF_SECRET_NAME = os.environ.get("SAFEPREFIX_HF_SECRET_NAME", "huggingface-token")
CONFIG_NAME = "k_densification_v1.yaml"
LAUNCHER_NAME = Path(__file__).name
MODEL_KEYS = ("family_a_small", "family_a_large", "family_b_small", "family_b_large")
INITIAL_AFFINITY = (
    "family_a_small",
    "family_a_small",
    "family_a_large",
    "family_a_large",
    "family_a_large",
    "family_b_small",
    "family_b_small",
    "family_b_large",
    "family_b_large",
    "family_b_large",
)

app = modal.App(APP_NAME)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME)

SOURCE_FILES = (
    "src/safeprefix/__init__.py",
    "src/safeprefix/config.py",
    "src/safeprefix/manifests.py",
    "src/safeprefix/reproducibility.py",
    "src/safeprefix/models/__init__.py",
    "src/safeprefix/models/cache_checkpoint.py",
    "src/safeprefix/models/cache_utils.py",
    "src/safeprefix/models/generation.py",
    "src/safeprefix/models/loader.py",
    "src/safeprefix/models/production_backend.py",
    "src/safeprefix/models/teacher_forcing.py",
    "src/safeprefix/parsing/__init__.py",
    "src/safeprefix/parsing/answer_parsers.py",
    "src/safeprefix/rollout/__init__.py",
    "src/safeprefix/rollout/production_engine.py",
    "src/safeprefix/rollout/verifier.py",
    "src/safeprefix/threshold_selection_tf_v1/__init__.py",
    "src/safeprefix/threshold_selection_tf_v1/data.py",
    "src/safeprefix/threshold_selection_tf_v1/runtime.py",
    "src/safeprefix/k_densification_v1/__init__.py",
    "src/safeprefix/k_densification_v1/calibration.py",
    "src/safeprefix/k_densification_v1/analysis.py",
    "src/safeprefix/k_densification_v1/preflight.py",
    "src/safeprefix/k_densification_v1/registry.py",
    "src/safeprefix/k_densification_v1/runtime.py",
    f"configs/{CONFIG_NAME}",
    "configs/models.yaml",
    f"scripts/{LAUNCHER_NAME}",
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "accelerate==1.10.1",
        "datasets==5.0.0",
        "huggingface-hub==0.34.4",
        "numpy==2.2.6",
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
    .env(
        {
            "PYTHONPATH": str(REMOTE_ROOT / "src"),
            "PYTHONUNBUFFERED": "1",
            "HF_HOME": "/cache/huggingface",
            "HF_HUB_CACHE": "/cache/huggingface/hub",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "SAFEPREFIX_K_DENSIFICATION_OUTPUT_VOLUME": OUTPUT_VOLUME_NAME,
        }
    )
)
for relative in SOURCE_FILES:
    image = image.add_local_file(LOCAL_ROOT / relative, str(REMOTE_ROOT / relative), copy=True)

WORKER_VOLUMES = {
    "/kdens": output_volume,
    "/cache": cache_volume.with_mount_options(read_only=True),
}
OUTPUT_VOLUMES = {"/kdens": output_volume}
CACHE_VOLUMES = {"/cache": cache_volume}
prefetch_image = image.env({"HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"})


def _safe_run_id(run_id: str) -> None:
    if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
        raise ValueError("run_id must be one safe component")


def _artifact_root(run_id: str) -> Path:
    _safe_run_id(run_id)
    return Path("/kdens") / run_id / "outputs/k_densification_v1"


def _config_path() -> Path:
    return REMOTE_ROOT / "configs" / CONFIG_NAME


def _source_digest() -> str:
    digest = hashlib.sha256()
    for relative in SOURCE_FILES:
        path = LOCAL_ROOT / relative
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _remote_source_digest() -> str:
    digest = hashlib.sha256()
    for relative in SOURCE_FILES:
        path = REMOTE_ROOT / relative
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _assert_source(expected: str) -> None:
    observed = _remote_source_digest()
    if observed != expected:
        raise RuntimeError(f"deployed source digest differs: {observed} != {expected}")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _assert_h100() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in K-densification worker")
    names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    if not names or any("H100" not in name.upper() for name in names):
        raise RuntimeError(f"literal H100 required; got {names}")
    return {"device_names": names, "torch": torch.__version__, "cuda": torch.version.cuda}


@app.function(
    image=prefetch_image,
    cpu=8,
    memory=32768,
    timeout=4 * 60 * 60,
    max_containers=1,
    volumes=CACHE_VOLUMES,
    secrets=[hf_secret],
)
def prefetch_models_remote(source_digest: str) -> dict[str, Any]:
    from huggingface_hub import snapshot_download
    from safeprefix.config import load_config

    _assert_source(source_digest)
    cache_volume.reload()
    config = load_config(_config_path()).data
    downloaded = {}
    for model_key in MODEL_KEYS:
        entry = config["models"][model_key]
        path = snapshot_download(
            repo_id=str(entry["hf_model_id"]),
            revision=str(entry["revision"]),
            cache_dir="/cache/huggingface/hub",
            local_files_only=False,
        )
        if Path(path).name != str(entry["revision"]):
            raise RuntimeError(f"{model_key}: cached revision differs")
        downloaded[model_key] = {
            "model_id": entry["hf_model_id"],
            "revision": entry["revision"],
            "snapshot_path": path,
        }
        cache_volume.commit()
    return {"status": "COMPLETE", "models": downloaded, "native_volume_mounted": False}


@app.cls(
    image=image,
    cpu=4,
    memory=16384,
    timeout=24 * 60 * 60,
    max_containers=1,
    volumes=OUTPUT_VOLUMES,
)
class RegistryCoordinator:
    run_id: str = modal.parameter()

    @modal.enter()
    def enter(self) -> None:
        from safeprefix.k_densification_v1.registry import connect_registry
        import pandas as pd

        output_volume.reload()
        self.root = _artifact_root(self.run_id)
        required = (
            self.root / "k32_rollout_registry.sqlite",
            self.root / "k32_confirmation_checkpoints.parquet",
            self.root / "generation_input/k32_generation_manifest.parquet",
        )
        if not all(path.is_file() for path in required):
            raise RuntimeError("staged K32 registry/input bundle is incomplete")
        self.connection = connect_registry(required[0])
        self.checkpoints = pd.read_parquet(required[1]).set_index("checkpoint_key")
        self.generation = pd.read_parquet(required[2])

    @modal.method()
    def initialize(self, source_digest: str, recover_orphaned: bool = False) -> dict[str, Any]:
        _assert_source(source_digest)
        recovered = 0
        if recover_orphaned:
            cursor = self.connection.execute(
                """UPDATE jobs SET status='pending', worker_id=NULL,
                          lease_expires_unix=NULL, claim_unix=NULL,
                          error_message='orchestrator_restart_requeued'
                   WHERE status='running'"""
            )
            recovered = int(cursor.rowcount)
            output_volume.commit()
        counts = {
            "checkpoints": self.connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0],
            "slots": self.connection.execute("SELECT COUNT(*) FROM rollout_slots").fetchone()[0],
            "jobs": self.connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0],
        }
        if counts["checkpoints"] != 256 or counts["jobs"] != 1024 or counts["slots"] < 4096:
            raise RuntimeError(f"staged registry counts differ: {counts}")
        return {"status": "READY", "recovered_orphaned_jobs": recovered, **counts}

    @modal.method()
    def transition(
        self,
        worker_id: str,
        from_model_key: str | None,
        to_model_key: str,
        reason: str,
    ) -> None:
        from safeprefix.k_densification_v1.registry import register_transition

        register_transition(
            self.connection,
            worker_id=worker_id,
            from_model_key=from_model_key,
            to_model_key=to_model_key,
            reason=reason,
        )
        output_volume.commit()

    @modal.method()
    def claim(self, worker_id: str, preferred_model_key: str) -> dict[str, Any]:
        from safeprefix.k_densification_v1.registry import claim_checkpoint_jobs

        claimed = claim_checkpoint_jobs(
            self.connection,
            worker_id=worker_id,
            preferred_model_key=preferred_model_key,
            lease_seconds=3600.0,
        )
        output_volume.commit()
        if claimed is None:
            running = self.connection.execute(
                "SELECT COUNT(*) FROM jobs WHERE status='running'"
            ).fetchone()[0]
            pending = self.connection.execute(
                "SELECT COUNT(*) FROM jobs WHERE status='pending'"
            ).fetchone()[0]
            return {"status": "WAIT" if running or pending else "DONE", "running": running, "pending": pending}
        checkpoint_key = str(claimed[0]["checkpoint_key"])
        checkpoint = self.checkpoints.loc[checkpoint_key].to_dict()
        claimed_slots = {
            slot
            for job in claimed
            for slot in range(int(job["block_start"]), int(job["block_end"]) + 1)
        }
        generation = self.generation.loc[
            self.generation["checkpoint_key"].astype(str).eq(checkpoint_key)
            & self.generation["rollout_slot"].astype(int).isin(claimed_slots)
        ].to_dict("records")
        return {
            "status": "CLAIMED",
            "jobs": claimed,
            "checkpoint": {"checkpoint_key": checkpoint_key, **checkpoint},
            "generation_rows": generation,
        }

    @modal.method()
    def complete(
        self,
        job_id: str,
        worker_id: str,
        rows: list[dict[str, Any]],
        artifact_path: str,
    ) -> None:
        from safeprefix.k_densification_v1.registry import complete_job

        complete_job(
            self.connection,
            job_id=job_id,
            worker_id=worker_id,
            rows=rows,
            artifact_path=artifact_path,
        )
        output_volume.commit()

    @modal.method()
    def fail(self, job_id: str, worker_id: str, error: str) -> None:
        from safeprefix.k_densification_v1.registry import fail_job

        fail_job(self.connection, job_id=job_id, worker_id=worker_id, error=error)
        output_volume.commit()

    @modal.method()
    def status(self) -> dict[str, Any]:
        counts = {
            row[0]: int(row[1])
            for row in self.connection.execute("SELECT status,COUNT(*) FROM jobs GROUP BY status")
        }
        workloads = {
            row[0]: float(row[1])
            for row in self.connection.execute(
                "SELECT model_key,COALESCE(SUM(estimated_token_work),0) FROM jobs WHERE status='pending' GROUP BY model_key"
            )
        }
        return {
            "job_status": counts,
            "remaining_estimated_token_work": workloads,
            "registered_slots": int(
                self.connection.execute("SELECT COUNT(*) FROM rollout_slots").fetchone()[0]
            ),
        }


def _load_sources(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    sources = {}
    for model_key in MODEL_KEYS:
        path = root / f"generation_input/source_traces/{model_key}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        sources[model_key] = {str(row["trace_id"]): row for row in rows}
    return sources


@app.function(
    image=image,
    gpu="H100!",
    cpu=12,
    memory=98304,
    timeout=24 * 60 * 60,
    max_containers=10,
    scaledown_window=300,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=10.0),
    volumes=WORKER_VOLUMES,
)
def dynamic_worker(
    run_id: str,
    worker_id: str,
    initial_model_key: str,
    source_digest: str,
) -> dict[str, Any]:
    import torch
    from safeprefix.config import load_config
    from safeprefix.k_densification_v1.runtime import execute_k32_checkpoint_bundle
    from safeprefix.models.loader import load_model

    _assert_source(source_digest)
    output_volume.reload()
    hardware = _assert_h100()
    root = _artifact_root(run_id)
    config = load_config(_config_path()).data
    sources = _load_sources(root)
    coordinator = RegistryCoordinator(run_id=run_id)
    current_model: str | None = None
    loaded = None
    preferred = initial_model_key
    transitions: list[dict[str, Any]] = []
    completed = 0
    failures = 0
    model_seconds = {key: 0.0 for key in MODEL_KEYS}
    started = time.time()
    while True:
        claim = coordinator.claim.remote(worker_id, preferred)
        if claim["status"] == "DONE":
            break
        if claim["status"] == "WAIT":
            time.sleep(5)
            continue
        jobs = claim["jobs"]
        model_key = str(jobs[0]["model_key"])
        if current_model != model_key:
            previous = current_model
            if loaded is not None:
                del loaded
                loaded = None
                gc.collect()
                torch.cuda.empty_cache()
            switch_started = time.time()
            loaded = load_model(config["models"][model_key])
            transition = {
                "worker_id": worker_id,
                "from_model_key": previous,
                "to_model_key": model_key,
                "reason": "initial_affinity" if previous is None else "current_queue_drained_spillover",
                "transition_unix": time.time(),
                "load_seconds": time.time() - switch_started,
            }
            coordinator.transition.remote(
                worker_id,
                previous,
                model_key,
                transition["reason"],
            )
            transitions.append(transition)
            current_model = model_key
        preferred = model_key
        job_started = time.time()
        try:
            source = sources[model_key][str(claim["checkpoint"]["trace_id"])]
            result = execute_k32_checkpoint_bundle(
                config,
                artifact_root=root,
                jobs=jobs,
                checkpoint=claim["checkpoint"],
                generation_rows=claim["generation_rows"],
                source_trace=source,
                loaded=loaded,
            )
            for job in jobs:
                job_result = result["job_results"][str(job["job_id"])]
                block_root = (
                    root
                    / "raw_blocks"
                    / model_key
                    / str(job["checkpoint_key"])
                    / f"slots-{int(job['block_start']):02d}-{int(job['block_end']):02d}"
                )
                coordinator.complete.remote(
                    str(job["job_id"]),
                    worker_id,
                    job_result["rows"],
                    str(block_root / "outcomes.parquet"),
                )
                completed += 1
            model_seconds[model_key] += time.time() - job_started
        except Exception as exc:
            failures += len(jobs)
            for job in jobs:
                coordinator.fail.remote(
                    str(job["job_id"]), worker_id, f"{type(exc).__name__}: {exc}"
                )
            if failures >= 5:
                raise
    payload = {
        "status": "COMPLETE",
        "worker_id": worker_id,
        "initial_model_key": initial_model_key,
        "completed_jobs": completed,
        "failures_requeued": failures,
        "transitions": transitions,
        "model_active_seconds": model_seconds,
        "wall_seconds": time.time() - started,
        "hardware": hardware,
    }
    _atomic_json(root / f"workers/{worker_id}.json", payload)
    output_volume.commit()
    return payload


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=24 * 60 * 60,
    max_containers=1,
    volumes=OUTPUT_VOLUMES,
)
def orchestrate_remote(
    run_id: str,
    source_digest: str,
    workspace_name: str,
    recover_orphaned: bool = False,
) -> dict[str, Any]:
    output_volume.reload()
    coordinator = RegistryCoordinator(run_id=run_id)
    initialized = coordinator.initialize.remote(source_digest, recover_orphaned)
    state_path = _artifact_root(run_id) / "orchestration_state.json"
    started = time.time()
    calls = {}
    for index, model_key in enumerate(INITIAL_AFFINITY):
        worker_id = f"gpu-worker-{index:02d}"
        call = dynamic_worker.spawn(run_id, worker_id, model_key, source_digest)
        calls[worker_id] = call
    _atomic_json(
        state_path,
        {
            "status": "GENERATION_RUNNING",
            "run_id": run_id,
            "workspace_name": workspace_name,
            "source_digest": source_digest,
            "initialized": initialized,
            "worker_call_ids": {key: call.object_id for key, call in calls.items()},
            "started_unix": started,
            "native_volume_mounted": False,
        },
    )
    output_volume.commit()
    results = {key: call.get() for key, call in calls.items()}
    status = coordinator.status.remote()
    if status["job_status"] != {"complete": 1024} or status["registered_slots"] != 8192:
        raise RuntimeError(f"terminal registry status differs: {status}")
    payload = {
        "status": "GENERATION_COMPLETE",
        "run_id": run_id,
        "workspace_name": workspace_name,
        "source_digest": source_digest,
        "worker_results": results,
        "registry_status": status,
        "wall_seconds": time.time() - started,
        "completed_unix": time.time(),
        "native_volume_mounted": False,
    }
    _atomic_json(state_path, payload)
    output_volume.commit()
    return payload


@app.function(image=image, cpu=8, memory=32768, timeout=2 * 60 * 60, volumes=OUTPUT_VOLUMES)
def finalize_generation_remote(run_id: str, source_digest: str) -> dict[str, Any]:
    import hashlib
    import pandas as pd
    import sqlite3
    from safeprefix.reproducibility import atomic_json, atomic_parquet

    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    _assert_source(source_digest)
    output_volume.reload()
    root = _artifact_root(run_id)
    state = json.loads((root / "orchestration_state.json").read_text())
    if state.get("status") != "GENERATION_COMPLETE":
        raise RuntimeError("generation is not complete")
    block_paths = sorted(root.glob("raw_blocks/*/*/slots-*/outcomes.parquet"))
    if len(block_paths) != 1024:
        raise RuntimeError(f"completed block count {len(block_paths)} != 1024")
    frames = [pd.read_parquet(path) for path in block_paths]
    generated = pd.concat(frames, ignore_index=True)
    if len(generated) != 4096 or generated.duplicated(["checkpoint_key", "rollout_index"]).any():
        raise RuntimeError("generated K32 rows are incomplete or duplicated")
    if set(generated["rollout_index"].astype(int)) != set(range(16, 32)):
        raise RuntimeError("generated slot set differs")
    counts = generated.groupby("checkpoint_key")["rollout_index"].nunique()
    if len(counts) != 256 or set(counts.astype(int)) != {16}:
        raise RuntimeError("not every confirmation checkpoint has 16 new slots")
    generated_path = root / "generated_slots_16_31.parquet"
    atomic_parquet(generated_path, generated.sort_values(["checkpoint_key", "rollout_index"]))
    registry_path = root / "k32_rollout_registry.sqlite"
    registry = sqlite3.connect(registry_path, timeout=60.0)
    registry.row_factory = sqlite3.Row
    try:
        registry_counts = {
            "complete_jobs": int(
                registry.execute("SELECT COUNT(*) FROM jobs WHERE status='complete'").fetchone()[0]
            ),
            "registered_slots": int(
                registry.execute("SELECT COUNT(*) FROM rollout_slots").fetchone()[0]
            ),
        }
        transition_rows = registry.execute(
            """SELECT transition_id, worker_id, from_model_key, to_model_key,
                      transition_unix, reason
                 FROM worker_transitions ORDER BY transition_id"""
        ).fetchall()
        transitions = [dict(row) for row in transition_rows]
        checkpoint_result = registry.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if checkpoint_result is not None and int(checkpoint_result[0]) != 0:
            raise RuntimeError(f"registry WAL checkpoint remained busy: {tuple(checkpoint_result)}")
    finally:
        registry.close()
    if registry_counts != {"complete_jobs": 1024, "registered_slots": 8192}:
        raise RuntimeError(f"terminal registry counts differ: {registry_counts}")
    worker_paths = sorted(root.glob("workers/gpu-worker-*.json"))
    workers = [json.loads(path.read_text()) for path in worker_paths]
    atomic_json(root / "gpu_worker_transitions.json", transitions)
    terminal_completed_jobs = sum(int(worker["completed_jobs"]) for worker in workers)
    utilization = {
        "status": "COMPLETE",
        "worker_count": len(workers),
        "completed_jobs": registry_counts["complete_jobs"],
        "completed_jobs_terminal_orchestrator": terminal_completed_jobs,
        "preexisting_completed_jobs": registry_counts["complete_jobs"] - terminal_completed_jobs,
        "failures_requeued": sum(int(worker["failures_requeued"]) for worker in workers),
        "transitions": len(transitions),
        "worker_wall_seconds": {worker["worker_id"]: worker["wall_seconds"] for worker in workers},
        "model_active_seconds": {
            model: sum(float(worker["model_active_seconds"][model]) for worker in workers)
            for model in MODEL_KEYS
        },
        "initial_affinity": list(INITIAL_AFFINITY),
        "dynamic_spillover": True,
        "generated_outcomes": len(generated),
        "generated_sha256": sha256_file(generated_path),
        "registry_complete_jobs": registry_counts["complete_jobs"],
        "registry_registered_slots": registry_counts["registered_slots"],
        "timing_scope": "worker timing fields cover the terminal resumed orchestrator only",
    }
    atomic_json(root / "gpu_utilization_summary.json", utilization)
    output_volume.commit()
    return utilization


@app.function(image=image, cpu=2, memory=8192, timeout=60 * 60, volumes=OUTPUT_VOLUMES)
def status_remote(run_id: str, source_digest: str) -> dict[str, Any]:
    _assert_source(source_digest)
    output_volume.reload()
    root = _artifact_root(run_id)
    state = json.loads((root / "orchestration_state.json").read_text()) if (root / "orchestration_state.json").is_file() else {}
    coordinator = RegistryCoordinator(run_id=run_id)
    return {"state": state, "registry": coordinator.status.remote()}


@app.local_entrypoint()
def main(
    action: str = "status",
    run_id: str = "safeprefix_k_densification_v1_20260729_r1",
) -> None:
    _safe_run_id(run_id)
    digest = _source_digest()
    workspace = os.environ.get("MODAL_PROFILE", "unknown")
    if action == "prefetch":
        function = modal.Function.from_name(APP_NAME, "prefetch_models_remote")
        print(json.dumps(function.remote(digest), indent=2, sort_keys=True))
        return
    if action in {"launch", "resume"}:
        function = modal.Function.from_name(APP_NAME, "orchestrate_remote")
        call = function.spawn(run_id, digest, workspace, action == "resume")
        print(
            json.dumps(
                {
                    "status": "SUBMITTED",
                    "run_id": run_id,
                    "source_digest": digest,
                    "workspace": workspace,
                    "function_call_id": call.object_id,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if action == "status":
        function = modal.Function.from_name(APP_NAME, "status_remote")
        print(json.dumps(function.remote(run_id, digest), indent=2, sort_keys=True))
        return
    if action == "finalize-generation":
        function = modal.Function.from_name(APP_NAME, "finalize_generation_remote")
        print(json.dumps(function.remote(run_id, digest), indent=2, sort_keys=True))
        return
    raise ValueError("action must be one of: prefetch, launch, resume, status, finalize-generation")
