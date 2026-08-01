#!/usr/bin/env python3
"""Portable, sharded Modal runner for teacher-forced threshold selection.

The prepared calibration-only bundle is uploaded to a workspace-local output
volume before deployment.  GPU workers therefore mount only that output and an
exact-revision model cache.  Boundary, completion, and native volumes are never
resolved or mounted by this worker app.
"""

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
REMOTE_ROOT = Path("/workspace/aaai_monkey")
APP_NAME = "safeprefix-threshold-selection-tf-v1"
OUTPUT_VOLUME_NAME = os.environ.get(
    "SAFEPREFIX_THRESHOLD_OUTPUT_VOLUME", "safeprefix-threshold-selection-tf-v1"
)
CACHE_VOLUME_NAME = os.environ.get("SAFEPREFIX_HF_CACHE_VOLUME", "safeprefix-hf-cache")
HF_SECRET_NAME = os.environ.get("SAFEPREFIX_HF_SECRET_NAME", "huggingface-token")
CONFIG_NAME = "safeprefix_threshold_selection_tf_v1.yaml"
RUNNER_NAME = "run_safeprefix_threshold_selection_tf_v1.py"
LAUNCHER_NAME = Path(__file__).name
MODEL_KEYS = ("family_a_small", "family_a_large", "family_b_small", "family_b_large")

app = modal.App(APP_NAME)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True, version=2)
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
    f"configs/{CONFIG_NAME}",
    "configs/models.yaml",
    f"scripts/{RUNNER_NAME}",
    f"scripts/{LAUNCHER_NAME}",
)
SOURCE_DIRS = (
    "src/safeprefix/boundary_v1",
    "src/safeprefix/threshold_selection_tf_v1",
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "accelerate==1.10.1",
        "datasets==5.0.0",
        "huggingface-hub==0.34.4",
        "matplotlib==3.10.5",
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
    .env(
        {
            "PYTHONPATH": str(REMOTE_ROOT / "src"),
            "PYTHONUNBUFFERED": "1",
            "HF_HOME": "/cache/huggingface",
            "HF_HUB_CACHE": "/cache/huggingface/hub",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "MPLCONFIGDIR": "/tmp/matplotlib",
            "SAFEPREFIX_THRESHOLD_OUTPUT_VOLUME": OUTPUT_VOLUME_NAME,
        }
    )
)
for relative in SOURCE_FILES:
    image = image.add_local_file(
        LOCAL_ROOT / relative, str(REMOTE_ROOT / relative), copy=True
    )
for relative in SOURCE_DIRS:
    image = image.add_local_dir(
        LOCAL_ROOT / relative,
        str(REMOTE_ROOT / relative),
        copy=True,
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )

WORKER_VOLUMES = {
    "/threshold": output_volume,
    "/cache": cache_volume.with_mount_options(read_only=True),
}
OUTPUT_VOLUMES = {"/threshold": output_volume}
CACHE_VOLUMES = {"/cache": cache_volume}
prefetch_image = image.env({"HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"})


def _safe_run_id(run_id: str) -> None:
    if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
        raise ValueError("run_id must be one safe path component")


def _artifact_root(run_id: str) -> Path:
    _safe_run_id(run_id)
    return Path("/threshold") / run_id / "artifacts/safeprefix_threshold_selection_tf_v1"


def _state_path(run_id: str) -> Path:
    return Path("/threshold") / run_id / "orchestration/state.json"


def _shard_state_path(run_id: str, shard_id: str) -> Path:
    if shard_id not in {"shard-00", "shard-01"}:
        raise ValueError(f"unknown shard_id: {shard_id}")
    return Path("/threshold") / run_id / f"orchestration/shards/{shard_id}/state.json"


def _config_path() -> Path:
    return REMOTE_ROOT / "configs" / CONFIG_NAME


def _source_paths(root: Path) -> list[Path]:
    files = [root / relative for relative in SOURCE_FILES]
    for relative in SOURCE_DIRS:
        files.extend(
            path
            for path in (root / relative).rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        )
    return sorted(files, key=lambda path: str(path.relative_to(root)))


def _source_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in _source_paths(root):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_state(run_id: str) -> dict[str, Any]:
    path = _state_path(run_id)
    return json.loads(path.read_text()) if path.is_file() else {}


def _write_state(run_id: str, **updates: Any) -> dict[str, Any]:
    payload = {**_read_state(run_id), **updates, "updated_unix": time.time()}
    _atomic_json(_state_path(run_id), payload)
    output_volume.commit()
    return payload


def _read_shard_state(run_id: str, shard_id: str) -> dict[str, Any]:
    path = _shard_state_path(run_id, shard_id)
    return json.loads(path.read_text()) if path.is_file() else {}


def _write_shard_state(run_id: str, shard_id: str, **updates: Any) -> dict[str, Any]:
    payload = {
        **_read_shard_state(run_id, shard_id),
        **updates,
        "shard_id": shard_id,
        "updated_unix": time.time(),
    }
    _atomic_json(_shard_state_path(run_id, shard_id), payload)
    output_volume.commit()
    return payload


def _runner(command: str, run_id: str, *arguments: str) -> list[str]:
    return [
        "python3",
        str(REMOTE_ROOT / "scripts" / RUNNER_NAME),
        command,
        "--config",
        str(_config_path()),
        "--artifact-root",
        str(_artifact_root(run_id)),
        *arguments,
    ]


def _run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("COMMAND: " + " ".join(command) + "\n")
        process = subprocess.Popen(
            command,
            cwd=REMOTE_ROOT,
            env=os.environ.copy(),
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
    output_volume.commit()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def _assert_source(source_digest: str) -> None:
    observed = _source_digest(REMOTE_ROOT)
    if observed != source_digest:
        raise RuntimeError(f"deployed source digest mismatch: {observed} != {source_digest}")


def _assert_literal_h100() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the threshold H100 worker")
    names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    if not names or any("H100" not in name.upper() for name in names):
        raise RuntimeError(f"literal H100 required; resolved devices were {names}")
    return {"device_names": names, "torch": torch.__version__, "cuda": torch.version.cuda}


def _validate_prepared(
    run_id: str, source_digest: str, source_commit: str
) -> dict[str, Any]:
    from safeprefix.config import load_config
    from safeprefix.threshold_selection_tf_v1.sharding import validate_execution_shards

    _assert_source(source_digest)
    root = _artifact_root(run_id)
    if not (root / "READY.json").is_file():
        raise RuntimeError("prepared calibration-only threshold bundle is absent")
    return validate_execution_shards(
        load_config(_config_path()).data,
        artifact_root=root,
        run_id=run_id,
        source_digest=source_digest,
        source_commit=source_commit,
    )


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
    """Populate a workspace-local cache at all four exact frozen revisions."""
    from huggingface_hub import snapshot_download
    from safeprefix.config import load_config

    _assert_source(source_digest)
    cache_volume.reload()
    config = load_config(_config_path()).data
    downloaded = {}
    for model_key in MODEL_KEYS:
        model = config["models"][model_key]
        model_id = str(model["hf_model_id"])
        revision = str(model["revision"])
        path = snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir="/cache/huggingface/hub",
            local_files_only=False,
        )
        if Path(path).name != revision:
            raise RuntimeError(f"{model_key}: resolved cache snapshot differs from pinned revision")
        downloaded[model_key] = {
            "model_id": model_id,
            "revision": revision,
            "snapshot_path": path,
        }
        cache_volume.commit()
    return {
        "status": "COMPLETE",
        "source_digest": source_digest,
        "models": downloaded,
        "native_volume_mounted": False,
    }


@app.cls(
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
class ThresholdWorker:
    run_id: str = modal.parameter()
    model_key: str = modal.parameter()

    @modal.enter()
    def load(self) -> None:
        from safeprefix.config import load_config
        from safeprefix.models.loader import load_model

        if self.model_key not in MODEL_KEYS:
            raise ValueError(f"unknown model key: {self.model_key}")
        output_volume.reload()
        self.hardware = _assert_literal_h100()
        self.config = load_config(_config_path()).data
        self.loaded = load_model(self.config["models"][self.model_key])

    @modal.method()
    def run_smoke(self, pack_id: str) -> dict[str, Any]:
        from safeprefix.threshold_selection_tf_v1.runtime import execute_threshold_pack

        first = execute_threshold_pack(
            self.config,
            artifact_root=_artifact_root(self.run_id),
            model_key=self.model_key,
            pack_id=pack_id,
            loaded=self.loaded,
            smoke=True,
        )
        second = execute_threshold_pack(
            self.config,
            artifact_root=_artifact_root(self.run_id),
            model_key=self.model_key,
            pack_id=pack_id,
            loaded=self.loaded,
            smoke=True,
        )
        if first.get("status") not in {"COMPLETE", "SKIPPED_VALID"}:
            raise RuntimeError("first smoke execution did not complete")
        if second.get("status") != "SKIPPED_VALID":
            raise RuntimeError("second smoke execution did not resume by validated skip")
        payload = {
            "status": "PASS",
            "model_key": self.model_key,
            "first_status": first.get("status"),
            "second_status": second.get("status"),
            "hardware": self.hardware,
        }
        _atomic_json(
            _artifact_root(self.run_id) / f"smoke/resume_checks/{self.model_key}.json",
            payload,
        )
        output_volume.commit()
        return payload

    @modal.method()
    def run_slot(
        self,
        shard_id: str,
        worker_slot: int,
        pack_ids: list[str],
        source_digest: str,
        source_commit: str,
    ) -> dict[str, Any]:
        from safeprefix.threshold_selection_tf_v1.runtime import execute_threshold_pack

        contract = _validate_prepared(
            self.run_id, source_digest=source_digest, source_commit=source_commit
        )["shards"][shard_id]
        owned = [
            row
            for row in contract["slots"]
            if row["model_key"] == self.model_key
            and int(row["worker_slot"]) == int(worker_slot)
        ]
        if len(owned) != 1 or list(owned[0]["pack_ids"]) != list(pack_ids):
            raise RuntimeError("worker request differs from frozen shard ownership")

        started = time.time()
        results = []
        for pack_id in pack_ids:
            output_volume.reload()
            results.append(
                execute_threshold_pack(
                    self.config,
                    artifact_root=_artifact_root(self.run_id),
                    model_key=self.model_key,
                    pack_id=pack_id,
                    loaded=self.loaded,
                    smoke=False,
                )
            )
            output_volume.commit()
        payload = {
            "status": "COMPLETE",
            "shard_id": shard_id,
            "model_key": self.model_key,
            "worker_slot": int(worker_slot),
            "pack_ids": pack_ids,
            "packs": len(results),
            "skipped_valid": sum(row.get("status") == "SKIPPED_VALID" for row in results),
            "wall_seconds": time.time() - started,
            "hardware": self.hardware,
        }
        _atomic_json(
            Path("/threshold")
            / self.run_id
            / f"orchestration/shards/{shard_id}/workers/{self.model_key}/slot-{int(worker_slot):02d}.json",
            payload,
        )
        output_volume.commit()
        return payload


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=24 * 60 * 60,
    max_containers=1,
    volumes=WORKER_VOLUMES,
)
def smoke_gate_remote(run_id: str, source_digest: str) -> dict[str, Any]:
    from safeprefix.threshold_selection_tf_v1.data import read_jsonl
    import pandas as pd

    output_volume.reload()
    _assert_source(source_digest)
    complete_path = _artifact_root(run_id) / "smoke/SMOKE_COMPLETE.json"
    if complete_path.is_file():
        completed = json.loads(complete_path.read_text())
        if completed.get("status") == "PASS":
            return completed
    packs = read_jsonl(_artifact_root(run_id) / "manifests/execution_packs.jsonl")
    trace_manifest = pd.read_parquet(
        _artifact_root(run_id) / "manifests/calibration_trace_manifest.parquet"
    )
    fold_zero = set(
        trace_manifest.loc[trace_manifest["fold"].eq(0), "trace_id"].astype(str)
    )
    selected: dict[str, str] = {}
    for model_key in MODEL_KEYS:
        candidates = [
            row for row in packs
            if row["model_key"] == model_key
            and int(row["trace_count"]) == 2
            and set(map(str, row["trace_ids"])) <= fold_zero
        ]
        if not candidates:
            raise RuntimeError(f"{model_key}: no two-trace pack exists for smoke")
        selected[model_key] = str(sorted(candidates, key=lambda row: row["pack_id"])[0]["pack_id"])
    calls = {}
    for model_key, pack_id in selected.items():
        worker = ThresholdWorker(run_id=run_id, model_key=model_key)
        calls[model_key] = worker.run_smoke.spawn(pack_id)
    results = {model_key: call.get() for model_key, call in calls.items()}
    # Spawned workers commit into the shared Volume from other containers.  A
    # reload is required before this long-lived coordinator can see their pack
    # files; otherwise the policy smoke can observe a stale mount snapshot.
    output_volume.reload()
    _run(
        _runner("smoke-policy", run_id),
        Path("/threshold") / run_id / "logs/smoke_policy.log",
    )
    summary = json.loads(complete_path.read_text())
    return {**summary, "gpu_results": results}


@app.function(image=image, cpu=16, memory=131072, timeout=6 * 60 * 60, volumes=OUTPUT_VOLUMES)
def analyze_and_report_remote(
    run_id: str, source_digest: str, source_commit: str
) -> dict[str, Any]:
    output_volume.reload()
    contract = _validate_prepared(run_id, source_digest, source_commit)
    from safeprefix.config import load_config
    from safeprefix.threshold_selection_tf_v1.sharding import validate_shard_results

    config = load_config(_config_path()).data
    for shard_id, shard in contract["shards"].items():
        validation = validate_shard_results(config, _artifact_root(run_id), shard)
        if validation["valid_packs"] != validation["expected_packs"]:
            raise RuntimeError(f"{shard_id}: cannot finalize before every pack validates")
    _run(_runner("analyze", run_id), Path("/threshold") / run_id / "logs/analyze.log")
    _run(
        _runner("report", run_id, "--run-id", run_id),
        Path("/threshold") / run_id / "logs/report.log",
    )
    output_volume.commit()
    return json.loads((_artifact_root(run_id) / "COMPLETE.json").read_text())


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=24 * 60 * 60,
    max_containers=1,
    volumes=OUTPUT_VOLUMES,
)
def orchestrate_shard_remote(
    run_id: str,
    shard_id: str,
    source_digest: str,
    source_commit: str,
    workspace_name: str,
) -> dict[str, Any]:
    from safeprefix.config import load_config
    from safeprefix.threshold_selection_tf_v1.sharding import validate_shard_results

    output_volume.reload()
    contract = _validate_prepared(run_id, source_digest, source_commit)
    if shard_id not in contract["shards"]:
        raise ValueError(f"unknown shard_id: {shard_id}")
    shard = contract["shards"][shard_id]
    receipt_path = (
        Path("/threshold") / run_id / f"orchestration/shards/{shard_id}/COMPLETE.json"
    )
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("status") == "COMPLETE" and receipt.get("shard_hash") == shard["shard_hash"]:
            validation = validate_shard_results(
                load_config(_config_path()).data, _artifact_root(run_id), shard
            )
            if validation["valid_packs"] == validation["expected_packs"]:
                return receipt
    started = float(_read_shard_state(run_id, shard_id).get("started_unix") or time.time())
    try:
        _write_shard_state(
            run_id,
            shard_id,
            status="CACHE_PREFETCH_RUNNING",
            started_unix=started,
            workspace_name=workspace_name,
            source_digest=source_digest,
            source_commit=source_commit,
            shard_hash=shard["shard_hash"],
            native_volume_mounted=False,
        )
        cache = prefetch_models_remote.remote(source_digest)
        output_volume.reload()
        _write_shard_state(run_id, shard_id, status="SMOKE_RUNNING", cache=cache)
        smoke = smoke_gate_remote.remote(run_id, source_digest)
        if smoke.get("status") != "PASS":
            raise RuntimeError("minimal threshold smoke did not pass")
        output_volume.reload()
        _write_shard_state(run_id, shard_id, status="PRODUCTION_RUNNING", smoke=smoke)
        config = load_config(_config_path()).data
        calls: dict[str, tuple[str, int, Any]] = {}
        for row in shard["slots"]:
            model_key = str(row["model_key"])
            slot = int(row["worker_slot"])
            pack_ids = list(map(str, row["pack_ids"]))
            worker = ThresholdWorker(run_id=run_id, model_key=model_key)
            call = worker.run_slot.spawn(
                shard_id, slot, pack_ids, source_digest, source_commit
            )
            calls[f"{model_key}:{slot}"] = (model_key, slot, call)
        _write_shard_state(
            run_id,
            shard_id,
            production_call_ids={key: value[2].object_id for key, value in calls.items()},
            production_worker_count=len(calls),
        )
        results = {key: value[2].get() for key, value in calls.items()}
        output_volume.reload()
        validation = validate_shard_results(config, _artifact_root(run_id), shard)
        if validation["valid_packs"] != validation["expected_packs"]:
            raise RuntimeError(f"{shard_id}: completed worker set failed pack validation")
        receipt = {
            "status": "COMPLETE",
            "run_id": run_id,
            "shard_id": shard_id,
            "shard_hash": shard["shard_hash"],
            "workspace_name": workspace_name,
            "source_digest": source_digest,
            "source_commit": source_commit,
            "validation": validation,
            "production_results": results,
            "completed_unix": time.time(),
            "wall_seconds": time.time() - started,
            "native_volume_mounted": False,
        }
        _atomic_json(receipt_path, receipt)
        output_volume.commit()
        return _write_shard_state(
            run_id,
            shard_id,
            status="COMPLETE",
            completed_unix=time.time(),
            wall_seconds=time.time() - started,
            receipt=receipt,
        )
    except Exception as exc:
        _write_shard_state(
            run_id,
            shard_id,
            status="FAILED_RESUMABLE",
            error=f"{type(exc).__name__}: {exc}",
            wall_seconds=time.time() - started,
        )
        raise


@app.function(image=image, cpu=2, memory=8192, timeout=60 * 60, volumes=OUTPUT_VOLUMES)
def status_remote(run_id: str, source_digest: str, source_commit: str) -> dict[str, Any]:
    from safeprefix.config import load_config
    from safeprefix.threshold_selection_tf_v1.sharding import validate_shard_results

    output_volume.reload()
    contract = _validate_prepared(run_id, source_digest, source_commit)
    root = _artifact_root(run_id)
    config = load_config(_config_path()).data
    shards = {
        shard_id: {
            "state": _read_shard_state(run_id, shard_id),
            "validation": validate_shard_results(config, root, shard),
        }
        for shard_id, shard in contract["shards"].items()
    }
    return {
        "run_id": run_id,
        "state": _read_state(run_id),
        "shards": shards,
        "smoke_complete": (root / "smoke/SMOKE_COMPLETE.json").is_file(),
        "analysis_complete": (root / "analysis_summary.json").is_file(),
        "terminal_complete": (root / "COMPLETE.json").is_file(),
        "native_volume_mounted": False,
    }


@app.function(image=image, cpu=16, memory=131072, timeout=6 * 60 * 60, volumes=OUTPUT_VOLUMES)
def report_remote(run_id: str, source_digest: str, source_commit: str) -> dict[str, Any]:
    output_volume.reload()
    _validate_prepared(run_id, source_digest, source_commit)
    _run(
        _runner("report", run_id, "--run-id", run_id),
        Path("/threshold") / run_id / "logs/report_regeneration.log",
    )
    output_volume.commit()
    return json.loads((_artifact_root(run_id) / "COMPLETE.json").read_text())


def _local_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=LOCAL_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _frozen_source_commit(run_id: str) -> str:
    local = (
        LOCAL_ROOT
        / "artifacts/safeprefix_threshold_selection_tf_v1/manifests/execution_shards.json"
    )
    if local.is_file():
        payload = json.loads(local.read_text())
        if payload.get("run_id") == run_id and payload.get("source_commit"):
            return str(payload["source_commit"])
    return _local_commit()


@app.local_entrypoint()
def main(
    action: str = "status",
    run_id: str = "safeprefix_threshold_selection_tf_v1_20260728_r1",
    shard_id: str = "shard-00",
) -> None:
    _safe_run_id(run_id)
    digest = _source_digest(LOCAL_ROOT)
    commit = _frozen_source_commit(run_id)
    workspace_name = os.environ.get("MODAL_PROFILE", "unknown")
    if action in {"submit-shard", "resume-shard"}:
        deployed = modal.Function.from_name(APP_NAME, "orchestrate_shard_remote")
        call = deployed.spawn(run_id, shard_id, digest, commit, workspace_name)
        print(
            json.dumps(
                {
                    "status": "SUBMITTED",
                    "action": action,
                    "run_id": run_id,
                    "shard_id": shard_id,
                    "workspace_name": workspace_name,
                    "source_digest": digest,
                    "source_commit": commit,
                    "function_call_id": call.object_id,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if action == "smoke":
        prefetch = modal.Function.from_name(APP_NAME, "prefetch_models_remote")
        gate = modal.Function.from_name(APP_NAME, "smoke_gate_remote")
        prefetch.remote(digest)
        print(json.dumps(gate.remote(run_id, digest), indent=2, sort_keys=True))
        return
    if action == "prefetch":
        deployed = modal.Function.from_name(APP_NAME, "prefetch_models_remote")
        print(json.dumps(deployed.remote(digest), indent=2, sort_keys=True))
        return
    if action == "status":
        deployed = modal.Function.from_name(APP_NAME, "status_remote")
        print(json.dumps(deployed.remote(run_id, digest, commit), indent=2, sort_keys=True))
        return
    if action == "finalize":
        deployed = modal.Function.from_name(APP_NAME, "analyze_and_report_remote")
        print(json.dumps(deployed.remote(run_id, digest, commit), indent=2, sort_keys=True))
        return
    if action == "report":
        deployed = modal.Function.from_name(APP_NAME, "report_remote")
        print(json.dumps(deployed.remote(run_id, digest, commit), indent=2, sort_keys=True))
        return
    raise ValueError(
        "action must be one of: prefetch, smoke, submit-shard, resume-shard, "
        "status, finalize, report"
    )
