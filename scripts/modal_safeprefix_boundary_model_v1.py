#!/usr/bin/env python3
"""Isolated Modal orchestration for SafePrefix boundary model v1."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import modal


LOCAL_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/workspace/aaai_monkey")
APP_NAME = "safeprefix-boundary-model-v1"
OUTPUT_VOLUME_NAME = "safeprefix-boundary-model-v1"
COMPLETION_VOLUME_NAME = "safeprefix-full-teacher-forced-runs-v2"
CONFIG_NAME = "boundary_model_v1.yaml"
RUNNER_NAME = "run_boundary_model_v1.py"
MODEL_KEYS = ("family_a_small", "family_a_large", "family_b_small", "family_b_large")

app = modal.App(APP_NAME)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True, version=2)
completion_volume = modal.Volume.from_name(
    COMPLETION_VOLUME_NAME, create_if_missing=False, version=2
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "matplotlib==3.10.5",
        "numpy==2.2.6",
        "pandas==2.3.2",
        "pyarrow==24.0.0",
        "PyYAML==6.0.2",
        "scikit-learn==1.7.1",
        "scipy==1.16.1",
        "torch==2.8.0",
    )
    .add_local_file(
        LOCAL_ROOT / "src/safeprefix/__init__.py",
        str(REMOTE_ROOT / "src/safeprefix/__init__.py"),
        copy=True,
    )
    .add_local_file(
        LOCAL_ROOT / "src/safeprefix/config.py",
        str(REMOTE_ROOT / "src/safeprefix/config.py"),
        copy=True,
    )
    .add_local_dir(
        LOCAL_ROOT / "src/safeprefix/boundary_v1",
        str(REMOTE_ROOT / "src/safeprefix/boundary_v1"),
        copy=True,
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
    .add_local_file(
        LOCAL_ROOT / "configs" / CONFIG_NAME,
        str(REMOTE_ROOT / "configs" / CONFIG_NAME),
        copy=True,
    )
    .add_local_file(
        LOCAL_ROOT / "scripts" / RUNNER_NAME,
        str(REMOTE_ROOT / "scripts" / RUNNER_NAME),
        copy=True,
    )
    .env(
        {
            "PYTHONPATH": str(REMOTE_ROOT / "src"),
            "PYTHONUNBUFFERED": "1",
            "MPLCONFIGDIR": "/tmp/matplotlib",
            "OMP_NUM_THREADS": "8",
        }
    )
)


def _safe_run_id(run_id: str) -> None:
    if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
        raise ValueError("run_id must be one safe path component")


def _artifact_root(run_id: str) -> Path:
    _safe_run_id(run_id)
    return Path("/boundary") / run_id / "artifacts/boundary_model_v1"


def _state_path(run_id: str) -> Path:
    return Path("/boundary") / run_id / "orchestration/state.json"


def _config_path() -> Path:
    return REMOTE_ROOT / "configs" / CONFIG_NAME


def _source_digest() -> str:
    digest = hashlib.sha256()
    roots = [
        REMOTE_ROOT / "src/safeprefix/boundary_v1",
        REMOTE_ROOT / "src/safeprefix/config.py",
        REMOTE_ROOT / "configs" / CONFIG_NAME,
        REMOTE_ROOT / "scripts" / RUNNER_NAME,
    ]
    files: list[Path] = []
    for root in roots:
        files.extend([root] if root.is_file() else [path for path in root.rglob("*") if path.is_file()])
    for path in sorted(files, key=str):
        digest.update(str(path.relative_to(REMOTE_ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_state(run_id: str, **updates: Any) -> dict[str, Any]:
    path = _state_path(run_id)
    prior = json.loads(path.read_text()) if path.is_file() else {}
    payload = {**prior, **updates, "updated_unix": time.time()}
    _atomic_json(path, payload)
    output_volume.commit()
    return payload


def _run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    if code:
        raise RuntimeError(f"command failed with exit code {code}: {' '.join(command)}")


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


VOLUMES = {
    "/boundary": output_volume,
    "/completion": completion_volume.with_mount_options(read_only=True),
}


@app.function(image=image, cpu=16, memory=65536, timeout=3 * 60 * 60, volumes=VOLUMES)
def prepare_remote(run_id: str, source_digest: str, source_commit: str) -> dict[str, Any]:
    output_volume.reload()
    if _source_digest() != source_digest:
        raise RuntimeError("deployed source digest mismatch")
    os.environ["SAFEPREFIX_SOURCE_COMMIT"] = source_commit
    _write_state(
        run_id,
        status="PREPARING",
        source_digest=source_digest,
        source_commit=source_commit,
        native_evaluation_used=False,
    )
    command = _runner(
        "prepare",
        run_id,
        "--old-root",
        "/completion/safeprefix_full_teacher_forced_20260726_r3/artifacts/full_teacher_forced_suite",
        "--completion-root",
        "/completion/safeprefix_teacher_forced_completion_20260727_r5/artifacts/teacher_forced_completion",
        "--manifest-root",
        "/completion/safeprefix_teacher_forced_completion_20260727_r5/artifacts/teacher_forced_completion/immutable_manifests",
    )
    _run(command, Path("/boundary") / run_id / "logs/prepare.log")
    output_volume.commit()
    return _write_state(run_id, status="DATA_READY")


@app.function(
    image=image,
    gpu="H100",
    cpu=12,
    memory=65536,
    timeout=12 * 60 * 60,
    retries=modal.Retries(max_retries=1, initial_delay=10.0),
    volumes=VOLUMES,
)
def train_remote(run_id: str, model_key: str) -> dict[str, Any]:
    if model_key not in MODEL_KEYS:
        raise ValueError(f"unknown model key {model_key}")
    output_volume.reload()
    _run(
        _runner("train-model", run_id, "--model-key", model_key, "--device", "cuda"),
        Path("/boundary") / run_id / f"logs/train_{model_key}.log",
    )
    output_volume.commit()
    return json.loads(
        (_artifact_root(run_id) / f"training/{model_key}/matrix_summary.json").read_text()
    )


@app.function(
    image=image,
    gpu="H100",
    cpu=16,
    memory=65536,
    timeout=6 * 60 * 60,
    volumes=VOLUMES,
)
def finalize_remote(
    run_id: str, source_digest: str, source_commit: str
) -> dict[str, Any]:
    output_volume.reload()
    if _source_digest() != source_digest:
        raise RuntimeError("deployed source digest mismatch")
    os.environ["SAFEPREFIX_RUN_ID"] = run_id
    _write_state(
        run_id,
        status="SELECTING",
        source_digest=source_digest,
        source_commit=source_commit,
    )
    _run(_runner("select", run_id), Path("/boundary") / run_id / "logs/select.log")
    _write_state(run_id, status="CALIBRATING_AND_EVALUATING")
    _run(
        _runner("finalize", run_id, "--device", "cuda"),
        Path("/boundary") / run_id / "logs/finalize.log",
    )
    _run(_runner("report", run_id), Path("/boundary") / run_id / "logs/report.log")
    output_volume.commit()
    return _write_state(
        run_id,
        status="COMPLETE",
        native_evaluation_used=False,
        final_tau_selected=False,
    )


@app.function(image=image, cpu=2, memory=4096, timeout=10 * 60, volumes=VOLUMES)
def status_remote(run_id: str) -> dict[str, Any]:
    output_volume.reload()
    state = json.loads(_state_path(run_id).read_text()) if _state_path(run_id).is_file() else {}
    matrices = {
        model: (
            json.loads(
                (_artifact_root(run_id) / f"training/{model}/matrix_summary.json").read_text()
            ).get("runs")
            if (_artifact_root(run_id) / f"training/{model}/matrix_summary.json").is_file()
            else len(
                list(
                    (_artifact_root(run_id) / f"training/{model}").glob("**/complete.json")
                )
            )
        )
        for model in MODEL_KEYS
    }
    return {"run_id": run_id, "state": state, "completed_runs_by_model": matrices}


@app.function(image=image, cpu=4, memory=8192, timeout=60 * 60, volumes=VOLUMES)
def report_remote(run_id: str, source_digest: str, source_commit: str) -> dict[str, Any]:
    output_volume.reload()
    if _source_digest() != source_digest:
        raise RuntimeError("deployed source digest mismatch")
    os.environ["SAFEPREFIX_RUN_ID"] = run_id
    _run(
        _runner("report", run_id),
        Path("/boundary") / run_id / "logs/report_regeneration.log",
    )
    output_volume.commit()
    return _write_state(
        run_id,
        status="COMPLETE",
        report_source_digest=source_digest,
        report_source_commit=source_commit,
        native_evaluation_used=False,
        final_tau_selected=False,
    )


def _local_source_digest() -> str:
    digest = hashlib.sha256()
    roots = [
        LOCAL_ROOT / "src/safeprefix/boundary_v1",
        LOCAL_ROOT / "src/safeprefix/config.py",
        LOCAL_ROOT / "configs" / CONFIG_NAME,
        LOCAL_ROOT / "scripts" / RUNNER_NAME,
    ]
    files: list[Path] = []
    for root in roots:
        files.extend([root] if root.is_file() else [path for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts])
    for path in sorted(files, key=str):
        digest.update(str(path.relative_to(LOCAL_ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


@app.local_entrypoint()
def main(
    action: str = "status",
    run_id: str = "safeprefix_boundary_model_v1_20260728_r1",
    local_root: str = "artifacts/boundary_model_v1_modal",
) -> None:
    _safe_run_id(run_id)
    if action in {"launch", "resume"}:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=LOCAL_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        digest = _local_source_digest()
        prepare_remote.remote(run_id, digest, commit)
        calls = [train_remote.spawn(run_id, model) for model in MODEL_KEYS]
        results = [call.get() for call in calls]
        if any(result.get("runs") != 27 for result in results):
            raise RuntimeError("one or more model matrices are incomplete")
        print(
            json.dumps(
                finalize_remote.remote(run_id, digest, commit), indent=2, sort_keys=True
            )
        )
    elif action == "finalize":
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=LOCAL_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        digest = _local_source_digest()
        print(
            json.dumps(
                finalize_remote.remote(run_id, digest, commit), indent=2, sort_keys=True
            )
        )
    elif action == "report":
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=LOCAL_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        digest = _local_source_digest()
        print(
            json.dumps(
                report_remote.remote(run_id, digest, commit), indent=2, sort_keys=True
            )
        )
    elif action == "status":
        print(json.dumps(status_remote.remote(run_id), indent=2, sort_keys=True))
    elif action == "fetch":
        source = f"{run_id}/artifacts/boundary_model_v1"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "modal",
                "volume",
                "get",
                OUTPUT_VOLUME_NAME,
                source,
                local_root,
            ],
            check=True,
        )
    else:
        raise ValueError(
            "action must be launch, resume, finalize, report, status, or fetch"
        )
