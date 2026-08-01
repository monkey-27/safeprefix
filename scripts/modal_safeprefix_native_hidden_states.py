#!/usr/bin/env python3
"""Persistent Modal runner for frozen-native hidden-state caching only."""

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import modal


LOCAL_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/workspace/safeprefix_code")
RUNS_ROOT = Path("/runs")
CONFIG_RELATIVE = Path("configs/native_hidden_state_cache.yaml")
RUN_VOLUME_NAME = os.environ.get(
    "SAFEPREFIX_NATIVE_ACQUISITION_VOLUME", "safeprefix-native-failed-trace-runs-v1"
)
CACHE_VOLUME_NAME = os.environ.get("SAFEPREFIX_HF_CACHE_VOLUME", "safeprefix-hf-cache")
APP_NAME = "safeprefix-native-hidden-state-cache"


def _source_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=LOCAL_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        # Modal imports this module in a deliberately minimal image without a
        # Git binary. The local import injects the immutable revision into the
        # image, so remote imports consume that value instead of rediscovering
        # repository state.
        return os.environ.get("SAFEPREFIX_SOURCE_COMMIT", "unknown")


SOURCE_COMMIT = _source_commit()
app = modal.App(APP_NAME)
run_volume = modal.Volume.from_name(RUN_VOLUME_NAME, create_if_missing=True)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name(
    os.environ.get("SAFEPREFIX_MODAL_HF_SECRET", "huggingface-token")
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
        "scipy==1.15.3",
        "torch==2.8.0",
        "transformers==4.55.4",
        "typer==0.17.4",
        "sentencepiece==0.2.1",
    )
    .add_local_dir(
        LOCAL_ROOT / "src" / "safeprefix",
        str(REMOTE_ROOT / "src" / "safeprefix"),
        copy=True,
    )
    .add_local_dir(LOCAL_ROOT / "configs", str(REMOTE_ROOT / "configs"), copy=True)
    .add_local_file(
        LOCAL_ROOT / "pyproject.toml", str(REMOTE_ROOT / "pyproject.toml"), copy=True
    )
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "HF_HUB_CACHE": "/cache/huggingface/hub",
            "PYTHONPATH": str(REMOTE_ROOT / "src"),
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "SAFEPREFIX_SOURCE_COMMIT": SOURCE_COMMIT,
        }
    )
)


def _safe(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError("run ID must be one safe path component")
    return value


def _config() -> dict[str, Any]:
    from safeprefix.config import load_config

    return load_config(REMOTE_ROOT / CONFIG_RELATIVE).data


def _acquisition_root(run_id: str) -> Path:
    return RUNS_ROOT / run_id / "artifacts/native_failed_trace_acquisition"


def _output_root(run_id: str) -> Path:
    return RUNS_ROOT / run_id / "artifacts/native_hidden_state_cache_v1"


def _state_path(run_id: str, model_key: str) -> Path:
    return RUNS_ROOT / run_id / "remote_run_state_native_hidden" / f"{model_key}.json"


def _write_state(run_id: str, model_key: str, **updates: Any) -> dict[str, Any]:
    from safeprefix.reproducibility import atomic_json

    path = _state_path(run_id, model_key)
    current = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    history = list(current.get("event_history", []))
    history.append({"time_unix": time.time(), **updates})
    current.update(updates, event_history=history)
    atomic_json(path, current)
    run_volume.commit()
    return current


def _cohort(run_id: str, model_key: str) -> list[dict[str, Any]]:
    from safeprefix.native_hidden_state_cache import read_jsonl, sha256_file

    root = _acquisition_root(run_id)
    cohort_path = root / "cohorts" / f"{model_key}.jsonl"
    marker_path = root / "cohorts" / f"{model_key}.immutable.json"
    if not cohort_path.is_file() or not marker_path.is_file():
        raise RuntimeError(f"missing frozen native cohort for {model_key}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("cohort_file_sha256") != sha256_file(cohort_path):
        raise RuntimeError("frozen native cohort checksum mismatch")
    return read_jsonl(cohort_path)


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=30 * 60,
    volumes={"/runs": run_volume},
)
def prepare_model(run_id: str, model_key: str) -> dict[str, Any]:
    from safeprefix.config import dump_resolved
    from safeprefix.native_hidden_state_cache import (
        build_hidden_state_packs,
        validate_source_cohort,
    )
    from safeprefix.reproducibility import atomic_json, atomic_jsonl, atomic_text, stable_hash

    _safe(run_id)
    run_volume.reload()
    config = _config()
    if model_key not in config["selected_models"]:
        raise ValueError(f"unconfigured model: {model_key}")
    entry = config["models"][model_key]
    cohort = _cohort(run_id, model_key)
    source_integrity = validate_source_cohort(
        cohort,
        model_key=model_key,
        model_revision=str(entry["revision"]),
        tokenizer_revision=str(entry["tokenizer_revision"]),
    )
    packs = build_hidden_state_packs(
        cohort,
        model_key=model_key,
        pack_size=int(config["representation"]["pack_size"]),
        configuration_hash=stable_hash(config),
    )
    root = _output_root(run_id)
    manifest_path = root / "immutable_manifests" / f"{model_key}_packs.jsonl"
    if manifest_path.is_file():
        from safeprefix.native_hidden_state_cache import read_jsonl

        if stable_hash(read_jsonl(manifest_path)) != stable_hash(packs):
            raise RuntimeError("native hidden-state pack manifest changed on resume")
    else:
        atomic_jsonl(manifest_path, packs)
    config_path = root / "immutable_manifests" / "resolved_config.yaml"
    resolved = dump_resolved(config)
    if config_path.is_file() and config_path.read_text(encoding="utf-8") != resolved:
        raise RuntimeError("native hidden-state resolved config changed on resume")
    if not config_path.is_file():
        atomic_text(config_path, resolved)
    protocol = {
        "status": "FROZEN",
        "run_id": run_id,
        "model_key": model_key,
        "source_commit": os.environ.get("SAFEPREFIX_SOURCE_COMMIT", "unknown"),
        "configuration_hash": stable_hash(config),
        "source_integrity": source_integrity,
        "pack_count": len(packs),
        "selected_layers": config["representation"]["selected_layers"],
        "token_scope": config["representation"]["token_scope"],
        "storage_dtype": config["representation"]["storage_dtype"],
        "kv_cache_persisted": False,
        "semantic_segmentation_enabled": False,
        "boundary_model_enabled": False,
        "repair_generation_enabled": False,
    }
    atomic_json(root / "immutable_manifests" / f"{model_key}_protocol.json", protocol)
    _write_state(
        run_id, model_key, status="PREPARED", current_error=None,
        preparation=protocol,
    )
    return protocol


@app.cls(
    image=image,
    gpu="H100!",
    cpu=12,
    memory=98304,
    timeout=8 * 60 * 60,
    max_containers=10,
    scaledown_window=30,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=5.0),
    volumes={"/runs": run_volume, "/cache": cache_volume},
    secrets=[hf_secret],
)
class HiddenStateWorker:
    run_id: str = modal.parameter()
    model_key: str = modal.parameter()

    @modal.enter()
    def load_context(self) -> None:
        from safeprefix.models.loader import load_model

        self.config = _config()
        self.entry = self.config["models"][self.model_key]
        self.cohort = _cohort(self.run_id, self.model_key)
        self.cohort_index = {str(row["trace_id"]): row for row in self.cohort}
        self.loaded = load_model(self.entry)
        if str(self.loaded.model_revision) != str(self.entry["revision"]):
            raise RuntimeError("loaded model revision differs from frozen extraction revision")
        if str(self.loaded.tokenizer_revision) != str(self.entry["tokenizer_revision"]):
            raise RuntimeError("loaded tokenizer revision differs from frozen extraction revision")

    @modal.method()
    def extract_pack(self, pack: dict[str, Any]) -> dict[str, Any]:
        from safeprefix.native_hidden_state_cache import execute_feature_pack

        run_volume.reload()
        result = execute_feature_pack(
            model=self.loaded.model,
            pack=pack,
            cohort_index=self.cohort_index,
            output_root=_output_root(self.run_id),
            chunk_size=int(self.config["representation"]["teacher_force_chunk_size"]),
            selected_layer=int(self.config["representation"]["selected_layers"][0]),
            model_revision=str(self.entry["revision"]),
            tokenizer_revision=str(self.entry["tokenizer_revision"]),
        )
        run_volume.commit()
        return result


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=8 * 60 * 60,
    volumes={"/runs": run_volume, "/cache": cache_volume},
    secrets=[hf_secret],
)
def orchestrate_model(run_id: str, model_key: str) -> dict[str, Any]:
    from safeprefix.native_hidden_state_cache import (
        aggregate_feature_cache,
        read_jsonl,
    )

    _safe(run_id)
    started = time.time()
    try:
        preparation = prepare_model.remote(run_id, model_key)
        run_volume.reload()
        config = _config()
        entry = config["models"][model_key]
        cohort = _cohort(run_id, model_key)
        root = _output_root(run_id)
        packs = read_jsonl(root / "immutable_manifests" / f"{model_key}_packs.jsonl")
        worker = HiddenStateWorker(run_id=run_id, model_key=model_key)
        _write_state(
            run_id, model_key, status="RUNNING_PRODUCTION_SHAPED_SMOKE",
            current_error=None, expected_pack_count=len(packs),
        )
        # The first pack is a real, reusable production pack. Running it alone
        # validates loading, exact-token teacher forcing, persistence, and the
        # complete marker before expensive fan-out.
        smoke = worker.extract_pack.remote(packs[0])
        if smoke.get("status") not in {"COMPLETE", "SKIPPED_VALID"}:
            raise RuntimeError("production-shaped native hidden-state smoke failed")
        run_volume.reload()
        completed = {
            str(pack["pack_id"])
            for pack in packs
            if (root / "packs" / model_key / str(pack["pack_id"]) / "complete.json").is_file()
        }
        missing = [pack for pack in packs if str(pack["pack_id"]) not in completed]
        _write_state(
            run_id, model_key, status="EXTRACTING_HIDDEN_STATES",
            current_error=None, completed_pack_count=len(completed),
            remaining_pack_count=len(missing), smoke=smoke,
        )
        calls = [worker.extract_pack.spawn(pack) for pack in missing]
        for call in calls:
            call.get()
        run_volume.reload()
        summary = aggregate_feature_cache(
            cohort=cohort,
            packs=packs,
            output_root=root,
            model_key=model_key,
            model_revision=str(entry["revision"]),
            tokenizer_revision=str(entry["tokenizer_revision"]),
        )
        summary.update(
            source_commit=os.environ.get("SAFEPREFIX_SOURCE_COMMIT", "unknown"),
            workspace_wall_seconds=time.time() - started,
            preparation=preparation,
        )
        from safeprefix.reproducibility import atomic_json

        atomic_json(root / "final" / model_key / "summary.json", summary)
        _write_state(
            run_id, model_key, status="COMPLETE", current_error=None,
            finished_unix=time.time(), summary=summary,
        )
        run_volume.commit()
        return summary
    except Exception as exc:
        run_volume.reload()
        _write_state(
            run_id, model_key, status="FAILED_RESUMABLE",
            current_error=f"{type(exc).__name__}: {exc}", failed_unix=time.time(),
        )
        raise


@app.local_entrypoint()
def main(run_id: str, model_key: str, prepare_only: bool = True) -> None:
    if prepare_only:
        print(json.dumps(prepare_model.remote(run_id, model_key), indent=2, sort_keys=True))
        return
    print(json.dumps(orchestrate_model.remote(run_id, model_key), indent=2, sort_keys=True))
