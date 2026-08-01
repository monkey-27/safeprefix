#!/usr/bin/env python3
"""Four-workspace native-failure acquisition launcher.

Deploy this same file once per authorized Modal workspace.  Each invocation
owns exactly one model and up to ten one-H100 workers.  The source census is
embedded read-only; the run volume contains all resumable mutable artifacts.

No function is submitted unless the local entrypoint receives the explicit
``--authorize-full-inference`` flag.  This protects the current prelaunch-only
state while leaving the post-approval command reproducible.
"""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import modal


LOCAL_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/workspace/safeprefix_code")
RUNS_ROOT = Path("/runs")
CONFIG_RELATIVE = Path("configs/native_failed_trace_acquisition.yaml")
PRELAUNCH_RELATIVE = Path("artifacts/native_failed_trace_acquisition/prelaunch")
RUN_VOLUME_NAME = os.environ.get(
    "SAFEPREFIX_NATIVE_ACQUISITION_VOLUME", "safeprefix-native-failed-trace-runs-v1"
)
CACHE_VOLUME_NAME = os.environ.get("SAFEPREFIX_HF_CACHE_VOLUME", "safeprefix-hf-cache")
APP_NAME = "safeprefix-native-failed-trace-acquisition"


def _local_source_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=LOCAL_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


SOURCE_COMMIT = _local_source_commit()

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
        "rapidfuzz==3.14.0",
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
    .add_local_dir(
        LOCAL_ROOT / PRELAUNCH_RELATIVE,
        str(REMOTE_ROOT / PRELAUNCH_RELATIVE),
        copy=True,
    )
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
        raise ValueError("run_id must be one safe path component")
    return value


def _config() -> dict[str, Any]:
    from safeprefix.config import load_config

    return load_config(REMOTE_ROOT / CONFIG_RELATIVE).data


def _source_paths() -> tuple[Path, Path]:
    root = REMOTE_ROOT / PRELAUNCH_RELATIVE
    return root / "eligible_source_manifest.jsonl", root / "source_manifest_metadata.json"


def _load_sources(config: dict[str, Any]) -> list[dict[str, Any]]:
    from safeprefix.native_failed_trace_acquisition import load_source_manifest

    manifest, metadata = _source_paths()
    rows = load_source_manifest(manifest, metadata)
    if not rows:
        raise RuntimeError("frozen source census is empty")
    return rows


def _state_path(run_id: str, model_key: str) -> Path:
    return RUNS_ROOT / run_id / "remote_run_state" / f"{model_key}.json"


def _write_state(run_id: str, model_key: str, **updates: Any) -> dict[str, Any]:
    from safeprefix.reproducibility import atomic_json

    path = _state_path(run_id, model_key)
    current = {}
    if path.is_file():
        current = json.loads(path.read_text(encoding="utf-8"))
    history = list(current.get("event_history", []))
    history.append({"time_unix": time.time(), **updates})
    current.update(updates, event_history=history)
    atomic_json(path, current)
    run_volume.commit()
    return current


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=30 * 60,
    volumes={"/runs": run_volume, "/cache": cache_volume},
)
def prepare_workspace(run_id: str, model_key: str) -> dict[str, Any]:
    from safeprefix.config import dump_resolved
    from safeprefix.native_failed_trace_pipeline import write_attempt_pack_manifests
    from safeprefix.reproducibility import atomic_json, atomic_text, package_versions, stable_hash

    _safe(run_id)
    config = _config()
    if model_key not in config["selected_models"]:
        raise ValueError(f"unconfigured model key: {model_key}")
    source_rows = _load_sources(config)
    root = RUNS_ROOT / run_id / "artifacts/native_failed_trace_acquisition"
    manifests = write_attempt_pack_manifests(
        config, source_rows, runs_root=RUNS_ROOT, run_id=run_id, model_key=model_key
    )
    source_manifest, source_metadata = _source_paths()
    frozen = root / "immutable_manifests"
    frozen.mkdir(parents=True, exist_ok=True)
    for source in (source_manifest, source_metadata):
        destination = frozen / source.name
        if destination.is_file() and destination.read_bytes() != source.read_bytes():
            raise RuntimeError("immutable source census changed on resume")
        if not destination.is_file():
            shutil.copyfile(source, destination)
    resolved_text = dump_resolved(config)
    resolved_path = frozen / "resolved_config.yaml"
    if resolved_path.is_file() and resolved_path.read_text(encoding="utf-8") != resolved_text:
        raise RuntimeError("resolved configuration changed after run preparation")
    if not resolved_path.is_file():
        atomic_text(resolved_path, resolved_text)
    payload = {
        "status": "PREPARED_NO_GPU_WORK_SUBMITTED",
        "run_id": run_id,
        "model_key": model_key,
        "model": config["models"][model_key],
        "configuration_hash": stable_hash(config),
        "source_rows": len(source_rows),
        "source_counts": {
            stratum: sum(row["stratum"] == stratum for row in source_rows)
            for stratum in ("gsm1k", "math_level_3", "math_level_4")
        },
        "attempt_pack_counts": {key: len(value) for key, value in manifests.items()},
        "generation": config["generation"]["shared"],
        "target": config["acquisition"]["target_by_stratum"],
        "git_commit": os.environ.get("SAFEPREFIX_SOURCE_COMMIT", "unknown"),
        "package_versions": package_versions(),
        "forbidden_stages": {
            key: config["acquisition"][key]
            for key in (
                "semantic_segmentation_enabled", "checkpoint_extraction_enabled",
                "hidden_state_extraction_enabled", "kv_cache_persistence_enabled",
                "boundary_model_enabled",
            )
        },
    }
    if any(payload["forbidden_stages"].values()):
        raise RuntimeError("a prohibited stage is enabled")
    prelaunch_path = root / "immutable_manifests" / f"{model_key}_prelaunch.json"
    if prelaunch_path.is_file():
        existing = json.loads(prelaunch_path.read_text(encoding="utf-8"))
        if (
            existing.get("configuration_hash") != payload["configuration_hash"]
            or existing.get("git_commit") != payload["git_commit"]
        ):
            raise RuntimeError("implementation/configuration revision changed after run preparation")
    else:
        atomic_json(prelaunch_path, payload)
    _write_state(run_id, model_key, status="PREPARED", current_error=None, preparation=payload)
    return payload


@app.cls(
    image=image,
    gpu="H100!",
    cpu=12,
    memory=98304,
    timeout=24 * 60 * 60,
    max_containers=10,
    scaledown_window=30,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=5.0),
    volumes={"/runs": run_volume, "/cache": cache_volume},
    secrets=[hf_secret],
)
class AcquisitionWorker:
    run_id: str = modal.parameter()
    model_key: str = modal.parameter()

    @modal.enter()
    def load_context(self) -> None:
        from safeprefix.models.loader import load_model
        from safeprefix.native_failed_trace_pipeline import attempt_pack_manifests

        _safe(self.run_id)
        self.config = _config()
        if self.model_key not in self.config["selected_models"]:
            raise ValueError(f"unconfigured model key: {self.model_key}")
        self.source_rows = _load_sources(self.config)
        self.source_index = {str(row["source_id"]): row for row in self.source_rows}
        self.attempt_packs = {
            pack["pack_id"]: pack
            for values in attempt_pack_manifests(
                self.config, self.source_rows, self.model_key
            ).values()
            for pack in values
        }
        self.loaded = load_model(self.config["models"][self.model_key])
        expected = str(self.config["models"][self.model_key]["revision"])
        if str(self.loaded.model_revision) != expected:
            raise RuntimeError(
                f"loaded model revision {self.loaded.model_revision!r} != pinned {expected!r}"
            )
        expected_tokenizer = str(
            self.config["models"][self.model_key]["tokenizer_revision"]
        )
        if str(self.loaded.tokenizer_revision) != expected_tokenizer:
            raise RuntimeError(
                f"loaded tokenizer revision {self.loaded.tokenizer_revision!r} "
                f"!= pinned {expected_tokenizer!r}"
            )

    @modal.method()
    def run_attempt_pack(self, pack_id: str, smoke_run_id: str = "") -> dict[str, Any]:
        from safeprefix.native_failed_trace_runtime import execute_attempt_pack

        if pack_id not in self.attempt_packs:
            raise ValueError(f"unknown attempt pack: {pack_id}")
        run_id = smoke_run_id or self.run_id
        run_volume.reload()
        result = execute_attempt_pack(
            config=self.config, runs_root=RUNS_ROOT, run_id=run_id,
            model_key=self.model_key, pack=self.attempt_packs[pack_id], loaded=self.loaded,
            source_index=self.source_index,
        )
        run_volume.commit()
        return result

    @modal.method()
    def run_regeneration_pack(
        self, pack: dict[str, Any], cohort_rows: list[dict[str, Any]],
        smoke_run_id: str = "",
    ) -> dict[str, Any]:
        from safeprefix.native_failed_trace_runtime import execute_regeneration_pack

        run_id = smoke_run_id or self.run_id
        cohort_index = {str(row["trace_id"]): row for row in cohort_rows}
        run_volume.reload()
        result = execute_regeneration_pack(
            config=self.config, runs_root=RUNS_ROOT, run_id=run_id,
            model_key=self.model_key, pack=pack, loaded=self.loaded,
            cohort_index=cohort_index,
        )
        run_volume.commit()
        return result


def _wait(calls: list[Any]) -> list[Any]:
    return [call.get() for call in calls]


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=30 * 60,
    volumes={"/runs": run_volume, "/cache": cache_volume},
)
def run_infrastructure_smoke(run_id: str, model_key: str) -> dict[str, Any]:
    """Exercise one real attempt pack and four regenerations in an isolated run."""

    from safeprefix.native_failed_trace_acquisition import build_regeneration_packs, read_jsonl
    from safeprefix.native_failed_trace_pipeline import attempt_pack_manifests
    from safeprefix.native_failed_trace_runtime import acquisition_root

    config = _config()
    sources = _load_sources(config)
    manifests = attempt_pack_manifests(config, sources, model_key)
    production_pack = manifests["gsm1k"][0]
    smoke_id = f"{run_id}__infrastructure_smoke"
    worker = AcquisitionWorker(run_id=run_id, model_key=model_key)
    first = worker.run_attempt_pack.remote(str(production_pack["pack_id"]), smoke_id)
    second = worker.run_attempt_pack.remote(str(production_pack["pack_id"]), smoke_id)
    if second.get("status") != "SKIPPED_VALID":
        raise RuntimeError("attempt pack resume did not skip the valid immutable smoke shard")
    # The worker commits through its own mounted-volume view. Refresh this
    # function's view before auditing the shard it just produced.
    run_volume.reload()
    attempt_path = (
        acquisition_root(RUNS_ROOT, smoke_id)
        / "attempts/raw" / model_key / "gsm1k" / str(production_pack["pack_id"])
        / "attempts.jsonl"
    )
    attempts = read_jsonl(attempt_path)
    if not attempts:
        raise RuntimeError("smoke attempt pack produced no persisted rows")
    attempt = attempts[0]
    # This isolated infrastructure trace is deliberately not a scientific
    # cohort member. It exists only to test four fresh prompt-root generations.
    smoke_trace = {
        **attempt,
        "trace_id": f"smoke-{attempt['attempt_key']}",
        "frozen_before_regeneration": True,
        "regeneration_conditioned_selection": False,
    }
    regen_pack = build_regeneration_packs(
        [smoke_trace], model_key=model_key, pack_size=1,
        configuration_hash=str(attempt["configuration_hash"]),
    )[0].to_dict()
    regen_first = worker.run_regeneration_pack.remote(regen_pack, [smoke_trace], smoke_id)
    regen_second = worker.run_regeneration_pack.remote(regen_pack, [smoke_trace], smoke_id)
    if regen_second.get("status") != "SKIPPED_VALID":
        raise RuntimeError("regeneration pack resume did not skip the valid immutable smoke shard")
    payload = {
        "status": "PASS",
        "model_key": model_key,
        "smoke_attempt_rows": int(first["row_count"]),
        "smoke_regeneration_rows": int(regen_first["row_count"]),
        "resume_skipped_valid": True,
        "note": "The isolated smoke trace never enters the scientific cohort.",
    }
    root = RUNS_ROOT / run_id / "artifacts/native_failed_trace_acquisition/smoke"
    from safeprefix.reproducibility import atomic_json
    atomic_json(root / f"{model_key}.json", payload)
    run_volume.commit()
    return payload


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    # Modal caps a single function call at 24 hours. The orchestration is
    # pack-resumable, so a retry or explicit resume continues from completed
    # immutable packs rather than requiring an over-limit function timeout.
    timeout=24 * 60 * 60,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=10.0),
    volumes={"/runs": run_volume, "/cache": cache_volume},
    secrets=[hf_secret],
)
def orchestrate_model_core(run_id: str, model_key: str, explicit_authorization: bool) -> dict[str, Any]:
    from safeprefix.native_failed_trace_pipeline import (
        attempt_pack_manifests,
        collect_attempt_rows,
        collect_regeneration_rows,
        finalize_model,
        freeze_model_cohort,
        next_attempt_wave,
        write_regeneration_pack_manifest,
    )

    if not explicit_authorization:
        raise PermissionError("full native inference requires an explicit launch authorization")
    config = _config()
    sources = _load_sources(config)
    state_path = _state_path(run_id, model_key)
    prior_state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    started_unix = float(prior_state.get("started_unix") or time.time())
    _write_state(
        run_id, model_key, status="STARTING_OR_RESUMING", current_error=None,
        started_unix=started_unix, explicit_launch_authorization=True,
    )
    preparation = prepare_workspace.remote(run_id, model_key)
    smoke = run_infrastructure_smoke.remote(run_id, model_key)
    manifests = attempt_pack_manifests(config, sources, model_key)
    worker = AcquisitionWorker(run_id=run_id, model_key=model_key)
    exhausted: dict[str, bool] = {}
    _write_state(
        run_id, model_key, status="ACQUIRING_INITIAL_FAILURES", current_error=None,
        preparation=preparation, smoke=smoke,
    )
    for stratum in ("gsm1k", "math_level_3", "math_level_4"):
        while True:
            run_volume.reload()
            attempts, completed = collect_attempt_rows(
                runs_root=RUNS_ROOT, run_id=run_id, model_key=model_key,
                manifests=manifests,
            )
            wave, state = next_attempt_wave(
                config, manifests, attempts, completed,
                stratum=stratum, prior_strata_exhausted=exhausted,
            )
            _write_state(run_id, model_key, status=f"ACQUIRING_{stratum}", stratum_state=state)
            if not wave:
                exhausted[stratum] = bool(state["source_exhausted"] and not state["quota_reached"])
                break
            calls = [worker.run_attempt_pack.spawn(str(pack["pack_id"])) for pack in wave]
            _wait(calls)
    run_volume.reload()
    attempts, _completed = collect_attempt_rows(
        runs_root=RUNS_ROOT, run_id=run_id, model_key=model_key, manifests=manifests,
    )
    cohort, cohort_summary = freeze_model_cohort(
        config, attempts, runs_root=RUNS_ROOT, run_id=run_id, model_key=model_key,
    )
    regeneration_packs = write_regeneration_pack_manifest(
        config, cohort, runs_root=RUNS_ROOT, run_id=run_id, model_key=model_key,
    )
    _write_state(
        run_id, model_key, status="COHORT_FROZEN_RUNNING_FULL_REGENERATIONS",
        cohort_summary=cohort_summary, regeneration_pack_count=len(regeneration_packs),
    )
    _existing, complete = collect_regeneration_rows(
        runs_root=RUNS_ROOT, run_id=run_id, model_key=model_key,
        packs=regeneration_packs,
    )
    completed_ids = set(complete)
    calls = [
        worker.run_regeneration_pack.spawn(pack, cohort)
        for pack in regeneration_packs
        if str(pack["pack_id"]) not in completed_ids
    ]
    _wait(calls)
    run_volume.reload()
    summary = finalize_model(
        config, sources, runs_root=RUNS_ROOT, run_id=run_id, model_key=model_key,
        manifests=manifests, regeneration_packs=regeneration_packs,
    )
    _write_state(
        run_id, model_key, status="COMPLETE", current_error=None,
        finished_unix=time.time(), summary=summary,
    )
    run_volume.commit()
    return summary


@app.function(
    image=image,
    cpu=2,
    memory=8192,
    timeout=24 * 60 * 60,
    volumes={"/runs": run_volume, "/cache": cache_volume},
)
def orchestrate_model(run_id: str, model_key: str, explicit_authorization: bool) -> dict[str, Any]:
    """Record terminal failures separately while allowing a clean resume."""

    try:
        return orchestrate_model_core.remote(run_id, model_key, explicit_authorization)
    except Exception as exc:
        run_volume.reload()
        _write_state(
            run_id,
            model_key,
            status="FAILED_RESUMABLE",
            current_error=f"{type(exc).__name__}: {exc}",
            failed_unix=time.time(),
        )
        raise


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=2 * 60 * 60,
    volumes={"/runs": run_volume},
)
def finalize_existing_artifacts(
    run_id: str, model_key: str, allow_partial: bool = False,
) -> dict[str, Any]:
    """CPU-only finalization; never launches or resumes model inference."""

    from safeprefix.config import load_config
    from safeprefix.native_failed_trace_acquisition import read_jsonl
    from safeprefix.native_failed_trace_pipeline import (
        finalize_model,
        finalize_partial_model,
    )

    _safe(run_id)
    run_volume.reload()
    root = RUNS_ROOT / run_id / "artifacts/native_failed_trace_acquisition"
    final_summary_path = root / "final" / model_key / "summary.json"
    final_integrity_path = root / "final" / model_key / "integrity_report.json"
    if final_summary_path.is_file() and final_integrity_path.is_file():
        summary = json.loads(final_summary_path.read_text(encoding="utf-8"))
        integrity = json.loads(final_integrity_path.read_text(encoding="utf-8"))
        if summary.get("status") == "COMPLETE" and integrity.get("status") == "PASS":
            return {
                "status": "SKIPPED_ALREADY_COMPLETE",
                "model_key": model_key,
                "summary": summary,
            }

    config_path = root / "immutable_manifests" / "resolved_config.yaml"
    source_path = root / "immutable_manifests" / "eligible_source_manifest.jsonl"
    if not config_path.is_file() or not source_path.is_file():
        raise RuntimeError("frozen config or source manifest is missing")
    config = load_config(config_path).data
    sources = read_jsonl(source_path)
    manifests = {
        stratum: read_jsonl(
            root / "execution_packs" / "attempts" / model_key / f"{stratum}.jsonl"
        )
        for stratum in ("gsm1k", "math_level_3", "math_level_4")
    }
    regeneration_packs = read_jsonl(
        root / "execution_packs" / "regenerations" / f"{model_key}.jsonl"
    )
    complete_pack_count = sum(
        (root / "regenerations" / "raw" / model_key / str(pack["pack_id"]) / "complete.json").is_file()
        for pack in regeneration_packs
    )
    if complete_pack_count == len(regeneration_packs):
        summary = finalize_model(
            config, sources, runs_root=RUNS_ROOT, run_id=run_id, model_key=model_key,
            manifests=manifests, regeneration_packs=regeneration_packs,
        )
        state_status = "COMPLETE"
    else:
        if not allow_partial:
            raise RuntimeError(
                f"{model_key} has {complete_pack_count}/{len(regeneration_packs)} packs; "
                "partial finalization was not authorized"
            )
        summary = finalize_partial_model(
            config, sources, runs_root=RUNS_ROOT, run_id=run_id, model_key=model_key,
            manifests=manifests, regeneration_packs=regeneration_packs,
        )
        state_status = "PARTIAL_COMPLETE_USER_DIRECTED"
    _write_state(
        run_id, model_key, status=state_status, current_error=None,
        finished_unix=time.time(), summary=summary,
    )
    run_volume.commit()
    return summary


@app.local_entrypoint()
def main(
    run_id: str,
    model_key: str,
    prepare_only: bool = True,
    authorize_full_inference: bool = False,
    finalize_existing: bool = False,
    allow_partial_finalization: bool = False,
) -> None:
    """Prepare by default; explicit authorization is required for any GPU work."""

    _safe(run_id)
    if finalize_existing:
        result = finalize_existing_artifacts.remote(
            run_id, model_key, allow_partial_finalization,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if prepare_only:
        result = prepare_workspace.remote(run_id, model_key)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if not authorize_full_inference:
        raise PermissionError(
            "refusing GPU submission without --authorize-full-inference; "
            "the current user instruction explicitly prohibits Modal submission"
        )
    call = orchestrate_model.spawn(run_id, model_key, True)
    print(json.dumps({"status": "SUBMITTED", "function_call_id": call.object_id, "model_key": model_key}, indent=2))
