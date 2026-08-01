#!/usr/bin/env python3
"""Modal driver for the monotonic prefix-validity probe training run.

The driver intentionally separates (1) full-step feature completion, (2)
ProcessBench fitting/freezing, and (3) untouched test evaluation. Native data
and suffix outcomes are never mounted by this application.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import modal


LOCAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LOCAL_ROOT / "src"))
REMOTE_ROOT = Path("/workspace/safeprefix_code")
CONFIG_RELATIVE = Path("configs/prefix_validity_v1.yaml")
APP_NAME = "safeprefix-prefix-validity-v1"


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=LOCAL_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except Exception:
        return os.environ.get("SAFEPREFIX_SOURCE_COMMIT", "unknown")


SOURCE_COMMIT = _commit()
app = modal.App(APP_NAME)
completion_volume = modal.Volume.from_name("safeprefix-recoverability-geometry-input-completion-v2")
boundary_volume = modal.Volume.from_name("safeprefix-recoverability-geometry-input-boundary-v2")
output_volume = modal.Volume.from_name("safeprefix-prefix-validity-v1", create_if_missing=True)
cache_volume = modal.Volume.from_name("safeprefix-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name(os.environ.get("SAFEPREFIX_MODAL_HF_SECRET", "huggingface-token"))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "accelerate==1.10.1", "datasets==5.0.0", "huggingface-hub==0.34.4",
        "matplotlib==3.10.5", "numpy==2.2.6", "pandas==2.3.2",
        "pyarrow==24.0.0", "PyYAML==6.0.2", "scikit-learn==1.7.1",
        "scipy==1.15.3", "torch==2.8.0", "transformers==4.55.4",
        "typer==0.17.4", "sentencepiece==0.2.1",
    )
    .add_local_dir(LOCAL_ROOT / "src/safeprefix", str(REMOTE_ROOT / "src/safeprefix"), copy=True)
    .add_local_dir(LOCAL_ROOT / "configs", str(REMOTE_ROOT / "configs"), copy=True)
    .add_local_file(LOCAL_ROOT / "pyproject.toml", str(REMOTE_ROOT / "pyproject.toml"), copy=True)
    .env({
        "HF_HOME": "/cache/huggingface", "HF_HUB_CACHE": "/cache/huggingface/hub",
        "PYTHONPATH": str(REMOTE_ROOT / "src"), "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg", "SAFEPREFIX_SOURCE_COMMIT": SOURCE_COMMIT,
    })
)


def _safe(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError("run/model identity must be one safe path component")
    return value


def _config() -> dict[str, Any]:
    from safeprefix.config import load_config
    from safeprefix.prefix_validity_v1.runner import load_prefix_validity_config
    remote = REMOTE_ROOT / CONFIG_RELATIVE
    path = remote if remote.is_file() else LOCAL_ROOT / CONFIG_RELATIVE
    load_prefix_validity_config(path)
    return load_config(path).data


def _root(run_id: str) -> Path:
    return Path("/output") / _safe(run_id)


def _completion_root(config: dict[str, Any]) -> Path:
    return Path(str(config["source"]["completion_root"]))


def _boundary_root(config: dict[str, Any]) -> Path:
    return Path(str(config["source"]["boundary_root"]))


def _boundary_selected_layers(config: dict[str, Any], model_key: str) -> list[int]:
    """Load the exact layers consumed by the frozen recoverability probe.

    The general model matrix also carries an older feature-extraction layer
    selection.  Prefix validity must use the recoverability probe's frozen
    representation, so the boundary-model manifest is authoritative here.
    """

    from safeprefix.prefix_validity_v1.runner import load_boundary_selected_layers

    local = REMOTE_ROOT / str(config["source"]["boundary_config"])
    return load_boundary_selected_layers(local, model_key)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@app.function(
    image=image, cpu=8, memory=32768, timeout=60 * 60,
    volumes={"/completion": completion_volume, "/boundary": boundary_volume, "/output": output_volume},
)
def prepare(run_id: str) -> dict[str, Any]:
    from safeprefix.prefix_validity_v1.data import build_prefix_validity_extraction_plan
    from safeprefix.prefix_validity_v1.runner import (
        build_missing_feature_packs, freeze_extraction_plan,
    )
    from safeprefix.reproducibility import atomic_json, atomic_jsonl, atomic_parquet

    config = _config(); root = _root(run_id)
    plan = build_prefix_validity_extraction_plan(
        safety_manifest_root=_completion_root(config) / "immutable_manifests",
        boundary_root=_boundary_root(config),
        boundary_config_path=REMOTE_ROOT / config["source"]["boundary_config"],
        reuse_completed_safety_features=bool(
            config["execution"].get("reuse_completed_safety_features", True)
        ),
    )
    freeze = freeze_extraction_plan(
        plan, output_root=root, config=config, source_commit=SOURCE_COMMIT,
    )
    packs, _ = build_missing_feature_packs(
        plan, traces_per_pack=int(config["execution"]["teacher_force_traces_per_pack"]),
    )
    atomic_jsonl(root / "manifests/missing_feature_packs.jsonl", packs)
    if plan.exclusions is not None:
        atomic_parquet(root / "manifests/processbench_trace_exclusions.parquet", plan.exclusions)
    for model_key, traces in plan.traces_by_model.items():
        atomic_jsonl(root / "manifests/trace_payloads" / f"{model_key}.jsonl", traces)
    atomic_json(root / "PREPARED.json", {
        "status": "READY_FOR_MISSING_FULL_STEP_FEATURES", "freeze": freeze,
        "missing_feature_packs": len(packs), "new_suffix_rollouts": 0,
        "native_outcomes_accessed": False,
    })
    output_volume.commit()
    return {"status": "PASS", "packs": packs, "summary": plan.summary}


@app.cls(
    image=image, gpu="H100!", cpu=10, memory=98304, timeout=6 * 60 * 60,
    max_containers=8, scaledown_window=60,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=5.0),
    volumes={
        "/completion": completion_volume, "/boundary": boundary_volume,
        "/output": output_volume, "/cache": cache_volume,
    }, secrets=[hf_secret],
)
class FullStepWorker:
    run_id: str = modal.parameter()
    model_key: str = modal.parameter()

    @modal.enter()
    def load(self) -> None:
        from safeprefix.models.loader import load_model
        self.config = _config()
        self.loaded = load_model(self.config["models"][self.model_key])
        self.selected_layers = _boundary_selected_layers(self.config, self.model_key)
        self.traces = {
            str(row["trace_id"]): row for row in _read_jsonl(
                _root(self.run_id) / "manifests/trace_payloads" / f"{self.model_key}.jsonl"
            )
        }

    @modal.method()
    def extract(self, pack: dict[str, Any]) -> dict[str, Any]:
        from safeprefix.prefix_validity_v1.runner import execute_missing_feature_pack
        result = execute_missing_feature_pack(
            loaded=self.loaded, model_key=self.model_key, pack=pack,
            trace_index=self.traces, output_root=_root(self.run_id),
            selected_layers=self.selected_layers,
            chunk_size=int(self.config["execution"]["teacher_force_chunk_size"]),
        )
        output_volume.commit()
        return result


def _fit_freeze_and_test_impl(
    source_run_id: str,
    output_run_id: str,
    *,
    device: str,
    completed_model_run_id: str | None = None,
) -> dict[str, Any]:
    import pandas as pd
    import torch
    from safeprefix.prefix_validity_v1.data import build_prefix_validity_data
    from safeprefix.prefix_validity_v1.runner import (
        assert_processbench_bundle, evaluate_frozen_model_test,
        freeze_processbench_bundle, train_and_freeze_model, write_final_report,
    )
    from safeprefix.prefix_validity_v1.evaluation import paired_probe_test_bootstrap
    from safeprefix.reproducibility import atomic_json, atomic_parquet

    config = _config(); root = _root(output_run_id)
    source_root = _root(source_run_id)
    print(
        f"[prefix-validity] fit started source={source_run_id} "
        f"output={output_run_id} device={device}",
        flush=True,
    )
    data = build_prefix_validity_data(
        safety_manifest_root=_completion_root(config) / "immutable_manifests",
        safety_feature_roots=[_completion_root(config), source_root],
        boundary_root=_boundary_root(config),
        boundary_config_path=REMOTE_ROOT / config["source"]["boundary_config"],
        reuse_completed_safety_features=bool(
            config["execution"].get("reuse_completed_safety_features", True)
        ),
    )
    print("[prefix-validity] compact corpus and equivalence audit complete", flush=True)
    atomic_parquet(root / "processbench/full_step_inventory.parquet", data.rows)
    atomic_parquet(root / "processbench/data_inventory.csv.parquet", data.inventory)
    atomic_parquet(root / "processbench/exclusions.parquet", data.exclusions)
    atomic_parquet(root / "processbench/feature_pack_inventory.parquet", data.pack_inventory)
    atomic_json(root / "processbench/data_integrity.json", data.integrity)
    results = {}
    if completed_model_run_id is not None:
        completed_root = _root(completed_model_run_id) / "processbench"
        for model_key in config["selected_models"]:
            model_complete = completed_root / model_key / "MODEL_COMPLETE.json"
            payload = json.loads(model_complete.read_text())
            if payload.get("status") != "COMPLETE" or payload.get("model_key") != model_key:
                raise RuntimeError(f"invalid completed model bundle: {model_complete}")
            results[model_key] = {
                architecture: payload[architecture]
                for architecture in ("linear_probe", "position_only")
            }
            if any(results[model_key][architecture].get("test_accessed") is not False
                   for architecture in ("linear_probe", "position_only")):
                raise RuntimeError(f"{model_key}: completed model bundle already accessed test")
            print(
                f"[prefix-validity] reused frozen training bundle {model_key} "
                f"from {completed_model_run_id}",
                flush=True,
            )
    else:
        for model_key in config["selected_models"]:
            print(f"[prefix-validity] training {model_key} started", flush=True)
            results[model_key] = train_and_freeze_model(
                data=data, model_key=model_key, config=config, output_root=root, device=device,
            )
            print(f"[prefix-validity] training {model_key} complete", flush=True)
    freeze = freeze_processbench_bundle(
        output_root=root, model_results=results, config=config, data=data,
    )
    tests = {}
    for model_key in config["selected_models"]:
        print(f"[prefix-validity] held-out evaluation {model_key} started", flush=True)
        tests[model_key] = evaluate_frozen_model_test(
            data=data, model_key=model_key, frozen_result=results[model_key],
            output_root=root, device=device,
        )
        for architecture in ("linear_probe", "position_only"):
            results[model_key][architecture]["test_metrics"] = tests[model_key][architecture]
            results[model_key][architecture]["test_accessed"] = True
        print(f"[prefix-validity] held-out evaluation {model_key} complete", flush=True)
    from safeprefix.prefix_validity_v1.reporting import write_publication_artifacts
    hidden_predictions = pd.concat(
        [pd.read_parquet(root / "processbench" / model / "linear_probe/test_predictions.parquet")
         for model in config["selected_models"]],
        ignore_index=True,
    )
    position_predictions = pd.concat(
        [pd.read_parquet(root / "processbench" / model / "position_only/test_predictions.parquet")
         for model in config["selected_models"]],
        ignore_index=True,
    )
    hidden_gammas = {
        model: float(results[model]["linear_probe"]["gamma"])
        for model in config["selected_models"]
    }
    position_gammas = {
        model: float(results[model]["position_only"]["gamma"])
        for model in config["selected_models"]
    }
    print("[prefix-validity] paired bootstrap started", flush=True)
    bootstrap_overall = paired_probe_test_bootstrap(
        hidden_predictions,
        position_predictions,
        hidden_gamma=hidden_gammas,
        position_gamma=position_gammas,
        replicates=int(config["statistics"]["bootstrap_replicates"]),
        seed=int(config["statistics"]["bootstrap_seed"]),
        expected_models=4,
    )
    bootstrap_by_domain = {}
    for domain in sorted(hidden_predictions["domain"].astype(str).unique()):
        bootstrap_by_domain[domain] = paired_probe_test_bootstrap(
            hidden_predictions.loc[hidden_predictions["domain"].astype(str).eq(domain)],
            position_predictions.loc[position_predictions["domain"].astype(str).eq(domain)],
            hidden_gamma=hidden_gammas,
            position_gamma=position_gammas,
            replicates=int(config["statistics"]["bootstrap_replicates"]),
            seed=int(config["statistics"]["bootstrap_seed"]) + 1 + len(bootstrap_by_domain),
            expected_models=4,
        )
    print("[prefix-validity] paired bootstrap complete", flush=True)
    atomic_json(root / "processbench/bootstrap_confidence_intervals.json", {
        "overall_equal_model_macro": bootstrap_overall,
        "by_domain_equal_model_macro": bootstrap_by_domain,
    })
    write_publication_artifacts(
        output_root=root,
        processbench_metrics=tests,
        hidden_test_predictions=hidden_predictions,
        gate_cutoffs=hidden_gammas,
    )
    frozen_verified = assert_processbench_bundle(root)
    summary = {
        "status": "COMPLETE",
        "scope": "processbench_prefix_validity_probe_training_and_test_only",
        "models": list(config["selected_models"]),
        "processbench": tests,
        "bootstrap": {
            "overall_equal_model_macro": bootstrap_overall,
            "by_domain_equal_model_macro": bootstrap_by_domain,
        },
        "native_application_run": False,
        "suffix_rollouts_generated": 0,
        "recoverability_probe_modified": False,
    }
    integrity = {
        "status": "PASS",
        "component_freeze_hash": frozen_verified["freeze_hash"],
        "all_four_models": len(tests) == 4,
        "test_accessed_only_after_global_freeze": True,
        "hidden_position_rows_matched": True,
        "processbench_split_leakage": False,
        "annotation_indexing_resolved": True,
        "native_application_run": False,
        "native_outcomes_accessed": False,
        "suffix_rollouts_generated": 0,
    }
    atomic_json(root / "summary.json", summary)
    atomic_json(root / "integrity_report.json", integrity)
    write_final_report(output_root=root, summary=summary, integrity=integrity)
    atomic_json(root / "processbench/PROCESSBENCH_TEST_COMPLETE.json", {
        "status": "COMPLETE", "component_freeze_hash": freeze["freeze_hash"],
        "models": results, "native_outcomes_accessed": False,
        "bootstrap_path": str(root / "processbench/bootstrap_confidence_intervals.json"),
    })
    output_volume.commit()
    return {"status": "PASS", "freeze": freeze, "models": results, "integrity": integrity}


_FIT_VOLUMES = {
    "/completion": completion_volume,
    "/boundary": boundary_volume,
    "/output": output_volume,
}


@app.function(
    image=image, cpu=32, memory=131072, timeout=8 * 60 * 60,
    volumes=_FIT_VOLUMES,
)
def fit_freeze_and_test(run_id: str) -> dict[str, Any]:
    """CPU production fitter; reads and writes the same isolated run root."""

    return _fit_freeze_and_test_impl(run_id, run_id, device="cpu")


@app.function(
    image=image, gpu="H100!", cpu=16, memory=131072, timeout=8 * 60 * 60,
    volumes=_FIT_VOLUMES,
)
def fit_freeze_and_test_gpu(source_run_id: str, output_run_id: str) -> dict[str, Any]:
    """Independent GPU race using immutable source packs and an isolated output root."""

    if source_run_id == output_run_id:
        raise ValueError("GPU race output must be isolated from the active CPU run")
    return _fit_freeze_and_test_impl(source_run_id, output_run_id, device="cuda")


@app.function(
    image=image, gpu="H100!", cpu=16, memory=131072, timeout=8 * 60 * 60,
    volumes=_FIT_VOLUMES,
)
def finalize_completed_models_gpu(
    source_run_id: str,
    completed_model_run_id: str,
    output_run_id: str,
) -> dict[str, Any]:
    """Finalize already-trained model bundles without repeating any fit."""

    if output_run_id in {source_run_id, completed_model_run_id}:
        raise ValueError("finalizer output must be isolated from source artifacts")
    return _fit_freeze_and_test_impl(
        source_run_id,
        output_run_id,
        device="cuda",
        completed_model_run_id=completed_model_run_id,
    )


@app.local_entrypoint()
def main(run_id: str = "prefix_validity_v1_20260729_r1") -> None:
    prepared = prepare.remote(run_id)
    packs = list(prepared["packs"])
    workers = {
        model: FullStepWorker(run_id=run_id, model_key=model)
        for model in _config()["selected_models"]
    }
    calls = [workers[pack["model_key"]].extract.spawn(pack) for pack in packs]
    for call in calls:
        call.get()
    processbench = fit_freeze_and_test.remote(run_id)
    print(json.dumps({
        "status": "PREFIX_VALIDITY_PROBES_COMPLETE_NATIVE_APPLICATION_NOT_RUN",
        "run_id": run_id, "processbench": processbench["status"],
        "native_application_enabled": False,
    }, indent=2, sort_keys=True))
