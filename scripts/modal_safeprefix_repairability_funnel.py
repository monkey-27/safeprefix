#!/usr/bin/env python3
"""Run the SafePrefix repairability-funnel analysis in the larpmonk workspace.

The only GPU work is deterministic teacher forcing of already persisted failed
token IDs to recover missing original-path hidden states.  No sampling API is
called.  The native-evaluation volume is intentionally not defined or mounted.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import modal


LOCAL_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/workspace/safeprefix_code")
GEOMETRY_ROOT = Path("/geometry")
BOUNDARY_ROOT = Path("/boundary")
SOURCE_RUN_ID = "safeprefix_recoverability_geometry_tf_v1_20260728_r1"
BOUNDARY_RUN_ID = "safeprefix_boundary_model_v1_20260728_r2"
DEFAULT_RUN_ID = "safeprefix_repairability_funnel_v1_20260729_r1"
MODEL_KEYS = (
    "family_a_small",
    "family_a_large",
    "family_b_small",
    "family_b_large",
)

APP_NAME = "safeprefix-repairability-funnel-v1"
GEOMETRY_VOLUME_NAME = os.environ.get(
    "SAFEPREFIX_GEOMETRY_VOLUME", "safeprefix-recoverability-geometry-output-v2"
)
BOUNDARY_VOLUME_NAME = os.environ.get(
    "SAFEPREFIX_BOUNDARY_VOLUME", "safeprefix-recoverability-geometry-input-boundary-v2"
)
CACHE_VOLUME_NAME = os.environ.get("SAFEPREFIX_HF_CACHE_VOLUME", "safeprefix-hf-cache")


def _source_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=LOCAL_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


SOURCE_COMMIT = _source_commit()
app = modal.App(APP_NAME)
geometry_volume = modal.Volume.from_name(GEOMETRY_VOLUME_NAME, create_if_missing=False, version=2)
boundary_volume = modal.Volume.from_name(BOUNDARY_VOLUME_NAME, create_if_missing=False, version=2)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True, version=1)
hf_secret = modal.Secret.from_name(os.environ.get("SAFEPREFIX_MODAL_HF_SECRET", "huggingface-token"))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "accelerate==1.10.1",
        "huggingface-hub==0.34.4",
        "matplotlib==3.10.5",
        "numpy==2.2.6",
        "pandas==2.3.2",
        "pyarrow==24.0.0",
        "PyYAML==6.0.2",
        "scikit-learn==1.7.1",
        "scipy==1.16.1",
        "torch==2.8.0",
        "transformers==4.55.4",
        "sentencepiece==0.2.1",
    )
    .add_local_dir(LOCAL_ROOT / "src/safeprefix", str(REMOTE_ROOT / "src/safeprefix"), copy=True)
    .add_local_dir(LOCAL_ROOT / "configs", str(REMOTE_ROOT / "configs"), copy=True)
    .add_local_file(
        LOCAL_ROOT / "scripts/modal_safeprefix_repairability_funnel.py",
        str(REMOTE_ROOT / "scripts/modal_safeprefix_repairability_funnel.py"),
        copy=True,
    )
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "HF_HUB_CACHE": "/cache/huggingface/hub",
            "MPLCONFIGDIR": "/tmp/matplotlib",
            "PYTHONPATH": str(REMOTE_ROOT / "src"),
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
)

VOLUMES = {
    "/geometry": geometry_volume,
    "/boundary": boundary_volume.with_mount_options(read_only=True),
    "/cache": cache_volume,
}


def _safe(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError("run_id must be one safe path component")
    return value


def _source_root() -> Path:
    return GEOMETRY_ROOT / SOURCE_RUN_ID / "artifacts/recoverability_geometry_tf_v1"


def _output_root(run_id: str) -> Path:
    return GEOMETRY_ROOT / _safe(run_id) / "artifacts/repairability_funnel_v1"


def _boundary_root() -> Path:
    return BOUNDARY_ROOT / BOUNDARY_RUN_ID / "artifacts/boundary_model_v1"


def _config() -> dict[str, Any]:
    from safeprefix.config import load_config

    return load_config(REMOTE_ROOT / "configs/full_teacher_forced_suite.yaml").data


def _digest() -> str:
    root = REMOTE_ROOT if (REMOTE_ROOT / "src/safeprefix/repairability_funnel.py").is_file() else LOCAL_ROOT
    paths = [
        root / "src/safeprefix/repairability_funnel.py",
        root / "src/safeprefix/recoverability_geometry/analysis.py",
        root / "src/safeprefix/recoverability_geometry/runner.py",
        root / "src/safeprefix/models/teacher_forcing.py",
        root / "src/safeprefix/models/loader.py",
        root / "src/safeprefix/reproducibility.py",
        root / "scripts/modal_safeprefix_repairability_funnel.py",
        root / "configs/repairability_funnel_v1.yaml",
        root / "configs/full_teacher_forced_suite.yaml",
        root / "configs/models.yaml",
        root / "configs/datasets.yaml",
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


@app.cls(
    image=image,
    gpu="H100!",
    cpu=12,
    memory=98304,
    timeout=24 * 60 * 60,
    max_containers=4,
    scaledown_window=30,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=5.0),
    volumes=VOLUMES,
    secrets=[hf_secret],
)
class OriginalPathReplayWorker:
    model_key: str = modal.parameter()

    @modal.enter()
    def load_context(self) -> None:
        from safeprefix.models.loader import load_model

        if self.model_key not in MODEL_KEYS:
            raise ValueError(f"unknown model key: {self.model_key}")
        config = _config()
        self.entry = config["models"][self.model_key]
        self.loaded = load_model(self.entry)
        if str(self.loaded.model_revision) != str(self.entry["revision"]):
            raise RuntimeError("loaded model revision differs from frozen configuration")
        if str(self.loaded.tokenizer_revision) != str(self.entry["tokenizer_revision"]):
            raise RuntimeError("loaded tokenizer revision differs from frozen configuration")
        self.prefill_chunk_size = int(config["production_execution"]["prefill_chunk_size"])

    @modal.method()
    def replay(self, run_id: str, source_digest: str) -> dict[str, Any]:
        import pandas as pd

        from safeprefix.models.teacher_forcing import teacher_force_token_ids_chunked
        from safeprefix.reproducibility import atomic_json, atomic_parquet

        if source_digest != _digest():
            raise RuntimeError("deployed source digest mismatch")
        geometry_volume.reload()
        output_state = _output_root(run_id) / f"inputs/original_path_states/{self.model_key}.parquet"
        output_marker = _output_root(run_id) / f"inputs/original_path_states/{self.model_key}.complete.json"
        if output_state.is_file() and output_marker.is_file():
            previous = json.loads(output_marker.read_text())
            if (
                previous.get("status") == "COMPLETE"
                and previous.get("model_key") == self.model_key
                and previous.get("source_digest") == source_digest
                and previous.get("deterministic_stored_token_replay") is True
                and previous.get("tokens_sampled") == 0
                and previous.get("suffix_rollouts_generated") == 0
                and previous.get("native_volume_mounted") is False
            ):
                return {**previous, "status": "SKIPPED_VALID"}
        source = _source_root()
        parents = pd.read_json(source / "analysis/local_parent_selection.jsonl", lines=True)
        parents = parents.loc[parents["base_model"].astype(str) == self.model_key].copy()
        traces = {
            str(row["trace_id"]): row
            for row in pd.read_json(
                source / f"manifests/source_traces/{self.model_key}.jsonl", lines=True
            ).to_dict("records")
        }
        child_path = source / "analysis/local_child_state_inputs.parquet"
        if child_path.is_file():
            children = pd.read_parquet(
                child_path,
                filters=[[("base_model", "==", self.model_key)]],
            )
            persisted_parent = (
                children.loc[children["parent_raw_hidden"].notna()]
                .sort_values(["trace_id", "checkpoint_index", "horizon", "branch_index"])
                .groupby(["trace_id", "checkpoint_index"], sort=True)
                .head(1)
                .set_index(["trace_id", "checkpoint_index"])["parent_raw_hidden"]
                .to_dict()
            )
        else:
            # Some authorized workspaces hold the parent/source bridge but not
            # the completed H4 child aggregate.  The boundary feature store is
            # the canonical source of the exact persisted parent raw hidden.
            torch = __import__("torch")
            feature_payload = torch.load(
                _boundary_root() / f"data/features/{self.model_key}.pt",
                map_location="cpu",
                weights_only=False,
            )
            raw_features = feature_payload["features"].to(torch.float32).numpy()
            persisted_parent = {
                (str(parent["trace_id"]), int(parent["checkpoint_index"])): raw_features[
                    int(parent["feature_row_index"])
                ]
                for parent in parents.to_dict("records")
            }
        rows: list[dict[str, Any]] = []
        audit_errors: list[float] = []
        audit_cosines: list[float] = []
        audit_relative_rmse: list[float] = []
        for parent in parents.sort_values(["trace_id", "checkpoint_index"]).to_dict("records"):
            trace_id = str(parent["trace_id"])
            source_row = traces.get(trace_id)
            if source_row is None:
                raise RuntimeError(f"source trace absent for selected parent: {trace_id}")
            prompt_ids = list(map(int, source_row["prompt_token_ids"]))
            completion_ids = list(map(int, source_row["completion_token_ids"]))
            token_ids = prompt_ids + completion_ids
            stored_token_count = len(token_ids)
            checkpoint = int(parent["checkpoint_token_offset"])
            requested = [checkpoint - 1]
            available_horizons: list[int] = []
            for horizon in (32, 64, 128):
                offset = checkpoint + horizon - 1
                if offset < len(token_ids):
                    requested.append(offset)
                    available_horizons.append(horizon)
            # Causal states at the requested offsets cannot depend on a later
            # stored token.  Truncation is deterministic replay, not generation.
            token_ids = token_ids[: max(requested) + 1]
            forced = teacher_force_token_ids_chunked(
                self.loaded.model,
                token_ids,
                prompt_count=len(prompt_ids),
                chunk_size=self.prefill_chunk_size,
                selected_layers=(-1,),
                selected_token_offsets=requested,
            )
            hidden = forced.selected_hidden_states[-1]
            replayed_parent = hidden[checkpoint - 1].to(dtype=__import__("torch").float32).numpy()
            expected_parent = persisted_parent[(trace_id, int(parent["checkpoint_index"]))]
            np = __import__("numpy")
            expected_parent = np.asarray(expected_parent, dtype="float32")
            difference = replayed_parent - expected_parent
            error = float(abs(difference).max())
            cosine = float(
                np.dot(replayed_parent, expected_parent)
                / max(np.linalg.norm(replayed_parent) * np.linalg.norm(expected_parent), 1e-12)
            )
            relative_rmse = float(
                np.sqrt(np.mean(np.square(difference)))
                / max(float(np.std(expected_parent)), 1e-12)
            )
            if cosine < 0.995 or relative_rmse > 0.10:
                raise RuntimeError(
                    "deterministic parent replay differs materially from persisted state: "
                    f"{trace_id} cosine={cosine} relative_rmse={relative_rmse} max_abs={error}"
                )
            audit_errors.append(error)
            audit_cosines.append(cosine)
            audit_relative_rmse.append(relative_rmse)
            for horizon in (32, 64, 128):
                available = horizon in available_horizons
                raw = (
                    hidden[checkpoint + horizon - 1].to(dtype=__import__("torch").float16).numpy()
                    if available
                    else None
                )
                rows.append(
                    {
                        "base_model": self.model_key,
                        "trace_id": trace_id,
                        "checkpoint_index": int(parent["checkpoint_index"]),
                        "checkpoint_token_offset": checkpoint,
                        "horizon": int(horizon),
                        "raw_hidden": raw,
                        "replayed_parent_raw_hidden": replayed_parent.astype("float16"),
                        "replay_status": "complete" if available else "stored_failed_sequence_too_short",
                        "stored_token_count": stored_token_count,
                        "selected_token_offset": checkpoint + horizon - 1,
                        "parent_replay_max_abs_error": error,
                        "parent_replay_cosine": cosine,
                        "parent_replay_relative_rmse": relative_rmse,
                        "deterministic_stored_token_replay": True,
                        "tokens_sampled": 0,
                        "suffix_rollouts_generated": 0,
                    }
                )
            del forced
        output = pd.DataFrame(rows)
        root = _output_root(run_id) / "inputs/original_path_states"
        atomic_parquet(root / f"{self.model_key}.parquet", output)
        summary = {
            "status": "COMPLETE",
            "model_key": self.model_key,
            "parent_count": int(len(parents)),
            "row_count": int(len(output)),
            "available_state_count": int((output["replay_status"] == "complete").sum()),
            "unavailable_stored_sequence_too_short": int((output["replay_status"] != "complete").sum()),
            "maximum_parent_replay_abs_error": max(audit_errors) if audit_errors else None,
            "minimum_parent_replay_cosine": min(audit_cosines) if audit_cosines else None,
            "maximum_parent_replay_relative_rmse": (
                max(audit_relative_rmse) if audit_relative_rmse else None
            ),
            "deterministic_stored_token_replay": True,
            "tokens_sampled": 0,
            "suffix_rollouts_generated": 0,
            "source_digest": source_digest,
            "source_commit": SOURCE_COMMIT,
            "native_volume_mounted": False,
        }
        atomic_json(root / f"{self.model_key}.complete.json", summary)
        geometry_volume.commit()
        return summary


@app.function(
    image=image,
    cpu=32,
    memory=131072,
    timeout=24 * 60 * 60,
    volumes=VOLUMES,
)
def analyze_remote(run_id: str, source_digest: str) -> dict[str, Any]:
    import pandas as pd

    from safeprefix.repairability_funnel import run_analysis
    from safeprefix.reproducibility import atomic_parquet

    if source_digest != _digest():
        raise RuntimeError("deployed source digest mismatch")
    geometry_volume.reload()
    root = _output_root(run_id)
    original_parts = []
    for model in MODEL_KEYS:
        state_path = root / f"inputs/original_path_states/{model}.parquet"
        marker_path = root / f"inputs/original_path_states/{model}.complete.json"
        if not state_path.is_file() or not marker_path.is_file():
            raise RuntimeError(f"original-path replay incomplete for {model}")
        marker = json.loads(marker_path.read_text())
        if (
            marker.get("status") != "COMPLETE"
            or marker.get("model_key") != model
            or marker.get("source_digest") != source_digest
            or marker.get("deterministic_stored_token_replay") is not True
            or marker.get("tokens_sampled") != 0
            or marker.get("suffix_rollouts_generated") != 0
            or marker.get("native_volume_mounted") is not False
        ):
            raise RuntimeError(f"invalid deterministic replay marker for {model}")
        state = pd.read_parquet(state_path)
        if (
            len(state) != int(marker.get("row_count", -1))
            or set(state["base_model"].astype(str)) != {model}
            or state.duplicated(["base_model", "trace_id", "checkpoint_index", "horizon"]).any()
            or not state["deterministic_stored_token_replay"].astype(bool).all()
            or int(state["tokens_sampled"].sum()) != 0
            or int(state["suffix_rollouts_generated"].sum()) != 0
        ):
            raise RuntimeError(f"invalid original-path state identity for {model}")
        original_parts.append(state)
    original = pd.concat(original_parts, ignore_index=True)
    combined = root / "inputs/original_path_states.parquet"
    atomic_parquet(combined, original)
    result = run_analysis(
        config_path=REMOTE_ROOT / "configs/repairability_funnel_v1.yaml",
        boundary_root=_boundary_root(),
        child_states_path=_source_root() / "analysis/local_child_state_inputs.parquet",
        parent_manifest_path=_source_root() / "analysis/local_parent_selection.jsonl",
        original_states_path=combined,
        output_root=root,
    )
    geometry_volume.commit()
    return result


@app.local_entrypoint()
def main(
    run_id: str = DEFAULT_RUN_ID,
    phase: str = "complete",
    model_key: str = "family_a_small",
) -> None:
    _safe(run_id)
    if model_key not in MODEL_KEYS:
        raise ValueError(f"unknown model key: {model_key}")
    digest = _digest()
    if phase == "replay-model":
        result = OriginalPathReplayWorker(model_key=model_key).replay.remote(run_id, digest)
    elif phase == "analyze":
        result = analyze_remote.remote(run_id, digest)
    elif phase == "complete":
        calls = [
            OriginalPathReplayWorker(model_key=model).replay.spawn(run_id, digest)
            for model in MODEL_KEYS
        ]
        replays = [call.get() for call in calls]
        result = {"replays": replays, "analysis": analyze_remote.remote(run_id, digest)}
    else:
        raise ValueError("phase must be replay-model, analyze, or complete")
    print(json.dumps(result, indent=2, sort_keys=True))
