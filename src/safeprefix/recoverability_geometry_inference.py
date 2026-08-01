"""Isolated teacher-forced inference for recoverability-geometry v1.

This module owns *inference artifacts only*: dense checkpoint continuations,
prompt-solvability generations, and local branch prefixes/children.  Geometry
analyses and diagnostic model training deliberately live elsewhere.  Native
artifacts are rejected both by path guards and by the Modal runner, which does
not mount a native volume.

The implementation reuses the production SafePrefix continuation machinery:
complete checkpoint caches, saved next-token logits, BF16 SDPA, deterministic
branch-specific inverse-CDF streams, and the frozen parser/verifier.  Every
logical record belongs to one immutable pack and is committed atomically.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import torch

from safeprefix.manifests import decode_reference_answer
from safeprefix.models.cache_checkpoint import (
    CacheCheckpoint,
    tokenizer_checkpoint_metadata,
)
from safeprefix.models.cache_utils import (
    cache_sequence_length,
    repeat_past_key_values,
    select_past_key_values_batch,
)
from safeprefix.models.loader import primary_device
from safeprefix.models.teacher_forcing import teacher_force_token_ids_chunked
from safeprefix.parsing.answer_parsers import parse_answer_region
from safeprefix.production_suite import parser_verifier_hashes, sha256_file
from safeprefix.reproducibility import (
    atomic_json,
    atomic_jsonl,
    atomic_parquet,
    stable_hash,
    stable_seed,
)
from safeprefix.rollout.production_engine import (
    ProductionRolloutRequest,
    _sample_with_uniforms,
    decode_execution_pack,
)
from safeprefix.models.generation import _uniform
from safeprefix.rollout.verifier import resolve_verifier


SCHEMA_VERSION = 1
GEOMETRY_SEED = 20260728
ORIGINAL_ROLLOUT_BASE_SEED = 2701
MODEL_KEYS = (
    "family_a_small",
    "family_a_large",
    "family_b_small",
    "family_b_large",
)
FORBIDDEN_PATH_TOKENS = (
    "native",
    "final_test",
    "final-test",
    "native_eval",
    "native-eval",
)
STAGE_DENSE = "dense_checkpoint_k32"
STAGE_PROMPT = "prompt_solvability_k16"
STAGE_LOCAL_PREFIX = "local_branch_prefix"
STAGE_LOCAL_CHILD = "local_branch_child"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def assert_teacher_forced_paths(paths: Iterable[str | Path]) -> None:
    """Fail before opening any path that could be a native artifact."""

    for raw in paths:
        text = Path(raw).as_posix().casefold()
        if any(token in text for token in FORBIDDEN_PATH_TOKENS):
            raise RuntimeError(f"native/final-test artifact path is forbidden: {raw}")


def _safe_run_component(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError("run identity must be one safe path component")
    return value


def geometry_seed(
    stage: str,
    *,
    model_key: str,
    trace_id: str,
    checkpoint_index: int | None = None,
    branch_index: int | None = None,
    horizon: int | None = None,
    rollout_index: int | None = None,
) -> int:
    """Derive a stage-separated seed independent of batching and resume order."""

    if stage not in {STAGE_PROMPT, STAGE_LOCAL_PREFIX, STAGE_LOCAL_CHILD}:
        raise ValueError(f"unsupported geometry seed stage: {stage}")
    return stable_seed(
        GEOMETRY_SEED,
        "recoverability-geometry-tf-v1",
        stage,
        str(model_key),
        str(trace_id),
        checkpoint_index,
        branch_index,
        horizon,
        rollout_index,
    )


def dense_rollout_seed(
    trace_id: str, checkpoint_index: int, rollout_index: int
) -> int:
    """Continue the exact original seed stream at indices 4..31."""

    index = int(rollout_index)
    if not 4 <= index < 32:
        raise ValueError("dense added rollout index must lie in [4, 31]")
    return stable_seed(
        ORIGINAL_ROLLOUT_BASE_SEED,
        str(trace_id),
        int(checkpoint_index),
        index,
    )


def logical_id(row: Mapping[str, Any]) -> str:
    fields = (
        row.get("stage"),
        row.get("model_key"),
        row.get("trace_id"),
        row.get("checkpoint_index"),
        row.get("branch_index"),
        row.get("horizon"),
        row.get("rollout_index"),
    )
    return stable_hash(["geometry-logical-v1", *fields])[:32]


def _logical_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("trace_id", "")),
        -1 if row.get("checkpoint_index") is None else int(row["checkpoint_index"]),
        -1 if row.get("branch_index") is None else int(row["branch_index"]),
        -1 if row.get("horizon") is None else int(row["horizon"]),
        -1 if row.get("rollout_index") is None else int(row["rollout_index"]),
        str(row.get("stage", "")),
    )


def _balanced_trace_packs(
    trace_rows: Sequence[Mapping[str, Any]],
    *,
    stage: str,
    model_key: str,
    keys_by_trace: Mapping[str, Sequence[Mapping[str, Any]]],
    traces_per_pack: int,
    configuration_hash: str,
) -> list[dict[str, Any]]:
    if int(traces_per_pack) < 1:
        raise ValueError("traces_per_pack must be positive")
    ordered = sorted(
        (dict(row) for row in trace_rows),
        key=lambda row: (
            -len(keys_by_trace[str(row["trace_id"])]),
            str(row.get("domain", row.get("source_bucket", ""))),
            str(row["trace_id"]),
        ),
    )
    if len({str(row["trace_id"]) for row in ordered}) != len(ordered):
        raise ValueError("pack input contains duplicate trace IDs")
    pack_count = max(1, (len(ordered) + int(traces_per_pack) - 1) // int(traces_per_pack))
    bins: list[list[dict[str, Any]]] = [[] for _ in range(pack_count)]
    work = [0] * pack_count
    for row in ordered:
        member_keys = keys_by_trace[str(row["trace_id"])]
        count = sum(
            int(key.get("checkpoint_token_offset") or 0)
            + (128 if key.get("stage") == STAGE_LOCAL_PREFIX else 4096)
            for key in member_keys
        )
        available = [i for i, values in enumerate(bins) if len(values) < int(traces_per_pack)]
        target = min(available, key=lambda i: (work[i], len(bins[i]), i))
        bins[target].append(row)
        work[target] += count
    packs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pack_index, members in enumerate(bins):
        members.sort(key=lambda row: str(row["trace_id"]))
        keys = sorted([
            dict(key)
            for row in members
            for key in keys_by_trace[str(row["trace_id"])]
        ], key=_logical_sort_key)
        ids = [str(key["logical_id"]) for key in keys]
        if len(ids) != len(set(ids)) or seen.intersection(ids):
            raise RuntimeError("logical record crossed immutable packs")
        seen.update(ids)
        identity = {
            "schema_version": SCHEMA_VERSION,
            "stage": stage,
            "model_key": model_key,
            "configuration_hash": configuration_hash,
            "pack_index": pack_index,
            "trace_ids": [str(row["trace_id"]) for row in members],
            "logical_keys": keys,
        }
        pack_hash = stable_hash(identity)
        packs.append(
            {
                **identity,
                "pack_id": f"{model_key}-{stage}-{pack_index:05d}-{pack_hash[:12]}",
                "pack_hash": pack_hash,
                "trace_count": len(members),
                "logical_record_count": len(keys),
                "estimated_work": work[pack_index],
            }
        )
    return packs


def build_dense_manifest(
    checkpoint_rows: Sequence[Mapping[str, Any]],
    *,
    model_key: str,
    configuration_hash: str,
    traces_per_pack: int = 4,
    rollout_indices: Sequence[int] = tuple(range(4, 32)),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
    trace_summary: dict[str, dict[str, Any]] = {}
    for raw in checkpoint_rows:
        row = dict(raw)
        if str(row.get("split")) != "teacher_forced_test":
            raise RuntimeError("non-test checkpoint entered dense geometry manifest")
        if str(row["base_model"]) != str(model_key):
            raise RuntimeError("checkpoint belongs to a different model")
        if int(row.get("num_rollouts", -1)) != 4:
            raise RuntimeError("dense relabeling requires an intact original K=4")
        trace_id = str(row["trace_id"])
        trace_summary.setdefault(
            trace_id,
            {
                "trace_id": trace_id,
                "domain": str(row["domain"]),
                "problem_group": str(row["problem_group"]),
            },
        )
        for index in map(int, rollout_indices):
            key = {
                "stage": STAGE_DENSE,
                "model_key": str(model_key),
                "trace_id": trace_id,
                "problem_group": str(row["problem_group"]),
                "checkpoint_id": str(row["checkpoint_id"]),
                "checkpoint_index": int(row["checkpoint_ordinal"]),
                "checkpoint_token_offset": int(row["checkpoint_token_offset"]),
                "rollout_index": index,
                "rollout_seed": dense_rollout_seed(
                    trace_id, int(row["checkpoint_ordinal"]), index
                ),
            }
            key["logical_id"] = logical_id(key)
            by_trace[trace_id].append(key)
    all_keys = sorted(
        [key for values in by_trace.values() for key in values],
        key=_logical_sort_key,
    )
    if len(all_keys) != len({key["logical_id"] for key in all_keys}):
        raise RuntimeError("dense manifest has duplicate logical records")
    packs = _balanced_trace_packs(
        list(trace_summary.values()),
        stage=STAGE_DENSE,
        model_key=model_key,
        keys_by_trace=by_trace,
        traces_per_pack=traces_per_pack,
        configuration_hash=configuration_hash,
    )
    return all_keys, packs


def build_prompt_manifest(
    checkpoint_rows: Sequence[Mapping[str, Any]],
    *,
    model_key: str,
    configuration_hash: str,
    traces_per_pack: int = 6,
    generations: int = 16,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    canonical: dict[str, dict[str, Any]] = {}
    for raw in checkpoint_rows:
        row = dict(raw)
        if str(row.get("split")) != "teacher_forced_test":
            raise RuntimeError("non-test checkpoint entered prompt geometry manifest")
        if str(row["base_model"]) != str(model_key):
            raise RuntimeError("checkpoint belongs to a different model")
        group = str(row["problem_group"])
        prior = canonical.get(group)
        if prior is None or str(row["trace_id"]) < str(prior["trace_id"]):
            canonical[group] = row
    by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
    trace_summary: list[dict[str, Any]] = []
    for group, row in sorted(canonical.items()):
        trace_id = str(row["trace_id"])
        trace_summary.append(
            {"trace_id": trace_id, "domain": str(row["domain"]), "problem_group": group}
        )
        for index in range(int(generations)):
            key = {
                "stage": STAGE_PROMPT,
                "model_key": str(model_key),
                "trace_id": trace_id,
                "problem_group": group,
                "checkpoint_index": None,
                "rollout_index": index,
                "rollout_seed": geometry_seed(
                    STAGE_PROMPT,
                    model_key=model_key,
                    trace_id=group,
                    rollout_index=index,
                ),
            }
            key["logical_id"] = logical_id(key)
            by_trace[trace_id].append(key)
    all_keys = sorted(
        [key for values in by_trace.values() for key in values],
        key=_logical_sort_key,
    )
    packs = _balanced_trace_packs(
        trace_summary,
        stage=STAGE_PROMPT,
        model_key=model_key,
        keys_by_trace=by_trace,
        traces_per_pack=traces_per_pack,
        configuration_hash=configuration_hash,
    )
    return all_keys, packs


def build_local_branch_manifest(
    parents: Sequence[Mapping[str, Any]],
    *,
    configuration_hash: str,
    branches_per_parent: int = 12,
    horizons: Sequence[int] = (32, 64, 128),
    children_per_horizon: int = 4,
    parents_per_pack: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len({(str(row["model_key"]), str(row["trace_id"])) for row in parents}) != len(parents):
        raise ValueError("local parents must contain at most one state per model-trace")
    all_keys: list[dict[str, Any]] = []
    all_packs: list[dict[str, Any]] = []
    for model_key in sorted({str(row["model_key"]) for row in parents}):
        model_parents = [dict(row) for row in parents if str(row["model_key"]) == model_key]
        by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for parent in model_parents:
            trace_id = str(parent["trace_id"])
            checkpoint_index = int(parent["checkpoint_index"])
            for branch_index in range(int(branches_per_parent)):
                prefix = {
                    "stage": STAGE_LOCAL_PREFIX,
                    "model_key": model_key,
                    "base_model": model_key,
                    "trace_id": trace_id,
                    "problem_group": str(parent.get("problem_group", "")),
                    "checkpoint_id": str(parent["checkpoint_id"]),
                    "checkpoint_index": checkpoint_index,
                    "checkpoint_token_offset": int(parent["checkpoint_token_offset"]),
                    "branch_index": branch_index,
                    "horizon": None,
                    "rollout_index": None,
                    "rollout_seed": geometry_seed(
                        STAGE_LOCAL_PREFIX,
                        model_key=model_key,
                        trace_id=trace_id,
                        checkpoint_index=checkpoint_index,
                        branch_index=branch_index,
                    ),
                }
                prefix["generation_seed"] = prefix["rollout_seed"]
                prefix["logical_id"] = logical_id(prefix)
                by_trace[trace_id].append(prefix)
                for horizon in map(int, horizons):
                    for child_index in range(int(children_per_horizon)):
                        child = {
                            "stage": STAGE_LOCAL_CHILD,
                            "model_key": model_key,
                            "base_model": model_key,
                            "trace_id": trace_id,
                            "problem_group": str(parent.get("problem_group", "")),
                            "checkpoint_id": str(parent["checkpoint_id"]),
                            "checkpoint_index": checkpoint_index,
                            "checkpoint_token_offset": int(parent["checkpoint_token_offset"]),
                            "branch_index": branch_index,
                            "horizon": horizon,
                            "rollout_index": child_index,
                            "rollout_seed": geometry_seed(
                                STAGE_LOCAL_CHILD,
                                model_key=model_key,
                                trace_id=trace_id,
                                checkpoint_index=checkpoint_index,
                                branch_index=branch_index,
                                horizon=horizon,
                                rollout_index=child_index,
                            ),
                        }
                        child["generation_seed"] = child["rollout_seed"]
                        child["logical_id"] = logical_id(child)
                        by_trace[trace_id].append(child)
        keys = sorted(
            [key for values in by_trace.values() for key in values],
            key=_logical_sort_key,
        )
        all_keys.extend(keys)
        all_packs.extend(
            _balanced_trace_packs(
                model_parents,
                stage="local_branch_geometry",
                model_key=model_key,
                keys_by_trace=by_trace,
                traces_per_pack=parents_per_pack,
                configuration_hash=configuration_hash,
            )
        )
    if len(all_keys) != len({str(row["logical_id"]) for row in all_keys}):
        raise RuntimeError("local branch manifest has duplicate logical records")
    return all_keys, all_packs


def _scan_original_rollouts(
    roots: Sequence[Path], selected_trace_ids: set[str]
) -> pd.DataFrame:
    assert_teacher_forced_paths(roots)
    frames: list[pd.DataFrame] = []
    for root in roots:
        for path in sorted(root.glob("*/rollouts.parquet")):
            marker_path = path.with_name("complete.json")
            if not marker_path.is_file():
                raise RuntimeError(f"original rollout shard lacks complete marker: {path}")
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if marker.get("rollouts_sha256") != sha256_file(path):
                raise RuntimeError(f"original rollout shard checksum mismatch: {path}")
            if marker.get("integrity", {}).get("passed") is False:
                raise RuntimeError(f"original rollout shard failed integrity: {path}")
            frame = pd.read_parquet(path)
            if "trace_id" not in frame.columns:
                raise RuntimeError(f"original rollout shard lacks trace_id: {path}")
            chosen = frame[frame["trace_id"].astype(str).isin(selected_trace_ids)].copy()
            if len(chosen):
                chosen["source_artifact"] = str(path)
                chosen["source_artifact_sha256"] = sha256_file(path)
                chosen["source_marker"] = str(marker_path)
                chosen["source_marker_sha256"] = sha256_file(marker_path)
                frames.append(chosen)
    if not frames:
        raise RuntimeError("no original K=4 outcomes found for geometry traces")
    result = pd.concat(frames, ignore_index=True)
    keys = ["model_key", "trace_id", "checkpoint_index", "rollout_index"]
    if result.duplicated(keys).any():
        raise RuntimeError("original K=4 outcomes contain duplicate logical rows")
    return result


def prepare_inference_manifests(
    *,
    boundary_root: Path,
    completion_manifest_root: Path,
    original_rollout_root: Path,
    extension_rollout_root: Path,
    output_root: Path,
    model_entries: Mapping[str, Mapping[str, Any]],
    configuration_hash: str,
    engine_digest: str,
    dense_traces_per_pack: int = 4,
    prompt_traces_per_pack: int = 6,
) -> dict[str, Any]:
    """Freeze the canonical teacher-forced inference census and packs."""

    assert_teacher_forced_paths(
        [
            boundary_root,
            completion_manifest_root,
            original_rollout_root,
            extension_rollout_root,
            output_root,
        ]
    )
    if not engine_digest:
        raise ValueError("an immutable geometry inference engine digest is required")
    ready_path = output_root / "manifests/INFERENCE_READY.json"
    canonical_path = boundary_root / "data/canonical_checkpoint_manifest.parquet"
    if ready_path.is_file():
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
        if ready.get("configuration_hash") != configuration_hash:
            raise RuntimeError("ready geometry configuration hash differs")
        if ready.get("engine_digest") != engine_digest:
            raise RuntimeError("ready geometry engine digest differs")
        if ready.get("canonical_manifest_sha256") != sha256_file(canonical_path):
            raise RuntimeError("ready canonical teacher-forced manifest changed")
        for source_path, expected_hash in ready.get("source_artifact_hashes", {}).items():
            path = Path(source_path)
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise RuntimeError(f"ready source artifact changed: {path}")
        return ready
    output_root.mkdir(parents=True, exist_ok=True)
    canonical = pd.read_parquet(canonical_path)
    test = canonical[canonical["split"].astype(str) == "teacher_forced_test"].copy()
    if len(test) == 0:
        raise RuntimeError("canonical boundary manifest has no teacher-forced test rows")
    if set(test["base_model"].astype(str)) != set(MODEL_KEYS):
        raise RuntimeError("teacher-forced test does not contain the frozen four-model matrix")
    if test.duplicated(["base_model", "trace_id", "checkpoint_ordinal"]).any():
        raise RuntimeError("canonical test manifest contains duplicate checkpoints")
    if not test["num_rollouts"].astype(int).eq(4).all():
        raise RuntimeError("canonical test manifest does not preserve K=4")
    shared_sets = {
        model: set(frame["common_trace_id"].astype(str))
        for model, frame in test.drop_duplicates(["base_model", "trace_id"]).groupby("base_model")
    }
    if len({frozenset(values) for values in shared_sets.values()}) != 1:
        raise RuntimeError("test trace identities are not shared across all four models")

    prediction_path = boundary_root / "test/teacher_forced_test_predictions.parquet"
    prediction_reconciliation: dict[str, Any] = {"available": prediction_path.is_file()}
    if prediction_path.is_file():
        predictions = pd.read_parquet(prediction_path)
        required_prediction = {
            "base_model", "trace_id", "checkpoint_id", "checkpoint_ordinal",
            "training_seed", "selected_architecture", "selected_learning_rate",
        }
        if required_prediction - set(predictions.columns):
            raise RuntimeError("test prediction table lacks frozen selection identity")
        if set(predictions["training_seed"].astype(int)) != {0, 1, 2}:
            raise RuntimeError("test prediction table does not contain probe seeds 0,1,2")
        if set(predictions["selected_architecture"].astype(str)) != {"linear_probe"}:
            raise RuntimeError("test prediction architecture differs from linear_probe")
        if set(predictions["selected_learning_rate"].astype(float)) != {0.001}:
            raise RuntimeError("test prediction learning rate differs from 1e-3")
        prediction_keys = predictions[
            ["base_model", "trace_id", "checkpoint_id", "checkpoint_ordinal"]
        ].drop_duplicates()
        canonical_keys = test[
            ["base_model", "trace_id", "checkpoint_id", "checkpoint_ordinal"]
        ].drop_duplicates()
        if set(map(tuple, prediction_keys.to_numpy())) != set(map(tuple, canonical_keys.to_numpy())):
            raise RuntimeError("test prediction rows do not reconcile to canonical checkpoints")
        per_key_seeds = predictions.groupby(
            ["base_model", "trace_id", "checkpoint_id", "checkpoint_ordinal"]
        )["training_seed"].apply(lambda values: sorted(map(int, values)))
        if any(values != [0, 1, 2] for values in per_key_seeds):
            raise RuntimeError("test prediction key lacks exactly three frozen seed rows")
        prediction_reconciliation.update(
            prediction_rows=len(predictions),
            canonical_checkpoint_model_rows=len(test),
            rows_per_checkpoint=len(predictions) / len(test),
        )

    source_rows: dict[str, dict[str, dict[str, Any]]] = {}
    original_frames: list[pd.DataFrame] = []
    dense_count = 0
    prompt_count = 0
    model_summary: dict[str, Any] = {}
    source_artifact_hashes: dict[str, str] = {
        str(canonical_path): sha256_file(canonical_path),
        str(prediction_path): sha256_file(prediction_path),
    }
    common_checkpoint_signatures: dict[str, dict[str, tuple[int, ...]]] = defaultdict(dict)
    for model_key in MODEL_KEYS:
        expected = model_entries[model_key]
        model_test = test[test["base_model"].astype(str) == model_key].copy()
        if set(model_test["model_revision"].astype(str)) != {str(expected["revision"])}:
            raise RuntimeError(f"{model_key}: model revision differs from frozen config")
        if set(model_test["tokenizer_revision"].astype(str)) != {
            str(expected["tokenizer_revision"])
        }:
            raise RuntimeError(f"{model_key}: tokenizer revision differs from frozen config")
        trace_path = (
            completion_manifest_root
            / "repairability/per_model"
            / model_key
            / "trace_manifest.jsonl"
        )
        source_artifact_hashes[str(trace_path)] = sha256_file(trace_path)
        rows = read_jsonl(trace_path)
        indexed = {str(row["trace_id"]): dict(row) for row in rows}
        selected_ids = set(model_test["trace_id"].astype(str))
        if not selected_ids.issubset(indexed):
            raise RuntimeError(f"{model_key}: missing canonical traces in completion manifest")
        selected = {trace_id: indexed[trace_id] for trace_id in selected_ids}
        for trace_id, row in selected.items():
            expected_offsets = list(
                model_test[model_test["trace_id"].astype(str) == trace_id]
                .sort_values("checkpoint_ordinal")["checkpoint_token_offset"]
                .astype(int)
            )
            if list(map(int, row["eligible_checkpoint_offsets"])) != expected_offsets:
                raise RuntimeError(f"{model_key}/{trace_id}: checkpoint offsets differ")
            common_id = str(
                model_test[model_test["trace_id"].astype(str) == trace_id][
                    "common_trace_id"
                ].iloc[0]
            )
            ordinals = tuple(
                model_test[model_test["trace_id"].astype(str) == trace_id]
                .sort_values("checkpoint_ordinal")["checkpoint_ordinal"]
                .astype(int)
            )
            common_checkpoint_signatures[common_id][model_key] = ordinals
        source_rows[model_key] = selected
        atomic_jsonl(
            output_root / f"manifests/source_traces/{model_key}.jsonl",
            [selected[key] for key in sorted(selected)],
        )
        original = _scan_original_rollouts(
            [
                original_rollout_root / model_key,
                extension_rollout_root / model_key,
            ],
            selected_ids,
        )
        expected_original = len(model_test) * 4
        if len(original) != expected_original:
            raise RuntimeError(
                f"{model_key}: original K=4 rows {len(original)} != {expected_original}"
            )
        required_original_fields = {
            "model_key", "trace_id", "checkpoint_index", "checkpoint_token_offset",
            "rollout_index", "rollout_seed", "binary_outcome", "verifier_pass",
            "parser_status", "parser_sha256", "verifier_sha256", "truncation_flag",
            "generated_token_ids", "generated_text", "stop_reason",
        }
        if required_original_fields - set(original.columns):
            raise RuntimeError(f"{model_key}: original K=4 schema is incomplete")
        if set(original["model_key"].astype(str)) != {model_key}:
            raise RuntimeError(f"{model_key}: original K=4 model identity differs")
        expected_hashes = parser_verifier_hashes()
        if set(original["parser_sha256"].astype(str)) != {expected_hashes["parser_sha256"]}:
            raise RuntimeError(f"{model_key}: original parser hash differs")
        if set(original["verifier_sha256"].astype(str)) != {expected_hashes["verifier_sha256"]}:
            raise RuntimeError(f"{model_key}: original verifier hash differs")
        grouped = original.groupby(["trace_id", "checkpoint_index"])
        if any(
            sorted(frame["rollout_index"].astype(int)) != [0, 1, 2, 3]
            for _, frame in grouped
        ):
            raise RuntimeError(f"{model_key}: original rollout indices are not 0..3")
        original_keys = set(
            map(tuple, original[["trace_id", "checkpoint_index"]].drop_duplicates().to_numpy())
        )
        canonical_keys = set(
            map(tuple, model_test[["trace_id", "checkpoint_ordinal"]].to_numpy())
        )
        if original_keys != canonical_keys:
            raise RuntimeError(f"{model_key}: original K=4 checkpoint keys differ")
        offset_lookup = {
            (str(row["trace_id"]), int(row["checkpoint_ordinal"])): int(
                row["checkpoint_token_offset"]
            )
            for row in model_test.to_dict("records")
        }
        for row in original.to_dict("records"):
            expected_seed = stable_seed(
                ORIGINAL_ROLLOUT_BASE_SEED,
                str(row["trace_id"]),
                int(row["checkpoint_index"]),
                int(row["rollout_index"]),
            )
            if int(row["rollout_seed"]) != expected_seed:
                raise RuntimeError(f"{model_key}: original K=4 seed formula differs")
            if int(row["checkpoint_token_offset"]) != offset_lookup[
                (str(row["trace_id"]), int(row["checkpoint_index"]))
            ]:
                raise RuntimeError(f"{model_key}: original checkpoint offset differs")
            # Parquet-backed booleans may materialize as numpy.bool_.  Accept the
            # two boolean scalar representations, but never integer 0/1 or a
            # truthy string: those would weaken the source-integrity contract.
            if (
                type(row["binary_outcome"]).__name__ not in {"bool", "bool_"}
                or type(row["verifier_pass"]).__name__ not in {"bool", "bool_"}
            ):
                raise RuntimeError(f"{model_key}: original outcome is not boolean")
            if bool(row.get("binary_outcome")) != bool(row.get("verifier_pass")):
                raise RuntimeError(f"{model_key}: original binary/verifier outcome differs")
            if bool(row.get("binary_outcome")) and (
                bool(row.get("truncation_flag"))
                or str(row.get("parser_status")) != "success"
            ):
                raise RuntimeError(f"{model_key}: invalid successful original outcome")
            source_artifact_hashes[str(row["source_artifact"])] = str(
                row["source_artifact_sha256"]
            )
            source_artifact_hashes[str(row["source_marker"])] = str(
                row["source_marker_sha256"]
            )
        original_frames.append(original)

        dense_keys, dense_packs = build_dense_manifest(
            model_test.to_dict("records"),
            model_key=model_key,
            configuration_hash=configuration_hash,
            traces_per_pack=dense_traces_per_pack,
        )
        prompt_keys, prompt_packs = build_prompt_manifest(
            model_test.to_dict("records"),
            model_key=model_key,
            configuration_hash=configuration_hash,
            traces_per_pack=prompt_traces_per_pack,
        )
        atomic_jsonl(output_root / f"manifests/dense/{model_key}_keys.jsonl", dense_keys)
        atomic_jsonl(output_root / f"manifests/dense/{model_key}_packs.jsonl", dense_packs)
        atomic_jsonl(output_root / f"manifests/prompt/{model_key}_keys.jsonl", prompt_keys)
        atomic_jsonl(output_root / f"manifests/prompt/{model_key}_packs.jsonl", prompt_packs)
        dense_count += len(dense_keys)
        prompt_count += len(prompt_keys)
        model_summary[model_key] = {
            "test_traces": int(model_test["trace_id"].nunique()),
            "test_problem_groups": int(model_test["problem_group"].nunique()),
            "test_checkpoints": len(model_test),
            "original_k4_rollouts": len(original),
            "new_dense_rollouts": len(dense_keys),
            "prompt_generations": len(prompt_keys),
            "dense_packs": len(dense_packs),
            "prompt_packs": len(prompt_packs),
        }

    for common_id, signatures in common_checkpoint_signatures.items():
        if set(signatures) != set(MODEL_KEYS):
            raise RuntimeError(f"shared trace {common_id} is missing a model")
        if len(set(signatures.values())) != 1:
            raise RuntimeError(
                f"shared trace {common_id} has different checkpoint ordinal/count alignment"
            )

    # A problem group may have two trace variants, but prompt-solvability must
    # use an equivalent problem/reference pair rather than silently choosing a
    # favorable variant.
    group_identity: dict[str, tuple[str, str]] = {}
    for model_key, rows in source_rows.items():
        model_test = test[test["base_model"].astype(str) == model_key]
        group_by_trace = dict(
            zip(model_test["trace_id"].astype(str), model_test["problem_group"].astype(str))
        )
        for trace_id, row in rows.items():
            key = group_by_trace[trace_id]
            value = (
                " ".join(str(row.get("problem_text", "")).split()),
                json.dumps(row.get("reference_answer"), sort_keys=True),
            )
            prior = group_identity.setdefault(key, value)
            if prior != value:
                raise RuntimeError(f"problem group {key} has non-equivalent variants")

    atomic_parquet(
        output_root / "manifests/eligible_teacher_forced_checkpoints.parquet", test
    )
    all_original = pd.concat(original_frames, ignore_index=True)
    atomic_parquet(output_root / "manifests/original_k4_outcomes.parquet", all_original)
    census = {
        "status": "READY",
        "configuration_hash": configuration_hash,
        "engine_digest": engine_digest,
        "canonical_manifest_sha256": sha256_file(canonical_path),
        "teacher_forced_only": True,
        "native_artifacts_accessed": False,
        "shared_test_trace_count": len(next(iter(shared_sets.values()))),
        "checkpoint_model_count": len(test),
        "original_k4_rollout_count": len(all_original),
        "new_dense_rollout_count": dense_count,
        "prompt_generation_count": prompt_count,
        "prediction_row_reconciliation": prediction_reconciliation,
        "models": model_summary,
        "source_artifact_hashes": source_artifact_hashes,
    }
    atomic_json(output_root / "manifests/prelaunch_census.json", census)
    atomic_json(ready_path, census)
    return census


def prepare_local_parent_packs(
    *,
    parent_manifest_path: Path,
    output_root: Path,
    configuration_hash: str,
    parents_per_pack: int = 2,
) -> dict[str, Any]:
    """Freeze the analysis-to-local-inference bridge after dense relabeling.

    Parent selection is performed by the predefined analysis layer. This
    function validates that result, freezes all 160 parents, and expands the
    exact 12-prefix/three-horizon/four-child inference packs. It never selects
    parents itself and therefore cannot inspect outcomes opportunistically.
    """

    assert_teacher_forced_paths([parent_manifest_path, output_root])
    parents = read_jsonl(parent_manifest_path)
    # The analysis package writes its native public field names.  The aliases
    # below are a schema bridge only; their values are independently validated
    # before any inference pack is frozen.
    for row in parents:
        if "model_key" not in row and "base_model" in row:
            row["model_key"] = row["base_model"]
        if "checkpoint_index" not in row and "checkpoint_ordinal" in row:
            row["checkpoint_index"] = row["checkpoint_ordinal"]
        if "selection_category" not in row and "requested_category" in row:
            row["selection_category"] = row["requested_category"]
        if "selection_rank_hash" not in row and "selection_hash" in row:
            row["selection_rank_hash"] = row["selection_hash"]
        if "substitution_reason" not in row and row.get("category_substitution"):
            row["substitution_reason"] = "closest_available_category"
    required = {
        "model_key", "trace_id", "checkpoint_id", "checkpoint_index",
        "checkpoint_token_offset", "domain", "selection_category",
        "dense_recoverability", "selection_rank_hash",
    }
    if any(required - set(row) for row in parents):
        raise RuntimeError("local-parent manifest lacks frozen selection provenance")
    if len(parents) != 160:
        raise RuntimeError(f"local-parent manifest must contain exactly 160 rows, got {len(parents)}")
    if len({(str(row["model_key"]), str(row["trace_id"])) for row in parents}) != 160:
        raise RuntimeError("local-parent manifest violates one parent per model-trace")
    expected_categories = {"near_boundary": 24, "high": 8, "low": 8}
    for model_key in MODEL_KEYS:
        subset = [row for row in parents if str(row["model_key"]) == model_key]
        if len(subset) != 40:
            raise RuntimeError(f"{model_key}: local-parent count is not 40")
        observed = Counter(str(row["selection_category"]) for row in subset)
        if dict(observed) != expected_categories:
            raise RuntimeError(f"{model_key}: local-parent category allocation differs")
        for row in subset:
            expected_rank = hashlib.sha256(
                "||".join(
                    (
                        str(row["model_key"]),
                        str(row["trace_id"]),
                        str(row["checkpoint_id"]),
                        str(GEOMETRY_SEED),
                    )
                ).encode("utf-8")
            ).hexdigest()
            if str(row["selection_rank_hash"]) != expected_rank:
                raise RuntimeError(f"{model_key}: local-parent stable selection hash differs")
            rate = float(row["dense_recoverability"])
            category = str(row["selection_category"])
            in_primary = (
                (category == "near_boundary" and 0.35 <= rate <= 0.65)
                or (category == "high" and rate >= 0.75)
                or (category == "low" and rate <= 0.25)
            )
            if not in_primary and not row.get("substitution_reason"):
                raise RuntimeError(
                    f"{model_key}: out-of-range local parent lacks substitution provenance"
                )
    manifest_sha = sha256_file(parent_manifest_path)
    marker_path = output_root / "manifests/local/LOCAL_PARENTS_READY.json"
    if marker_path.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (
            marker.get("parent_manifest_sha256") != manifest_sha
            or marker.get("configuration_hash") != configuration_hash
        ):
            raise RuntimeError("frozen local-parent inference identity changed")
        return marker
    keys, packs = build_local_branch_manifest(
        parents,
        configuration_hash=configuration_hash,
        parents_per_pack=parents_per_pack,
    )
    atomic_jsonl(output_root / "manifests/local/local_parent_manifest.jsonl", parents)
    atomic_jsonl(output_root / "manifests/local/all_local_keys.jsonl", keys)
    for model_key in MODEL_KEYS:
        atomic_jsonl(
            output_root / f"manifests/local/{model_key}_packs.jsonl",
            [pack for pack in packs if str(pack["model_key"]) == model_key],
        )
    marker = {
        "status": "READY",
        "parent_manifest_sha256": manifest_sha,
        "configuration_hash": configuration_hash,
        "parent_count": len(parents),
        "prefix_count": sum(row["stage"] == STAGE_LOCAL_PREFIX for row in keys),
        "planned_child_count": sum(row["stage"] == STAGE_LOCAL_CHILD for row in keys),
        "pack_count": len(packs),
        "native_artifacts_accessed": False,
    }
    atomic_json(marker_path, marker)
    return marker


def _atomic_torch(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _pack_marker_valid(pack: Mapping[str, Any], pack_root: Path) -> bool:
    marker_path = pack_root / "complete.json"
    rows_path = pack_root / "outcomes.parquet"
    if not marker_path.is_file() or not rows_path.is_file():
        return False
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("pack_hash") != pack.get("pack_hash"):
        return False
    if marker.get("outcomes_sha256") != sha256_file(rows_path):
        return False
    frame = pd.read_parquet(rows_path)
    expected = {str(row["logical_id"]) for row in pack["logical_keys"]}
    observed = list(frame["logical_id"].astype(str))
    return len(observed) == len(set(observed)) and set(observed) == expected


def _verify_generation_result(
    *,
    result: Any,
    source: Mapping[str, Any],
    verifier: Any,
) -> dict[str, Any]:
    parsed = parse_answer_region(result.text)
    truncated = str(result.stop_reason) == "length"
    passed = bool(
        parsed.success
        and not truncated
        and verifier(
            parsed.parsed_answer,
            decode_reference_answer(source["reference_answer"]),
            {},
        )
    )
    return {
        "generated_token_ids": list(map(int, result.token_ids)),
        "generated_token_count": len(result.token_ids),
        "generated_text": result.text,
        "stop_reason": str(result.stop_reason),
        "truncation_status": truncated,
        "parser_status": "success" if parsed.success else "failure",
        "parser_method": parsed.method,
        "normalized_extracted_answer": (
            None if not parsed.success else str(parsed.parsed_answer)
        ),
        "verifier_outcome": passed,
        "binary_outcome": passed,
        "infrastructure_status": "complete",
        "latency_seconds": float(result.latency_seconds),
    }


def _make_request(
    key: Mapping[str, Any], checkpoint: CacheCheckpoint, pack_id: str
) -> ProductionRolloutRequest:
    # ProductionRolloutRequest's 0..3 index is an implementation-local branch
    # lane. The scientifically meaningful geometry rollout index remains in
    # immutable metadata and in the output row.
    return ProductionRolloutRequest(
        branch_id=str(key["logical_id"]),
        rollout_seed=int(key["rollout_seed"]),
        rollout_index=int(key.get("rollout_index") or 0) % 4,
        checkpoint=checkpoint,
        metadata={"pack_id": pack_id, **dict(key)},
    )


def _teacher_force_trace(
    *,
    model: Any,
    source: Mapping[str, Any],
    checkpoint_offsets: Sequence[int],
    prefill_chunk_size: int,
) -> Any:
    prompt_ids = list(map(int, source["prompt_token_ids"]))
    completion_ids = list(map(int, source["completion_token_ids"]))
    return teacher_force_token_ids_chunked(
        model,
        prompt_ids + completion_ids,
        prompt_count=len(prompt_ids),
        chunk_size=int(prefill_chunk_size),
        # Dense relabeling and local-parent restoration reuse already persisted
        # checkpoint features; they need cache/logits only. Requesting hidden
        # states here would materialize every layer for no scientific output.
        selected_layers=(),
        selected_token_offsets=(),
        selected_checkpoint_offsets=list(map(int, checkpoint_offsets)),
    )


def execute_dense_or_prompt_pack(
    *,
    stage: str,
    pack: Mapping[str, Any],
    source_rows: Mapping[str, Mapping[str, Any]],
    loaded: Any,
    generation: Mapping[str, Any],
    output_root: Path,
    selected_layers: Sequence[int],
    maximum_batch_size: int,
    compaction_quantum: int,
    maximum_context_length: int,
    maximum_decode_kv_bytes: int | None,
    prefill_chunk_size: int,
) -> dict[str, Any]:
    if stage not in {STAGE_DENSE, STAGE_PROMPT}:
        raise ValueError("unsupported ordinary inference stage")
    assert_teacher_forced_paths([output_root])
    pack_root = output_root / "packs" / stage / str(pack["model_key"]) / str(pack["pack_id"])
    if _pack_marker_valid(pack, pack_root):
        return {"status": "SKIPPED_VALID", "pack_id": pack["pack_id"]}
    generation = dict(generation)
    if int(generation.get("max_new_tokens", -1)) != 4096:
        raise RuntimeError("scientific dense/prompt generation requires max_new_tokens=4096")
    requests: list[ProductionRolloutRequest] = []
    keys_by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key in pack["logical_keys"]:
        keys_by_trace[str(key["trace_id"])].append(dict(key))
    tokenizer = loaded.tokenizer
    for trace_id in map(str, pack["trace_ids"]):
        source = source_rows[trace_id]
        keys = keys_by_trace[trace_id]
        if stage == STAGE_DENSE:
            offsets = sorted({int(key["checkpoint_token_offset"]) for key in keys})
            forced = _teacher_force_trace(
                model=loaded.model,
                source=source,
                checkpoint_offsets=offsets,
                prefill_chunk_size=prefill_chunk_size,
            )
            checkpoint_by_offset = {
                offset: forced.cache_checkpoint(
                    offset,
                    model_id=loaded.model_id,
                    model_revision=loaded.model_revision,
                    tokenizer_id=loaded.tokenizer_id,
                    tokenizer_revision=loaded.tokenizer_revision,
                    tokenizer_metadata=tokenizer_checkpoint_metadata(tokenizer),
                    generation_metadata=generation,
                    clone_cache=False,
                )
                for offset in offsets
            }
            requests.extend(
                _make_request(key, checkpoint_by_offset[int(key["checkpoint_token_offset"])], str(pack["pack_id"]))
                for key in keys
            )
        else:
            prompt_ids = list(map(int, source["prompt_token_ids"]))
            forced = teacher_force_token_ids_chunked(
                loaded.model,
                prompt_ids,
                prompt_count=len(prompt_ids),
                chunk_size=int(prefill_chunk_size),
                selected_checkpoint_offsets=[len(prompt_ids)],
            )
            root = forced.cache_checkpoint(
                len(prompt_ids),
                model_id=loaded.model_id,
                model_revision=loaded.model_revision,
                tokenizer_id=loaded.tokenizer_id,
                tokenizer_revision=loaded.tokenizer_revision,
                tokenizer_metadata=tokenizer_checkpoint_metadata(tokenizer),
                generation_metadata=generation,
                clone_cache=False,
            )
            requests.extend(_make_request(key, root, str(pack["pack_id"])) for key in keys)
    results, metrics = decode_execution_pack(
        loaded.model,
        tokenizer,
        requests,
        generation=generation,
        maximum_batch_size=int(maximum_batch_size),
        compaction_quantum=int(compaction_quantum),
        maximum_context_length=int(maximum_context_length),
        maximum_decode_kv_bytes=maximum_decode_kv_bytes,
    )
    verifier = resolve_verifier(str(generation.get("verifier", "exact_answer")))
    hashes = parser_verifier_hashes()
    rows: list[dict[str, Any]] = []
    for result in results:
        key = dict(result.request.metadata)
        source = source_rows[str(key["trace_id"])]
        outcome = _verify_generation_result(result=result, source=source, verifier=verifier)
        row = {
            "schema_version": SCHEMA_VERSION,
            "stage": stage,
            "logical_id": str(key["logical_id"]),
            "pack_id": str(pack["pack_id"]),
            "pack_hash": str(pack["pack_hash"]),
            "configuration_hash": str(pack["configuration_hash"]),
            "model_key": str(pack["model_key"]),
            "base_model": str(pack["model_key"]),
            "model_id": loaded.model_id,
            "model_revision": loaded.model_revision,
            "tokenizer_revision": loaded.tokenizer_revision,
            "trace_id": str(key["trace_id"]),
            "problem_id": str(source["problem_id"]),
            "problem_group": str(
                key.get("problem_group", source.get("production_problem_group", ""))
            ),
            "domain": str(source["source_bucket"]),
            "checkpoint_id": key.get("checkpoint_id"),
            "checkpoint_index": key.get("checkpoint_index"),
            "checkpoint_token_offset": key.get("checkpoint_token_offset"),
            "rollout_index": int(key["rollout_index"]),
            "rollout_seed": int(key["rollout_seed"]),
            "generation_seed": int(key["rollout_seed"]),
            "parser_sha256": hashes["parser_sha256"],
            "verifier_sha256": hashes["verifier_sha256"],
            **outcome,
        }
        row["artifact_hash"] = stable_hash(
            [row["logical_id"], row["generated_token_ids"], row["binary_outcome"]]
        )
        rows.append(row)
    expected = {str(key["logical_id"]) for key in pack["logical_keys"]}
    observed = [str(row["logical_id"]) for row in rows]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise RuntimeError("ordinary geometry inference pack failed exact-once integrity")
    pack_root.mkdir(parents=True, exist_ok=True)
    outcomes_path = pack_root / "outcomes.parquet"
    atomic_parquet(outcomes_path, pd.DataFrame(rows))
    marker = {
        "status": "COMPLETE",
        "stage": stage,
        "pack_id": pack["pack_id"],
        "pack_hash": pack["pack_hash"],
        "logical_record_count": len(rows),
        "outcomes_sha256": sha256_file(outcomes_path),
        "decode_metrics": metrics.to_dict(),
        "native_artifacts_accessed": False,
    }
    atomic_json(pack_root / "complete.json", marker)
    return marker


@dataclass
class LocalPrefixState:
    branch_index: int
    rollout_seed: int
    token_ids: list[int]
    stop_reason: str
    hidden_states: dict[int, torch.Tensor]
    checkpoints: dict[int, CacheCheckpoint]


def _local_model_forward(
    model: Any,
    *,
    token_ids: torch.Tensor,
    cache: Any,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    cache_position: torch.Tensor,
) -> Any:
    kwargs = {
        "input_ids": token_ids[:, None],
        "attention_mask": attention_mask,
        "position_ids": position_ids[:, None],
        "cache_position": cache_position,
        "past_key_values": cache,
        "use_cache": True,
        "return_dict": True,
        "output_hidden_states": True,
    }
    with torch.inference_mode():
        try:
            return model(**kwargs)
        except TypeError as exc:
            if "cache_position" not in str(exc):
                raise
            kwargs.pop("cache_position")
            return model(**kwargs)


def generate_local_prefix_states(
    *,
    model: Any,
    tokenizer: Any,
    parent: CacheCheckpoint,
    branch_seeds: Sequence[int],
    generation: Mapping[str, Any],
    horizons: Sequence[int] = (32, 64, 128),
) -> list[LocalPrefixState]:
    """Generate actual batched branch states and capture exact horizon caches."""

    if model.training:
        raise RuntimeError("local branch generation requires model.eval()")
    horizons = tuple(sorted(set(map(int, horizons))))
    if not horizons or horizons[0] < 1:
        raise ValueError("local branch horizons must be positive")
    branch_count = len(branch_seeds)
    if branch_count < 1:
        raise ValueError("at least one local branch is required")
    device = primary_device(model)
    cache = repeat_past_key_values(parent.past_key_values, branch_count)
    logits = parent.next_token_logits.repeat_interleave(branch_count, dim=0).to(device)
    prefix = int(parent.prefix_token_count)
    maximum = max(horizons)
    attention = torch.zeros(
        (branch_count, prefix + maximum), dtype=torch.long, device=device
    )
    attention[:, :prefix] = 1
    active = torch.ones(branch_count, dtype=torch.bool, device=device)
    token_rows: list[list[int]] = [[] for _ in range(branch_count)]
    stop_reason: list[str | None] = [None] * branch_count
    hidden_by_branch: list[dict[int, torch.Tensor]] = [dict() for _ in range(branch_count)]
    checkpoint_by_branch: list[dict[int, CacheCheckpoint]] = [dict() for _ in range(branch_count)]
    eos = tokenizer.eos_token_id
    eos_set = {int(eos)} if isinstance(eos, int) else set(map(int, eos or []))
    stop_ids = eos_set | set(map(int, generation.get("stop_token_ids", [])))
    pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else next(iter(eos_set), 0))
    for step in range(maximum):
        uniforms = torch.tensor(
            [
                _uniform(int(seed), step)
                for seed in branch_seeds
            ],
            dtype=torch.float32,
            device=device,
        )
        # Use the exact production logits stack. Branch-specific uniforms keep
        # results invariant to batch ordering and resume position.
        selected = _sample_with_uniforms(logits, generation, uniforms)
        selected = torch.where(active, selected, torch.full_like(selected, pad))
        attention[:, prefix + step] = active.to(torch.long)
        positions = torch.where(
            active,
            torch.full_like(selected, prefix + step),
            torch.zeros_like(selected),
        )
        output = _local_model_forward(
            model,
            token_ids=selected,
            cache=cache,
            attention_mask=attention[:, : prefix + step + 1],
            position_ids=positions,
            cache_position=torch.tensor([prefix + step], dtype=torch.long, device=device),
        )
        cache = output.past_key_values
        logits = output.logits[:, -1]
        current_hidden = output.hidden_states[-1][:, -1]
        selected_cpu = selected.detach().cpu().tolist()
        active_before = active.clone()
        for row, token in enumerate(selected_cpu):
            if bool(active_before[row]):
                token_rows[row].append(int(token))
        for token_id in stop_ids:
            stopped = active_before & (selected == int(token_id))
            for row in torch.nonzero(stopped, as_tuple=False).flatten().cpu().tolist():
                stop_reason[row] = "eos" if int(token_id) in eos_set else "stop"
            active &= ~stopped
        horizon = step + 1
        if horizon in horizons:
            for row in range(branch_count):
                if len(token_rows[row]) < horizon:
                    continue
                hidden_by_branch[row][horizon] = current_hidden[row].detach().cpu().to(torch.float16)
                # A branch that emits EOS/stop exactly at the requested horizon
                # has a terminal state but no valid child continuation state.
                # Retain its hidden vector and terminal verifier outcome while
                # marking this and later child horizons unavailable.
                if not bool(active[row]):
                    continue
                child_cache = select_past_key_values_batch(cache, row, prefix + horizon)
                prefix_tokens = torch.cat(
                    [
                        parent.prefix_token_ids[0].detach().cpu(),
                        torch.tensor(token_rows[row], dtype=torch.long),
                    ]
                ).unsqueeze(0).to(device)
                child = CacheCheckpoint(
                    past_key_values=child_cache,
                    next_token_logits=logits[row : row + 1].detach().clone(),
                    prefix_token_count=prefix + horizon,
                    attention_mask=torch.ones((1, prefix + horizon), dtype=torch.long, device=device),
                    position_ids=torch.tensor([[prefix + horizon]], dtype=torch.long, device=device),
                    cache_position=torch.tensor([prefix + horizon], dtype=torch.long, device=device),
                    prefix_token_ids=prefix_tokens,
                    model_id=parent.model_id,
                    model_revision=parent.model_revision,
                    tokenizer_id=parent.tokenizer_id,
                    tokenizer_revision=parent.tokenizer_revision,
                    dtype=parent.dtype,
                    checkpoint_token_offset=prefix + horizon,
                    tokenizer_metadata=dict(parent.tokenizer_metadata),
                    generation_metadata=dict(parent.generation_metadata),
                    rng_metadata={"geometry_branch_seed": int(branch_seeds[row])},
                    cache_storage="independent",
                )
                child.validate()
                checkpoint_by_branch[row][horizon] = child
        if not bool(active.any()):
            break
    return [
        LocalPrefixState(
            branch_index=index,
            rollout_seed=int(branch_seeds[index]),
            token_ids=token_rows[index],
            stop_reason=stop_reason[index] or "length",
            hidden_states=hidden_by_branch[index],
            checkpoints=checkpoint_by_branch[index],
        )
        for index in range(branch_count)
    ]


def execute_local_pack(
    *,
    pack: Mapping[str, Any],
    source_rows: Mapping[str, Mapping[str, Any]],
    loaded: Any,
    generation: Mapping[str, Any],
    output_root: Path,
    selected_layers: Sequence[int],
    maximum_batch_size: int,
    compaction_quantum: int,
    maximum_context_length: int,
    maximum_decode_kv_bytes: int | None,
    prefill_chunk_size: int,
) -> dict[str, Any]:
    assert_teacher_forced_paths([output_root])
    pack_root = output_root / "packs/local_branch_geometry" / str(pack["model_key"]) / str(pack["pack_id"])
    if _pack_marker_valid(pack, pack_root):
        return {"status": "SKIPPED_VALID", "pack_id": pack["pack_id"]}
    tokenizer = loaded.tokenizer
    verifier = resolve_verifier(str(generation.get("verifier", "exact_answer")))
    prefix_generation = {**dict(generation), "max_new_tokens": 128}
    keys_by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key in pack["logical_keys"]:
        keys_by_trace[str(key["trace_id"])].append(dict(key))
    output_rows: list[dict[str, Any]] = []
    child_features: dict[str, torch.Tensor] = {}
    hashes = parser_verifier_hashes()
    for trace_id in map(str, pack["trace_ids"]):
        source = source_rows[trace_id]
        keys = keys_by_trace[trace_id]
        prefix_keys = sorted(
            [key for key in keys if key["stage"] == STAGE_LOCAL_PREFIX],
            key=lambda key: int(key["branch_index"]),
        )
        if len({int(key["checkpoint_index"]) for key in prefix_keys}) != 1:
            raise RuntimeError("one local pack trace must reference one parent checkpoint")
        parent_offset = int(prefix_keys[0]["checkpoint_token_offset"])
        forced = _teacher_force_trace(
            model=loaded.model,
            source=source,
            checkpoint_offsets=[parent_offset],
            prefill_chunk_size=prefill_chunk_size,
        )
        parent = forced.cache_checkpoint(
            parent_offset,
            model_id=loaded.model_id,
            model_revision=loaded.model_revision,
            tokenizer_id=loaded.tokenizer_id,
            tokenizer_revision=loaded.tokenizer_revision,
            tokenizer_metadata=tokenizer_checkpoint_metadata(tokenizer),
            generation_metadata=generation,
            clone_cache=False,
        )
        states = generate_local_prefix_states(
            model=loaded.model,
            tokenizer=tokenizer,
            parent=parent,
            branch_seeds=[int(key["rollout_seed"]) for key in prefix_keys],
            generation=prefix_generation,
        )
        state_by_branch = {state.branch_index: state for state in states}
        child_requests: list[ProductionRolloutRequest] = []
        child_key_by_id: dict[str, dict[str, Any]] = {}
        for prefix_key in prefix_keys:
            branch = int(prefix_key["branch_index"])
            state = state_by_branch[branch]
            prefix_text = tokenizer.decode(
                state.token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            terminal_proxy = type(
                "PrefixResult",
                (),
                {
                    "text": prefix_text,
                    "token_ids": state.token_ids,
                    "stop_reason": state.stop_reason,
                    "latency_seconds": 0.0,
                },
            )()
            prefix_outcome = _verify_generation_result(
                result=terminal_proxy, source=source, verifier=verifier
            )
            output_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "stage": STAGE_LOCAL_PREFIX,
                    "logical_id": prefix_key["logical_id"],
                    "pack_id": pack["pack_id"],
                    "pack_hash": pack["pack_hash"],
                    "configuration_hash": pack["configuration_hash"],
                    "model_key": pack["model_key"],
                    "base_model": pack["model_key"],
                    "model_id": loaded.model_id,
                    "model_revision": loaded.model_revision,
                    "tokenizer_revision": loaded.tokenizer_revision,
                    "trace_id": trace_id,
                    "problem_id": source["problem_id"],
                    "problem_group": str(
                        prefix_key.get(
                            "problem_group",
                            source.get("production_problem_group", ""),
                        )
                    ),
                    "domain": source["source_bucket"],
                    "checkpoint_id": prefix_key["checkpoint_id"],
                    "checkpoint_index": prefix_key["checkpoint_index"],
                    "checkpoint_token_offset": parent_offset,
                    "branch_index": branch,
                    "horizon": None,
                    "rollout_index": None,
                    "rollout_seed": prefix_key["rollout_seed"],
                    "generation_seed": prefix_key["rollout_seed"],
                    "available_horizons": sorted(state.checkpoints),
                    "early_termination_status": state.stop_reason != "length",
                    "parser_sha256": hashes["parser_sha256"],
                    "verifier_sha256": hashes["verifier_sha256"],
                    **prefix_outcome,
                }
            )
            for horizon, hidden in state.hidden_states.items():
                feature_id = f"{trace_id}:{branch}:{horizon}"
                child_features[feature_id] = hidden
            child_keys = [
                key
                for key in keys
                if key["stage"] == STAGE_LOCAL_CHILD
                and int(key["branch_index"]) == branch
            ]
            for key in child_keys:
                horizon = int(key["horizon"])
                if horizon not in state.checkpoints:
                    continue
                request = _make_request(
                    key, state.checkpoints[horizon], str(pack["pack_id"])
                )
                child_requests.append(request)
                child_key_by_id[str(key["logical_id"])] = key
        if child_requests:
            results, _ = decode_execution_pack(
                loaded.model,
                tokenizer,
                child_requests,
                generation=generation,
                maximum_batch_size=int(maximum_batch_size),
                compaction_quantum=int(compaction_quantum),
                maximum_context_length=int(maximum_context_length),
                maximum_decode_kv_bytes=maximum_decode_kv_bytes,
            )
            for result in results:
                key = child_key_by_id[result.request.branch_id]
                outcome = _verify_generation_result(
                    result=result, source=source, verifier=verifier
                )
                output_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "stage": STAGE_LOCAL_CHILD,
                        "logical_id": key["logical_id"],
                        "pack_id": pack["pack_id"],
                        "pack_hash": pack["pack_hash"],
                        "configuration_hash": pack["configuration_hash"],
                        "model_key": pack["model_key"],
                        "base_model": pack["model_key"],
                        "model_id": loaded.model_id,
                        "model_revision": loaded.model_revision,
                        "tokenizer_revision": loaded.tokenizer_revision,
                        "trace_id": trace_id,
                        "problem_id": source["problem_id"],
                        "problem_group": str(
                            key.get(
                                "problem_group",
                                source.get("production_problem_group", ""),
                            )
                        ),
                        "domain": source["source_bucket"],
                        "checkpoint_id": key["checkpoint_id"],
                        "checkpoint_index": key["checkpoint_index"],
                        "checkpoint_token_offset": key["checkpoint_token_offset"],
                        "branch_index": key["branch_index"],
                        "horizon": key["horizon"],
                        "rollout_index": key["rollout_index"],
                        "rollout_seed": key["rollout_seed"],
                        "generation_seed": key["rollout_seed"],
                        "early_termination_status": False,
                        "parser_sha256": hashes["parser_sha256"],
                        "verifier_sha256": hashes["verifier_sha256"],
                        **outcome,
                    }
                )
    expected_prefix = {
        str(key["logical_id"])
        for key in pack["logical_keys"]
        if key["stage"] == STAGE_LOCAL_PREFIX
    }
    observed_prefix = {
        str(row["logical_id"]) for row in output_rows if row["stage"] == STAGE_LOCAL_PREFIX
    }
    if observed_prefix != expected_prefix:
        raise RuntimeError("local prefix pack lost a logical prefix record")
    expected_children = {
        str(key["logical_id"])
        for key in pack["logical_keys"]
        if key["stage"] == STAGE_LOCAL_CHILD
    }
    observed_children = {
        str(row["logical_id"]) for row in output_rows if row["stage"] == STAGE_LOCAL_CHILD
    }
    unavailable = expected_children - observed_children
    # Missing child rows are valid only when their branch terminated before the
    # requested horizon. Persist explicit unavailable infrastructure-neutral
    # statuses so every planned logical identity still has one row.
    key_index = {str(key["logical_id"]): key for key in pack["logical_keys"]}
    for identity in sorted(unavailable):
        key = key_index[identity]
        output_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                **dict(key),
                "pack_id": pack["pack_id"],
                "pack_hash": pack["pack_hash"],
                "configuration_hash": pack["configuration_hash"],
                "model_id": loaded.model_id,
                "model_revision": loaded.model_revision,
                "tokenizer_revision": loaded.tokenizer_revision,
                "problem_id": source_rows[str(key["trace_id"])]["problem_id"],
                "domain": source_rows[str(key["trace_id"])]["source_bucket"],
                "generated_token_ids": [],
                "generated_token_count": 0,
                "generated_text": "",
                "stop_reason": "parent_terminated_before_horizon",
                "truncation_status": False,
                "parser_status": "not_run",
                "normalized_extracted_answer": None,
                "verifier_outcome": None,
                "binary_outcome": None,
                "infrastructure_status": "unavailable_by_protocol",
                "early_termination_status": True,
                "parser_sha256": hashes["parser_sha256"],
                "verifier_sha256": hashes["verifier_sha256"],
            }
        )
    observed = [str(row["logical_id"]) for row in output_rows]
    expected = {str(key["logical_id"]) for key in pack["logical_keys"]}
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise RuntimeError("local pack failed exact-once logical integrity")
    pack_root.mkdir(parents=True, exist_ok=True)
    outcomes_path = pack_root / "outcomes.parquet"
    features_path = pack_root / "child_hidden_states.pt"
    atomic_parquet(outcomes_path, pd.DataFrame(output_rows))
    _atomic_torch(features_path, child_features)
    marker = {
        "status": "COMPLETE",
        "stage": "local_branch_geometry",
        "pack_id": pack["pack_id"],
        "pack_hash": pack["pack_hash"],
        "logical_record_count": len(output_rows),
        "prefix_count": len(expected_prefix),
        "child_completion_count": len(observed_children),
        "unavailable_child_count": len(unavailable),
        "outcomes_sha256": sha256_file(outcomes_path),
        "features_sha256": sha256_file(features_path),
        "native_artifacts_accessed": False,
    }
    atomic_json(pack_root / "complete.json", marker)
    return marker


def aggregate_stage(
    *, output_root: Path, stage: str, packs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    assert_teacher_forced_paths([output_root])
    if not packs:
        raise ValueError("at least one immutable pack is required")
    model_keys = {str(pack["model_key"]) for pack in packs}
    if len(model_keys) != 1:
        raise RuntimeError("per-worker stage aggregation requires exactly one model")
    model_key = next(iter(model_keys))
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for pack in packs:
        root = output_root / "packs" / stage / str(pack["model_key"]) / str(pack["pack_id"])
        if not _pack_marker_valid(pack, root):
            missing.append(str(pack["pack_id"]))
            continue
        frames.append(pd.read_parquet(root / "outcomes.parquet"))
    if missing:
        raise RuntimeError(f"stage aggregation has incomplete packs: {missing[:5]}")
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    expected = {
        str(key["logical_id"]) for pack in packs for key in pack["logical_keys"]
    }
    observed = list(result["logical_id"].astype(str)) if len(result) else []
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise RuntimeError("aggregated stage failed exact-once integrity")
    destination = output_root / f"aggregated/{stage}/{model_key}_outcomes.parquet"
    atomic_parquet(destination, result)
    summary = {
        "status": "COMPLETE",
        "stage": stage,
        "model_key": model_key,
        "pack_count": len(packs),
        "logical_record_count": len(result),
        "outcomes_sha256": sha256_file(destination),
        "native_artifacts_accessed": False,
    }
    atomic_json(output_root / f"aggregated/{stage}/{model_key}_summary.json", summary)
    return summary
