"""End-to-end bridges for the teacher-forced recoverability-geometry study.

This module deliberately contains no model inference.  It joins the immutable
per-model inference artifacts, invokes the pre-registered CPU analyses, freezes
the Phase-2 parent-selection handoff, and materializes the exact Phase-4 child
state table from:

* the canonical boundary-model feature store (parent state);
* the local-branch ``child_hidden_states.pt`` shards (child state); and
* four terminal verifier outcomes per available child state.

Every public entry point rejects native/final-test paths before reading them.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from safeprefix.production_suite import sha256_file
from safeprefix.recoverability_geometry_inference import (
    MODEL_KEYS,
    STAGE_DENSE,
    STAGE_LOCAL_CHILD,
    STAGE_LOCAL_PREFIX,
    STAGE_PROMPT,
    assert_teacher_forced_paths,
    read_jsonl,
)
from safeprefix.reproducibility import atomic_json, atomic_jsonl, atomic_parquet


def _normalize_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    """Add analysis aliases without dropping the production schema."""

    output = frame.copy()
    if "base_model" not in output:
        if "model_key" not in output:
            raise KeyError("inference outcomes lack model_key/base_model")
        output["base_model"] = output["model_key"].astype(str)
    if "model_key" not in output:
        output["model_key"] = output["base_model"].astype(str)
    if "verifier_outcome" not in output:
        if "binary_outcome" not in output:
            raise KeyError("inference outcomes lack verifier/binary outcome")
        output["verifier_outcome"] = output["binary_outcome"]
    if "binary_outcome" not in output:
        output["binary_outcome"] = output["verifier_outcome"]
    if "generation_seed" not in output:
        if "rollout_seed" not in output:
            raise KeyError("inference outcomes lack rollout/generation seed")
        output["generation_seed"] = output["rollout_seed"]
    if "rollout_seed" not in output:
        output["rollout_seed"] = output["generation_seed"]
    if "problem_group" not in output:
        if "problem_id" not in output:
            raise KeyError("inference outcomes lack problem_group/problem_id")
        output["problem_group"] = output["problem_id"].astype(str)
    return output


def combine_model_stage_outcomes(
    *,
    output_root: Path,
    stage: str,
    model_keys: Sequence[str] = MODEL_KEYS,
) -> tuple[Path, dict[str, Any]]:
    """Validate and concatenate the exact per-model stage aggregates."""

    assert_teacher_forced_paths([output_root])
    if stage not in {STAGE_DENSE, STAGE_PROMPT}:
        raise ValueError("only dense and prompt outcomes enter Phase 2")
    frames: list[pd.DataFrame] = []
    per_model: dict[str, int] = {}
    input_hashes: dict[str, str] = {}
    for model_key in map(str, model_keys):
        path = output_root / f"aggregated/{stage}/{model_key}_outcomes.parquet"
        summary_path = output_root / f"aggregated/{stage}/{model_key}_summary.json"
        if not path.is_file() or not summary_path.is_file():
            raise RuntimeError(f"{model_key}: missing completed {stage} aggregate")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "COMPLETE":
            raise RuntimeError(f"{model_key}: {stage} summary is not complete")
        if summary.get("outcomes_sha256") != sha256_file(path):
            raise RuntimeError(f"{model_key}: {stage} aggregate checksum differs")
        frame = _normalize_outcomes(pd.read_parquet(path))
        if set(frame["model_key"].astype(str)) != {model_key}:
            raise RuntimeError(f"{model_key}: {stage} aggregate has mixed model keys")
        if set(frame["base_model"].astype(str)) != {model_key}:
            raise RuntimeError(f"{model_key}: analysis model alias differs")
        if set(frame["stage"].astype(str)) != {stage}:
            raise RuntimeError(f"{model_key}: aggregate stage identity differs")
        if set(frame["infrastructure_status"].astype(str)) != {"complete"}:
            raise RuntimeError(f"{model_key}: unresolved {stage} infrastructure status")
        if frame["logical_id"].astype(str).duplicated().any():
            raise RuntimeError(f"{model_key}: duplicate {stage} logical records")
        if not frame["rollout_seed"].astype("int64").equals(
            frame["generation_seed"].astype("int64")
        ):
            raise RuntimeError(f"{model_key}: rollout/generation seed alias differs")
        per_model[model_key] = int(len(frame))
        input_hashes[str(path)] = sha256_file(path)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if combined["logical_id"].astype(str).duplicated().any():
        raise RuntimeError(f"cross-model duplicate logical IDs in {stage}")
    census_path = output_root / "manifests/prelaunch_census.json"
    if not census_path.is_file():
        raise RuntimeError("prelaunch census is absent")
    census = json.loads(census_path.read_text(encoding="utf-8"))
    expected_field = (
        "new_dense_rollout_count" if stage == STAGE_DENSE else "prompt_generation_count"
    )
    expected = int(census[expected_field])
    if len(combined) != expected:
        raise RuntimeError(
            f"{stage}: combined rows {len(combined)} != frozen census {expected}"
        )
    destination = output_root / f"aggregated/{stage}/all_models_outcomes.parquet"
    atomic_parquet(destination, combined)
    summary = {
        "status": "COMPLETE",
        "stage": stage,
        "models": list(map(str, model_keys)),
        "per_model_rows": per_model,
        "logical_record_count": int(len(combined)),
        "outcomes_sha256": sha256_file(destination),
        "input_hashes": input_hashes,
        "native_artifacts_accessed": False,
    }
    atomic_json(output_root / f"aggregated/{stage}/all_models_summary.json", summary)
    return destination, summary


def freeze_local_parent_bridge(
    *,
    output_root: Path,
    expected_parent_count: int = 160,
) -> tuple[Path, dict[str, Any]]:
    """Write the exact JSONL handoff consumed by local-branch inference."""

    assert_teacher_forced_paths([output_root])
    source = output_root / "manifests/local_parent_manifest.parquet"
    if not source.is_file():
        raise RuntimeError("Phase-2 local-parent parquet is absent")
    parents = pd.read_parquet(source).copy()
    aliases = {
        "base_model": "model_key",
        "checkpoint_ordinal": "checkpoint_index",
        "requested_category": "selection_category",
        "selection_hash": "selection_rank_hash",
    }
    for source_name, destination_name in aliases.items():
        if destination_name not in parents and source_name in parents:
            parents[destination_name] = parents[source_name]
    if "substitution_reason" not in parents:
        substituted = (
            parents["category_substitution"].astype(bool)
            if "category_substitution" in parents
            else pd.Series(False, index=parents.index)
        )
        parents["substitution_reason"] = substituted.map(
            {True: "closest_available_category", False: None}
        )
    required = {
        "model_key",
        "trace_id",
        "checkpoint_id",
        "checkpoint_index",
        "checkpoint_token_offset",
        "domain",
        "selection_category",
        "dense_recoverability",
        "selection_rank_hash",
    }
    if missing := required - set(parents):
        raise RuntimeError(f"local-parent analysis bridge lacks {sorted(missing)}")
    if len(parents) != int(expected_parent_count):
        raise RuntimeError(
            f"local-parent analysis bridge has {len(parents)}, expected {expected_parent_count}"
        )
    if parents.duplicated(["model_key", "trace_id"]).any():
        raise RuntimeError("local-parent bridge violates one parent per model-trace")
    destination = output_root / "analysis/local_parent_selection.jsonl"
    records = parents.sort_values(
        ["model_key", "selection_category", "selection_rank_hash"],
        kind="stable",
    ).to_dict("records")
    atomic_jsonl(destination, records)
    summary = {
        "status": "READY",
        "parent_count": int(len(records)),
        "per_model": Counter(map(str, parents["model_key"])),
        "source_sha256": sha256_file(source),
        "destination_sha256": sha256_file(destination),
        "native_artifacts_accessed": False,
    }
    atomic_json(output_root / "analysis/local_parent_bridge.json", summary)
    return destination, summary


def run_phase2_bridge(
    *,
    boundary_root: Path,
    output_root: Path,
    published_root: Path | None = None,
) -> dict[str, Any]:
    """Aggregate Phase 1, run Phase 2, and freeze the H4 parent bridge."""

    assert_teacher_forced_paths(
        [value for value in (boundary_root, output_root, published_root) if value]
    )
    dense_path, dense_summary = combine_model_stage_outcomes(
        output_root=output_root, stage=STAGE_DENSE
    )
    prompt_path, prompt_summary = combine_model_stage_outcomes(
        output_root=output_root, stage=STAGE_PROMPT
    )
    from safeprefix.recoverability_geometry.runner import run_post_download_analysis

    analysis_summary = run_post_download_analysis(
        boundary_root=boundary_root,
        dense_outcomes_path=dense_path,
        prompt_outcomes_path=prompt_path,
        output_root=output_root,
        published_root=published_root,
    )
    parent_path, parent_summary = freeze_local_parent_bridge(output_root=output_root)
    result = {
        "status": "PHASE2_COMPLETE_H4_PARENT_BRIDGE_READY",
        "dense": dense_summary,
        "prompt": prompt_summary,
        "analysis": analysis_summary,
        "local_parent_bridge": {
            **parent_summary,
            "path": str(parent_path),
        },
        "native_artifacts_accessed": False,
    }
    atomic_json(output_root / "analysis/phase2_bridge_summary.json", result)
    return result


def _load_parent_feature_lookup(
    *,
    boundary_root: Path,
    parents: pd.DataFrame,
    model_keys: Sequence[str],
) -> dict[tuple[str, str, str], np.ndarray]:
    canonical = pd.read_parquet(
        boundary_root / "data/canonical_checkpoint_manifest.parquet"
    )
    test = canonical.loc[
        canonical["split"].astype(str).eq("teacher_forced_test")
    ].copy()
    lookup: dict[tuple[str, str, str], np.ndarray] = {}
    for model_key in map(str, model_keys):
        payload = torch.load(
            boundary_root / f"data/features/{model_key}.pt",
            map_location="cpu",
            weights_only=False,
        )
        features = payload["features"].to(torch.float32).numpy()
        model_parents = parents.loc[parents["model_key"].astype(str).eq(model_key)]
        model_test = test.loc[test["base_model"].astype(str).eq(model_key)]
        indexed = model_test.set_index(["trace_id", "checkpoint_id"], drop=False)
        for row in model_parents.to_dict("records"):
            key = (str(model_key), str(row["trace_id"]), str(row["checkpoint_id"]))
            try:
                canonical_row = indexed.loc[(key[1], key[2])]
            except KeyError as exc:
                raise RuntimeError(f"local parent is absent from canonical test: {key}") from exc
            if isinstance(canonical_row, pd.DataFrame):
                raise RuntimeError(f"canonical parent identity is not unique: {key}")
            index = int(canonical_row["feature_row_index"])
            if not 0 <= index < len(features):
                raise RuntimeError(f"canonical feature index is out of range: {key}")
            lookup[key] = np.asarray(features[index], dtype=np.float32)
    if len(lookup) != len(parents):
        raise RuntimeError("not every local parent resolved to one canonical feature")
    return lookup


def _validate_local_pack(
    *,
    pack: Mapping[str, Any],
    pack_root: Path,
) -> tuple[pd.DataFrame, Mapping[str, torch.Tensor]]:
    marker_path = pack_root / "complete.json"
    outcomes_path = pack_root / "outcomes.parquet"
    features_path = pack_root / "child_hidden_states.pt"
    if not all(path.is_file() for path in (marker_path, outcomes_path, features_path)):
        raise RuntimeError(f"incomplete local pack: {pack['pack_id']}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("status") != "COMPLETE" or marker.get("pack_hash") != pack["pack_hash"]:
        raise RuntimeError(f"invalid local pack marker: {pack['pack_id']}")
    if marker.get("outcomes_sha256") != sha256_file(outcomes_path):
        raise RuntimeError(f"local outcomes checksum differs: {pack['pack_id']}")
    if marker.get("features_sha256") != sha256_file(features_path):
        raise RuntimeError(f"local child-feature checksum differs: {pack['pack_id']}")
    outcomes = pd.read_parquet(outcomes_path)
    expected = {str(row["logical_id"]) for row in pack["logical_keys"]}
    observed = list(outcomes["logical_id"].astype(str))
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise RuntimeError(f"local pack exact-once integrity failed: {pack['pack_id']}")
    features = torch.load(features_path, map_location="cpu", weights_only=False)
    if not isinstance(features, Mapping):
        raise RuntimeError(f"local feature shard is not a mapping: {pack['pack_id']}")
    return outcomes, features


def build_phase4_child_state_table(
    *,
    boundary_root: Path,
    output_root: Path,
    model_keys: Sequence[str] = MODEL_KEYS,
) -> tuple[Path, dict[str, Any]]:
    """Join local child features, four outcomes, and canonical parent features."""

    assert_teacher_forced_paths([boundary_root, output_root])
    parent_path = output_root / "analysis/local_parent_selection.jsonl"
    ready_path = output_root / "manifests/local/LOCAL_PARENTS_READY.json"
    if not parent_path.is_file() or not ready_path.is_file():
        raise RuntimeError("local-parent inference bridge is not frozen")
    parents = pd.DataFrame(read_jsonl(parent_path))
    if "model_key" not in parents and "base_model" in parents:
        parents["model_key"] = parents["base_model"].astype(str)
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    if ready.get("status") != "READY":
        raise RuntimeError("local-parent inference marker is not ready")
    if len(parents) != int(ready["parent_count"]):
        raise RuntimeError("local-parent JSONL count differs from ready marker")
    parent_features = _load_parent_feature_lookup(
        boundary_root=boundary_root,
        parents=parents,
        model_keys=model_keys,
    )
    parent_lookup = {
        (str(row["model_key"]), str(row["trace_id"]), str(row["checkpoint_id"])): row
        for row in parents.to_dict("records")
    }
    rows: list[dict[str, Any]] = []
    pack_hashes: dict[str, dict[str, str]] = {}
    planned_child_records = 0
    for model_key in map(str, model_keys):
        pack_manifest = output_root / f"manifests/local/{model_key}_packs.jsonl"
        packs = read_jsonl(pack_manifest)
        if not packs:
            raise RuntimeError(f"{model_key}: local pack manifest is empty")
        for pack in packs:
            planned_child_records += sum(
                str(key["stage"]) == STAGE_LOCAL_CHILD
                for key in pack["logical_keys"]
            )
            if str(pack["model_key"]) != model_key:
                raise RuntimeError(f"{model_key}: local pack model identity differs")
            pack_root = (
                output_root
                / "packs/local_branch_geometry"
                / model_key
                / str(pack["pack_id"])
            )
            outcomes, feature_map = _validate_local_pack(
                pack=pack, pack_root=pack_root
            )
            pack_hashes[str(pack["pack_id"])] = {
                "outcomes_sha256": sha256_file(pack_root / "outcomes.parquet"),
                "features_sha256": sha256_file(
                    pack_root / "child_hidden_states.pt"
                ),
            }
            child = outcomes.loc[
                outcomes["stage"].astype(str).eq(STAGE_LOCAL_CHILD)
            ].copy()
            prefix = outcomes.loc[
                outcomes["stage"].astype(str).eq(STAGE_LOCAL_PREFIX)
            ].copy()
            expected_prefix = sum(
                str(key["stage"]) == STAGE_LOCAL_PREFIX
                for key in pack["logical_keys"]
            )
            if len(prefix) != expected_prefix:
                raise RuntimeError(f"{pack['pack_id']}: local prefix count differs")
            group_columns = [
                "trace_id",
                "checkpoint_id",
                "checkpoint_index",
                "branch_index",
                "horizon",
            ]
            for identity, group in child.groupby(group_columns, sort=True):
                trace_id, checkpoint_id, checkpoint_index, branch_index, horizon = identity
                if len(group) != 4 or sorted(group["rollout_index"].astype(int)) != [
                    0,
                    1,
                    2,
                    3,
                ]:
                    raise RuntimeError(
                        f"{pack['pack_id']}: child state lacks exactly four outcomes"
                    )
                statuses = set(group["infrastructure_status"].astype(str))
                if statuses not in ({"complete"}, {"unavailable_by_protocol"}):
                    raise RuntimeError(
                        f"{pack['pack_id']}: mixed/unresolved child-state status {statuses}"
                    )
                feature_id = f"{trace_id}:{int(branch_index)}:{int(horizon)}"
                feature = feature_map.get(feature_id)
                if feature is not None:
                    feature = torch.as_tensor(feature).to(torch.float32).numpy()
                available = statuses == {"complete"}
                if available and feature is None:
                    raise RuntimeError(
                        f"{pack['pack_id']}: available child lacks hidden state {feature_id}"
                    )
                parent_key = (model_key, str(trace_id), str(checkpoint_id))
                if parent_key not in parent_lookup:
                    raise RuntimeError(f"local child has unknown parent: {parent_key}")
                parent = parent_lookup[parent_key]
                parent_raw = parent_features[parent_key]
                if feature is not None and tuple(feature.shape) != tuple(parent_raw.shape):
                    raise RuntimeError(
                        f"{pack['pack_id']}: parent/child feature dimensions differ"
                    )
                verifier_values = group["verifier_outcome"]
                if available and verifier_values.isna().any():
                    raise RuntimeError(
                        f"{pack['pack_id']}: completed child has missing verifier outcome"
                    )
                success_count = (
                    int(verifier_values.astype(bool).sum()) if available else 0
                )
                parent_id = f"{model_key}:{checkpoint_id}"
                rows.append(
                    {
                        "base_model": model_key,
                        "model_key": model_key,
                        "model_id": str(group["model_id"].iloc[0]),
                        "model_revision": str(group["model_revision"].iloc[0]),
                        "trace_id": str(trace_id),
                        "problem_id": str(group["problem_id"].iloc[0]),
                        "problem_group": str(parent.get("problem_group", "")),
                        "domain": str(group["domain"].iloc[0]),
                        "checkpoint_id": str(checkpoint_id),
                        "checkpoint_index": int(checkpoint_index),
                        "parent_id": parent_id,
                        "branch_id": (
                            f"{parent_id}:branch-{int(branch_index):02d}"
                        ),
                        "branch_index": int(branch_index),
                        "horizon": int(horizon),
                        "raw_hidden": (
                            None if feature is None else feature.astype(np.float32)
                        ),
                        "parent_raw_hidden": parent_raw.astype(np.float32),
                        "child_success_count": success_count,
                        "child_num_rollouts": 4 if available else 0,
                        "horizon_available": bool(available),
                        "parent_recoverability": float(
                            parent["dense_recoverability"]
                        ),
                        "early_termination_status": bool(
                            group["early_termination_status"].astype(bool).any()
                        ),
                        "child_rollout_seeds": sorted(
                            map(int, group["rollout_seed"])
                        ),
                        "child_logical_ids": sorted(
                            map(str, group["logical_id"])
                        ),
                        "infrastructure_status": next(iter(statuses)),
                        "source_pack_id": str(pack["pack_id"]),
                    }
                )
            expected_feature_ids = {
                (
                    str(pack["pack_id"]),
                    f"{key['trace_id']}:{int(key['branch_index'])}:{int(key['horizon'])}",
                )
                for key in pack["logical_keys"]
                if str(key["stage"]) == STAGE_LOCAL_CHILD
            }
            unknown = {
                (str(pack["pack_id"]), str(feature_id))
                for feature_id in feature_map
            } - expected_feature_ids
            if unknown:
                raise RuntimeError(
                    f"{pack['pack_id']}: child feature shard contains unknown keys"
                )
    frame = pd.DataFrame(rows)
    state_identity = ["base_model", "parent_id", "branch_id", "horizon"]
    if frame.duplicated(state_identity).any():
        raise RuntimeError("Phase-4 child-state table contains duplicate states")
    if planned_child_records % 4:
        raise RuntimeError("planned local children are not divisible into K=4 states")
    if (
        ready.get("planned_child_count") is not None
        and int(ready["planned_child_count"]) != planned_child_records
    ):
        raise RuntimeError("local ready marker and pack manifests disagree")
    expected_states = planned_child_records // 4
    if len(frame) != expected_states:
        raise RuntimeError(
            f"Phase-4 child states {len(frame)} != frozen expected {expected_states}"
        )
    if int(frame["child_num_rollouts"].sum()) != 4 * int(
        frame["horizon_available"].sum()
    ):
        raise RuntimeError("Phase-4 child rollout count is inconsistent")
    destination = output_root / "analysis/local_child_state_inputs.parquet"
    atomic_parquet(destination, frame)
    summary = {
        "status": "COMPLETE",
        "child_state_rows": int(len(frame)),
        "available_child_states": int(frame["horizon_available"].sum()),
        "unavailable_child_states": int((~frame["horizon_available"]).sum()),
        "terminal_child_completions": int(frame["child_num_rollouts"].sum()),
        "per_model": {
            str(model): {
                "states": int(len(part)),
                "available_states": int(part["horizon_available"].sum()),
                "terminal_completions": int(part["child_num_rollouts"].sum()),
            }
            for model, part in frame.groupby("base_model", sort=True)
        },
        "pack_hashes": pack_hashes,
        "table_sha256": sha256_file(destination),
        "native_artifacts_accessed": False,
    }
    atomic_json(output_root / "analysis/local_child_state_inputs_summary.json", summary)
    return destination, summary


def run_phase4_bridge(
    *,
    boundary_root: Path,
    output_root: Path,
    published_root: Path | None = None,
) -> dict[str, Any]:
    """Build the local child table and execute the frozen Phase-4 analysis."""

    assert_teacher_forced_paths(
        [value for value in (boundary_root, output_root, published_root) if value]
    )
    child_path, child_summary = build_phase4_child_state_table(
        boundary_root=boundary_root,
        output_root=output_root,
    )
    from safeprefix.recoverability_geometry.runner import run_phase4_analysis

    analysis_summary = run_phase4_analysis(
        boundary_root=boundary_root,
        child_states_path=child_path,
        output_root=output_root,
        published_root=published_root,
    )
    result = {
        "status": analysis_summary.get("status"),
        "child_state_bridge": child_summary,
        "analysis": analysis_summary,
        "native_artifacts_accessed": False,
    }
    atomic_json(output_root / "analysis/phase4_bridge_summary.json", result)
    return result


__all__ = [
    "build_phase4_child_state_table",
    "combine_model_stage_outcomes",
    "freeze_local_parent_bridge",
    "run_phase2_bridge",
    "run_phase4_bridge",
]
