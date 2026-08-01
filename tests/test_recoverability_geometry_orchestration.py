from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from safeprefix.production_suite import sha256_file
from safeprefix.recoverability_geometry_inference import (
    STAGE_DENSE,
    STAGE_LOCAL_CHILD,
    STAGE_LOCAL_PREFIX,
)
from safeprefix.recoverability_geometry_orchestration import (
    build_phase4_child_state_table,
    combine_model_stage_outcomes,
    freeze_local_parent_bridge,
)
from safeprefix.reproducibility import atomic_json, atomic_jsonl, atomic_parquet


def test_stage_bridge_normalizes_production_schema_without_losing_fields(
    tmp_path: Path,
) -> None:
    root = tmp_path / "teacher_forced_geometry"
    rows = pd.DataFrame(
        [
            {
                "stage": STAGE_DENSE,
                "logical_id": f"logical-{index}",
                "model_key": model,
                "problem_id": f"problem-{index}",
                "problem_group": f"group-{index}",
                "trace_id": f"trace-{index}",
                "checkpoint_id": f"checkpoint-{index}",
                "rollout_seed": 100 + index,
                "binary_outcome": index == 0,
                "infrastructure_status": "complete",
            }
            for index, model in enumerate(("a", "b"))
        ]
    )
    for model in ("a", "b"):
        path = root / f"aggregated/{STAGE_DENSE}/{model}_outcomes.parquet"
        frame = rows.loc[rows["model_key"] == model]
        atomic_parquet(path, frame)
        atomic_json(
            root / f"aggregated/{STAGE_DENSE}/{model}_summary.json",
            {
                "status": "COMPLETE",
                "outcomes_sha256": sha256_file(path),
            },
        )
    atomic_json(
        root / "manifests/prelaunch_census.json",
        {"new_dense_rollout_count": 2, "prompt_generation_count": 0},
    )
    path, summary = combine_model_stage_outcomes(
        output_root=root, stage=STAGE_DENSE, model_keys=("a", "b")
    )
    result = pd.read_parquet(path)
    assert summary["logical_record_count"] == 2
    assert result["base_model"].tolist() == result["model_key"].tolist()
    assert result["generation_seed"].tolist() == result["rollout_seed"].tolist()
    assert result["verifier_outcome"].tolist() == result["binary_outcome"].tolist()
    assert result["problem_group"].tolist() == ["group-0", "group-1"]


def test_parent_bridge_writes_exact_modal_handoff_schema(tmp_path: Path) -> None:
    root = tmp_path / "teacher_forced_geometry"
    parent = pd.DataFrame(
        [
            {
                "base_model": "a",
                "trace_id": "trace",
                "checkpoint_id": "trace:1",
                "checkpoint_ordinal": 1,
                "checkpoint_token_offset": 42,
                "domain": "math",
                "requested_category": "near_boundary",
                "dense_recoverability": 0.5,
                "selection_hash": "abc",
                "category_substitution": False,
            }
        ]
    )
    atomic_parquet(root / "manifests/local_parent_manifest.parquet", parent)
    path, summary = freeze_local_parent_bridge(
        output_root=root, expected_parent_count=1
    )
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert path == root / "analysis/local_parent_selection.jsonl"
    assert summary["parent_count"] == 1
    assert rows[0]["model_key"] == "a"
    assert rows[0]["checkpoint_index"] == 1
    assert rows[0]["selection_category"] == "near_boundary"
    assert rows[0]["selection_rank_hash"] == "abc"


def _local_key(stage: str, logical_id: str, rollout: int | None) -> dict:
    return {
        "stage": stage,
        "logical_id": logical_id,
        "model_key": "a",
        "trace_id": "trace",
        "checkpoint_id": "trace:0",
        "checkpoint_index": 0,
        "checkpoint_token_offset": 20,
        "branch_index": 0,
        "horizon": None if stage == STAGE_LOCAL_PREFIX else 32,
        "rollout_index": rollout,
        "rollout_seed": 100 if rollout is None else 200 + rollout,
    }


def test_phase4_bridge_joins_parent_child_and_four_outcomes(tmp_path: Path) -> None:
    boundary = tmp_path / "teacher_forced_boundary"
    root = tmp_path / "teacher_forced_geometry"
    parent = {
        "model_key": "a",
        "base_model": "a",
        "trace_id": "trace",
        "checkpoint_id": "trace:0",
        "checkpoint_index": 0,
        "checkpoint_token_offset": 20,
        "problem_group": "group",
        "domain": "math",
        "dense_recoverability": 0.5,
    }
    atomic_jsonl(root / "analysis/local_parent_selection.jsonl", [parent])
    atomic_json(
        root / "manifests/local/LOCAL_PARENTS_READY.json",
        {"status": "READY", "parent_count": 1, "prefix_count": 1},
    )
    canonical = pd.DataFrame(
        [
            {
                "base_model": "a",
                "split": "teacher_forced_test",
                "trace_id": "trace",
                "checkpoint_id": "trace:0",
                "feature_row_index": 0,
            }
        ]
    )
    atomic_parquet(
        boundary / "data/canonical_checkpoint_manifest.parquet", canonical
    )
    (boundary / "data/features").mkdir(parents=True)
    torch.save(
        {"features": torch.tensor([[1.0, 2.0]], dtype=torch.float16)},
        boundary / "data/features/a.pt",
    )
    prefix_key = _local_key(STAGE_LOCAL_PREFIX, "prefix", None)
    child_keys = [
        _local_key(STAGE_LOCAL_CHILD, f"child-{index}", index)
        for index in range(4)
    ]
    pack = {
        "model_key": "a",
        "pack_id": "pack",
        "pack_hash": "hash",
        "logical_keys": [prefix_key, *child_keys],
    }
    atomic_jsonl(root / "manifests/local/a_packs.jsonl", [pack])
    outcomes = pd.DataFrame(
        [
            {
                **prefix_key,
                "model_id": "model/a",
                "model_revision": "revision",
                "problem_id": "problem",
                "domain": "math",
                "infrastructure_status": "complete",
                "verifier_outcome": False,
                "early_termination_status": False,
            },
            *[
                {
                    **key,
                    "model_id": "model/a",
                    "model_revision": "revision",
                    "problem_id": "problem",
                    "domain": "math",
                    "infrastructure_status": "complete",
                    "verifier_outcome": index < 2,
                    "early_termination_status": False,
                }
                for index, key in enumerate(child_keys)
            ],
        ]
    )
    pack_root = root / "packs/local_branch_geometry/a/pack"
    outcomes_path = pack_root / "outcomes.parquet"
    features_path = pack_root / "child_hidden_states.pt"
    atomic_parquet(outcomes_path, outcomes)
    pack_root.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"trace:0:32": torch.tensor([3.0, 4.0], dtype=torch.float16)},
        features_path,
    )
    atomic_json(
        pack_root / "complete.json",
        {
            "status": "COMPLETE",
            "pack_hash": "hash",
            "outcomes_sha256": sha256_file(outcomes_path),
            "features_sha256": sha256_file(features_path),
        },
    )
    path, summary = build_phase4_child_state_table(
        boundary_root=boundary,
        output_root=root,
        model_keys=("a",),
    )
    result = pd.read_parquet(path)
    assert summary["child_state_rows"] == 1
    assert summary["terminal_child_completions"] == 4
    assert result.loc[0, "base_model"] == "a"
    assert result.loc[0, "parent_id"] == "a:trace:0"
    assert result.loc[0, "child_success_count"] == 2
    assert result.loc[0, "child_num_rollouts"] == 4
    assert result.loc[0, "horizon_available"]
    assert list(result.loc[0, "raw_hidden"]) == pytest.approx([3.0, 4.0])
    assert list(result.loc[0, "parent_raw_hidden"]) == pytest.approx([1.0, 2.0])
