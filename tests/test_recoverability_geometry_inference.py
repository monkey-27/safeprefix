from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pandas as pd
import pytest

from safeprefix.recoverability_geometry_inference import (
    MODEL_KEYS,
    GEOMETRY_SEED,
    STAGE_DENSE,
    STAGE_LOCAL_CHILD,
    STAGE_LOCAL_PREFIX,
    STAGE_PROMPT,
    _pack_marker_valid,
    aggregate_stage,
    assert_teacher_forced_paths,
    build_dense_manifest,
    build_local_branch_manifest,
    build_prompt_manifest,
    dense_rollout_seed,
    geometry_seed,
    prepare_local_parent_packs,
    prepare_inference_manifests,
)
from safeprefix.reproducibility import atomic_json, atomic_parquet
from safeprefix.reproducibility import stable_seed
from safeprefix.production_suite import parser_verifier_hashes, sha256_file


def checkpoint_row(
    model_key: str,
    trace_id: str,
    checkpoint: int,
    *,
    group: str = "group-0",
) -> dict:
    return {
        "base_model": model_key,
        "model_id": f"model/{model_key}",
        "model_revision": f"revision-{model_key}",
        "tokenizer_revision": f"revision-{model_key}",
        "trace_id": trace_id,
        "common_trace_id": "common-0",
        "problem_id": "problem-0",
        "problem_group": group,
        "domain": "processbench_math",
        "split": "teacher_forced_test",
        "checkpoint_id": f"{trace_id}:{checkpoint}",
        "checkpoint_ordinal": checkpoint,
        "checkpoint_token_offset": 10 + checkpoint,
        "num_rollouts": 4,
    }


def test_native_artifact_paths_are_rejected() -> None:
    assert_teacher_forced_paths(["/completion/teacher_forced", "/geometry/tf_v1"])
    with pytest.raises(RuntimeError, match="forbidden"):
        assert_teacher_forced_paths(["/runs/native_evaluation/final"])


def test_seeds_are_deterministic_stage_separated_and_dense_continues_k4() -> None:
    first = geometry_seed(
        STAGE_PROMPT,
        model_key="family_a_small",
        trace_id="trace",
        rollout_index=0,
    )
    assert first == geometry_seed(
        STAGE_PROMPT,
        model_key="family_a_small",
        trace_id="trace",
        rollout_index=0,
    )
    assert first != geometry_seed(
        STAGE_LOCAL_PREFIX,
        model_key="family_a_small",
        trace_id="trace",
        branch_index=0,
    )
    assert dense_rollout_seed("trace", 0, 4) != dense_rollout_seed("trace", 0, 5)
    with pytest.raises(ValueError):
        dense_rollout_seed("trace", 0, 3)


def test_dense_manifest_is_exact_deterministic_and_work_balanced() -> None:
    rows = [
        checkpoint_row("family_a_small", "trace-b", 0, group="group-b"),
        checkpoint_row("family_a_small", "trace-a", 0, group="group-a"),
        checkpoint_row("family_a_small", "trace-a", 1, group="group-a"),
    ]
    keys, packs = build_dense_manifest(
        rows,
        model_key="family_a_small",
        configuration_hash="config",
        traces_per_pack=1,
    )
    keys_again, packs_again = build_dense_manifest(
        list(reversed(rows)),
        model_key="family_a_small",
        configuration_hash="config",
        traces_per_pack=1,
    )
    assert len(keys) == 3 * 28
    assert {key["rollout_index"] for key in keys} == set(range(4, 32))
    assert len({key["logical_id"] for key in keys}) == len(keys)
    assert keys_again == keys
    assert packs_again == packs
    assert sum(pack["logical_record_count"] for pack in packs) == len(keys)


def test_prompt_manifest_deduplicates_problem_groups() -> None:
    rows = [
        checkpoint_row("family_a_small", "trace-z", 0),
        checkpoint_row("family_a_small", "trace-a", 1),
        checkpoint_row("family_a_small", "trace-a", 0),
    ]
    keys, packs = build_prompt_manifest(
        rows,
        model_key="family_a_small",
        configuration_hash="config",
    )
    assert len(keys) == 16
    assert {key["trace_id"] for key in keys} == {"trace-a"}
    assert {key["rollout_index"] for key in keys} == set(range(16))
    assert sum(pack["logical_record_count"] for pack in packs) == 16


def test_local_manifest_has_prefix_and_four_children_at_every_horizon() -> None:
    parents = [
        {
            "model_key": "family_a_small",
            "trace_id": "trace-a",
            "checkpoint_id": "trace-a:2",
            "checkpoint_index": 2,
            "checkpoint_token_offset": 42,
            "domain": "processbench_math",
        }
    ]
    keys, packs = build_local_branch_manifest(
        parents,
        configuration_hash="config",
        branches_per_parent=12,
    )
    prefix = [key for key in keys if key["stage"] == STAGE_LOCAL_PREFIX]
    children = [key for key in keys if key["stage"] == STAGE_LOCAL_CHILD]
    assert len(prefix) == 12
    assert len(children) == 12 * 3 * 4
    assert {key["horizon"] for key in children} == {32, 64, 128}
    assert len({key["rollout_seed"] for key in keys}) == len(keys)
    assert packs[0]["logical_record_count"] == 12 + 12 * 3 * 4
    with pytest.raises(ValueError, match="at most one"):
        build_local_branch_manifest(
            [parents[0], {**parents[0], "checkpoint_index": 3}],
            configuration_hash="config",
        )


def test_pack_marker_and_aggregation_enforce_exact_once(tmp_path: Path) -> None:
    rows = [checkpoint_row("family_a_small", "trace-a", 0)]
    _, packs = build_dense_manifest(
        rows,
        model_key="family_a_small",
        configuration_hash="config",
        rollout_indices=(4, 5),
    )
    pack = packs[0]
    pack_root = tmp_path / "packs" / STAGE_DENSE / "family_a_small" / pack["pack_id"]
    outcomes = pd.DataFrame(
        [{"logical_id": key["logical_id"], "binary_outcome": False} for key in pack["logical_keys"]]
    )
    outcomes_path = pack_root / "outcomes.parquet"
    atomic_parquet(outcomes_path, outcomes)
    atomic_json(
        pack_root / "complete.json",
        {
            "pack_hash": pack["pack_hash"],
            "outcomes_sha256": sha256_file(outcomes_path),
        },
    )
    assert _pack_marker_valid(pack, pack_root)
    summary = aggregate_stage(output_root=tmp_path, stage=STAGE_DENSE, packs=packs)
    assert summary["logical_record_count"] == 2
    duplicate = pd.concat([outcomes, outcomes.iloc[[0]]], ignore_index=True)
    atomic_parquet(outcomes_path, duplicate)
    atomic_json(
        pack_root / "complete.json",
        {
            "pack_hash": pack["pack_hash"],
            "outcomes_sha256": sha256_file(outcomes_path),
        },
    )
    assert not _pack_marker_valid(pack, pack_root)


def test_prepare_reconciles_canonical_checkpoints_not_seed_expanded_rows(
    tmp_path: Path,
) -> None:
    boundary = tmp_path / "boundary"
    completion_manifest = tmp_path / "completion_manifests"
    original = tmp_path / "original_rollouts"
    extension = tmp_path / "extension_rollouts"
    output = tmp_path / "geometry"
    canonical_rows = []
    prediction_rows = []
    model_entries = {}
    for model_key in MODEL_KEYS:
        trace_id = f"{model_key}-trace"
        for checkpoint in range(2):
            row = checkpoint_row(model_key, trace_id, checkpoint)
            canonical_rows.append(row)
            for seed in range(3):
                prediction_rows.append(
                    {
                        **row,
                        "training_seed": seed,
                        "selected_architecture": "linear_probe",
                        "selected_learning_rate": 0.001,
                        "raw_logit": seed,
                    }
                )
        model_entries[model_key] = {
            "revision": f"revision-{model_key}",
            "tokenizer_revision": f"revision-{model_key}",
        }
        trace_root = completion_manifest / f"repairability/per_model/{model_key}"
        trace_root.mkdir(parents=True, exist_ok=True)
        (trace_root / "trace_manifest.jsonl").write_text(
            json.dumps(
                {
                    "trace_id": trace_id,
                    "prompt_token_ids": [1, 2],
                    "completion_token_ids": [3, 4, 5, 6, 7, 8, 9, 10],
                    "eligible_checkpoint_offsets": [10, 11],
                    "problem_id": "problem-0",
                        "source_bucket": "processbench_math",
                        "problem_text": "Compute one plus zero.",
                        "reference_answer": "1",
                }
            )
            + "\n"
        )
        rollout_rows = []
        frozen_hashes = parser_verifier_hashes()
        for checkpoint in range(2):
            for rollout_index in range(4):
                rollout_rows.append(
                    {
                        "model_key": model_key,
                        "trace_id": trace_id,
                        "checkpoint_index": checkpoint,
                        "checkpoint_token_offset": 10 + checkpoint,
                        "rollout_index": rollout_index,
                        "rollout_seed": stable_seed(
                            2701, trace_id, checkpoint, rollout_index
                        ),
                        "verifier_pass": rollout_index == 0,
                        "binary_outcome": rollout_index == 0,
                        "truncation_flag": False,
                        "parser_status": "success",
                        "generated_token_ids": [11, 12],
                        "generated_text": "The answer is 1.",
                        "stop_reason": "eos",
                        **frozen_hashes,
                    }
                )
        rollout_path = original / model_key / "pack-0/rollouts.parquet"
        atomic_parquet(rollout_path, pd.DataFrame(rollout_rows))
        atomic_json(
            rollout_path.with_name("complete.json"),
            {"rollouts_sha256": sha256_file(rollout_path), "integrity": {"passed": True}},
        )
    atomic_parquet(
        boundary / "data/canonical_checkpoint_manifest.parquet",
        pd.DataFrame(canonical_rows),
    )
    atomic_parquet(
        boundary / "test/teacher_forced_test_predictions.parquet",
        pd.DataFrame(prediction_rows),
    )
    summary = prepare_inference_manifests(
        boundary_root=boundary,
        completion_manifest_root=completion_manifest,
        original_rollout_root=original,
        extension_rollout_root=extension,
        output_root=output,
        model_entries=model_entries,
        configuration_hash="config",
        engine_digest="engine",
    )
    assert summary["checkpoint_model_count"] == 8
    assert summary["prediction_row_reconciliation"]["prediction_rows"] == 24
    assert summary["prediction_row_reconciliation"]["rows_per_checkpoint"] == 3
    assert summary["new_dense_rollout_count"] == 8 * 28
    assert summary["prompt_generation_count"] == 4 * 16
    assert summary["native_artifacts_accessed"] is False

    # The ready marker is identity-bound rather than a blind resume shortcut.
    with pytest.raises(RuntimeError, match="engine digest differs"):
        prepare_inference_manifests(
            boundary_root=boundary,
            completion_manifest_root=completion_manifest,
            original_rollout_root=original,
            extension_rollout_root=extension,
            output_root=output,
            model_entries=model_entries,
            configuration_hash="config",
            engine_digest="different-engine",
        )


def test_local_parent_bridge_freezes_exact_160_parent_design(tmp_path: Path) -> None:
    rows = []
    for model_key in MODEL_KEYS:
        categories = ["near_boundary"] * 24 + ["high"] * 8 + ["low"] * 8
        for index, category in enumerate(categories):
            recoverability = {
                "near_boundary": 0.5,
                "high": 0.875,
                "low": 0.125,
            }[category]
            trace_id = f"{model_key}-trace-{index}"
            checkpoint_id = f"{trace_id}:0"
            rank_hash = hashlib.sha256(
                "||".join(
                    (model_key, trace_id, checkpoint_id, str(GEOMETRY_SEED))
                ).encode("utf-8")
            ).hexdigest()
            rows.append(
                {
                    "model_key": model_key,
                    "trace_id": trace_id,
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_index": 0,
                    "checkpoint_token_offset": 20,
                    "domain": "processbench_math",
                    "selection_category": category,
                    "dense_recoverability": recoverability,
                    "selection_rank_hash": rank_hash,
                }
            )
    parent_path = tmp_path / "analysis/local_parent_selection.jsonl"
    parent_path.parent.mkdir(parents=True)
    parent_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = prepare_local_parent_packs(
        parent_manifest_path=parent_path,
        output_root=tmp_path / "geometry",
        configuration_hash="config",
    )
    assert summary["parent_count"] == 160
    assert summary["prefix_count"] == 160 * 12
    assert summary["planned_child_count"] == 160 * 12 * 3 * 4
