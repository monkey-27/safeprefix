from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from safeprefix.config import load_config
from safeprefix.reproducibility import atomic_parquet
from safeprefix.threshold_selection_tf_v1.analysis import (
    make_bootstrap_plan,
    select_frozen_threshold,
    select_threshold_actions,
)
from safeprefix.threshold_selection_tf_v1.data import (
    MODEL_KEYS,
    assign_crossfit_folds,
    row_artifact_hash,
    sha256_file,
)
from safeprefix.threshold_selection_tf_v1.reporting import REPORT_NAMES
from safeprefix.threshold_selection_tf_v1.runtime import _valid_pack
from safeprefix.threshold_selection_tf_v1.sharding import (
    SHARD_IDS,
    freeze_execution_shards,
    validate_execution_shards,
)


ROOT = Path(__file__).resolve().parents[1]


def test_threshold_config_loads_with_frozen_protocol_values() -> None:
    config = load_config(ROOT / "configs/safeprefix_threshold_selection_tf_v1.yaml").data
    assert config["seed"] == 20260728
    assert config["artifacts_root"] == "artifacts/safeprefix_threshold_selection_tf_v1"
    assert config["selected_models"] == list(MODEL_KEYS)
    assert config["generation"]["checkpoint_rollout_indices"] == list(range(4, 16))
    assert config["generation"]["full_regeneration_indices"] == list(range(16))
    assert config["generation"]["preallocate_kv_cache"] is True
    assert config["source"]["expected_operational_seeds"] == {
        "family_a_small": 0,
        "family_a_large": 0,
        "family_b_small": 1,
        "family_b_large": 1,
    }


def test_problem_group_folds_are_deterministic_balanced_and_disjoint() -> None:
    rows = []
    for domain in ("d0", "d1"):
        for group in range(10):
            for duplicate in range(2 if group == 0 else 1):
                rows.append(
                    {
                        "common_trace_id": f"{domain}-g{group}-t{duplicate}",
                        "problem_group": f"{domain}-g{group}",
                        "domain": domain,
                    }
                )
    frame = pd.DataFrame(rows)
    first = assign_crossfit_folds(frame, folds=5, seed=20260728)
    second = assign_crossfit_folds(frame.sample(frac=1, random_state=3), folds=5, seed=20260728)
    columns = ["common_trace_id", "fold"]
    pd.testing.assert_frame_equal(
        first[columns].sort_values("common_trace_id").reset_index(drop=True),
        second[columns].sort_values("common_trace_id").reset_index(drop=True),
    )
    assert first.groupby("problem_group")["fold"].nunique().max() == 1
    assert first.groupby(["domain", "fold"]).size().min() >= 2


def _prediction_rows() -> pd.DataFrame:
    rows = []
    probabilities = [0.8, 0.4, 0.7]
    for model_key in MODEL_KEYS:
        for ordinal, probability in enumerate(probabilities):
            rows.append(
                {
                    "base_model": model_key,
                    "trace_id": f"{model_key}-trace",
                    "common_trace_id": "shared",
                    "problem_id": "problem",
                    "problem_group": "group",
                    "domain": "domain",
                    "checkpoint_id": f"{model_key}-c{ordinal}",
                    "checkpoint_ordinal": ordinal,
                    "checkpoint_token_offset": 10 + ordinal,
                    "oof_probability": probability,
                }
            )
    return pd.DataFrame(rows)


def test_outcome_blind_selector_uses_latest_qualifying_checkpoint_and_anchor() -> None:
    predictions = _prediction_rows()
    actions = select_threshold_actions(predictions, [0.0, 0.5, 0.75, 1.0])
    one = actions[actions["base_model"].eq(MODEL_KEYS[0])].set_index("threshold")
    assert one.loc[0.0, "selected_checkpoint_ordinal"] == 2
    assert one.loc[0.5, "selected_checkpoint_ordinal"] == 2
    assert one.loc[0.75, "selected_checkpoint_ordinal"] == 0
    assert bool(one.loc[1.0, "fallback"])
    contaminated = predictions.assign(dense_success_rate=1.0)
    with pytest.raises(RuntimeError, match="outcome column"):
        select_threshold_actions(contaminated, [0.5])


def test_domain_stratified_bootstrap_preserves_complete_shared_trace_count() -> None:
    rows = []
    for model_key in MODEL_KEYS:
        for index in range(8):
            rows.append(
                {
                    "base_model": model_key,
                    "common_trace_id": f"t{index}",
                    "domain": "left" if index < 3 else "right",
                }
            )
    plan, trace_order = make_bootstrap_plan(pd.DataFrame(rows), replicates=50, seed=7)
    assert plan.shape == (50, 8)
    assert len(trace_order) == 8
    assert np.allclose(plan.sum(axis=1), 1.0)
    left = [trace_order.index(f"t{index}") for index in range(3)]
    right = [trace_order.index(f"t{index}") for index in range(3, 8)]
    assert np.allclose(plan[:, left].sum(axis=1), 3 / 8)
    assert np.allclose(plan[:, right].sum(axis=1), 5 / 8)


def test_threshold_selection_uses_one_percent_conservative_tie() -> None:
    curve = pd.DataFrame(
        [
            {"threshold": 0.2, "feasible": True, "macro_mean_fresh_tokens": 100.0, "macro_fallback_rate": 0.3},
            {"threshold": 0.4, "feasible": True, "macro_mean_fresh_tokens": 100.9, "macro_fallback_rate": 0.4},
            {"threshold": 0.6, "feasible": True, "macro_mean_fresh_tokens": 101.1, "macro_fallback_rate": 0.2},
            {"threshold": 1.0, "feasible": False, "macro_mean_fresh_tokens": 200.0, "macro_fallback_rate": 1.0},
        ]
    )
    selected = select_frozen_threshold(curve)
    assert selected["selected_tau"] == 0.4
    assert selected["tie_candidates"] == [0.2, 0.4]


def _outcome_row(*, full: bool, seed: int, index: int) -> dict[str, object]:
    row: dict[str, object] = {
        "schema_version": 1,
        "scientific": True,
        "model_key": "family_a_small",
        "trace_id": "trace",
        "common_trace_id": "common",
        "pack_id": "pack",
        "pack_hash": "hash",
        "rollout_index": index,
        "rollout_seed": seed,
        "generated_token_ids": [1, 2, 3],
        "generated_token_count": 3,
        "raw_suffix": "answer",
        "normalized_final_answer": None,
        "parser_status": "failure",
        "truncation_status": False,
        "binary_outcome": False,
        "infrastructure_status": "executed",
        "latency_seconds": 0.25,
    }
    if full:
        row.update(checkpoint_id=None, checkpoint_ordinal=None)
    else:
        row.update(checkpoint_id="checkpoint", checkpoint_ordinal=0)
    row["artifact_hash"] = row_artifact_hash(row)
    return row


def test_row_hash_survives_parquet_and_pack_validation_checks_exact_seeds(tmp_path: Path) -> None:
    dense = pd.DataFrame([_outcome_row(full=False, seed=101, index=4)])
    full = pd.DataFrame([_outcome_row(full=True, seed=202, index=0)])
    root = tmp_path / "pack"
    dense_path = root / "added_checkpoint_suffixes.parquet"
    full_path = root / "full_regenerations.parquet"
    atomic_parquet(dense_path, dense)
    atomic_parquet(full_path, full)
    dense_roundtrip = pd.read_parquet(dense_path).iloc[0].to_dict()
    assert row_artifact_hash(dense_roundtrip) == dense_roundtrip["artifact_hash"]
    pack = {
        "pack_id": "pack",
        "pack_hash": "hash",
        "checkpoint_count": 1,
        "trace_count": 1,
    }
    marker = {
        "status": "COMPLETE",
        "scientific": True,
        "pack_hash": "hash",
        "dense_sha256": sha256_file(dense_path),
        "full_regeneration_sha256": sha256_file(full_path),
    }
    (root / "complete.json").write_text(json.dumps(marker))
    checkpoint_manifest = pd.DataFrame(
        [
            {
                "base_model": "family_a_small",
                "trace_id": "trace",
                "checkpoint_id": "checkpoint",
                "checkpoint_ordinal": 0,
                "rollout_index": 4,
                "rollout_seed": 101,
            }
        ]
    )
    regeneration_manifest = pd.DataFrame(
        [
            {
                "base_model": "family_a_small",
                "trace_id": "trace",
                "rollout_index": 0,
                "rollout_seed": 202,
            }
        ]
    )
    assert _valid_pack(
        pack,
        root,
        added_rollouts=1,
        full_regenerations=1,
        scientific=True,
        checkpoint_manifest=checkpoint_manifest,
        regeneration_manifest=regeneration_manifest,
    )
    wrong = regeneration_manifest.assign(rollout_seed=999)
    assert not _valid_pack(
        pack,
        root,
        added_rollouts=1,
        full_regenerations=1,
        scientific=True,
        checkpoint_manifest=checkpoint_manifest,
        regeneration_manifest=wrong,
    )


def test_publication_contract_and_modal_native_volume_is_absent() -> None:
    assert len(REPORT_NAMES) == 12
    launcher = (ROOT / "scripts/modal_safeprefix_threshold_selection_tf_v1.py").read_text()
    assert "safeprefix-native-failed-trace-runs-v1" not in launcher
    assert 'gpu="H100!"' in launcher
    assert '"/boundary_source"' not in launcher
    assert '"/completion"' not in launcher
    assert "max_containers=10" in launcher
    assert "results = {model_key: call.get()" in launcher
    assert "reload is required before this long-lived coordinator" in launcher


def _synthetic_shard_root(tmp_path: Path) -> tuple[dict[str, object], Path]:
    config: dict[str, object] = {
        "execution": {
            "gpu_workers_by_model": {
                "family_a_small": 3,
                "family_a_large": 3,
                "family_b_small": 5,
                "family_b_large": 9,
            }
        }
    }
    root = tmp_path / "artifacts"
    manifests = root / "manifests"
    manifests.mkdir(parents=True)
    from safeprefix.reproducibility import stable_hash

    (manifests / "frozen_protocol.json").write_text(
        json.dumps({"configuration_hash": stable_hash(config)})
    )
    rows = []
    for model_key in MODEL_KEYS:
        for index in range(60):
            rows.append(
                {
                    "model_key": model_key,
                    "pack_id": f"{model_key}-{index:03d}",
                    "estimated_work": 1 + (index % 7),
                }
            )
    (manifests / "execution_packs.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    pd.DataFrame({"pack_id": [row["pack_id"] for row in rows]}).to_parquet(
        manifests / "dense_checkpoint_rollout_manifest.parquet", index=False
    )
    pd.DataFrame({"pack_id": [row["pack_id"] for row in rows]}).to_parquet(
        manifests / "full_regeneration_manifest.parquet", index=False
    )
    return config, root


def test_cross_workspace_shards_are_deterministic_disjoint_and_complete(tmp_path: Path) -> None:
    config, root = _synthetic_shard_root(tmp_path)
    first = freeze_execution_shards(
        config,
        artifact_root=root,
        run_id="run",
        source_digest="digest",
        source_commit="commit",
    )
    second = validate_execution_shards(
        config,
        artifact_root=root,
        run_id="run",
        source_digest="digest",
        source_commit="commit",
    )
    assert first["master"] == second["master"]
    assert first["master"]["total_slots"] == 20
    assert first["master"]["total_packs"] == 240
    assert set(first["shards"]) == set(SHARD_IDS)
    assert all(first["shards"][shard_id]["slot_count"] == 10 for shard_id in SHARD_IDS)
    owned = [
        pack_id
        for shard in first["shards"].values()
        for slot in shard["slots"]
        for pack_id in slot["pack_ids"]
    ]
    assert len(owned) == len(set(owned)) == 240


def test_cross_workspace_shard_tampering_is_rejected(tmp_path: Path) -> None:
    config, root = _synthetic_shard_root(tmp_path)
    freeze_execution_shards(
        config,
        artifact_root=root,
        run_id="run",
        source_digest="digest",
        source_commit="commit",
    )
    path = root / "manifests/execution_shards/shard-00.json"
    payload = json.loads(path.read_text())
    payload["slots"][0]["pack_ids"] = payload["slots"][0]["pack_ids"][1:]
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="shard hash differs"):
        validate_execution_shards(
            config,
            artifact_root=root,
            run_id="run",
            source_digest="digest",
            source_commit="commit",
        )
