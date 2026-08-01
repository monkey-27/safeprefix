from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from safeprefix.prefix_validity_v1.data import (
    PrefixValidityDataError,
    SOURCE_REPRESENTATION,
    TARGET_REPRESENTATION,
    _discover_pack_dirs,
    _final_answer_only,
    _load_model_features,
    _selected_trace_rows,
    completed_step_and_label,
    indexing_convention,
    validate_recoverability_feature_equivalence,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trace() -> dict:
    return {
        "trace_id": "trace-a",
        "feature_trace_id": "trace-a",
        "feature_origin": "reused_teacher_forced_completion_r5",
        "source_trace_id": "source-a",
        "source_dataset": "Qwen/ProcessBench",
        "source_subset": "math",
        "problem_id": "math-1",
        "production_problem_group": "group-a",
        "model_revision": "model-sha",
        "tokenizer_revision": "tokenizer-sha",
        "representation": SOURCE_REPRESENTATION,
        "reasoning_steps": ["clean", "wrong", "still wrong"],
        "final_answer_text": "9",
        "first_error_zero_based": 1,
        "checkpoint_indices": [0, 1, 2, 3],
        "checkpoint_offsets": [10, 20, 30, 40],
        "full_token_count": 50,
        "source_step_boundary_audits": [
            {
                "boundary_policy": "snap_after_complete_source_step_v1",
                "character_displacement": 2 if index == 0 else 0,
                "complete_step_included": True,
                "exact": index != 0,
                "next_step_content_included": False,
                "snap_policy": "after",
                "step_index": index,
                "token_offset": offset,
            }
            for index, offset in enumerate([10, 20, 30])
        ],
    }


def _write_pack(root: Path, *, conflict: bool = False) -> Path:
    pack = root / "safety" / "family_a_small" / "pack-a"
    pack.mkdir(parents=True)
    hidden = 2
    # Layout: current three layers, delta three layers, final summary three
    # layers, and four auxiliary features.  The production representation is
    # the third/current block, columns [2h:3h].
    raw = torch.arange(4 * (9 * hidden + 4), dtype=torch.float16).reshape(4, -1)
    if conflict:
        raw += 1
    torch.save(
        {
            "trace-a": {
                "features": raw,
                "checkpoint_offsets": [10, 20, 30, 40],
                "checkpoint_indices": [0, 1, 2, 3],
                "visible_safety_labels": [True, True, False, False],
                "safety_supervision_mask": [True, True, True, True],
                "first_error_zero_based": 1,
                "selected_layers": [1, 2, -1],
                "model_revision": "model-sha",
                "tokenizer_revision": "tokenizer-sha",
                "representation": SOURCE_REPRESENTATION,
            }
        },
        pack / "features.pt",
    )
    pd.DataFrame(
        [
            {
                "model_key": "family_a_small",
                "trace_id": "trace-a",
                "checkpoint_index": index,
                "checkpoint_token_offset": offset,
                "visible_safety_label": label,
                "safety_supervision_mask": True,
                "feature_row_index": index,
            }
            for index, (offset, label) in enumerate(
                zip([10, 20, 30, 40], [True, True, False, False])
            )
        ]
    ).to_parquet(pack / "checkpoint_metadata.parquet", index=False)
    marker = {
        "status": "COMPLETE",
        "pack_id": "pack-a",
        "model_key": "family_a_small",
        "features_sha256": _sha(pack / "features.pt"),
        "metadata_sha256": _sha(pack / "checkpoint_metadata.parquet"),
    }
    (pack / "complete.json").write_text(json.dumps(marker))
    return pack


def test_indexing_resolves_root_first_error_and_later_steps() -> None:
    assert indexing_convention()["status"] == "RESOLVED"
    assert completed_step_and_label(0, 1) == (
        None,
        None,
        "prompt_root_is_not_training_checkpoint",
    )
    assert completed_step_and_label(1, 1) == (0, 1, None)
    assert completed_step_and_label(2, 1) == (1, 0, None)
    assert completed_step_and_label(3, 1) == (2, 0, None)
    assert completed_step_and_label(1, None)[2] == (
        "missing_or_ambiguous_first_error_annotation"
    )


def test_processbench_filter_preserves_exclusion_lineage() -> None:
    valid = _trace()
    valid["index_base"] = "zero"
    controlled = {**valid, "trace_id": "crv", "source_dataset": "facebook/crv"}
    selected, excluded = _selected_trace_rows(
        [valid, controlled], {"group-a": "train"}, model_key="family_a_small"
    )
    assert [row["trace_id"] for row in selected] == ["trace-a"]
    assert excluded[0]["trace_id"] == "crv"
    assert excluded[0]["exclusion_reason"] == "excluded_non_processbench_or_crv"
    assert excluded[0]["split"] == "train"


def test_final_answer_only_filter_is_conservative() -> None:
    assert _final_answer_only(r"\boxed{9}", "9")
    assert _final_answer_only("Final answer: 9", "9")
    assert not _final_answer_only("Therefore the result is \\boxed{9}", "9")
    assert not _final_answer_only("We compute 3 squared, so the answer is 9", "9")


def test_frozen_pack_extracts_identical_final_current_layer_and_labels(tmp_path: Path) -> None:
    pack = _write_pack(tmp_path)
    packs = _discover_pack_dirs([tmp_path], "family_a_small")
    records, features, inventory = _load_model_features(
        model_key="family_a_small",
        selected=[_trace()],
        pack_dirs=packs,
        spec={
            "model_id": "model",
            "model_revision": "model-sha",
            "tokenizer_revision": "tokenizer-sha",
            "hidden_size": 2,
            "selected_layers": [1, 2, -1],
        },
        assignments={"group-a": "train"},
    )
    raw = torch.load(pack / "features.pt", map_location="cpu", weights_only=False)[
        "trace-a"
    ]["features"]
    assert torch.equal(features, raw[1:, 4:6])
    assert [row["prefix_valid"] for row in records] == [1, 0, 0]
    assert [row["completed_step_zero_based"] for row in records] == [0, 1, 2]
    assert [row["true_last_valid_checkpoint"] for row in records] == [1, 1, 1]
    assert [row["hidden_feature_row_index"] for row in records] == [0, 1, 2]
    assert all(row["hidden_state_representation"] == TARGET_REPRESENTATION for row in records)
    assert records[0]["alignment_exact"] is False
    assert records[0]["alignment_character_displacement"] == 2
    assert len(inventory) == 1


def test_missing_selected_feature_trace_fails_instead_of_recomputing(tmp_path: Path) -> None:
    with pytest.raises(PrefixValidityDataError, match="teacher forcing would be required"):
        _load_model_features(
            model_key="family_a_small",
            selected=[_trace()],
            pack_dirs={},
            spec={
                "model_id": "model",
                "model_revision": "model-sha",
                "tokenizer_revision": "tokenizer-sha",
                "hidden_size": 2,
                "selected_layers": [1, 2, -1],
            },
            assignments={"group-a": "train"},
        )


def test_conflicting_duplicate_pack_fails_loudly(tmp_path: Path) -> None:
    one = tmp_path / "one"
    two = tmp_path / "two"
    _write_pack(one)
    _write_pack(two, conflict=True)
    with pytest.raises(PrefixValidityDataError, match="conflicting duplicate safety pack"):
        _discover_pack_dirs([one, two], "family_a_small")


def test_shared_production_feature_equivalence_is_bitwise(tmp_path: Path) -> None:
    features = {"family_a_small": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float16)}
    rows = pd.DataFrame(
        [
            {
                "model_key": "family_a_small",
                "source_trace_id": "source-a",
                "checkpoint_index": 1,
                "hidden_feature_row_index": 0,
            },
            {
                "model_key": "family_a_small",
                "source_trace_id": "source-a",
                "checkpoint_index": 2,
                "hidden_feature_row_index": 1,
            },
        ]
    )
    data = tmp_path / "data"
    (data / "features").mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "base_model": "family_a_small",
                "source_trace_id": "source-a",
                "checkpoint_ordinal": 1,
                "feature_row_index": 0,
            },
            {
                "base_model": "family_a_small",
                "source_trace_id": "source-a",
                "checkpoint_ordinal": 2,
                "feature_row_index": 1,
            },
        ]
    ).to_parquet(data / "canonical_checkpoint_manifest.parquet", index=False)
    torch.save({"features": features["family_a_small"]}, data / "features/family_a_small.pt")
    result = validate_recoverability_feature_equivalence(rows, features, tmp_path)
    assert result["status"] == "PASS"
    assert result["models"]["family_a_small"]["bitwise_equal_rate"] == 1.0

    features["family_a_small"][1, 1] += 1
    with pytest.raises(PrefixValidityDataError, match="exceeds frozen numerical bounds"):
        validate_recoverability_feature_equivalence(rows, features, tmp_path)


def test_small_cross_run_feature_variation_is_audited_not_hidden(tmp_path: Path) -> None:
    production = torch.tensor([[10.0, 20.0], [30.0, 40.0]], dtype=torch.float16)
    features = {"family_a_small": production.clone()}
    features["family_a_small"][1] += torch.tensor([0.03125, -0.03125])
    rows = pd.DataFrame([
        {
            "model_key": "family_a_small",
            "source_trace_id": "source-a",
            "checkpoint_index": index + 1,
            "hidden_feature_row_index": index,
        }
        for index in range(2)
    ])
    data = tmp_path / "data"
    (data / "features").mkdir(parents=True)
    pd.DataFrame([
        {
            "base_model": "family_a_small",
            "source_trace_id": "source-a",
            "checkpoint_ordinal": index + 1,
            "feature_row_index": index,
        }
        for index in range(2)
    ]).to_parquet(data / "canonical_checkpoint_manifest.parquet", index=False)
    torch.save({"features": production}, data / "features/family_a_small.pt")
    result = validate_recoverability_feature_equivalence(rows, features, tmp_path)
    model = result["models"]["family_a_small"]
    assert model["bitwise_equal_rate"] == 0.5
    assert len(model["non_bitwise_rows"]) == 1
    assert model["minimum_cosine_similarity"] >= 0.99
