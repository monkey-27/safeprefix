from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from safeprefix.recoverability_geometry.analysis import (
    AffineAxis,
    classify_trajectory,
    compute_eta,
    margin_probability_equivalence,
    select_canonical_seed,
    select_local_branch_parents,
    summarize_seed_stability,
    trajectory_sensitivity,
)
from safeprefix.recoverability_geometry.runner import (
    _build_seed_stability_table,
    _guard_non_native,
    _normalize_inference_outcomes,
    aggregate_prompt_solvability,
    merge_dense_k32,
)
from safeprefix.recoverability_geometry.statistics import holm_primary_families


def test_layernorm_probe_is_audited_in_exact_affine_feature_space() -> None:
    state = {
        "local_model.0.weight": torch.tensor([1.2, 0.8, 1.1]),
        "local_model.0.bias": torch.tensor([0.1, -0.2, 0.3]),
        "local_model.1.weight": torch.tensor([[2.0, -1.0, 0.5]]),
        "local_model.1.bias": torch.tensor([0.25]),
    }
    axis = AffineAxis.from_state_dict(state, layernorm_epsilon=1e-5)
    raw = np.asarray([[1.0, 3.0, -2.0], [0.5, 0.1, 4.0]])
    expected = torch.nn.functional.linear(
        torch.nn.functional.layer_norm(
            torch.tensor(raw, dtype=torch.float32),
            (3,), state["local_model.0.weight"], state["local_model.0.bias"], 1e-5,
        ),
        state["local_model.1.weight"], state["local_model.1.bias"],
    ).squeeze(1).numpy()
    assert np.allclose(axis.logit_from_raw(raw), expected, atol=1e-6)
    assert axis.audit()["raw_hidden_affine"] is False
    decomposition = axis.decompose(axis.transform_raw(raw))
    assert np.allclose(
        decomposition["parallel_component"] + decomposition["orthogonal_component"],
        axis.transform_raw(raw),
    )


def test_canonical_seed_and_eta_are_exact_and_unclipped() -> None:
    assert select_canonical_seed({0: 0.4, 1: 0.2, 2: 0.3}) == 2
    assert compute_eta(0.6, 0.3, 0.5) == pytest.approx(3.0)


def test_dense_merge_preserves_k4_and_requires_exact_k28() -> None:
    canonical = pd.DataFrame(
        [{"base_model": "m", "trace_id": "t", "checkpoint_id": "c", "split": "teacher_forced_test", "num_rollouts": 4, "success_count": 1}]
    )
    added = pd.DataFrame(
        [{"base_model": "m", "trace_id": "t", "checkpoint_id": "c", "rollout_seed": seed, "verifier_outcome": seed % 2, "infrastructure_status": "complete"} for seed in range(28)]
    )
    merged = merge_dense_k32(canonical, added)
    assert merged.loc[0, "original_k4_success_count"] == 1
    assert merged.loc[0, "new_success_count"] == 14
    assert merged.loc[0, "total_success_count"] == 15
    assert merged.loc[0, "num_rollouts"] == 32
    with pytest.raises(RuntimeError, match="exactly 28"):
        merge_dense_k32(canonical, added.iloc[:-1])


def test_phase2_adapter_accepts_real_inference_schema_without_mutating_rows() -> None:
    canonical = pd.DataFrame(
        [{"base_model": "m", "trace_id": "t", "checkpoint_id": "c", "split": "teacher_forced_test", "num_rollouts": 4, "success_count": 1}]
    )
    added = pd.DataFrame(
        [
            {
                "model_key": "m",
                "base_model": "m",
                "trace_id": "t",
                "checkpoint_id": "c",
                "rollout_seed": seed,
                "generation_seed": seed,
                "binary_outcome": seed % 2,
                "verifier_outcome": bool(seed % 2),
                "infrastructure_status": "complete",
                "logical_id": f"dense-{seed}",
            }
            for seed in range(28)
        ]
    )
    original = added.copy(deep=True)
    merged = merge_dense_k32(canonical, added)
    pd.testing.assert_frame_equal(added, original)
    assert merged.loc[0, "new_success_count"] == 14
    assert merged.loc[0, "num_rollouts"] == 32


@pytest.mark.parametrize(
    ("column", "replacement", "message"),
    [
        ("base_model", "b", "base_model != model_key"),
        ("verifier_outcome", False, "verifier_outcome != binary_outcome"),
        ("generation_seed", 2, "generation_seed != rollout_seed"),
    ],
)
def test_phase2_adapter_rejects_conflicting_inference_aliases(
    column: str, replacement: object, message: str
) -> None:
    row = {
        "model_key": "a",
        "base_model": "a",
        "binary_outcome": True,
        "verifier_outcome": True,
        "rollout_seed": 1,
        "generation_seed": 1,
    }
    row[column] = replacement
    with pytest.raises(RuntimeError, match=message):
        _normalize_inference_outcomes(pd.DataFrame([row]))


def test_seed_stability_join_keeps_k4_and_k32_trial_counts_explicit() -> None:
    canonical_rows = []
    dense_rows = []
    scored_rows = []
    for checkpoint, original_successes in (("c0", 1), ("c1", 3)):
        canonical_rows.append(
            {
                "base_model": "m",
                "trace_id": "t",
                "checkpoint_id": checkpoint,
                "split": "teacher_forced_test",
                "num_rollouts": 4,
                "success_count": original_successes,
            }
        )
        scored_rows.append(
            {
                "base_model": "m",
                "trace_id": "t",
                "checkpoint_id": checkpoint,
                "split": "teacher_forced_test",
                "num_rollouts": 4,
                "seed_0_raw_logit": -0.5 if checkpoint == "c0" else 0.5,
                "seed_1_raw_logit": -0.4 if checkpoint == "c0" else 0.6,
                "seed_2_raw_logit": -0.6 if checkpoint == "c0" else 0.4,
            }
        )
        for seed in range(28):
            dense_rows.append(
                {
                    "model_key": "m",
                    "trace_id": "t",
                    "checkpoint_id": checkpoint,
                    "rollout_seed": seed,
                    "binary_outcome": (seed + (checkpoint == "c1")) % 2,
                    "infrastructure_status": "complete",
                }
            )
    dense = merge_dense_k32(pd.DataFrame(canonical_rows), pd.DataFrame(dense_rows))
    stability_table = _build_seed_stability_table(
        pd.DataFrame(scored_rows), dense, model="m"
    )
    assert set(stability_table["num_rollouts"].astype(int)) == {4}
    assert set(stability_table["dense_num_rollouts"].astype(int)) == {32}
    assert not any(column.endswith(("_x", "_y")) for column in stability_table)
    summary = summarize_seed_stability(
        stability_table,
        score_columns={seed: f"seed_{seed}_raw_logit" for seed in (0, 1, 2)},
        success_column="total_success_count",
        trials_column="dense_num_rollouts",
    )
    assert set(summary["seed_k32_metrics"]) == {"0", "1", "2"}


def test_prompt_solvability_is_exact_16_and_jeffreys_smoothed() -> None:
    rows = pd.DataFrame(
        [{"base_model": "m", "problem_id": "p", "generation_seed": seed, "verifier_outcome": seed < 4, "infrastructure_status": "complete"} for seed in range(16)]
    )
    result = aggregate_prompt_solvability(rows)
    assert result.loc[0, "prompt_solvability_raw"] == 0.25
    assert result.loc[0, "prompt_solvability_smoothed"] == pytest.approx(4.5 / 17)


def test_h3_short_trace_is_descriptive_only_and_sensitivity_is_frozen() -> None:
    short = classify_trajectory([1, 2, 3, 4], [32] * 4)
    assert short["category"] == "not_formally_classified"
    assert short["formal_classification_eligible"] is False
    sensitivity = trajectory_sensitivity([30, 29, 3, 2, 1], [32] * 5)
    assert set(sensitivity) == {"magnitude_0.15", "magnitude_0.20", "magnitude_0.25"}


def test_h4_selection_enforces_one_parent_per_model_trace_and_fixed_quota() -> None:
    rows = []
    rates = [0.0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0]
    for model in ("a", "b"):
        for index in range(60):
            rows.append({"base_model": model, "trace_id": f"{model}-{index}", "checkpoint_id": f"c{index}", "domain": f"d{index%4}", "dense_recoverability": rates[index % len(rates)]})
    selected, audit = select_local_branch_parents(pd.DataFrame(rows), quotas={"near_boundary": 4, "high": 2, "low": 2})
    assert len(selected) == 16
    assert not selected.duplicated(["base_model", "trace_id"]).any()
    shuffled, _ = select_local_branch_parents(pd.DataFrame(rows).sample(frac=1, random_state=9), quotas={"near_boundary": 4, "high": 2, "low": 2})
    assert set(selected["selection_hash"]) == set(shuffled["selection_hash"])
    assert not audit.empty


def test_native_paths_and_primary_holm_family_mismatch_fail_loudly(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="native artifact"):
        _guard_non_native(tmp_path / "native_evaluation" / "rows.parquet")
    corrected = holm_primary_families({"H1": 0.01, "H2": 0.02, "H3": 0.03, "H4": 0.04})
    assert corrected["H1"] == pytest.approx(0.04)
    with pytest.raises(ValueError, match="exactly H1"):
        holm_primary_families({"H1": 0.1})


def test_margin_is_not_misrepresented_as_independent_information() -> None:
    margin = np.asarray([-2.0, -1.0, 0.5, 3.0])
    probability = 1 / (1 + np.exp(-(2 * margin + 0.3)))
    audit = margin_probability_equivalence(margin, probability)
    assert audit["rank_order_identical"] is True
    assert audit["incremental_information_identifiable"] is False
