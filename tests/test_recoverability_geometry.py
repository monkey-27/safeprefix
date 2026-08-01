from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from safeprefix.recoverability_geometry import (
    AffineAxis,
    ShapeThresholds,
    affine_axis_audit,
    axis_energy_fraction_eta,
    branch_metrics,
    classify_binomial_trajectory,
    clustered_bootstrap,
    cross_domain_transfer_summary,
    generate_geometry_figures,
    h1_axis_orthogonal_diagnostics,
    holm_correction,
    margin_metrics,
    parent_clustered_bootstrap,
    render_geometry_report,
    select_canonical_median_seed,
    select_h4_parents,
    seed_axis_stability,
    shape_sensitivity_analysis,
    trace_clustered_bootstrap,
    within_trace_metrics,
)
from safeprefix.recoverability_geometry.analysis import (
    compute_branch_metrics,
    compute_h1_metrics,
    compute_h2_metrics,
    summarize_trajectory_prevalence,
    trace_weighted_nll,
    margin_probability_equivalence,
)
from safeprefix.recoverability_geometry.reporting import (
    required_output_paths,
    validate_required_outputs,
)
from safeprefix.recoverability_geometry.runner import _score_all_splits


def test_canonical_seed_is_nearest_median_and_input_order_invariant() -> None:
    records = [
        {"seed": 29, "dev_score": 0.7, "test_score": 999},
        {"seed": 47, "dev_score": 0.4, "test_score": -999},
        {"seed": 11, "dev_score": 0.5, "test_score": 0},
    ]
    forward = select_canonical_median_seed(records)
    reverse = select_canonical_median_seed(list(reversed(records)))
    assert forward == reverse
    assert forward["seed"] == 11
    assert forward["median_dev_score"] == pytest.approx(0.5)
    assert forward["record"]["test_score"] == 0


def test_canonical_seed_even_tie_breaks_by_seed_and_rejects_duplicates() -> None:
    selected = select_canonical_median_seed(
        [{"seed": 9, "dev_score": 1.0}, {"seed": 3, "dev_score": 3.0}]
    )
    assert selected["median_dev_score"] == 2.0
    assert selected["seed"] == 3
    with pytest.raises(ValueError, match="duplicate seed"):
        select_canonical_median_seed(
            [{"seed": 1, "dev_score": 1.0}, {"seed": 1, "dev_score": 2.0}]
        )


def test_affine_axis_projection_reconstructs_and_hits_boundary() -> None:
    hidden = np.asarray([[1.0, 2.0, -1.0], [3.0, -2.0, 4.0]])
    axis = AffineAxis(np.asarray([2.0, -1.0, 0.5]), bias=-0.25, seed=11)
    expected = hidden @ axis.weight + axis.bias
    audit = affine_axis_audit(hidden, axis, expected_logits=expected)
    decomposition = axis.decompose(hidden)
    assert audit["passed"] is True
    assert audit["maximum_reconstruction_error"] < 1e-12
    assert np.allclose(axis.logits(decomposition["boundary_projection"]), 0.0)
    assert np.allclose(
        decomposition["parallel_component"] + decomposition["orthogonal_component"],
        hidden,
    )


def test_affine_axis_audit_detects_wrong_logits_and_dimensions() -> None:
    axis = AffineAxis(np.asarray([1.0, 0.0]), bias=0.0)
    assert not affine_axis_audit([[1.0, 2.0]], axis, expected_logits=[9.0])["passed"]
    with pytest.raises(ValueError, match="hidden dimension"):
        axis.logits([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="nonzero"):
        AffineAxis(np.zeros(2))


def test_learned_layernorm_affine_is_folded_into_common_geometry_exactly() -> None:
    raw = np.asarray([[1.0, 3.0, -2.0], [4.0, -1.0, 2.0]])
    head_weight = np.asarray([0.5, -2.0, 1.5])
    gamma = np.asarray([2.0, 0.25, -1.0])
    beta = np.asarray([0.1, -0.3, 0.7])
    head_bias = -0.2
    axis = AffineAxis(
        head_weight,
        bias=head_bias,
        layernorm_weight=gamma,
        layernorm_bias=beta,
        layernorm_epsilon=1e-5,
    )
    normalized = (raw - raw.mean(axis=1, keepdims=True)) / np.sqrt(
        ((raw - raw.mean(axis=1, keepdims=True)) ** 2).mean(axis=1, keepdims=True)
        + 1e-5
    )
    frozen_probe = (normalized * gamma + beta) @ head_weight + head_bias
    assert np.allclose(axis.weight, head_weight * gamma)
    assert axis.bias == pytest.approx(head_bias + head_weight @ beta)
    assert np.allclose(axis.transform_raw(raw), normalized)
    assert np.allclose(axis.logit_from_raw(raw), frozen_probe)
    assert axis.audit()["learned_layernorm_affine_folded_into_head"] is True


def test_feature_store_indices_are_applied_in_manifest_row_order() -> None:
    manifest = pd.DataFrame(
        {
            "feature_row_index": [2, 0, 1],
            "trace_id": ["t2", "t0", "t1"],
            "checkpoint_id": ["c2", "c0", "c1"],
            "base_model": ["m"] * 3,
            "split": ["train", "architecture_dev", "teacher_forced_test"],
        }
    )
    features = np.asarray([[0.0, 1.0], [2.0, 0.0], [1.0, 3.0]])
    axes = {seed: AffineAxis(np.asarray([1.0, -0.25])) for seed in (0, 1, 2)}
    scored, scores, common, _ = _score_all_splits(manifest, features, axes, 0)
    expected_common = axes[0].transform_raw(features[[2, 0, 1]])
    assert scored["trace_id"].tolist() == ["t2", "t0", "t1"]
    assert np.allclose(common, expected_common)
    assert np.allclose(scores[0], axes[0].logit_from_feature(expected_common))


def test_seed_stability_reports_orientation_and_projection_correlations() -> None:
    hidden = np.asarray([[0.0, 1.0], [1.0, 0.0], [2.0, -1.0]])
    report = seed_axis_stability(
        {
            11: AffineAxis(np.asarray([1.0, 0.0]), seed=11),
            29: AffineAxis(np.asarray([0.99, 0.01]), seed=29),
            47: AffineAxis(np.asarray([-1.0, 0.0]), seed=47),
        },
        hidden=hidden,
    )
    assert report["pair_count"] == 3
    assert report["minimum_oriented_cosine"] == pytest.approx(-1.0)
    opposite = next(
        row for row in report["pairs"] if {row["left_seed"], row["right_seed"]} == {11, 47}
    )
    assert opposite["logit_pearson"] == pytest.approx(-1.0)


def test_eta_is_axis_energy_fraction_with_zero_displacement_convention() -> None:
    axis = AffineAxis(np.asarray([1.0, 0.0]))
    displacements = np.asarray([[3.0, 4.0], [0.0, 2.0]])
    assert axis_energy_fraction_eta(displacements, axis) == pytest.approx(9.0 / 29.0)
    assert axis_energy_fraction_eta(np.zeros((2, 2)), axis) == 0.0


def test_h1_interface_compares_axis_orthogonal_and_full_without_training() -> None:
    hidden = np.asarray([[-2.0, 1.0], [-1.0, -1.0], [1.0, 2.0], [2.0, -2.0]])
    axis = AffineAxis(np.asarray([1.0, 0.0]))
    report = h1_axis_orthogonal_diagnostics(
        hidden,
        axis,
        successes=[0, 1, 3, 4],
        trials=[4, 4, 4, 4],
        orthogonal_score=[0.0, 0.0, 0.0, 0.0],
        full_score=[-3.0, -1.0, 1.0, 3.0],
        displacements=np.asarray([[1.0, 0.0], [1.0, 1.0]]),
    )
    assert report["observations"] == 4
    assert report["axis_observed_spearman"] > 0.9
    assert report["full"]["nll_gain_over_axis"] > 0
    assert report["eta"] == pytest.approx(2.0 / 3.0)


def _within_trace_frame() -> pd.DataFrame:
    rows = []
    for trace, offset in (("a", 0.0), ("b", 100.0), ("c", -50.0)):
        for checkpoint in range(4):
            rows.append(
                {
                    "trace_id": trace,
                    "checkpoint_ordinal": checkpoint,
                    "axis_score": offset + checkpoint,
                    "position_control": checkpoint,
                    "noise_control": (checkpoint * 7 + len(trace)) % 3,
                    "success_count": checkpoint,
                    "trial_count": 4,
                }
            )
    return pd.DataFrame(rows)


def test_h2_within_trace_metrics_remove_static_trace_difficulty() -> None:
    report = within_trace_metrics(
        _within_trace_frame(),
        control_score_cols=["noise_control"],
        permutation_seed=91,
    )
    assert report["traces"] == 3
    assert report["within_trace_pearson"] == pytest.approx(1.0)
    assert report["within_trace_spearman"] == pytest.approx(1.0)
    assert report["trace_fixed_effect_slope"] == pytest.approx(0.25)
    assert report["adjacent_direction_agreement"] == pytest.approx(1.0)
    assert report["median_trace_spearman"] == pytest.approx(1.0)
    assert report["permutation_seed"] == 91


def test_h2_rejects_duplicate_positions_and_invalid_counts() -> None:
    frame = _within_trace_frame()
    with pytest.raises(ValueError, match="unique"):
        within_trace_metrics(pd.concat([frame, frame.iloc[[0]]], ignore_index=True))
    frame.loc[0, "success_count"] = 5
    with pytest.raises(ValueError, match="binomial"):
        within_trace_metrics(frame)


def _dense_control_frame() -> pd.DataFrame:
    rows = []
    for trace_index, trace in enumerate(("a", "b", "c", "d")):
        for checkpoint in range(4):
            score = checkpoint - 1.5 + trace_index * 0.1
            successes = int(np.clip(round(16 + 7 * score), 0, 32))
            rows.append(
                {
                    "trace_id": trace,
                    "domain": "x" if trace_index < 2 else "y",
                    "checkpoint_ordinal": checkpoint,
                    "normalized_checkpoint_ordinal": checkpoint / 3,
                    "normalized_checkpoint_token_position": (checkpoint + 1) / 5,
                    "prompt_solvability_smoothed": 0.2 + 0.2 * trace_index,
                    "total_trace_token_count": 100 + trace_index,
                    "total_checkpoint_count": 4,
                    "canonical_raw_logit": score,
                    "success_count": successes,
                    "num_rollouts": 32,
                    "axis": score,
                    "residual": score + 0.01,
                    "control": checkpoint * 0.05,
                }
            )
    return pd.DataFrame(rows)


def test_h1_runs_trace_paired_bootstrap_and_preserves_unclipped_eta() -> None:
    frame = _dense_control_frame()
    report = compute_h1_metrics(
        frame,
        model_logits={
            "axis_only": "axis",
            "axis_plus_residual": "residual",
            "full_state": "axis",
            "position_prompt_control": "control",
        },
        bootstrap_replicates=100,
    )
    paired = report["axis_plus_residual_improvement"]
    assert paired["replicates"] == 100
    assert "ci_includes_zero" in paired
    assert np.isfinite(report["eta"])


def test_h2_reports_fixed_position_prompt_and_domain_diagnostics_with_bootstrap() -> None:
    report = compute_h2_metrics(_dense_control_frame(), bootstrap_replicates=100)
    assert report["strict_within_trace"]["trace_slope_bootstrap"]["replicates"] == 100
    assert report["matched_checkpoint_position_bins"]
    assert report["narrow_prompt_solvability_bins"]
    assert set(report["by_domain"]) == {"x", "y"}


def test_nll_is_finite_for_extreme_logits_and_endpoint_outcomes() -> None:
    value = trace_weighted_nll(
        ["a", "a", "b", "b"],
        [0, 32, 0, 32],
        [32, 32, 32, 32],
        [-1e6, 1e6, -1e3, 1e3],
    )
    assert np.isfinite(value)
    assert value == pytest.approx(0.0, abs=1e-12)


def test_h2_complete_separation_keeps_strict_nll_finite() -> None:
    frame = _dense_control_frame()
    frame["success_count"] = np.where(
        frame["checkpoint_ordinal"] < 2, 0, frame["num_rollouts"]
    )
    report = compute_h2_metrics(frame, bootstrap_replicates=20)
    assert np.isfinite(report["strict_within_trace"]["trace_weighted_nll"])
    assert np.isfinite(report["pooled_controlled"]["group_crossfit_full_nll"])


def test_margin_probability_equivalence_clips_only_exact_endpoints() -> None:
    report = margin_probability_equivalence(
        [-30.0, 30.0], [0.0, 1.0]
    )
    assert report["endpoint_probabilities_clipped_for_logit"] == 2
    assert np.isfinite(report["margin_probability_logit_correlation"])


def test_h3_prevalence_counts_short_traces_but_bootstraps_only_formal_rows() -> None:
    frame = pd.DataFrame(
        [
            {"base_model": "m1", "trace_id": "a1", "common_trace_id": "a", "category": "single_collapse", "formal_classification_eligible": True},
            {"base_model": "m2", "trace_id": "a2", "common_trace_id": "a", "category": "gradual_decline", "formal_classification_eligible": True},
            {"base_model": "m1", "trace_id": "b1", "common_trace_id": "b", "category": "not_formally_classified", "formal_classification_eligible": False},
        ]
    )
    report = summarize_trajectory_prevalence(frame, bootstrap_replicates=50)
    assert report["all_trace_model_rows"] == 3
    assert report["formal_trace_model_rows"] == 2
    assert report["shared_trace_clusters"] == 2
    assert report["formal_prevalence_bootstrap"]["replicates"] == 50


def _counts(probability: np.ndarray, trials: int = 400) -> tuple[np.ndarray, np.ndarray]:
    total = np.full(len(probability), trials, dtype=int)
    return np.rint(probability * total).astype(int), total


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (np.asarray([0.1, 0.2, 0.4, 0.7, 0.9]), "monotone_increasing"),
        (np.asarray([0.9, 0.7, 0.4, 0.2, 0.1]), "monotone_decreasing"),
        (np.asarray([0.6, 0.35, 0.2, 0.35, 0.6]), "u_shaped"),
        (np.asarray([0.2, 0.5, 0.75, 0.5, 0.2]), "inverted_u"),
        (np.asarray([0.5, 0.5, 0.5, 0.5, 0.5]), "flat"),
    ],
)
def test_h3_binomial_bic_classifies_known_trajectory_shapes(
    probability: np.ndarray, expected: str
) -> None:
    positions = np.linspace(0.0, 1.0, len(probability))
    success, trials = _counts(probability)
    result = classify_binomial_trajectory(positions, success, trials)
    assert result["shape"] == expected
    assert result["thresholds"] == ShapeThresholds().to_dict()
    assert result["bernoulli_trials"] == int(trials.sum())


def test_h3_thresholds_can_reclassify_small_effect_as_flat() -> None:
    positions = np.linspace(0.0, 1.0, 5)
    success, trials = _counts(np.asarray([0.48, 0.49, 0.50, 0.51, 0.52]), trials=10_000)
    strict = classify_binomial_trajectory(
        positions,
        success,
        trials,
        thresholds=ShapeThresholds(minimum_probability_range=0.10),
    )
    permissive = classify_binomial_trajectory(
        positions,
        success,
        trials,
        thresholds=ShapeThresholds(minimum_probability_range=0.01),
    )
    assert strict["shape"] == "flat"
    assert permissive["shape"] == "monotone_increasing"


def test_h3_sensitivity_enumerates_exact_grid_deterministically() -> None:
    positions = np.linspace(0.0, 1.0, 5)
    increasing = _counts(np.asarray([0.1, 0.2, 0.4, 0.7, 0.9]))
    flat = _counts(np.asarray([0.5] * 5))
    trajectories = {
        "z": {"positions": positions, "successes": flat[0], "trials": flat[1]},
        "a": {
            "positions": positions,
            "successes": increasing[0],
            "trials": increasing[1],
        },
    }
    first = shape_sensitivity_analysis(
        trajectories,
        bic_thresholds=[2.0, 6.0],
        probability_ranges=[0.05, 0.10],
        turning_point_margins=[0.10],
    )
    second = shape_sensitivity_analysis(
        dict(reversed(list(trajectories.items()))),
        bic_thresholds=[2.0, 6.0],
        probability_ranges=[0.05, 0.10],
        turning_point_margins=[0.10],
    )
    assert first == second
    assert first["setting_count"] == 4
    assert len(first["per_trace"]) == 8


def _parent_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"trace_id": "a", "checkpoint_ordinal": 0, "success_count": 0, "trial_count": 4},
            {"trace_id": "a", "checkpoint_ordinal": 1, "success_count": 2, "trial_count": 4},
            {"trace_id": "a", "checkpoint_ordinal": 2, "success_count": 3, "trial_count": 6},
            {"trace_id": "b", "checkpoint_ordinal": 0, "success_count": 4, "trial_count": 4},
            {"trace_id": "b", "checkpoint_ordinal": 1, "success_count": 0, "trial_count": 4},
            {"trace_id": "c", "checkpoint_ordinal": 0, "success_count": 1, "trial_count": 4},
            {"trace_id": "c", "checkpoint_ordinal": 1, "success_count": 3, "trial_count": 4},
        ]
    )


def test_h4_parent_selection_is_deterministic_and_requires_mixed_outcomes() -> None:
    frame = _parent_candidates()
    forward = select_h4_parents(frame)
    reverse = select_h4_parents(frame.iloc[::-1])
    pd.testing.assert_frame_equal(forward, reverse)
    selected = dict(zip(forward["trace_id"], forward["checkpoint_ordinal"]))
    assert selected == {"a": 2, "c": 1}
    assert "b" not in selected
    assert forward["parent_id"].tolist() == ["a:2", "c:1"]


def _branches() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"parent_id": "a:2", "binary_outcome": 1, "parent_axis_score": 0.0, "child_axis_score": 2.0, "orthogonal_distance": 1.0},
            {"parent_id": "a:2", "binary_outcome": 0, "parent_axis_score": 0.0, "child_axis_score": -1.0, "orthogonal_distance": 2.0},
            {"parent_id": "a:2", "binary_outcome": 0, "parent_axis_score": 0.0, "child_axis_score": 0.0, "orthogonal_distance": 3.0},
            {"parent_id": "c:1", "binary_outcome": 1, "parent_axis_score": 1.0, "child_axis_score": 4.0, "orthogonal_distance": 0.5},
            {"parent_id": "c:1", "binary_outcome": 0, "parent_axis_score": 1.0, "child_axis_score": 0.0, "orthogonal_distance": 1.5},
            {"parent_id": "no-mix", "binary_outcome": 1, "parent_axis_score": 0.0, "child_axis_score": 1.0, "orthogonal_distance": 0.0},
        ]
    )


def test_h4_branch_metrics_are_within_parent_and_ignore_nonmixed_parent() -> None:
    report = branch_metrics(_branches())
    assert report["parents_total"] == 3
    assert report["parents_with_both_outcomes"] == 2
    assert report["mean_parent_axis_delta_gap"] > 3.0
    assert report["mean_parent_pairwise_axis_order_accuracy"] == pytest.approx(1.0)
    assert report["pooled_pairwise_axis_order_accuracy"] == pytest.approx(1.0)


def test_h4_orthogonal_separation_uses_centroid_vectors_not_radius_difference() -> None:
    frame = pd.DataFrame(
        [
            {"parent_id": "p", "horizon": 32, "horizon_available": True, "child_score": 2.0, "delta_r": 2.0, "orthogonal_displacement_norm": 1.0, "orthogonal_displacement": [1.0, 0.0], "child_success_count": 4, "child_num_rollouts": 4},
            {"parent_id": "p", "horizon": 32, "horizon_available": True, "child_score": -2.0, "delta_r": -2.0, "orthogonal_displacement_norm": 1.0, "orthogonal_displacement": [-1.0, 0.0], "child_success_count": 0, "child_num_rollouts": 4},
        ]
    )
    report = compute_branch_metrics(frame)["by_horizon"]["32"]
    assert report["axis_outcome_separation"] == pytest.approx(4.0)
    assert report["orthogonal_outcome_centroid_separation"] == pytest.approx(2.0)
    assert report["axis_to_orthogonal_separation_ratio"] == pytest.approx(2.0)


def test_h4_pooled_orthogonal_separation_is_model_stratified_across_dimensions() -> None:
    frame = pd.DataFrame(
        [
            {"base_model": "m2", "parent_id": "m2:p", "horizon": 32, "horizon_available": True, "child_score": 2.0, "delta_r": 2.0, "orthogonal_displacement_norm": 1.0, "orthogonal_displacement": [1.0, 0.0], "child_success_count": 4, "child_num_rollouts": 4},
            {"base_model": "m2", "parent_id": "m2:p", "horizon": 32, "horizon_available": True, "child_score": -2.0, "delta_r": -2.0, "orthogonal_displacement_norm": 1.0, "orthogonal_displacement": [-1.0, 0.0], "child_success_count": 0, "child_num_rollouts": 4},
            {"base_model": "m3", "parent_id": "m3:p", "horizon": 32, "horizon_available": True, "child_score": 4.0, "delta_r": 4.0, "orthogonal_displacement_norm": 2.0, "orthogonal_displacement": [2.0, 0.0, 0.0], "child_success_count": 4, "child_num_rollouts": 4},
            {"base_model": "m3", "parent_id": "m3:p", "horizon": 32, "horizon_available": True, "child_score": -4.0, "delta_r": -4.0, "orthogonal_displacement_norm": 2.0, "orthogonal_displacement": [-2.0, 0.0, 0.0], "child_success_count": 0, "child_num_rollouts": 4},
        ]
    )

    report = compute_branch_metrics(frame)["by_horizon"]["32"]

    assert report["orthogonal_outcome_centroid_separation_aggregation"] == "equal_model_macro"
    assert report["orthogonal_outcome_centroid_separation_by_model"] == pytest.approx(
        {"m2": 2.0, "m3": 4.0}
    )
    assert report["orthogonal_outcome_centroid_separation"] == pytest.approx(3.0)
    assert report["orthogonal_outcome_separation"] == pytest.approx(3.0)
    assert report["axis_outcome_separation_by_model"] == pytest.approx(
        {"m2": 4.0, "m3": 8.0}
    )
    assert report["axis_to_orthogonal_separation_ratio_by_model"] == pytest.approx(
        {"m2": 2.0, "m3": 2.0}
    )
    assert report["axis_to_orthogonal_separation_ratio"] == pytest.approx(2.0)
    assert report["models_contributing_to_orthogonal_separation"] == 2

    # Per-model calls preserve the pre-existing metric definition exactly.
    m2 = compute_branch_metrics(frame.loc[frame["base_model"] == "m2"])["by_horizon"]["32"]
    m3 = compute_branch_metrics(frame.loc[frame["base_model"] == "m3"])["by_horizon"]["32"]
    assert m2["orthogonal_outcome_centroid_separation"] == pytest.approx(2.0)
    assert m3["orthogonal_outcome_centroid_separation"] == pytest.approx(4.0)


def test_h4_rejects_inconsistent_orthogonal_dimensions_within_model() -> None:
    frame = pd.DataFrame(
        [
            {"base_model": "m", "parent_id": "p", "horizon": 32, "horizon_available": True, "child_score": 2.0, "delta_r": 2.0, "orthogonal_displacement_norm": 1.0, "orthogonal_displacement": [1.0, 0.0], "child_success_count": 4, "child_num_rollouts": 4},
            {"base_model": "m", "parent_id": "p", "horizon": 32, "horizon_available": True, "child_score": 1.0, "delta_r": 1.0, "orthogonal_displacement_norm": 1.0, "orthogonal_displacement": [1.0, 0.0, 0.0], "child_success_count": 4, "child_num_rollouts": 4},
            {"base_model": "m", "parent_id": "p", "horizon": 32, "horizon_available": True, "child_score": -2.0, "delta_r": -2.0, "orthogonal_displacement_norm": 1.0, "orthogonal_displacement": [-1.0, 0.0], "child_success_count": 0, "child_num_rollouts": 4},
        ]
    )

    with pytest.raises(ValueError, match="inconsistent high-outcome orthogonal vector dimensions"):
        compute_branch_metrics(frame)


def test_cross_domain_transfer_reports_diagonal_cross_and_worst() -> None:
    frame = pd.DataFrame(
        [
            {"train_domain": "a", "eval_domain": "a", "metric": 0.9},
            {"train_domain": "a", "eval_domain": "b", "metric": 0.6},
            {"train_domain": "b", "eval_domain": "a", "metric": 0.5},
            {"train_domain": "b", "eval_domain": "b", "metric": 0.8},
        ]
    )
    report = cross_domain_transfer_summary(frame)
    assert report["in_domain_mean"] == pytest.approx(0.85)
    assert report["cross_domain_mean"] == pytest.approx(0.55)
    assert report["transfer_gap"] == pytest.approx(0.30)
    assert report["worst_cell"]["mean"] == pytest.approx(0.5)


def test_margin_metrics_support_logits_and_probabilities() -> None:
    logits = margin_metrics([-2.0, -1.0, 1.0, 2.0], [0, 0, 1, 1])
    probabilities = margin_metrics(
        [0.1, 0.25, 0.75, 0.9], [0, 0, 1, 1], scores_are_probabilities=True
    )
    assert logits["correct_sign_fraction"] == 1.0
    assert logits["pairwise_order_accuracy"] == 1.0
    assert probabilities["mean_signed_margin"] > 1.0
    with pytest.raises(ValueError, match="probabilities"):
        margin_metrics([1.1], [1], scores_are_probabilities=True)


def _cluster_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"trace_id": "a", "parent_id": "p1", "domain": "x", "value": 1.0},
            {"trace_id": "a", "parent_id": "p1", "domain": "x", "value": 3.0},
            {"trace_id": "b", "parent_id": "p2", "domain": "x", "value": -1.0},
            {"trace_id": "c", "parent_id": "p3", "domain": "y", "value": 5.0},
        ]
    )


def test_clustered_bootstrap_is_deterministic_and_resamples_whole_clusters() -> None:
    frame = _cluster_frame()

    def statistic(sample: pd.DataFrame) -> dict[str, float]:
        per_cluster = sample.groupby("trace_id")["value"].mean()
        return {"macro_mean": float(per_cluster.mean())}

    first = trace_clustered_bootstrap(
        frame, statistic, replicates=200, seed=77, stratify_col="domain"
    )
    second = clustered_bootstrap(
        frame,
        cluster_col="trace_id",
        statistic=statistic,
        replicates=200,
        seed=77,
        stratify_col="domain",
    )
    assert first.to_dict(include_replicates=True) == second.to_dict(
        include_replicates=True
    )
    assert first.cluster_count == 3
    assert first.estimates["macro_mean"] == pytest.approx((2.0 - 1.0 + 5.0) / 3.0)
    assert len(first.replicate_values["macro_mean"]) == 200


def test_parent_bootstrap_uses_parent_as_independent_unit() -> None:
    frame = _cluster_frame()
    result = parent_clustered_bootstrap(
        frame,
        lambda sample: float(sample.groupby("parent_id")["value"].mean().mean()),
        replicates=50,
        seed=5,
    )
    assert result.cluster_column == "parent_id"
    assert result.cluster_count == 3


def test_clustered_bootstrap_rejects_cluster_crossing_strata() -> None:
    frame = _cluster_frame()
    frame.loc[1, "domain"] = "y"
    with pytest.raises(ValueError, match="exactly one stratum"):
        trace_clustered_bootstrap(
            frame, lambda sample: float(sample["value"].mean()), stratify_col="domain"
        )


def test_holm_correction_is_monotone_and_preserves_input_identity() -> None:
    corrected = holm_correction({"c": 0.04, "a": 0.01, "b": 0.03})
    assert corrected["a"]["holm_p"] == pytest.approx(0.03)
    assert corrected["b"]["holm_p"] == pytest.approx(0.06)
    assert corrected["c"]["holm_p"] == pytest.approx(0.06)
    assert corrected["a"]["reject"] is True
    assert corrected["b"]["reject"] is False
    sequence = holm_correction([0.01, 0.03, 0.04])
    assert isinstance(sequence, list) and len(sequence) == 3


def test_reporting_hooks_create_only_requested_outputs(tmp_path: Path) -> None:
    checkpoint = _within_trace_frame()
    branch = _branches()
    transfer = pd.DataFrame(
        [
            {"train_domain": "a", "eval_domain": "a", "metric": 0.9},
            {"train_domain": "a", "eval_domain": "b", "metric": 0.6},
            {"train_domain": "b", "eval_domain": "a", "metric": 0.5},
            {"train_domain": "b", "eval_domain": "b", "metric": 0.8},
        ]
    )
    figures = generate_geometry_figures(
        tmp_path / "figures",
        checkpoint_frame=checkpoint,
        shape_records=[{"shape": "flat"}, {"shape": "u_shaped"}],
        branch_frame=branch,
        transfer_frame=transfer,
    )
    assert set(figures) == {
        "h1_h2_axis_geometry",
        "h3_shape_counts",
        "h4_branch_separation",
        "cross_domain_transfer",
    }
    assert all(Path(path).is_file() for path in figures.values())
    report = render_geometry_report(
        tmp_path / "report.md",
        {"H1": {"eta": 0.4}, "H2": {"correlation": 0.8}, "status": "diagnostic"},
        figure_paths=figures,
    )
    text = report.read_text(encoding="utf-8")
    assert "does not run inference" in text
    assert "## H1" in text and "## H2" in text
    assert "h1_h2_axis_geometry" in text


def test_report_json_conversion_handles_numpy_without_reading_inputs(tmp_path: Path) -> None:
    path = render_geometry_report(
        tmp_path / "nested" / "report.md",
        {"H3": {"values": np.asarray([1, 2]), "score": np.float64(0.5)}},
    )
    text = path.read_text(encoding="utf-8")
    assert '"values": [' in text
    assert "    1," in text and "    2" in text


def test_required_output_contract_points_into_report_subdirectory(tmp_path: Path) -> None:
    paths = required_output_paths(tmp_path)
    assert all(path.parent == tmp_path / "reports" for path in paths["reports"])
    assert validate_required_outputs(tmp_path)["passed"] is False
