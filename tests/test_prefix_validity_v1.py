from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from safeprefix.prefix_validity_v1.calibration import (
    apply_calibrator,
    assign_trace_folds,
    fit_fivefold_oof_calibration,
)
from safeprefix.prefix_validity_v1.evaluation import (
    align_matched_probe_predictions,
    boundary_metrics,
    checkpoint_metrics,
    paired_trace_bootstrap,
    paired_probe_test_bootstrap,
)
from safeprefix.prefix_validity_v1.monotonic import (
    apply_monotonic_projection,
    project_nonincreasing,
    select_gate_cutoff,
)
from safeprefix.prefix_validity_v1.training import (
    PrefixValidityCorpus,
    PrefixValidityTrainingConfig,
    fit_probe_matrix,
    select_learning_rate_and_median_seed,
    trace_weighted_binary_bce,
)


def _frame() -> pd.DataFrame:
    rows = []
    splits = ["train", "architecture_dev", "calibration", "teacher_forced_test"]
    for split_index, split in enumerate(splits):
        for trace_number in range(10):
            trace_id = f"{split}_{trace_number}"
            true_boundary = 1 + trace_number % 2
            for ordinal in (1, 2, 3):
                rows.append(
                    {
                        "base_model": "model",
                        "domain": "processbench_math" if trace_number % 2 else "processbench_omni",
                        "split": split,
                        "trace_id": trace_id,
                        "checkpoint_id": f"{trace_id}:{ordinal}",
                        "checkpoint_ordinal": ordinal,
                        "prefix_token_count": ordinal * 10,
                        "checkpoint_token_offset": ordinal * 10,
                        "total_checkpoint_count": 3,
                        "total_trace_token_count": 40,
                        "prefix_valid": int(ordinal <= true_boundary),
                        "true_last_valid_checkpoint": true_boundary,
                        "raw_logit": float(3 - ordinal + (trace_number % 3) * 0.1),
                    }
                )
    return pd.DataFrame(rows)


def test_pava_is_closest_nonincreasing_projection() -> None:
    projected = project_nonincreasing([0.9, 0.2, 0.8, 0.1])
    assert projected == pytest.approx([0.9, 0.5, 0.5, 0.1])
    assert np.all(np.diff(projected) <= 0)
    assert project_nonincreasing([0.3, 0.3, 0.2]).tolist() == pytest.approx([0.3, 0.3, 0.2])


def test_monotonic_projection_preserves_row_order_and_separates_traces() -> None:
    frame = pd.DataFrame(
        {
            "base_model": ["m"] * 4,
            "trace_id": ["a", "a", "b", "b"],
            "checkpoint_ordinal": [2, 1, 1, 2],
            "calibrated_probability": [0.8, 0.2, 0.7, 0.4],
        }
    )
    result = apply_monotonic_projection(frame)
    assert result.index.tolist() == frame.index.tolist()
    assert result.loc[[1, 0], "monotonic_probability"].tolist() == pytest.approx([0.5, 0.5])
    assert result.loc[[2, 3], "monotonic_probability"].tolist() == pytest.approx([0.7, 0.4])


def test_fivefold_oof_calibration_is_trace_disjoint_and_complete() -> None:
    calibration = _frame().query("split == 'calibration'").copy()
    folds = assign_trace_folds(calibration)
    assert folds.nunique() == 5
    assert pd.DataFrame({"trace": calibration.trace_id, "fold": folds}).groupby("trace").fold.nunique().max() == 1
    result = fit_fivefold_oof_calibration(calibration)
    assert len(result.fold_calibrators) == 5
    assert result.predictions["calibrated_probability"].between(0, 1).all()
    assert result.predictions["calibrated_probability"].notna().all()
    test = _frame().query("split == 'teacher_forced_test'").copy()
    scored = apply_calibrator(test, result.final_calibrator)
    assert scored["calibrated_probability"].between(0, 1).all()


def test_gate_cutoff_obeys_late_constraint_and_uses_monotonic_oof_rows() -> None:
    calibration = _frame().query("split == 'calibration'").copy()
    calibration["calibrated_probability"] = 1 / (1 + np.exp(-calibration["raw_logit"]))
    calibration = apply_monotonic_projection(calibration)
    cutoff, audit = select_gate_cutoff(calibration, max_late_rate=0.05)
    assert cutoff.late_boundary_rate <= 0.05
    assert audit["selected"].sum() == 1
    assert cutoff.calibration_traces == calibration.trace_id.nunique()


def test_gate_reports_when_only_all_root_is_validated() -> None:
    frame = pd.DataFrame(
        {
            "trace_id": ["a", "a", "b", "b"],
            "checkpoint_ordinal": [1, 2, 1, 2],
            "true_last_valid_checkpoint": [0, 0, 0, 0],
            "monotonic_probability": [0.8, 0.7, 0.6, 0.5],
            "split": ["calibration"] * 4,
        }
    )
    cutoff, _ = select_gate_cutoff(frame, max_late_rate=0.05)
    assert cutoff.non_root_coverage == 0
    assert cutoff.validated_gate_exists is False


def test_trace_weighted_bce_gives_each_trace_equal_mass() -> None:
    logits = torch.zeros((2, 2))
    targets = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    mask = torch.tensor([[True, False], [True, True]])
    assert float(trace_weighted_binary_bce(logits, targets, mask)) == pytest.approx(np.log(2))
    with pytest.raises(ValueError, match="binary"):
        trace_weighted_binary_bce(logits, torch.tensor([[0.5, 0], [1, 0]]), mask)


def test_training_uses_two_lrs_three_seeds_and_median_seed() -> None:
    frame = _frame().drop(columns=["raw_logit"])
    rng = np.random.default_rng(4)
    features = rng.normal(size=(len(frame), 6)).astype(np.float32)
    corpus = PrefixValidityCorpus(frame, features)
    config = PrefixValidityTrainingConfig(max_epochs=2, patience=1, batch_size_traces=16)
    candidates = fit_probe_matrix(corpus, architecture="linear_probe", config=config)
    assert len(candidates) == 6
    selected, record = select_learning_rate_and_median_seed(candidates)
    assert selected.learning_rate in {1e-3, 3e-4}
    assert record["test_used_for_selection"] is False
    assert all(candidate.metadata()["class_balanced_loss"] is False for candidate in candidates)


def test_corpus_accepts_exact_complete_step_manifest_column_names() -> None:
    frame = _frame().drop(columns=["raw_logit", "prefix_token_count"]).rename(
        columns={
            "base_model": "model_key",
            "checkpoint_ordinal": "checkpoint_index",
            "total_trace_token_count": "full_trace_token_count",
        }
    ).drop(columns=["total_checkpoint_count"])
    corpus = PrefixValidityCorpus(frame, np.ones((len(frame), 3), dtype=np.float32))
    assert corpus.frame["base_model"].eq("model").all()
    assert corpus.frame["total_checkpoint_count"].eq(3).all()


def test_checkpoint_boundary_metrics_and_paired_trace_bootstrap() -> None:
    test = _frame().query("split == 'teacher_forced_test'").copy()
    test["monotonic_probability"] = np.where(test["prefix_valid"].eq(1), 0.9, 0.1)
    check = checkpoint_metrics(test)
    assert check["balanced_accuracy"] == pytest.approx(1.0)
    boundary = boundary_metrics(test, gamma=0.5)
    assert boundary["late_boundary_rate"] == 0.0
    assert boundary["exact_last_valid_checkpoint_accuracy"] == 1.0

    test["method_a"] = test["prefix_valid"]
    test["method_b"] = 0.0

    def statistic(frame: pd.DataFrame) -> dict[str, float]:
        return {"difference": float((frame["method_a"] - frame["method_b"]).mean())}

    result = paired_trace_bootstrap(test, statistic, replicates=100, seed=7)
    assert result["unit"] == "complete failed trace"
    assert result["macro"]["difference"]["finite_replicates"] == 100


def test_matched_probe_bootstrap_is_test_only_and_paired() -> None:
    hidden = _frame().query("split == 'teacher_forced_test'").copy()
    hidden["monotonic_probability"] = np.where(hidden["prefix_valid"].eq(1), 0.9, 0.1)
    position = hidden.copy()
    position["raw_logit"] = 0.0
    position["monotonic_probability"] = 0.5
    matched = align_matched_probe_predictions(hidden, position)
    assert len(matched) == len(hidden)
    result = paired_probe_test_bootstrap(
        hidden,
        position,
        hidden_gamma=0.5,
        position_gamma=0.5,
        replicates=50,
        seed=4,
        expected_models=1,
    )
    assert result["test_only"] is True
    assert result["row_identity_matched"] is True
    assert (
        result["macro"][
            "hidden_minus_position_boundary_exact_last_valid_checkpoint_accuracy"
        ]["estimate"]
        > 0
    )


def test_matched_probe_alignment_rejects_row_drift() -> None:
    hidden = _frame().query("split == 'teacher_forced_test'").copy()
    hidden["monotonic_probability"] = 0.5
    position = hidden.iloc[:-1].copy()
    with pytest.raises(RuntimeError, match="different checkpoint counts"):
        align_matched_probe_predictions(hidden, position)
