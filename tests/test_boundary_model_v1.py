from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from safeprefix.boundary_v1.evaluation import (
    apply_calibration,
    fit_positive_affine_calibrator,
    threshold_sweep,
)
from safeprefix.boundary_v1.inference import BoundaryScorer
from safeprefix.boundary_v1.models import build_predictor
from safeprefix.boundary_v1.training import normalized_binomial_nll


def test_normalized_binomial_nll_weights_traces_equally() -> None:
    logits = torch.zeros((2, 3))
    successes = torch.tensor([[4.0, 0.0, 0.0], [4.0, 4.0, 4.0]])
    trials = torch.full((2, 3), 4.0)
    mask = torch.tensor([[True, False, False], [True, True, True]])
    loss = normalized_binomial_nll(logits, successes, trials, mask)
    assert float(loss) == pytest.approx(np.log(2.0))


@pytest.mark.parametrize(
    "architecture",
    ["position_only", "linear_probe", "local_mlp", "change_aware_mlp", "causal_gru"],
)
def test_predictor_shapes(architecture: str) -> None:
    model = build_predictor(architecture, 12).eval()
    hidden = torch.randn(3, 5, 12)
    position = torch.randn(3, 5, 4)
    mask = torch.tensor(
        [[True] * 5, [True, True, True, False, False], [True, False, False, False, False]]
    )
    output = model(hidden, position, mask)
    assert output.shape == (3, 5)
    assert torch.equal(output[~mask], torch.zeros_like(output[~mask]))


def test_causal_gru_prefix_matches_full_and_padded_batch() -> None:
    torch.manual_seed(4)
    model = build_predictor("causal_gru", 16).eval()
    hidden = torch.randn(1, 6, 16)
    position = torch.zeros(1, 6, 4)
    full_mask = torch.ones(1, 6, dtype=torch.bool)
    full = model(hidden, position, full_mask)
    prefix = model(hidden[:, :4], position[:, :4], full_mask[:, :4])
    assert torch.allclose(prefix, full[:, :4], atol=1e-6, rtol=1e-6)

    padded_hidden = torch.cat([hidden, torch.randn(1, 6, 16)], dim=0)
    padded_position = torch.zeros(2, 6, 4)
    padded_mask = torch.tensor([[True] * 6, [True] * 4 + [False] * 2])
    padded = model(padded_hidden, padded_position, padded_mask)
    isolated = model(
        padded_hidden[1:2, :4], padded_position[1:2, :4], padded_mask[1:2, :4]
    )
    assert torch.allclose(isolated, padded[1:2, :4], atol=1e-6, rtol=1e-6)


def _synthetic_predictions(split: str = "calibration") -> pd.DataFrame:
    rows = []
    for trace_index in range(30):
        for checkpoint in range(3):
            raw_logit = -2.0 + checkpoint + (trace_index % 5) * 0.2
            observed = float(raw_logit > -0.4)
            rows.append(
                {
                    "trace_id": f"t{trace_index}",
                    "split": split,
                    "raw_logit": raw_logit,
                    "raw_probability": 1 / (1 + np.exp(-raw_logit)),
                    "observed_success_rate": observed,
                }
            )
    return pd.DataFrame(rows)


def test_calibrator_is_positive_and_calibration_only() -> None:
    frame = _synthetic_predictions()
    calibrator = fit_positive_affine_calibrator(frame, max_iterations=80)
    calibrated = apply_calibration(frame, calibrator)
    assert calibrator["a"] > 0
    assert calibrator["fit_split"] == "calibration"
    assert calibrated["calibrated_probability"].between(0, 1).all()
    with pytest.raises(RuntimeError, match="calibration split"):
        fit_positive_affine_calibrator(_synthetic_predictions("architecture_dev"), 10)


def test_threshold_sweep_does_not_freeze_tau() -> None:
    frame = pd.DataFrame(
        [
            {
                "trace_id": "a",
                "checkpoint_ordinal": index,
                "total_checkpoint_count": 3,
                "prefix_token_count": (index + 1) * 10,
                "total_trace_token_count": 40,
                "success_count": success,
                "observed_success_rate": success / 4,
                "calibrated_probability": probability,
            }
            for index, (success, probability) in enumerate([(4, 0.9), (1, 0.6), (0, 0.2)])
        ]
    )
    result = threshold_sweep(frame, [0.1, 0.7, 0.95])
    assert not result["final_tau_selected"].any()
    assert result.loc[result["tau"] == 0.95, "full_regeneration_fallback_fraction"].item() == 1.0
    assert result.loc[result["tau"] == 0.1, "dangerous_late_selection_frequency"].item() == 1.0


def test_boundary_scorer_latest_or_fallback(tmp_path: Path) -> None:
    model = build_predictor("linear_probe", 6).eval()
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "predictor_config": model.config.to_dict(),
        },
        checkpoint,
    )
    calibration = tmp_path / "calibrator.json"
    calibration.write_text(
        json.dumps(
            {
                "base_model": "family_test",
                "fit_split": "calibration",
                "a": 1.0,
                "b": 0.0,
            }
        )
    )
    scorer = BoundaryScorer(
        base_model="family_test",
        predictor_checkpoint=checkpoint,
        calibration_artifact=calibration,
    )
    metadata = [
        {
            "checkpoint_ordinal": index,
            "prefix_token_count": 10 * (index + 1),
            "total_checkpoint_count": 3,
            "total_trace_token_count": 40,
        }
        for index in range(3)
    ]
    fallback = scorer.score(torch.randn(3, 6), metadata, tau=1.0)
    assert fallback["decision"]["decision"] == "full_regeneration_fallback"
    selected = scorer.score(torch.randn(3, 6), metadata, tau=0.0)
    assert selected["decision"]["selected_checkpoint_index"] == 2
