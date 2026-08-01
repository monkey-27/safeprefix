from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from safeprefix.repairability_funnel import (
    _adherence,
    _atomic_json,
    _cross_fit,
    _fold_assignment,
    _funnel_metrics,
    _grouped_ridge_predictions,
    _holm,
    _orientation,
)


def test_atomic_json_serializes_nonfinite_values_as_strict_null(tmp_path: Path) -> None:
    path = tmp_path / "strict.json"
    _atomic_json(path, {"missing": float("nan"), "infinite": np.float64(np.inf)})
    payload = json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    assert payload == {"infinite": None, "missing": None}


def test_success_contrast_direction_and_concentration() -> None:
    directions = np.asarray(
        [[1.0, 0.0], [0.98, 0.2], [-1.0, 0.0], [-0.98, -0.2]], dtype=float
    )
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    recoverability = np.asarray([0.9, 0.8, 0.2, 0.1])
    result = _funnel_metrics(directions, recoverability)
    assert result["directional_concentration"] > 0.95
    assert result["orientation"][0] > 0.99
    assert result["leading_spectral_mass"] > 0.5


def test_adherence_uses_continuous_success_and_failure_weights() -> None:
    directions = np.asarray([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    recoverability = np.asarray([0.9, 0.1, 0.5])
    positive = _adherence(np.asarray([1.0, 0.0]), directions, recoverability)
    negative = _adherence(np.asarray([-1.0, 0.0]), directions, recoverability)
    assert positive > 0
    assert negative < 0
    assert np.isclose(positive, -negative)


def test_orientation_is_undefined_when_support_weights_do_not_vary() -> None:
    directions = np.eye(3)
    assert _orientation(directions, np.asarray([0.5, 0.5, 0.5])) is None


def test_trace_fold_assignment_is_deterministic_and_trace_grouped() -> None:
    traces = ["a", "a", "b", "c", "d", "e", "f"]
    first = _fold_assignment(traces, folds=3, seed=17)
    second = _fold_assignment(list(reversed(traces)), folds=3, seed=17)
    assert first == second
    assert set(first) == set(traces)
    assert len(set(first.values())) == 3


def test_crossfit_never_uses_evaluated_trace_in_fit() -> None:
    rng = np.random.default_rng(4)
    values = {index: rng.normal(size=8) for index in range(12)}
    traces = {index: f"trace-{index // 2}" for index in values}
    transformed, transforms, assignments, records = _cross_fit(
        values,
        traces,
        folds=3,
        variance_target=0.95,
        maximum_dimensions=4,
        ridge_fraction=1e-3,
        seed=9,
    )
    assert set(transformed) == set(values)
    assert set(transforms) == set(assignments.values())
    for trace in set(traces.values()):
        assert len({assignments[index] for index in values if traces[index] == trace}) == 1
    assert all(record["fit_trace_count"] + record["heldout_trace_count"] == 6 for record in records)


def test_holm_correction_is_monotone_and_keeps_all_families() -> None:
    result = _holm({"F1": 0.01, "F2": 0.02, "F3": 0.5, "F4": 1.0})
    assert set(result) == {"F1", "F2", "F3", "F4"}
    ordered = [result[name]["holm_adjusted_p"] for name in ("F1", "F2", "F3", "F4")]
    assert ordered == sorted(ordered)
    assert result["F1"]["holm_adjusted_p"] == 0.04


def test_grouped_ridge_imputes_support_only_undefined_geometry() -> None:
    frame = pd.DataFrame(
        {
            "trace_id": [f"t-{index // 2}" for index in range(12)],
            "base_model": ["m"] * 12,
            "domain": ["d"] * 12,
            "horizon": [32] * 12,
            "parent_recoverability": np.linspace(0.1, 0.9, 12),
            "undefined_support_breadth": [np.nan, 1.0] * 6,
            "target": np.linspace(0.2, 0.8, 12),
        }
    )
    predictions = _grouped_ridge_predictions(
        frame,
        target="target",
        numeric_features=["parent_recoverability", "undefined_support_breadth"],
        categorical_features=["base_model", "domain", "horizon"],
        folds=3,
        alpha=1.0,
        seed=4,
    )
    assert np.isfinite(predictions).all()
