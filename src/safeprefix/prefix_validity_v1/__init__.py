"""Monotonic first-visible-error / prefix-validity probe primitives."""

from .calibration import (
    OOFCalibrationResult,
    PositiveAffineCalibrator,
    apply_calibrator,
    fit_fivefold_oof_calibration,
    fit_positive_affine_calibrator,
)
from .evaluation import (
    boundary_metrics,
    checkpoint_metrics,
    evaluate_by_domain,
    paired_trace_bootstrap,
)
from .monotonic import (
    GateCutoff,
    apply_monotonic_projection,
    project_nonincreasing,
    select_gate_cutoff,
)
from .training import (
    PrefixValidityCorpus,
    PrefixValidityTrainingConfig,
    ProbeCandidate,
    fit_probe_matrix,
    load_frozen_probe,
    predict_probe,
    save_frozen_probe,
    select_learning_rate_and_median_seed,
    trace_weighted_binary_bce,
)

__all__ = [
    "GateCutoff",
    "OOFCalibrationResult",
    "PositiveAffineCalibrator",
    "PrefixValidityCorpus",
    "PrefixValidityTrainingConfig",
    "ProbeCandidate",
    "apply_calibrator",
    "apply_monotonic_projection",
    "boundary_metrics",
    "checkpoint_metrics",
    "evaluate_by_domain",
    "fit_fivefold_oof_calibration",
    "fit_positive_affine_calibrator",
    "fit_probe_matrix",
    "load_frozen_probe",
    "paired_trace_bootstrap",
    "predict_probe",
    "project_nonincreasing",
    "save_frozen_probe",
    "select_gate_cutoff",
    "select_learning_rate_and_median_seed",
    "trace_weighted_binary_bce",
]
