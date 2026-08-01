"""Pre-registered CPU analyses for teacher-forced recoverability geometry.

This package deliberately contains no artifact discovery or model-inference code.
Callers must pass already validated, non-native teacher-forced tables and arrays.
"""

from .analysis import (
    AffineAxis,
    ShapeThresholds,
    affine_axis_audit,
    axis_energy_fraction_eta,
    branch_metrics,
    classify_binomial_trajectory,
    classify_trajectory,
    compute_branch_metrics,
    compute_h1_metrics,
    compute_h2_metrics,
    compute_margin,
    cross_domain_transfer_summary,
    h1_axis_orthogonal_diagnostics,
    margin_metrics,
    select_canonical_median_seed,
    select_canonical_seed,
    select_h4_parents,
    select_local_branch_parents,
    seed_axis_stability,
    shape_sensitivity_analysis,
    summarize_seed_stability,
    summarize_trajectory_prevalence,
    within_trace_metrics,
)
from .reporting import generate_geometry_figures, render_geometry_report
from .statistics import (
    clustered_bootstrap,
    clustered_paired_bootstrap,
    holm_adjust,
    holm_correction,
    parent_clustered_bootstrap,
    trace_clustered_bootstrap,
)

__all__ = [
    "AffineAxis",
    "ShapeThresholds",
    "affine_axis_audit",
    "axis_energy_fraction_eta",
    "branch_metrics",
    "classify_binomial_trajectory",
    "classify_trajectory",
    "clustered_bootstrap",
    "clustered_paired_bootstrap",
    "compute_branch_metrics",
    "compute_h1_metrics",
    "compute_h2_metrics",
    "compute_margin",
    "cross_domain_transfer_summary",
    "generate_geometry_figures",
    "h1_axis_orthogonal_diagnostics",
    "holm_adjust",
    "holm_correction",
    "margin_metrics",
    "parent_clustered_bootstrap",
    "render_geometry_report",
    "select_canonical_median_seed",
    "select_canonical_seed",
    "select_h4_parents",
    "select_local_branch_parents",
    "seed_axis_stability",
    "shape_sensitivity_analysis",
    "summarize_seed_stability",
    "summarize_trajectory_prevalence",
    "trace_clustered_bootstrap",
    "within_trace_metrics",
]
