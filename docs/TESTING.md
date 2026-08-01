# Testing

The test suite is local and code-focused. It uses temporary files and small
in-memory fixtures; it should not require checked-in artifacts or model weights.

## Test Tiers

| Command | Purpose |
| --- | --- |
| `make test-fast` | Parser, segmentation, boundary, prefix-validity, and K-densification unit checks. |
| `python3 -m pytest` | Full local suite for package logic and orchestration contracts. |
| `make audit-code-only` | Git-tree audit for accidentally tracked data, results, weights, or archives. |
| `safeprefix doctor` | Repo-shape and tracked-file policy check. |

## Coverage Map

| Area | Representative tests |
| --- | --- |
| Parsing and token alignment | `tests/test_answer_parsers.py`, `tests/test_segmenters.py`, `tests/test_token_alignment.py` |
| Dataset and reference prep | `tests/test_problem_splits.py`, `tests/test_reference_join.py`, `tests/test_native_data_compilation.py` |
| Cache and generation plumbing | `tests/test_cache_slicing.py`, `tests/test_cache_production_v3.py`, `tests/test_full_teacher_forced_rollout_engine.py` |
| Boundary and prefix validity | `tests/test_boundary_model_v1.py`, `tests/test_prefix_validity_v1*.py` |
| Threshold and geometry analysis | `tests/test_safeprefix_threshold_selection_tf_v1.py`, `tests/test_recoverability_geometry*.py` |
| Native trace and hidden-state flows | `tests/test_native_failed_trace_acquisition.py`, `tests/test_native_hidden_state_cache.py` |
| K-densification | `tests/test_k_densification_v1.py`, `tests/test_k_densification_reporting.py` |

Cloud jobs and GPU model execution are launched through Modal scripts and are
not part of the default local test suite.
