# Navigation

This repository is organized around the runnable SafePrefix pipeline, not by
historical experiment date.

## Top-Level Files

| Path | Use |
| --- | --- |
| `README.md` | First stop: install, quick checks, and main workflow links. |
| `docs/WORKFLOWS.md` | Command map for local and Modal execution. |
| `docs/ARTIFACT_POLICY.md` | What must never be checked in and how to audit it. |
| `docs/TESTING.md` | Test tiers and coverage map. |
| `docs/MODAL.md` | Cloud launcher conventions, secrets, and volumes. |
| `PILOT_PROTOCOL.md` | Scientific protocol and claims boundary. |
| `DATA_SCHEMA.md` | Runtime data shapes written by the pipeline. |
| `configs/` | YAML configuration for each maintained stage. |
| `scripts/` | Local CLIs and Modal launchers. |
| `src/safeprefix/` | Importable Python package. |
| `tests/` | Unit and orchestration tests for the maintained code. |

Generated data, outputs, reports, weights, and checkpoints are not versioned.
The ignored runtime roots are `artifacts/`, `outputs/`, `results/`, and
`data/`.

## Package Map

| Package | Responsibility |
| --- | --- |
| `safeprefix.data` | Dataset loading, normalization, deduplication, reference joining, and problem-level splits. |
| `safeprefix.parsing` | Answer parsers, reasoning segmentation, and token/span alignment. |
| `safeprefix.models` | Model loading, cache restoration, generation, teacher forcing, hidden features, and validation manifests. |
| `safeprefix.rollout` | Branch generation, posterior utilities, scheduling, and verifier integration. |
| `safeprefix.boundary_v1` | Boundary dataset construction, model training, selection, inference, and reports. |
| `safeprefix.prefix_validity_v1` | Monotonic prefix-validity training, calibration, evaluation, and reporting. |
| `safeprefix.threshold_selection_tf_v1` | Teacher-forced threshold selection, sharding, execution, and reports. |
| `safeprefix.recoverability_geometry` | Analysis and reporting for teacher-forced recoverability geometry. |
| `safeprefix.k_densification_v1` | Nested rollout-count registry, generation, calibration, and reporting. |

## Script Families

| Script family | Stage |
| --- | --- |
| `00_*`, `16_*`, `33_*`, `35_*`, `37_*`, `38_*` | Dataset/reference/native-trace preparation. |
| `25_*`, `26_*`, `27_*`, `28_*` | Teacher-forced rollout and completion preparation. |
| `run_boundary_model_v1.py` | Boundary-model prepare/train/select/finalize/report stages. |
| `run_safeprefix_prefix_validity_v1.py` | Prefix-validity config validation and local entry point. |
| `run_safeprefix_threshold_selection_tf_v1.py` | Threshold-selection preparation, sharding, analysis, and reporting. |
| `run_recoverability_geometry_tf.py` | Geometry preparation and bridge stages. |
| `run_k_densification_v1.py` | K-densification preflight, registry, generation finalization, and reporting. |
| `modal_safeprefix_*.py` | Cloud launchers for the corresponding long-running stages. |

## Fresh Checkout Checklist

```bash
python3 -m pip install -e ".[dev]"
safeprefix doctor
safeprefix commands
python3 -m pytest
```

For cloud launchers, install the cloud extra and use the command forms in
`docs/WORKFLOWS.md`.
