# Workflows

Run commands from the repository root after installing the package.

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
```

Add Modal support only when launching cloud jobs:

```bash
python3 -m pip install -e ".[dev,cloud]"
```

## Workflow Contract

| Stage | Entry point | External inputs | Ignored outputs |
| --- | --- | --- | --- |
| Dataset audit | `scripts/00_audit_datasets.py` | Hugging Face datasets or local CRV files | `artifacts/dataset_audit/audit/` |
| Teacher-forced rollouts | `scripts/25_run_full_teacher_forced_rollouts.py` | Frozen manifests from an external run volume | `artifacts/full_teacher_forced_suite/` |
| Teacher-forced completion | `scripts/26_*`, `scripts/27_*`, `scripts/28_*` | Frozen manifests and completed rollout root | `artifacts/count_reconciliation/`, `artifacts/teacher_forced_completion*/` |
| Boundary model | `scripts/run_boundary_model_v1.py` | Original/completed rollout roots and manifest root | `artifacts/boundary_model_v1/` |
| Prefix validity | `scripts/run_safeprefix_prefix_validity_v1.py` | Boundary and completion roots configured in YAML | `outputs/prefix_validity_v1/` |
| Threshold selection | `scripts/run_safeprefix_threshold_selection_tf_v1.py` | Boundary, completion, and old-rollout roots | `artifacts/safeprefix_threshold_selection_tf_v1/` |
| Native traces | `scripts/33_*`, `scripts/35_*`, `scripts/37_*`, `scripts/38_*` | Frozen manifests and Modal workspace outputs | `artifacts/native_*` |
| Geometry/funnel | `scripts/run_recoverability_geometry_tf.py`, `scripts/modal_safeprefix_repairability_funnel.py` | Boundary and teacher-forced geometry roots | `artifacts/recoverability_geometry_tf_v1/`, `artifacts/repairability_funnel_v1/` |
| K-densification | `scripts/run_k_densification_v1.py` | Threshold, boundary, completion, and original roots | `outputs/k_densification_v1/` |

Fresh checkouts can run help, config validation, and tests immediately. Full
experiment stages require the external inputs listed above; those inputs must
remain outside Git.

## Data and Manifest Preparation

```bash
python3 scripts/00_audit_datasets.py --config configs/dataset_audit.yaml

python3 scripts/25_run_full_teacher_forced_rollouts.py --help
python3 scripts/25_run_full_teacher_forced_rollouts.py prepare \
  --config configs/full_teacher_forced_suite.yaml \
  --run-id <run-id>
python3 scripts/26_reconcile_safeprefix_counts.py --help
python3 scripts/27_complete_teacher_forced_corpora.py --help
python3 scripts/28_freeze_teacher_forced_completion_manifest.py --help

python3 scripts/33_prepare_native_data_compilation.py \
  --config configs/native_data_compilation.yaml

python3 scripts/35_prepare_native_failed_trace_acquisition.py \
  --config configs/native_failed_trace_acquisition.yaml
```

## Teacher-Forced Rollouts

Local orchestration:

```bash
python3 scripts/25_run_full_teacher_forced_rollouts.py \
  --help
```

Teacher-forced completion expects two generated prerequisite directories under
the ignored `artifacts/` tree:

```bash
python3 scripts/26_reconcile_safeprefix_counts.py \
  --frozen-root <frozen-manifest-root> \
  --production-root <completed-teacher-forced-root>

python3 scripts/28_freeze_teacher_forced_completion_manifest.py \
  --frozen-source-root <frozen-manifest-root> \
  --completed-root <completed-teacher-forced-root>
```

Modal launchers:

```bash
MODAL_PROFILE=<profile> python3 -m modal run --detach \
  scripts/modal_safeprefix_full_teacher_forced.py \
  --action submit \
  --run-id safeprefix_full_teacher_forced_<date>_r1

MODAL_PROFILE=<profile> python3 -m modal run --detach \
  scripts/modal_safeprefix_teacher_forced_completion.py \
  --action submit \
  --run-id safeprefix_teacher_forced_completion_<date>_r1
```

## Boundary Model V1

```bash
python3 scripts/run_boundary_model_v1.py prepare \
  --config configs/boundary_model_v1.yaml \
  --artifact-root artifacts/boundary_model_v1 \
  --old-root <old-rollout-root> \
  --completion-root <completion-root> \
  --manifest-root <manifest-root>

python3 scripts/run_boundary_model_v1.py train-model \
  --config configs/boundary_model_v1.yaml \
  --artifact-root artifacts/boundary_model_v1 \
  --model-key family_a_small \
  --device cuda

python3 scripts/run_boundary_model_v1.py select \
  --config configs/boundary_model_v1.yaml \
  --artifact-root artifacts/boundary_model_v1

python3 scripts/run_boundary_model_v1.py finalize \
  --config configs/boundary_model_v1.yaml \
  --artifact-root artifacts/boundary_model_v1 \
  --device cuda

python3 scripts/run_boundary_model_v1.py report \
  --config configs/boundary_model_v1.yaml \
  --artifact-root artifacts/boundary_model_v1
```

Cloud runner:

```bash
MODAL_PROFILE=<profile> python3 -m modal run --detach \
  scripts/modal_safeprefix_boundary_model_v1.py \
  --action launch \
  --run-id safeprefix_boundary_model_v1_<date>_r1
```

## Prefix Validity

```bash
python3 scripts/run_safeprefix_prefix_validity_v1.py \
  --config configs/prefix_validity_v1.yaml

MODAL_PROFILE=<profile> python3 -m modal run --detach \
  scripts/modal_safeprefix_prefix_validity_v1.py \
  --run-id safeprefix_prefix_validity_v1_<date>_r1
```

## Threshold Selection and Native Traces

```bash
python3 scripts/run_safeprefix_threshold_selection_tf_v1.py --help

MODAL_PROFILE=<profile> python3 -m modal run --detach \
  scripts/modal_safeprefix_threshold_selection_tf_v1.py \
  --action submit-shard \
  --run-id safeprefix_threshold_selection_tf_v1_<date>_r1 \
  --shard-id shard-00

MODAL_PROFILE=<profile> python3 -m modal run --detach \
  scripts/modal_safeprefix_native_failed_trace_acquisition.py \
  --run-id safeprefix_native_failed_trace_acquisition_<date>_r1 \
  --model-key family_a_small

python3 scripts/37_merge_native_failed_trace_workspaces.py --help
python3 scripts/38_validate_native_failed_trace_prelaunch.py --help
```

## Geometry and Repairability Funnel

```bash
python3 scripts/run_recoverability_geometry_tf.py prepare
python3 scripts/run_recoverability_geometry_tf.py phase2
python3 scripts/run_recoverability_geometry_tf.py phase4

MODAL_PROFILE=<profile> python3 -m modal run --detach \
  scripts/modal_safeprefix_repairability_funnel.py \
  --phase complete \
  --run-id safeprefix_repairability_funnel_<date>_r1
```

## K-Densification

```bash
python3 scripts/run_k_densification_v1.py --help

MODAL_PROFILE=<profile> python3 -m modal run --detach \
  scripts/modal_safeprefix_k_densification_v1.py \
  --action launch \
  --run-id safeprefix_k_densification_v1_<date>_r1
```
