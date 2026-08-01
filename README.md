# SafePrefix

SafePrefix tests whether a verifier-safe reasoning prefix can be preserved by
restoring an exact KV-cache checkpoint and regenerating only the suffix. This
repository is the curated executable code surface from `aaai_monkey`.

The repo is intentionally focused. It omits old pilot scaffolding, mock-only
smoke harnesses, unrelated experiments, generated datasets, result reports,
tensors, Parquet rollout tables, and large archives. The original research
workspace remains untouched.

## Layout

| Path | Purpose |
| --- | --- |
| `src/safeprefix/` | Python package for data prep, cache restoration, teacher-forced rollouts, boundary models, prefix validity, native trace acquisition, geometry, and K-densification. |
| `scripts/` | Maintained command-line and Modal entry points. |
| `configs/` | Frozen YAML configs for production and analysis workflows. |
| `docs/WORKFLOWS.md` | Command map for data generation, training, evaluation, and cloud runs. |
| `tests/` | Focused tests for the maintained code path. |

Generated files belong in ignored runtime directories such as `artifacts/`,
`outputs/`, `results/`, and `data/`. They are intentionally absent from Git.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
```

For Modal launchers:

```bash
python3 -m pip install -e ".[dev,cloud]"
```

## Quick Checks

```bash
make test-fast
safeprefix commands
```

## Main Workflows

See [docs/WORKFLOWS.md](docs/WORKFLOWS.md) for the full command map.

```bash
# Validate the prefix-validity config.
python3 scripts/run_safeprefix_prefix_validity_v1.py \
  --config configs/prefix_validity_v1.yaml

# Train/evaluate boundary-model v1 in resumable stages.
python3 scripts/run_boundary_model_v1.py prepare --help
python3 scripts/run_boundary_model_v1.py train-model --help
python3 scripts/run_boundary_model_v1.py select --help
python3 scripts/run_boundary_model_v1.py finalize --help
python3 scripts/run_boundary_model_v1.py report --help

# Run the K-densification robustness analysis.
python3 scripts/run_k_densification_v1.py --help
```
