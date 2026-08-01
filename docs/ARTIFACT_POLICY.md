# Artifact Policy

This repository tracks code, configuration, and prose documentation only.

## Not Tracked

Do not commit generated or downloaded experiment material:

| Kind | Examples |
| --- | --- |
| Runtime roots | `artifacts/`, `outputs/`, `results/`, `published_artifacts/`, `data/` |
| Model material | weights, checkpoints, adapters, cached hidden states, KV caches |
| Tables and logs | Parquet, JSONL, CSV/TSV result tables, sqlite databases |
| Binary arrays | PyTorch tensors, NumPy arrays, safetensors, ONNX files |
| Archives | tarballs, zip files, compressed run bundles |

These paths and extensions are ignored in `.gitignore`; runtime commands should
write there or to external Modal volumes.

## Allowed

The repo may contain:

- Python source under `src/safeprefix/`;
- runnable scripts under `scripts/`;
- YAML configs under `configs/`;
- prose docs under `docs/` and the repository root;
- tests that construct temporary data inside pytest-managed temp directories.

## Audit Before Pushing

```bash
make audit-code-only
safeprefix doctor
git status --short
```
