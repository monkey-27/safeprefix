# Modal

Modal launchers live in `scripts/modal_safeprefix_*.py`. They package source
code and configs into Modal images, then write run material to Modal volumes or
ignored local runtime directories.

## Setup

```bash
python3 -m pip install -e ".[dev,cloud]"
modal setup
```

The launchers assume access to:

| Resource | Use |
| --- | --- |
| `huggingface-token` secret | Authenticated model and dataset access. |
| `safeprefix-hf-cache` volume | Shared Hugging Face cache. |
| `safeprefix-runs` volume | Read-only source runs for stages that reuse prior generated inputs. |
| Stage-specific output volumes | Durable runtime outputs for each long-running stage. |

## Command Pattern

Most launchers use explicit `action` and `run_id` parameters:

```bash
MODAL_PROFILE=<profile> python3 -m modal run \
  scripts/modal_safeprefix_full_teacher_forced.py \
  --action submit \
  --run-id <run-id>
```

Check a launcher's local entrypoint with:

```bash
python3 -m modal run scripts/<launcher>.py --help
```

Do not copy downloaded Modal results into Git. Keep them in ignored runtime
directories or external storage.
