.PHONY: help install install-cloud doctor audit-code-only test test-fast help-data help-teacher-forced help-boundary help-threshold help-geometry help-k-densification run-audit-data run-prefix-validity

PYTHON ?= python3
PYTHONPATH ?= src

help:
	@printf "%s\n" \
	  "SafePrefix targets:" \
	  "  install              Install package with test dependencies" \
	  "  install-cloud        Install package with Modal support" \
	  "  doctor               Check repo layout and tracked-file policy" \
	  "  audit-code-only      Fail if generated data/results/weights are tracked" \
	  "  test-fast            Run fast local tests" \
	  "  test                 Run full local tests" \
	  "  help-*               Print workflow-specific command help" \
	  "  run-audit-data       Generate dataset audit outputs under ignored artifacts/" \
	  "  run-prefix-validity  Validate prefix-validity config"

install:
	$(PYTHON) -m pip install -e ".[dev]"

install-cloud:
	$(PYTHON) -m pip install -e ".[dev,cloud]"

doctor:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m safeprefix.cli doctor

audit-code-only:
	! git ls-files | rg '^(artifacts|outputs|published_artifacts|results|data|weights|checkpoints|models)/'
	! git ls-files | rg -i '\.(parquet|jsonl|pt|pth|safetensors|bin|ckpt|onnx|npy|npz|h5|hdf5|pkl|pickle|tar|tgz|gz|zip|sqlite|db|csv|tsv)$$'
	git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | awk '$$1=="blob" && $$3 > 1048576 {print; found=1} END {exit found}'

test:
	$(PYTHON) -m pytest

test-fast:
	$(PYTHON) -m pytest tests/test_answer_parsers.py tests/test_segmenters.py tests/test_boundary_model_v1.py tests/test_prefix_validity_v1.py tests/test_k_densification_v1.py

help-data:
	$(PYTHON) scripts/00_audit_datasets.py --help

help-teacher-forced:
	$(PYTHON) scripts/25_run_full_teacher_forced_rollouts.py --help
	$(PYTHON) scripts/26_reconcile_safeprefix_counts.py --help
	$(PYTHON) scripts/27_complete_teacher_forced_corpora.py --help
	$(PYTHON) scripts/28_freeze_teacher_forced_completion_manifest.py --help

help-boundary:
	$(PYTHON) scripts/run_boundary_model_v1.py --help

help-threshold:
	$(PYTHON) scripts/run_safeprefix_threshold_selection_tf_v1.py --help

help-geometry:
	$(PYTHON) scripts/run_recoverability_geometry_tf.py --help

help-k-densification:
	$(PYTHON) scripts/run_k_densification_v1.py --help

run-audit-data:
	$(PYTHON) scripts/00_audit_datasets.py --config configs/dataset_audit.yaml

run-prefix-validity:
	$(PYTHON) scripts/run_safeprefix_prefix_validity_v1.py --config configs/prefix_validity_v1.yaml
