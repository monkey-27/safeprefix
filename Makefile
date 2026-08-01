.PHONY: install install-cloud test test-fast audit-data reconcile-counts freeze-completion boundary-prepare boundary-train prefix-validity k-densification

PYTHON ?= python3

install:
	$(PYTHON) -m pip install -e ".[dev]"

install-cloud:
	$(PYTHON) -m pip install -e ".[dev,cloud]"

test:
	$(PYTHON) -m pytest

test-fast:
	$(PYTHON) -m pytest tests/test_answer_parsers.py tests/test_segmenters.py tests/test_boundary_model_v1.py tests/test_prefix_validity_v1.py tests/test_k_densification_v1.py

audit-data:
	$(PYTHON) scripts/00_audit_datasets.py --config configs/datasets.yaml

reconcile-counts:
	$(PYTHON) scripts/26_reconcile_safeprefix_counts.py --help

freeze-completion:
	$(PYTHON) scripts/28_freeze_teacher_forced_completion_manifest.py --help

boundary-prepare:
	$(PYTHON) scripts/run_boundary_model_v1.py prepare --help

boundary-train:
	$(PYTHON) scripts/run_boundary_model_v1.py train-model --help

prefix-validity:
	$(PYTHON) scripts/run_safeprefix_prefix_validity_v1.py --config configs/prefix_validity_v1.yaml

k-densification:
	$(PYTHON) scripts/run_k_densification_v1.py --help
