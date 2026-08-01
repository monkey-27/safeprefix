from __future__ import annotations

import pandas as pd
import pytest

from safeprefix.k_densification_v1.preflight import (
    _candidate_census,
    CohortInsufficientError,
)


def _config(required: int = 2) -> dict:
    return {
        "cohort": {
            "minimum_reasoning_checkpoints_per_model_trace": 4,
            "problems_per_domain": required,
            "stable_hash_namespace": "k-densification-v1-test",
            "domains": {
                "crv_arithmetic": "CRV",
                "processbench_math": "Math",
                "processbench_olympiadbench": "Olympiad",
                "processbench_omnimath": "Omni",
            },
        }
    }


def _metadata(counts: dict[str, list[int]]) -> pd.DataFrame:
    rows = []
    models = (
        "family_a_small",
        "family_a_large",
        "family_b_small",
        "family_b_large",
    )
    for domain, values in counts.items():
        for problem_index, reasoning_count in enumerate(values):
            for model in models:
                rows.append(
                    {
                        "domain": domain,
                        "problem_id": f"{domain}-{problem_index}",
                        "common_trace_id": f"trace-{domain}-{problem_index}",
                        "base_model": model,
                        "trace_id": f"{model}-{domain}-{problem_index}",
                        "reasoning_checkpoint_count": reasoning_count,
                        "all_checkpoint_count": reasoning_count + 1,
                        "median_response_length_component": 100 + problem_index,
                    }
                )
    return pd.DataFrame(rows)


def test_candidate_gate_excludes_prompt_root_and_never_reads_outcomes() -> None:
    counts = {
        "crv_arithmetic": [4, 4],
        "processbench_math": [4, 3],
        "processbench_olympiadbench": [4, 4],
        "processbench_omnimath": [4, 4],
    }
    frame, summary = _candidate_census(config=_config(), metadata=_metadata(counts))
    assert summary["strict_eligible_problem_count_by_model_domain"]["family_a_small"]["Math"] == 1
    assert summary["permissive_root_inclusive_problem_count_by_model_domain"]["family_a_small"]["Math"] == 2
    assert summary["cohort_gate_passed"] is False
    assert summary["rollout_success_values_loaded"] is False
    assert not frame["rollout_success_inspected"].any()


def test_candidate_gate_passes_only_when_every_domain_meets_quota() -> None:
    counts = {
        "crv_arithmetic": [4, 5],
        "processbench_math": [4, 5],
        "processbench_olympiadbench": [4, 5],
        "processbench_omnimath": [4, 5],
    }
    _, summary = _candidate_census(config=_config(), metadata=_metadata(counts))
    assert summary["cohort_gate_passed"] is True
    assert summary["strict_total_eligible_model_problems"] == 32
    assert summary["cross_model_problem_intersection_required"] is False


def test_candidate_gate_does_not_require_cross_model_problem_ids() -> None:
    rows = []
    for model_index, model in enumerate(
        ("family_a_small", "family_a_large", "family_b_small", "family_b_large")
    ):
        for domain in (
            "crv_arithmetic",
            "processbench_math",
            "processbench_olympiadbench",
            "processbench_omnimath",
        ):
            for problem_index in range(2):
                rows.append(
                    {
                        "domain": domain,
                        "problem_id": f"{model}-{domain}-{problem_index}",
                        "common_trace_id": f"trace-{model_index}-{domain}-{problem_index}",
                        "base_model": model,
                        "trace_id": f"{model}-{domain}-{problem_index}",
                        "reasoning_checkpoint_count": 4,
                        "all_checkpoint_count": 5,
                        "median_response_length_component": 100 + problem_index,
                    }
                )
    _, summary = _candidate_census(config=_config(), metadata=pd.DataFrame(rows))
    assert summary["cohort_gate_passed"] is True
    assert summary["strict_total_eligible_model_problems"] == 32
    assert summary["bootstrap_unit"] == "complete_model_problem_trace_within_model"
    assert summary["cross_model_paired_problem_bootstrap"] is False


def test_blocker_exception_preserves_machine_summary() -> None:
    summary = {"status": "BLOCKED_BEFORE_ROLLOUT_GENERATION"}
    error = CohortInsufficientError("blocked", summary=summary)
    assert str(error) == "blocked"
    assert error.summary == summary
