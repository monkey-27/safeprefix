import json

import pytest

from safeprefix.native_failed_trace_acquisition import (
    bootstrap_trace_metrics,
    build_attempt_packs,
    build_regeneration_packs,
    initial_seed,
    regeneration_seed,
    select_frozen_cohort,
    validate_final_integrity,
    validate_partial_regeneration_integrity,
)
from safeprefix.native_failed_trace_pipeline import next_attempt_wave, target_for_stratum


def _config():
    return {
        "seed": 2701,
        "acquisition": {
            "target_by_stratum": {"gsm1k": 2, "math_level_3": 3, "math_level_4": 4},
            "workers_per_workspace": 2,
            "semantic_segmentation_enabled": False,
            "checkpoint_extraction_enabled": False,
            "hidden_state_extraction_enabled": False,
            "kv_cache_persistence_enabled": False,
            "boundary_model_enabled": False,
        },
        "generation": {"rollout_indices": [0, 1, 2, 3]},
    }


def _source(stratum: str, rank: int) -> dict:
    return {"source_id": f"{stratum}-{rank}", "stratum": stratum, "source_order_rank": rank}


def _attempt(model: str, stratum: str, rank: int, status: str = "valid_incorrect") -> dict:
    return {
        "attempt_key": f"attempt-{model}-{stratum}-{rank}",
        "model_key": model,
        "source_id": f"{stratum}-{rank}",
        "stratum": stratum,
        "source_order_rank": rank,
        "initial_generation_seed": rank + 10,
        "raw_initial_response": "reasoning\\n\\boxed{0}",
        "parser_status": "parsed",
        "verifier_status": "completed",
        "initial_status": status,
        "configuration_hash": "cfg",
        "problem_text": "p",
        "gold_answer": "1",
        "rendered_prompt": "prompt",
        "prompt_token_ids": [1, 2],
        "completion_token_ids": [3, 4],
        "truncation_flag": False,
        "parser_version": "parser-v1",
        "verifier_version": "verifier-v1",
    }


def test_attempt_packs_preserve_global_source_order_and_identity():
    rows = [_source("gsm1k", rank) for rank in range(5)]
    first = build_attempt_packs(rows, model_key="m", stratum="gsm1k", pack_size=2, configuration_hash="cfg")
    second = build_attempt_packs(rows, model_key="m", stratum="gsm1k", pack_size=2, configuration_hash="cfg")
    assert [pack.to_dict() for pack in first] == [pack.to_dict() for pack in second]
    assert [item for pack in first for item in pack.source_ids] == [f"gsm1k-{rank}" for rank in range(5)]
    assert [len(pack.source_ids) for pack in first] == [2, 2, 1]


def test_seed_assignment_is_identity_bound_not_resume_order():
    config = _config()
    assert initial_seed(config, "m", "p") == initial_seed(config, "m", "p")
    trace_seeds = [regeneration_seed(config, "m", "trace", index) for index in range(4)]
    assert len(set(trace_seeds)) == 4
    assert initial_seed(config, "m", "p") != initial_seed(config, "m2", "p")
    with pytest.raises(ValueError):
        regeneration_seed(config, "m", "trace", 4)


def test_frozen_quota_transfer_is_exact_and_never_uses_results_of_regeneration():
    attempts = [
        _attempt("m", "gsm1k", 0),
        _attempt("m", "math_level_3", 0),
        _attempt("m", "math_level_3", 1),
        _attempt("m", "math_level_3", 2),
        _attempt("m", "math_level_3", 3),
        *[_attempt("m", "math_level_4", rank) for rank in range(5)],
    ]
    cohort, summary = select_frozen_cohort(attempts, _config()["acquisition"]["target_by_stratum"])
    assert summary["quota_transfers"] == {"gsm1k_to_math_level_3": 1, "math_level_3_to_math_level_4": 0}
    assert summary["selected_by_stratum"] == {"gsm1k": 1, "math_level_3": 4, "math_level_4": 4}
    assert len(cohort) == 9
    assert all(row["frozen_before_regeneration"] for row in cohort)
    assert all(not row["regeneration_conditioned_selection"] for row in cohort)


def test_sequential_target_does_not_transfer_a_deficit_before_exhaustion():
    config = _config()
    attempts = [_attempt("m", "gsm1k", 0)]
    assert target_for_stratum(config, attempts, "math_level_3", prior_strata_exhausted={"gsm1k": False}) == 3
    assert target_for_stratum(config, attempts, "math_level_3", prior_strata_exhausted={"gsm1k": True}) == 4


def test_next_wave_stops_at_quota_and_uses_at_most_workspace_workers():
    config = _config()
    rows = [_source("gsm1k", rank) for rank in range(10)]
    packs = build_attempt_packs(rows, model_key="m", stratum="gsm1k", pack_size=2, configuration_hash="cfg")
    manifests = {"gsm1k": [pack.to_dict() for pack in packs]}
    wave, state = next_attempt_wave(
        config, manifests, [], {"gsm1k": []}, stratum="gsm1k", prior_strata_exhausted={},
    )
    assert len(wave) == 2
    assert not state["quota_reached"]
    attempts = [_attempt("m", "gsm1k", 0), _attempt("m", "gsm1k", 1)]
    for row in attempts:
        row["pack_id"] = manifests["gsm1k"][0]["pack_id"]
    wave, state = next_attempt_wave(
        config, manifests, attempts, {"gsm1k": [manifests["gsm1k"][0]["pack_id"]]},
        stratum="gsm1k", prior_strata_exhausted={},
    )
    assert wave == []
    assert state["quota_reached"]


def test_noncontiguous_completed_pack_cannot_satisfy_ordered_quota():
    config = _config()
    rows = [_source("gsm1k", rank) for rank in range(6)]
    packs = build_attempt_packs(rows, model_key="m", stratum="gsm1k", pack_size=2, configuration_hash="cfg")
    manifests = {"gsm1k": [pack.to_dict() for pack in packs]}
    attempts = [_attempt("m", "gsm1k", 2), _attempt("m", "gsm1k", 3)]
    for row in attempts:
        row["pack_id"] = manifests["gsm1k"][1]["pack_id"]
    wave, state = next_attempt_wave(
        config, manifests, attempts,
        {"gsm1k": [manifests["gsm1k"][1]["pack_id"]]},
        stratum="gsm1k", prior_strata_exhausted={},
    )
    assert not state["quota_reached"]
    assert state["completed_contiguous_prefix_packs"] == 0
    assert wave[0]["pack_id"] == manifests["gsm1k"][0]["pack_id"]


def test_regeneration_pack_has_exact_trace_membership():
    cohort = [{"trace_id": f"t{index}"} for index in range(7)]
    packs = build_regeneration_packs(cohort, model_key="m", pack_size=3, configuration_hash="cfg")
    assert [len(pack.trace_ids) for pack in packs] == [3, 3, 1]
    assert [trace for pack in packs for trace in pack.trace_ids] == [f"t{index}" for index in range(7)]


def test_trace_bootstrap_treats_success_counts_as_problem_units():
    metrics = bootstrap_trace_metrics([0, 1, 4], replicates=100, seed=7, confidence_level=0.95)
    assert metrics["n"] == 3
    assert metrics["fr_at_1"] == pytest.approx(5 / 12)
    assert metrics["fr_at_4"] == pytest.approx(2 / 3)
    assert metrics["pf_at_4"] == pytest.approx(1 / 3)


def test_integrity_requires_four_unique_rollouts_and_forbids_native_boundary_work():
    config = _config()
    attempt = _attempt("m", "gsm1k", 0)
    cohort, _ = select_frozen_cohort([attempt], {"gsm1k": 1, "math_level_3": 0, "math_level_4": 0})
    trace_id = cohort[0]["trace_id"]
    source_rows = [_source("gsm1k", 0)]
    rollouts = [
        {
            "rollout_key": f"r{index}", "model_key": "m", "trace_id": trace_id,
            "rollout_index": index,
            "rollout_seed": regeneration_seed(config, "m", trace_id, index),
            "raw_regenerated_response": "x", "parser_status": "parsed",
            "verifier_status": "completed", "binary_verifier_outcome": 0,
            "configuration_hash": "cfg",
            "parser_version": "parser-v1", "verifier_version": "verifier-v1",
        }
        for index in range(4)
    ]
    result = validate_final_integrity(
        config={**config, "parser": {"version": "parser-v1"}, "verifier": {"version": "verifier-v1"}},
        source_rows=source_rows, attempts=[attempt], cohort=cohort, rollouts=rollouts,
    )
    assert result["status"] == "PASS"
    with pytest.raises(RuntimeError, match="exactly four"):
        validate_final_integrity(
            config={**config, "parser": {"version": "parser-v1"}, "verifier": {"version": "verifier-v1"}},
            source_rows=source_rows, attempts=[attempt], cohort=cohort,
            rollouts=rollouts[:3],
        )
    bad = json.loads(json.dumps(config))
    bad["parser"] = {"version": "parser-v1"}
    bad["verifier"] = {"version": "verifier-v1"}
    bad["acquisition"]["boundary_model_enabled"] = True
    with pytest.raises(RuntimeError, match="forbidden"):
        validate_final_integrity(
            config=bad, source_rows=source_rows, attempts=[attempt], cohort=cohort, rollouts=rollouts,
        )


def test_partial_integrity_is_explicit_and_does_not_weaken_full_gate():
    config = _config()
    config.update(parser={"version": "parser-v1"}, verifier={"version": "verifier-v1"})
    attempts = [_attempt("m", "gsm1k", rank) for rank in range(2)]
    frozen, _ = select_frozen_cohort(
        attempts, {"gsm1k": 2, "math_level_3": 0, "math_level_4": 0},
    )
    completed = frozen[:1]
    trace_id = completed[0]["trace_id"]
    rollouts = [
        {
            "rollout_key": f"partial-r{index}", "model_key": "m", "trace_id": trace_id,
            "rollout_index": index,
            "rollout_seed": regeneration_seed(config, "m", trace_id, index),
            "raw_regenerated_response": "x", "parser_status": "parsed",
            "verifier_status": "completed", "binary_verifier_outcome": 0,
            "configuration_hash": "cfg", "parser_version": "parser-v1",
            "verifier_version": "verifier-v1",
        }
        for index in range(4)
    ]
    result = validate_partial_regeneration_integrity(
        config=config,
        source_rows=[_source("gsm1k", rank) for rank in range(2)],
        attempts=attempts,
        frozen_cohort=frozen,
        completed_cohort=completed,
        rollouts=rollouts,
        expected_pack_count=2,
        completed_pack_count=1,
    )
    assert result["status"] == "PARTIAL_PASS"
    assert not result["full_completion_gate_passed"]
    assert result["completed_cohort_rows"] == 1
    assert result["missing_cohort_rows"] == 1
    assert result["four_rollouts_per_completed_trace"]

    with pytest.raises(RuntimeError, match="membership differ"):
        validate_partial_regeneration_integrity(
            config=config,
            source_rows=[_source("gsm1k", rank) for rank in range(2)],
            attempts=attempts,
            frozen_cohort=frozen,
            completed_cohort=frozen,
            rollouts=rollouts,
            expected_pack_count=2,
            completed_pack_count=1,
        )
