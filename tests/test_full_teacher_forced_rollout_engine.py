from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from safeprefix.full_teacher_forced import (
    aggregate_checkpoint_outcomes,
    build_execution_packs,
    logical_rollout_keys,
    reasoning_steps_list,
    rollout_seed,
    select_frozen_manifest_failures,
    validate_pack_rows,
)
from safeprefix.models.cache_checkpoint import (
    capture_cache_checkpoint,
    clone_cache_checkpoint,
)
from safeprefix.models.cache_utils import (
    CacheRowSource,
    cache_to_legacy,
    collate_right_aligned_cache_rows,
    preallocate_decode_cache,
)
from safeprefix.models.teacher_forcing import (
    teacher_force_token_ids,
    teacher_force_token_ids_chunked,
)
from safeprefix.rollout.production_engine import (
    ProductionRolloutRequest,
    _exact_nucleus_probabilities,
    _sample_with_uniforms,
    decode_execution_pack,
)
from safeprefix.models.generation import _uniform, select_tokens_from_saved_logits
from safeprefix.production_suite import (
    _load_authoritative_frozen_cohort,
    ANSWER_REGION_AUDIT_POLICY,
    SOURCE_STEP_BOUNDARY_POLICY,
    answer_region_checkpoint_audit,
    canonical_completion,
    engine_revision,
    run_root,
    select_validation_pack,
    source_step_boundary_audits,
    validate_prepared_manifests,
)
from safeprefix.parsing.token_alignment import TokenAlignment
from safeprefix.reproducibility import stable_hash
from safeprefix.testing import (
    CharacterTokenizer,
    StochasticToyCausalModel,
    ToyCausalModel,
)


BASE_SEED = 2701

SOURCE_IDENTITIES = {
    "crv_arithmetic": ("facebook/crv", "arithmetic_expressions"),
    "processbench_math": ("Qwen/ProcessBench", "math"),
    "processbench_olympiadbench": ("Qwen/ProcessBench", "olympiadbench"),
    "processbench_omnimath": ("Qwen/ProcessBench", "omnimath"),
}


def test_source_step_boundaries_snap_after_complete_step_without_next_content() -> None:
    completion, ranges = canonical_completion(["abc", "def"])
    assert completion == "\nabc\n\ndef"
    # Token 12 spans the final ``c`` and exactly the canonical paragraph
    # delimiter. Snapping before would omit a character from the source step.
    alignment = TokenAlignment(
        input_ids=(10, 11, 12, 13),
        offsets=((0, 1), (1, 3), (3, 6), (6, 9)),
        text_length=len(completion),
        method="synthetic_cross_boundary",
    )

    audits = source_step_boundary_audits(alignment, ranges)

    first = audits[0]
    assert first["boundary_policy"] == SOURCE_STEP_BOUNDARY_POLICY
    assert first["snap_policy"] == "after"
    assert first["requested_char_offset"] == 4
    assert first["resolved_char_offset"] == 6
    assert first["token_offset"] == 3
    assert first["character_displacement"] == 2
    assert first["exact"] is False
    assert first["complete_step_included"] is True
    assert first["next_step_content_included"] is False
    assert first["resolved_char_offset"] == first["next_step_char_start"]
    assert audits[1]["exact"] is True


def test_source_step_boundary_rejects_token_entering_next_step_content() -> None:
    completion, ranges = canonical_completion(["abc", "def"])
    alignment = TokenAlignment(
        input_ids=(10, 11, 12, 13),
        offsets=((0, 1), (1, 3), (3, 7), (7, 9)),
        text_length=len(completion),
        method="synthetic_crosses_next_step",
    )

    with pytest.raises(RuntimeError, match="entered the next dataset step"):
        source_step_boundary_audits(alignment, ranges)


def test_answer_region_audit_accepts_only_pre_answer_checkpoints() -> None:
    completion, ranges = canonical_completion(
        ["Clean reasoning.", "First bad step.", "Final answer: 42"]
    )
    alignment = TokenAlignment(
        input_ids=tuple(range(len(completion))),
        offsets=tuple((index, index + 1) for index in range(len(completion))),
        text_length=len(completion),
        method="character_exact",
    )
    boundaries = source_step_boundary_audits(alignment, ranges)

    audit = answer_region_checkpoint_audit(
        completion,
        ranges,
        final_answer_text="42",
        first_error_zero_based=1,
        eligible_step_boundaries=boundaries,
    )

    assert audit["policy"] == ANSWER_REGION_AUDIT_POLICY
    assert audit["status"] == "PASS"
    assert audit["answer_region_identified"] is True
    assert audit["answer_region_method"] in {
        "source_final_answer_terminal_match",
        "final_answer_marker",
    }
    assert audit["latest_eligible_resolved_char_offset"] <= audit["answer_char_start"]
    assert audit["all_eligible_checkpoints_before_answer_region"] is True


def test_answer_region_audit_rejects_answer_inside_eligible_prefix() -> None:
    completion, ranges = canonical_completion(["Final answer: 42", "First bad step."])
    alignment = TokenAlignment(
        input_ids=tuple(range(len(completion))),
        offsets=tuple((index, index + 1) for index in range(len(completion))),
        text_length=len(completion),
        method="character_exact",
    )
    boundaries = source_step_boundary_audits(alignment, ranges)

    with pytest.raises(RuntimeError, match="enters the deterministic final-answer"):
        answer_region_checkpoint_audit(
            completion,
            ranges,
            final_answer_text="42",
            first_error_zero_based=1,
            eligible_step_boundaries=boundaries,
        )


def test_answer_region_audit_records_but_does_not_promote_weak_fallback() -> None:
    completion, ranges = canonical_completion(
        ["Clean derivation.", "First bad step.", "Therefore the result is 17."]
    )
    alignment = TokenAlignment(
        input_ids=tuple(range(len(completion))),
        offsets=tuple((index, index + 1) for index in range(len(completion))),
        text_length=len(completion),
        method="character_exact",
    )
    boundaries = source_step_boundary_audits(alignment, ranges)

    audit = answer_region_checkpoint_audit(
        completion,
        ranges,
        final_answer_text="",
        first_error_zero_based=1,
        eligible_step_boundaries=boundaries,
    )

    assert audit["status"] == "PASS"
    assert audit["answer_region_identified"] is False
    assert audit["answer_region_exclusion_boundary_identified"] is True
    assert audit["answer_region_method"] == "conservative_final_source_step_container"
    assert audit["answer_region_schema_proof"] == (
        "entire_final_dataset_step_treated_as_answer_container"
    )
    assert audit["parser_success"] is True
    assert audit["parser_candidate_accepted"] is False


def _trace(trace_index: int, first_error: int) -> dict[str, Any]:
    return {
        "trace_id": f"trace-{trace_index:03d}",
        "source_trace_id": f"source-trace-{trace_index:03d}",
        "problem_id": f"problem-{trace_index:03d}",
        "source_bucket": "crv_arithmetic" if trace_index % 2 == 0 else "processbench_math",
        "first_error_zero_based": first_error,
        "reasoning_steps": [f"reasoning step {index}" for index in range(first_error + 2)],
    }


def _frozen_trace(
    trace_index: int,
    source_bucket: str,
    manifest_role: str,
    *,
    problem_group: str | None = None,
    final_answer_correct: bool = False,
    final_answer_text: str = "",
    reasoning_steps: Any = None,
) -> dict[str, Any]:
    source_dataset, source_subset = SOURCE_IDENTITIES[source_bucket]
    return {
        "source_trace_id": f"frozen-{manifest_role}-{trace_index:03d}",
        "problem_id": f"frozen-problem-{trace_index:03d}",
        "problem_group_hash": problem_group or f"frozen-group-{manifest_role}-{trace_index:03d}",
        "source_dataset": source_dataset,
        "source_subset": source_subset,
        "problem_text": f"Problem {trace_index}",
        "reasoning_steps": (
            ["clean setup", "first visible error", "later consequence"]
            if reasoning_steps is None
            else reasoning_steps
        ),
        "reference_answer": "42",
        "final_answer_text": final_answer_text,
        "first_error_index": 1,
        "index_base": "zero",
        "final_answer_correct": final_answer_correct,
        "manifest_role": manifest_role,
    }


def _complete_pack_rows(pack: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key in pack["logical_rollout_keys"]:
        rows.append(
            {
                **key,
                "pack_id": pack["pack_id"],
                "pack_hash": pack["pack_hash"],
                "configuration_hash": pack["configuration_hash"],
                "engine_revision": pack["engine_revision"],
                "problem_id": f"problem-for-{key['trace_id']}",
                "source_bucket": "crv_arithmetic",
                "checkpoint_token_offset": 17 + int(key["checkpoint_index"]),
                "first_visible_error_zero_based": 3,
                "generated_token_ids": [11, 12],
                "generated_token_count": 2,
                "generated_text": "mock generation",
                "stop_reason": "eos",
                "truncation_flag": False,
                "parser_status": "success",
                "parser_method": "boxed",
                "verifier_pass": int(key["rollout_index"]) in {0, 2},
                "binary_outcome": int(key["rollout_index"]) in {0, 2},
            }
        )
    return rows


def _checkpoint(
    model: ToyCausalModel,
    token_ids: list[int],
):
    ids = torch.tensor([token_ids], dtype=torch.long)
    output = model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        use_cache=True,
        return_dict=True,
    )
    return capture_cache_checkpoint(
        output.past_key_values,
        output.logits[:, -1],
        ids,
        len(token_ids),
        model_id="mock/toy",
        model_revision="mock-v1",
        tokenizer_id="mock/character",
        tokenizer_revision="mock-v1",
    )


def _request(
    branch_id: str,
    seed: int,
    rollout_index: int,
    checkpoint: Any,
) -> ProductionRolloutRequest:
    return ProductionRolloutRequest(
        branch_id=branch_id,
        rollout_seed=seed,
        rollout_index=rollout_index,
        checkpoint=checkpoint,
        metadata={"test": True},
    )


def _write_jsonl(path: Any, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_frozen_legacy_reasoning_steps_are_losslessly_normalized() -> None:
    expected = [
        "First line\nwith a continuation",
        "Second step with 'quoted text' and $x^2$",
        "Third step",
    ]
    frozen_numpy_representation = str(np.asarray(expected, dtype=object))

    assert reasoning_steps_list(expected) == expected
    assert reasoning_steps_list(tuple(expected)) == expected
    assert reasoning_steps_list(np.asarray(expected, dtype=object)) == expected
    assert reasoning_steps_list(frozen_numpy_representation) == expected
    assert reasoning_steps_list(repr(expected)) == expected

    with pytest.raises((TypeError, ValueError), match="reasoning_steps"):
        reasoning_steps_list("unquoted free-form text is not a frozen step array")


def test_frozen_train_dev_selection_preserves_roles_and_is_order_independent() -> None:
    buckets = list(SOURCE_IDENTITIES)
    train = [
        _frozen_trace(index, bucket, "teacher_forced_train")
        for index, bucket in enumerate(buckets)
    ]
    dev = [
        _frozen_trace(index + 100, bucket, "teacher_forced_dev")
        for index, bucket in enumerate(buckets)
    ]
    # A non-failure must be documented as excluded rather than silently entering
    # the repairability cohort or perturbing the frozen role assignment.
    train.append(
        _frozen_trace(
            999,
            "crv_arithmetic",
            "teacher_forced_train",
            final_answer_correct=True,
        )
    )

    selected, summary, exclusions = select_frozen_manifest_failures(train, dev)
    reversed_selected, reversed_summary, reversed_exclusions = (
        select_frozen_manifest_failures(list(reversed(train)), list(reversed(dev)))
    )

    assert selected == reversed_selected
    assert summary == reversed_summary
    assert exclusions == reversed_exclusions
    assert summary["new_split_created"] is False
    assert summary["split_counts"] == {"train": 4, "dev": 4}
    assert {row["source_bucket"] for row in selected} == set(buckets)
    assert all(
        row["pipeline_split"]
        == ("train" if row["manifest_role"] == "teacher_forced_train" else "dev")
        for row in selected
    )
    assert all(isinstance(row["reasoning_steps"], list) for row in selected)
    assert [row["reason"] for row in exclusions] == [
        "not_failed_with_visible_error_annotation"
    ]


def test_frozen_train_dev_selection_rejects_role_drift_and_problem_leakage() -> None:
    buckets = list(SOURCE_IDENTITIES)
    train = [
        _frozen_trace(index, bucket, "teacher_forced_train")
        for index, bucket in enumerate(buckets)
    ]
    dev = [
        _frozen_trace(index + 100, bucket, "teacher_forced_dev")
        for index, bucket in enumerate(buckets)
    ]

    wrong_role = deepcopy(train)
    wrong_role[0]["manifest_role"] = "teacher_forced_dev"
    with pytest.raises(ValueError, match="role drift"):
        select_frozen_manifest_failures(wrong_role, dev)

    leaked_dev = deepcopy(dev)
    leaked_dev[0]["problem_group_hash"] = train[0]["problem_group_hash"]
    with pytest.raises(ValueError, match="train/dev leakage"):
        select_frozen_manifest_failures(train, leaked_dev)


def test_frozen_selection_retains_nonempty_unlocalizable_final_answer_for_conservative_audit() -> None:
    buckets = list(SOURCE_IDENTITIES)
    train = [
        _frozen_trace(index, bucket, "teacher_forced_train")
        for index, bucket in enumerate(buckets)
    ]
    dev = [
        _frozen_trace(index + 100, bucket, "teacher_forced_dev")
        for index, bucket in enumerate(buckets)
    ]
    train.append(
        _frozen_trace(
            999,
            "processbench_math",
            "teacher_forced_train",
            final_answer_text="- ...",
            reasoning_steps=[
                "The intermediate expression is - ... but not the result.",
                "First visible error.",
            ],
        )
    )

    selected, summary, exclusions = select_frozen_manifest_failures(train, dev)

    assert len(selected) == 9
    assert "unlocalizable_final_answer_region" not in summary["exclusion_counts"]
    assert not any(
        row["source_trace_id"] == "frozen-teacher_forced_train-999"
        for row in exclusions
    )


def test_authoritative_cohort_opens_only_frozen_train_dev_and_checks_hashes(
    tmp_path: Any,
) -> None:
    source_root = tmp_path / "frozen-source"
    train = [
        _frozen_trace(index, bucket, "teacher_forced_train")
        for index, bucket in enumerate(SOURCE_IDENTITIES)
    ]
    dev = [
        _frozen_trace(index + 100, bucket, "teacher_forced_dev")
        for index, bucket in enumerate(SOURCE_IDENTITIES)
    ]
    hashes = {
        "teacher_forced_train": stable_hash(train),
        "teacher_forced_dev": stable_hash(dev),
    }
    summary = {
        "status": "COMPLETE",
        "configuration_hash": "frozen-source-config-v1",
        "manifest_hashes": hashes,
    }
    source_root.mkdir(parents=True)
    (source_root / "summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    _write_jsonl(source_root / "train.jsonl", train)
    _write_jsonl(source_root / "dev.jsonl", dev)
    # These deliberately invalid decoys make an accidental scan/open fail.
    (source_root / "native_final_test.jsonl").write_text("not-json\n", encoding="utf-8")
    (source_root / "native_development.jsonl").write_text("not-json\n", encoding="utf-8")
    config = {
        "frozen_teacher_forced_manifests": {
            "root": str(source_root),
            "summary_file": "summary.json",
            "train_file": "train.jsonl",
            "dev_file": "dev.jsonl",
            "source_configuration_hash": "frozen-source-config-v1",
            "expected_counts": {
                "teacher_forced_train": len(train),
                "teacher_forced_dev": len(dev),
            },
            "manifest_hashes": hashes,
        },
        "full_teacher_forced_suite": {"first_error_bins": [0, 1, 3, 6]},
    }
    artifact_root = tmp_path / "artifacts"

    selected, details = _load_authoritative_frozen_cohort(config, artifact_root)

    assert len(selected) == len(train) + len(dev)
    access = details["access"]
    assert set(access["files_opened"]) == {
        "summary",
        "teacher_forced_train",
        "teacher_forced_dev",
    }
    assert access["native_configuration_development_opened"] is False
    assert access["native_final_test_manifest_opened"] is False
    assert access["native_or_final_outputs_opened"] is False

    tampered = deepcopy(train)
    tampered[0]["problem_text"] = "tampered after freeze"
    _write_jsonl(source_root / "train.jsonl", tampered)
    with pytest.raises(RuntimeError, match="content hash differs"):
        _load_authoritative_frozen_cohort(config, tmp_path / "tampered-artifacts")


def test_prepared_manifest_validation_rejects_prohibited_source_access(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SAFEPREFIX_RUNS_ROOT", str(tmp_path / "runs"))
    config = {"artifacts_root": "unused", "selected_models": []}
    root = run_root(config, "source-access-guard")
    immutable = root / "immutable_manifests"
    immutable.mkdir(parents=True)
    (immutable / "immutable_protocol_manifest.json").write_text(
        json.dumps(
            {
                "configuration_hash": stable_hash(config),
                "engine_revision": engine_revision(),
                "models": {},
            }
        ),
        encoding="utf-8",
    )
    (immutable / "source_summary.json").write_text("{}", encoding="utf-8")
    (immutable / "source_exclusions.jsonl").write_text("", encoding="utf-8")
    (root / "source_manifest_access_ledger.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "native_configuration_development_opened": False,
                "prompt_pilot_manifest_opened": False,
                "native_final_test_manifest_opened": True,
                "native_or_final_outputs_opened": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="source access|native|final"):
        validate_prepared_manifests(config, run_id="source-access-guard")


@pytest.mark.parametrize("tamper", ["policy", "displacement"])
def test_prepared_manifest_validation_rejects_boundary_audit_tampering(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    monkeypatch.setenv("SAFEPREFIX_RUNS_ROOT", str(tmp_path / "runs"))
    config = {"artifacts_root": "unused", "selected_models": ["model"]}
    configuration_hash = stable_hash(config)
    root = run_root(config, "boundary-audit-guard")
    immutable = root / "immutable_manifests"
    model_root = immutable / "per_model" / "model"
    model_root.mkdir(parents=True)
    completion, ranges = canonical_completion(["Clean step.", "Final answer: 42"])
    alignment = TokenAlignment(
        input_ids=tuple(range(len(completion))),
        offsets=tuple((index, index + 1) for index in range(len(completion))),
        text_length=len(completion),
        method="character_exact",
    )
    audits = source_step_boundary_audits(alignment, ranges)
    prompt_ids = [7, 8]
    offsets = [len(prompt_ids)] + [
        len(prompt_ids) + int(item["token_offset"]) for item in audits
    ]
    checkpoint_audits = [
        {
            "checkpoint_index": 0,
            "checkpoint_kind": "prompt_root",
            "checkpoint_token_offset": len(prompt_ids),
            "source_step_index": None,
            "boundary_policy": "exact_prompt_token_boundary",
            "exact": True,
            "character_displacement": 0,
        }
    ] + [
        {
            **item,
            "checkpoint_index": int(item["step_index"]) + 1,
            "checkpoint_kind": "after_complete_source_step",
            "source_step_index": int(item["step_index"]),
            "checkpoint_token_offset": len(prompt_ids)
            + int(item["token_offset"]),
        }
        for item in audits
    ]
    row = {
        "trace_id": "trace-boundary-audit",
        "source_trace_id": "source-boundary-audit",
        "problem_id": "problem-boundary-audit",
        "source_bucket": "processbench_math",
        "reasoning_steps": ["Clean step.", "Final answer: 42"],
        "first_error_zero_based": 1,
        "prompt_token_ids": prompt_ids,
        "checkpoint_offsets": offsets,
        "eligible_checkpoint_offsets": offsets[:2],
        "source_step_boundary_policy": SOURCE_STEP_BOUNDARY_POLICY,
        "source_step_boundary_audits": audits,
        "checkpoint_boundary_audits": checkpoint_audits,
        "eligible_checkpoint_boundary_audits": checkpoint_audits[:2],
        "answer_region_checkpoint_audit": answer_region_checkpoint_audit(
            completion,
            ranges,
            final_answer_text="42",
            first_error_zero_based=1,
            eligible_step_boundaries=audits,
        ),
    }
    pack = build_execution_packs(
        [row],
        model_key="model",
        base_seed=BASE_SEED,
        configuration_hash=configuration_hash,
        engine_revision=engine_revision(),
        traces_per_pack=1,
    )[0]
    _write_jsonl(model_root / "trace_manifest.jsonl", [row])
    _write_jsonl(model_root / "execution_packs.jsonl", [pack])
    (model_root / "worker_assignments.json").write_text(
        json.dumps({"0": [pack["pack_id"]]}), encoding="utf-8"
    )
    (model_root / "checkpoint_boundary_manifest.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "traces": 1,
                "source_step_boundary_policy": SOURCE_STEP_BOUNDARY_POLICY,
                "answer_region_audit_policy": ANSWER_REGION_AUDIT_POLICY,
                "answer_region_audit_failures": 0,
            }
        ),
        encoding="utf-8",
    )
    (immutable / "immutable_protocol_manifest.json").write_text(
        json.dumps(
            {
                "configuration_hash": configuration_hash,
                "engine_revision": engine_revision(),
                "models": {"model": {"traces": 1, "rollouts": 8}},
            }
        ),
        encoding="utf-8",
    )
    (immutable / "source_summary.json").write_text("{}", encoding="utf-8")
    (immutable / "source_exclusions.jsonl").write_text("", encoding="utf-8")
    (root / "source_manifest_access_ledger.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "files_opened": {
                    "summary": {},
                    "teacher_forced_train": {},
                    "teacher_forced_dev": {},
                },
                "native_configuration_development_opened": False,
                "prompt_pilot_manifest_opened": False,
                "native_final_test_manifest_opened": False,
                "native_or_final_outputs_opened": False,
            }
        ),
        encoding="utf-8",
    )
    assert validate_prepared_manifests(
        config, run_id="boundary-audit-guard"
    )["passed"]

    if tamper == "policy":
        row["source_step_boundary_policy"] = "snap_before"
    else:
        row["source_step_boundary_audits"][0]["character_displacement"] = 99
    _write_jsonl(model_root / "trace_manifest.jsonl", [row])

    with pytest.raises(RuntimeError, match="prepared immutable manifests"):
        validate_prepared_manifests(config, run_id="boundary-audit-guard")


def test_execution_pack_construction_is_input_order_independent() -> None:
    rows = [
        _trace(0, 0),
        _trace(1, 4),
        _trace(2, 2),
        _trace(3, 1),
        _trace(4, 3),
    ]
    kwargs = {
        "model_key": "family_a_small",
        "base_seed": BASE_SEED,
        "configuration_hash": "configuration-hash",
        "engine_revision": "engine-revision",
        "traces_per_pack": 2,
    }
    forward = build_execution_packs(rows, **kwargs)
    reverse = build_execution_packs(list(reversed(rows)), **kwargs)
    shuffled = build_execution_packs([rows[index] for index in (2, 0, 4, 1, 3)], **kwargs)

    assert forward == reverse == shuffled
    assert sum(pack["trace_count"] for pack in forward) == len(rows)
    assert max(pack["trace_count"] for pack in forward) <= 2
    observed_trace_ids = [
        trace_id for pack in forward for trace_id in pack["trace_ids"]
    ]
    assert Counter(observed_trace_ids) == Counter(row["trace_id"] for row in rows)


def test_each_eligible_checkpoint_has_exactly_four_frozen_seed_keys() -> None:
    row = _trace(7, 3)
    keys = logical_rollout_keys(
        row,
        model_key="family_b_large",
        base_seed=BASE_SEED,
    )

    assert len(keys) == 4 * (row["first_error_zero_based"] + 1)
    by_checkpoint = Counter(key.checkpoint_index for key in keys)
    assert by_checkpoint == {0: 4, 1: 4, 2: 4, 3: 4}
    assert len({key.as_tuple() for key in keys}) == len(keys)
    for checkpoint in range(4):
        checkpoint_keys = [
            key for key in keys if key.checkpoint_index == checkpoint
        ]
        assert [key.rollout_index for key in checkpoint_keys] == [0, 1, 2, 3]
        assert [key.rollout_seed for key in checkpoint_keys] == [
            rollout_seed(
                BASE_SEED,
                row["trace_id"],
                checkpoint,
                rollout_index,
            )
            for rollout_index in range(4)
        ]


def test_pack_validation_exposes_duplicate_missing_and_unexpected_records() -> None:
    pack = build_execution_packs(
        [_trace(0, 1)],
        model_key="family_a_small",
        base_seed=BASE_SEED,
        configuration_hash="configuration-hash",
        engine_revision="engine-revision",
        traces_per_pack=1,
    )[0]
    complete = _complete_pack_rows(pack)
    assert validate_pack_rows(pack, complete)["passed"]

    duplicate = complete + [deepcopy(complete[0])]
    duplicate_result = validate_pack_rows(pack, duplicate)
    assert not duplicate_result["passed"]
    assert duplicate_result["duplicate_rows"] == 1

    missing_result = validate_pack_rows(pack, complete[:-1])
    assert not missing_result["passed"]
    assert missing_result["missing_rows"] == 1

    unexpected = deepcopy(complete)
    unexpected[-1]["trace_id"] = "not-in-this-pack"
    unexpected_result = validate_pack_rows(pack, unexpected)
    assert not unexpected_result["passed"]
    assert unexpected_result["missing_rows"] == 1
    assert unexpected_result["unexpected_rows"] == 1

    schema_failure = deepcopy(complete)
    schema_failure[0].pop("generated_token_ids")
    schema_result = validate_pack_rows(pack, schema_failure)
    assert not schema_result["passed"]
    assert schema_result["schema_failure_rows"] == [0]


def test_pack_validation_accepts_parquet_list_array_round_trip(tmp_path: Any) -> None:
    pack = build_execution_packs(
        [_trace(0, 1)],
        model_key="family_a_small",
        base_seed=BASE_SEED,
        configuration_hash="configuration-hash",
        engine_revision="engine-revision",
        traces_per_pack=1,
    )[0]
    path = tmp_path / "rollouts.parquet"
    pd.DataFrame(_complete_pack_rows(pack)).to_parquet(path, index=False)
    reloaded = pd.read_parquet(path).to_dict("records")

    assert isinstance(reloaded[0]["generated_token_ids"], np.ndarray)
    assert validate_pack_rows(pack, reloaded)["passed"]


def test_pack_validation_rejects_semantic_row_corruption() -> None:
    trace = _trace(0, 1)
    trace.update(
        problem_id="problem-for-trace-000",
        source_trace_id="source-trace-000",
        pipeline_split="train",
        eligible_checkpoint_offsets=[17, 18],
    )
    pack = build_execution_packs(
        [trace],
        model_key="family_a_small",
        base_seed=BASE_SEED,
        configuration_hash="configuration-hash",
        engine_revision="engine-revision",
        traces_per_pack=1,
    )[0]
    complete = _complete_pack_rows(pack)
    for row in complete:
        row.update(
            problem_id=trace["problem_id"],
            source_bucket=trace["source_bucket"],
            source_trace_id=trace["source_trace_id"],
            pipeline_split=trace["pipeline_split"],
            first_visible_error_zero_based=trace["first_error_zero_based"],
            execution_freeze_digest="freeze-v1",
            model_execution_settings_hash="settings-v1",
            parser_sha256="parser-v1",
            verifier_sha256="verifier-v1",
        )
    validation_kwargs = {
        "trace_rows": {trace["trace_id"]: trace},
        "expected_freeze_digest": "freeze-v1",
        "expected_settings_hash": "settings-v1",
        "expected_parser_sha256": "parser-v1",
        "expected_verifier_sha256": "verifier-v1",
    }
    assert validate_pack_rows(pack, complete, **validation_kwargs)["passed"]

    corruptions = {
        "configuration_hash": lambda row: row.update(configuration_hash="wrong"),
        "engine_revision": lambda row: row.update(engine_revision="wrong"),
        "generated_token_count": lambda row: row.update(generated_token_count=99),
        "binary_outcome": lambda row: row.update(binary_outcome=not row["verifier_pass"]),
        "truncation_flag": lambda row: row.update(stop_reason="length", truncation_flag=False),
        "invalid_success": lambda row: row.update(
            parser_status="failure", verifier_pass=True, binary_outcome=True
        ),
        "execution_freeze_digest": lambda row: row.update(execution_freeze_digest="wrong"),
        "model_execution_settings_hash": lambda row: row.update(
            model_execution_settings_hash="wrong"
        ),
        "parser_sha256": lambda row: row.update(parser_sha256="wrong"),
        "verifier_sha256": lambda row: row.update(verifier_sha256="wrong"),
        "checkpoint_token_offset": lambda row: row.update(checkpoint_token_offset=999),
        "problem_id": lambda row: row.update(problem_id="wrong"),
        "source_trace_id": lambda row: row.update(source_trace_id="wrong"),
        "pipeline_split": lambda row: row.update(pipeline_split="dev"),
        "first_visible_error_zero_based": lambda row: row.update(
            first_visible_error_zero_based=999
        ),
    }
    for expected_reason, corrupt in corruptions.items():
        candidate = deepcopy(complete)
        corrupt(candidate[0])
        result = validate_pack_rows(pack, candidate, **validation_kwargs)
        observed_reasons = {
            reason
            for failure in result["semantic_failure_rows"]
            for reason in failure["reasons"]
        }
        assert not result["passed"], expected_reason
        assert expected_reason in observed_reasons


def test_pack_validation_rejects_trace_manifest_checkpoint_coverage_drift() -> None:
    trace = _trace(0, 1)
    trace.update(
        problem_id="problem-for-trace-000",
        source_trace_id="source-trace-000",
        pipeline_split="train",
        eligible_checkpoint_offsets=[17, 18],
    )
    pack = build_execution_packs(
        [trace],
        model_key="family_a_small",
        base_seed=BASE_SEED,
        configuration_hash="configuration-hash",
        engine_revision="engine-revision",
        traces_per_pack=1,
    )[0]
    rows = _complete_pack_rows(pack)

    # Simulate a trace manifest that now claims a third eligible checkpoint. A
    # row-only validator must not accept a pack whose frozen logical membership
    # no longer covers the authoritative trace checkpoint set.
    drifted_trace = {
        **trace,
        "first_error_zero_based": 2,
        "reasoning_steps": ["clean 0", "clean 1", "error", "later"],
        "eligible_checkpoint_offsets": [17, 18, 19],
    }
    for row in rows:
        row.update(
            problem_id=drifted_trace["problem_id"],
            source_bucket=drifted_trace["source_bucket"],
            source_trace_id=drifted_trace["source_trace_id"],
            pipeline_split=drifted_trace["pipeline_split"],
            first_visible_error_zero_based=2,
        )

    result = validate_pack_rows(
        pack,
        rows,
        trace_rows={trace["trace_id"]: drifted_trace},
    )
    assert not result["passed"]


def test_checkpoint_aggregation_preserves_raw_outcomes_without_discretization() -> None:
    pack = build_execution_packs(
        [_trace(0, 0)],
        model_key="family_a_small",
        base_seed=BASE_SEED,
        configuration_hash="configuration-hash",
        engine_revision="engine-revision",
        traces_per_pack=1,
    )[0]
    rows = _complete_pack_rows(pack)
    aggregate = aggregate_checkpoint_outcomes(list(reversed(rows)))

    assert len(aggregate) == 1
    checkpoint = aggregate[0]
    assert checkpoint["binary_outcomes"] == [True, False, True, False]
    assert checkpoint["success_count"] == 2
    assert checkpoint["trial_count"] == 4
    assert checkpoint["repairability_discretized"] is False
    assert checkpoint["rollout_seeds"] == [
        rollout_seed(BASE_SEED, "trace-000", 0, rollout_index)
        for rollout_index in range(4)
    ]
    assert "repairability_label" not in checkpoint
    assert "first_unsafe_label" not in checkpoint

    with pytest.raises(AssertionError, match="does not contain rollout indices 0..3"):
        aggregate_checkpoint_outcomes(rows[:-1])


def test_chunked_teacher_forcing_preserves_requested_states_on_toy_model() -> None:
    model = ToyCausalModel(vocab_size=128).eval()
    ids = [10, 11, 12, 13, 14, 15, 16]
    kwargs = {
        "prompt_count": 2,
        "selected_layers": [0, 2, -1],
        "selected_token_offsets": [1, 3, 6],
        "selected_checkpoint_offsets": [2, 4, 7],
    }
    reference = teacher_force_token_ids(model, ids, **kwargs)
    chunked = teacher_force_token_ids_chunked(model, ids, chunk_size=3, **kwargs)

    assert torch.equal(reference.token_ids, chunked.token_ids)
    assert torch.allclose(
        reference.token_log_probabilities,
        chunked.token_log_probabilities,
        atol=1e-6,
    )
    assert set(chunked.checkpoint_next_token_logits) == {2, 4, 7}
    for offset in (2, 4, 7):
        assert torch.equal(
            reference.checkpoint_next_token_logits[offset],
            chunked.checkpoint_next_token_logits[offset],
        )
    for layer in (0, 2, -1):
        for offset in (1, 3, 6):
            assert torch.equal(
                reference.selected_hidden_states[layer][offset],
                chunked.selected_hidden_states[layer][offset],
            )


def test_readonly_checkpoint_views_share_source_but_clones_are_independent() -> None:
    model = ToyCausalModel(vocab_size=128).eval()
    token_ids = torch.tensor([[10, 11, 12, 13, 14, 15]], dtype=torch.long)
    with torch.inference_mode():
        output = model(
            input_ids=token_ids,
            attention_mask=torch.ones_like(token_ids),
            use_cache=True,
            return_dict=True,
        )
    source_key, source_value = cache_to_legacy(output.past_key_values)[0][:2]
    source_key_before = source_key.clone()
    source_value_before = source_value.clone()

    view = capture_cache_checkpoint(
        output.past_key_values,
        output.logits[:, 3],
        token_ids,
        4,
        model_id="mock/toy",
        model_revision="mock-v1",
        tokenizer_id="mock/character",
        tokenizer_revision="mock-v1",
        clone_cache=False,
    )
    view_key, view_value = cache_to_legacy(view.past_key_values)[0][:2]

    assert view.cache_storage == "readonly_teacher_forced_view"
    assert view_key.shape[-2] == view_value.shape[-2] == 4
    assert view_key.untyped_storage().data_ptr() == source_key.untyped_storage().data_ptr()
    assert view_value.untyped_storage().data_ptr() == source_value.untyped_storage().data_ptr()
    assert torch.equal(view_key, source_key[..., :4, :])
    assert torch.equal(view_value, source_value[..., :4, :])

    independent = clone_cache_checkpoint(view)
    clone_key, clone_value = cache_to_legacy(independent.past_key_values)[0][:2]
    assert independent.cache_storage == "independent"
    assert clone_key.untyped_storage().data_ptr() != source_key.untyped_storage().data_ptr()
    assert clone_value.untyped_storage().data_ptr() != source_value.untyped_storage().data_ptr()

    clone_key.add_(1000)
    clone_value.sub_(1000)
    assert torch.equal(source_key, source_key_before)
    assert torch.equal(source_value, source_value_before)
    assert torch.equal(view_key, source_key_before[..., :4, :])
    assert torch.equal(view_value, source_value_before[..., :4, :])


def test_heterogeneous_cache_collation_is_right_aligned_and_independent() -> None:
    first_key = torch.arange(8, dtype=torch.float32).reshape(1, 1, 4, 2)
    first_cache = ((first_key, first_key + 100.0),)
    second_key = torch.arange(24, dtype=torch.float32).reshape(2, 1, 6, 2)
    second_cache = ((second_key, second_key + 200.0),)
    expected_first = first_key.clone()
    expected_second_tail = second_key[1:2, ..., -2:, :].clone()

    collated, mask, lengths = collate_right_aligned_cache_rows(
        [
            CacheRowSource(first_cache, row=0, valid_length=4),
            CacheRowSource(second_cache, row=1, valid_length=2),
        ]
    )
    key, value = cache_to_legacy(collated)[0][:2]

    assert lengths == [4, 2]
    assert key.shape == (2, 1, 4, 2)
    assert value.shape == (2, 1, 4, 2)
    assert mask.tolist() == [[1, 1, 1, 1], [0, 0, 1, 1]]
    assert torch.equal(key[0:1], expected_first)
    assert torch.count_nonzero(key[1:2, ..., :2, :]) == 0
    assert torch.equal(key[1:2, ..., -2:, :], expected_second_tail)
    assert key.dtype == first_key.dtype

    key.add_(1000.0)
    assert torch.equal(first_key, expected_first)
    assert torch.equal(second_key[1:2, ..., -2:, :], expected_second_tail)

    with pytest.raises(ValueError, match="at least one cache row"):
        collate_right_aligned_cache_rows([])
    with pytest.raises(ValueError, match="at least one valid token"):
        collate_right_aligned_cache_rows(
            [CacheRowSource(first_cache, row=0, valid_length=0)]
        )


def test_static_wave_cache_preserves_compacted_prefix_tensors() -> None:
    from transformers import LlamaConfig

    model = ToyCausalModel().eval()
    model.config = LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    key = torch.arange(24, dtype=torch.float32).reshape(2, 1, 6, 2)
    value = key + 100.0
    static = preallocate_decode_cache(model, ((key, value),), max_cache_len=10)
    observed_key, observed_value = cache_to_legacy(static)[0][:2]

    assert torch.equal(observed_key[..., :6, :], key)
    assert torch.equal(observed_value[..., :6, :], value)
    assert torch.count_nonzero(observed_key[..., 6:, :]) == 0
    assert torch.count_nonzero(observed_value[..., 6:, :]) == 0


def test_single_softmax_top_p_sampler_matches_frozen_sampler() -> None:
    generation = {"temperature": 0.7, "top_p": 0.95, "top_k": 0}
    for trial in range(8):
        logits = torch.randn(6, 257, generator=torch.Generator().manual_seed(trial))
        seeds = [1000 + 6 * trial + index for index in range(6)]
        reference, _ = select_tokens_from_saved_logits(logits, generation, seeds)
        uniforms = torch.tensor([_uniform(seed, 0) for seed in seeds])
        optimized = _sample_with_uniforms(logits, generation, uniforms)
        assert torch.equal(optimized, reference)


def test_adaptive_nucleus_sampler_matches_frozen_sampler_without_full_sort() -> None:
    vocabulary = 8192
    logits = torch.linspace(250.0, -250.0, vocabulary).repeat(4, 1)
    logits += torch.tensor([0.0, 0.01, -0.02, 0.03])[:, None]
    generation = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 0,
        "nucleus_initial_top_k": 64,
    }
    seeds = [8101, 8102, 8103, 8104]
    reference, _ = select_tokens_from_saved_logits(logits, generation, seeds)
    uniforms = torch.tensor([_uniform(seed, 0) for seed in seeds])
    optimized = _sample_with_uniforms(logits, generation, uniforms)

    assert torch.equal(optimized, reference)
    probabilities = _exact_nucleus_probabilities(
        logits / generation["temperature"],
        top_p=generation["top_p"],
        initial_top_k=generation["nucleus_initial_top_k"],
    )
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(4), atol=1e-6)
    assert int((probabilities > 0).sum(dim=-1).max()) < 64


def test_adaptive_nucleus_sampler_falls_back_safely_on_boundary_ties() -> None:
    logits = torch.zeros(3, 257)
    generation = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 0,
        "nucleus_initial_top_k": 16,
    }
    seeds = [9101, 9102, 9103]
    reference, _ = select_tokens_from_saved_logits(logits, generation, seeds)
    uniforms = torch.tensor([_uniform(seed, 0) for seed in seeds])
    optimized = _sample_with_uniforms(logits, generation, uniforms)

    assert torch.equal(optimized, reference)


def test_validation_pack_selection_is_deterministic_and_source_stratified() -> None:
    rows: list[dict[str, Any]] = []
    for source_index, source in enumerate(SOURCE_IDENTITIES):
        for variant in range(3):
            first_error = 1 + ((source_index + variant) % 3)
            row = _trace(source_index * 10 + variant, first_error)
            row.update(
                source_bucket=source,
                eligible_checkpoint_offsets=[
                    11 + source_index + 7 * checkpoint
                    for checkpoint in range(first_error + 1)
                ],
                full_token_count=100 + 50 * variant + 10 * source_index,
            )
            rows.append(row)
    kwargs = {
        "model_key": "family_a_small",
        "count": 4,
        "mode": "benchmark",
        "configuration_hash": "configuration-hash",
        "revision": "engine-revision",
        "base_seed": BASE_SEED,
    }

    forward = select_validation_pack(rows, **kwargs)
    reverse = select_validation_pack(list(reversed(rows)), **kwargs)
    shuffled = select_validation_pack(rows[::2] + rows[1::2], **kwargs)

    assert forward == reverse == shuffled
    selected_ids = set(forward["trace_ids"])
    selected_sources = {
        str(row["source_bucket"])
        for row in rows
        if str(row["trace_id"]) in selected_ids
    }
    assert selected_sources == set(SOURCE_IDENTITIES)
    assert forward["trace_count"] == 4
    assert forward["rollout_count"] == sum(
        4 * (int(row["first_error_zero_based"]) + 1)
        for row in rows
        if str(row["trace_id"]) in selected_ids
    )


def test_production_decode_is_branch_order_invariant_with_heterogeneous_prefixes() -> None:
    model = StochasticToyCausalModel(vocab_size=64).eval()
    tokenizer = CharacterTokenizer()
    checkpoints = [
        _checkpoint(model, [10, 11, 12]),
        _checkpoint(model, [20, 21, 22, 23, 24]),
        _checkpoint(model, [30, 31, 32, 33]),
        _checkpoint(model, [40, 41, 42, 43, 44, 45]),
    ]
    requests = [
        _request(f"branch-{index}", 700 + index, index, checkpoint)
        for index, checkpoint in enumerate(checkpoints)
    ]
    generation = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 0,
        "max_new_tokens": 7,
        "stop_token_ids": [],
    }

    forward, forward_metrics = decode_execution_pack(
        model,
        tokenizer,
        requests,
        generation=generation,
        maximum_batch_size=3,
        compaction_quantum=2,
        maximum_context_length=64,
    )
    reverse, reverse_metrics = decode_execution_pack(
        model,
        tokenizer,
        list(reversed(requests)),
        generation=generation,
        maximum_batch_size=3,
        compaction_quantum=2,
        maximum_context_length=64,
    )

    expected = {
        result.request.branch_id: (
            result.token_ids,
            result.text,
            result.stop_reason,
        )
        for result in forward
    }
    observed = {
        result.request.branch_id: (
            result.token_ids,
            result.text,
            result.stop_reason,
        )
        for result in reverse
    }
    assert observed == expected
    assert [result.request.branch_id for result in forward] == [
        request.branch_id for request in requests
    ]
    assert [result.request.branch_id for result in reverse] == [
        request.branch_id for request in reversed(requests)
    ]
    assert forward_metrics.useful_output_tokens == reverse_metrics.useful_output_tokens
    assert forward_metrics.forwarded_row_steps == reverse_metrics.forwarded_row_steps
    assert forward_metrics.completed_branches == len(requests)


def test_production_decode_compacts_after_eos_and_refills_pending_branches() -> None:
    model = ToyCausalModel(vocab_size=64).eval()
    tokenizer = CharacterTokenizer()
    tokenizer.eos_token_id = 14
    requests = [
        _request("eos-long-prefix", 11, 0, _checkpoint(model, [7, 8, 9, 10, 11, 12, 13])),
        _request("length-a", 12, 1, _checkpoint(model, [16, 17, 18, 19, 20, 21])),
        _request("eos-refill", 13, 2, _checkpoint(model, [9, 10, 11, 12, 13])),
        _request("length-b", 14, 3, _checkpoint(model, [27, 28, 29, 30])),
    ]
    source_cache_snapshots = {
        request.branch_id: tuple(
            tuple(value.clone() if isinstance(value, torch.Tensor) else value for value in layer)
            for layer in cache_to_legacy(request.checkpoint.past_key_values)
        )
        for request in requests
    }

    results, metrics = decode_execution_pack(
        model,
        tokenizer,
        requests,
        generation={
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": 3,
            "stop_token_ids": [],
        },
        maximum_batch_size=2,
        compaction_quantum=2,
        maximum_context_length=64,
    )
    by_id = {result.request.branch_id: result for result in results}

    assert by_id["eos-long-prefix"].token_ids == [14]
    assert by_id["eos-long-prefix"].stop_reason == "eos"
    assert by_id["eos-refill"].token_ids == [14]
    assert by_id["eos-refill"].stop_reason == "eos"
    assert len(by_id["length-a"].token_ids) == 3
    assert by_id["length-a"].stop_reason == "length"
    assert len(by_id["length-b"].token_ids) == 3
    assert by_id["length-b"].stop_reason == "length"
    assert metrics.completed_branches == 4
    assert metrics.admitted_branches == 4
    assert metrics.compaction_refill_events >= 2
    assert metrics.useful_output_tokens == 8
    assert metrics.forwarded_row_steps > metrics.useful_output_tokens

    for request in requests:
        original = source_cache_snapshots[request.branch_id]
        observed = cache_to_legacy(request.checkpoint.past_key_values)
        for original_layer, observed_layer in zip(original, observed):
            assert torch.equal(original_layer[0], observed_layer[0])
            assert torch.equal(original_layer[1], observed_layer[1])


def test_production_decode_rejects_duplicate_branch_ids_and_context_overflow() -> None:
    model = ToyCausalModel(vocab_size=64).eval()
    tokenizer = CharacterTokenizer()
    checkpoint = _checkpoint(model, [10, 11, 12])
    duplicate = [
        _request("same", 1, 0, checkpoint),
        _request("same", 2, 1, checkpoint),
    ]
    generation = {"temperature": 0.0, "max_new_tokens": 2}

    with pytest.raises(ValueError, match="duplicate branch IDs"):
        decode_execution_pack(
            model,
            tokenizer,
            duplicate,
            generation=generation,
            maximum_batch_size=2,
            compaction_quantum=1,
            maximum_context_length=64,
        )

    with pytest.raises(ValueError, match="exceeds configured context"):
        decode_execution_pack(
            model,
            tokenizer,
            [_request("overflow", 1, 0, checkpoint)],
            generation={"temperature": 0.0, "max_new_tokens": 62},
            maximum_batch_size=1,
            compaction_quantum=1,
            maximum_context_length=64,
        )
