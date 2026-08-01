from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from safeprefix.config import load_config
from safeprefix.prompting.chat_format import render_chat_prompt
from safeprefix.teacher_forced_completion import (
    completion_worker_allocation,
    completed_integrity_status_valid,
    feature_reuse_equivalence,
    safety_label_sequence,
    validate_completed_aggregate_counts,
)


ROOT = Path(__file__).resolve().parents[1]


def test_visible_safety_label_convention() -> None:
    labels, mask = safety_label_sequence(
        checkpoint_count=5, terminal_correct=True, first_error_zero_based=None
    )
    assert labels == [True] * 5 and mask == [True] * 5

    labels, mask = safety_label_sequence(
        checkpoint_count=5, terminal_correct=False, first_error_zero_based=1
    )
    assert labels == [True, True, False, False, False]
    assert mask == [True] * 5

    labels, mask = safety_label_sequence(
        checkpoint_count=5, terminal_correct=False, first_error_zero_based=None
    )
    assert labels == [None] * 5 and mask == [False] * 5


def test_completed_run_integrity_status_is_the_frozen_terminal_value() -> None:
    # The completed production run writes the same terminal status to both its
    # summary and integrity report; reuse validation must not expect the older
    # intermediate string "PASS".
    final = {"status": "INTEGRITY_VALIDATED", "native_final_test_access_count": 0}
    integrity = {"status": "INTEGRITY_VALIDATED"}
    assert completed_integrity_status_valid(final, integrity)
    assert not completed_integrity_status_valid(final, {"status": "PASS"})
    assert not completed_integrity_status_valid(
        {**final, "native_final_test_access_count": 1}, integrity
    )


def test_completed_aggregate_uses_frozen_trial_count_schema() -> None:
    aggregate = pd.DataFrame(
        {
            "trace_id": [f"trace-{index}" for index in range(3895)],
            "checkpoint_index": [0] * 3895,
            "trial_count": [4] * 3895,
        }
    )
    validate_completed_aggregate_counts(aggregate)
    with pytest.raises(RuntimeError, match="schema"):
        validate_completed_aggregate_counts(aggregate.rename(columns={"trial_count": "rollout_count"}))
    aggregate.loc[0, "trial_count"] = 3
    with pytest.raises(RuntimeError, match="count"):
        validate_completed_aggregate_counts(aggregate)


def test_feature_reuse_requires_exact_values_but_recomputation_is_valid() -> None:
    import torch

    old = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    exact = feature_reuse_equivalence(old.clone(), old)
    assert exact["shape_equal"] and exact["reuse_permitted"]
    assert exact["action"] == "reuse"

    changed = feature_reuse_equivalence(
        torch.tensor([[1.0, 3.0]], dtype=torch.bfloat16), old
    )
    assert changed["shape_equal"] and not changed["reuse_permitted"]
    assert changed["max_abs_difference"] == 1.0
    assert changed["action"] == "recompute_all_safety_features"

    wrong_shape = feature_reuse_equivalence(torch.ones(2, 2), old)
    assert not wrong_shape["shape_equal"]
    assert not wrong_shape["reuse_permitted"]


def test_completion_gpu_allocation_is_frozen_to_measured_work_ratio() -> None:
    config = load_config(ROOT / "configs/teacher_forced_completion.yaml").data
    allocation = completion_worker_allocation(config)
    assert allocation == {
        "family_a_small": 5,
        "family_a_large": 5,
        "family_b_small": 10,
        "family_b_large": 20,
    }
    assert sum(allocation.values()) == 40
    assert config["models"]["family_b_small"]["chat_template_kwargs"] == {
        "date_string": "26 Jul 2026"
    }
    assert config["models"]["family_b_large"]["chat_template_kwargs"] == {
        "date_string": "26 Jul 2024"
    }


def test_chat_template_receives_frozen_date_string() -> None:
    class RecordingTokenizer:
        chat_template = "native-template"

        def __init__(self) -> None:
            self.kwargs = None

        def apply_chat_template(self, messages, **kwargs):
            self.kwargs = kwargs
            return f"date={kwargs['date_string']} messages={len(messages)}"

    tokenizer = RecordingTokenizer()
    rendered = render_chat_prompt(
        tokenizer,
        "problem",
        template_kwargs={"date_string": "26 Jul 2024"},
    )
    assert rendered == "date=26 Jul 2024 messages=1"
    assert tokenizer.kwargs == {
        "tokenize": False,
        "add_generation_prompt": True,
        "date_string": "26 Jul 2024",
    }
