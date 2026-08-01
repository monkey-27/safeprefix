import pytest

from safeprefix.data.common import source_trace_identity
from safeprefix.data.splits import assign_problem_splits, assert_problem_disjoint, problem_group_key
from safeprefix.manifests import NormalizedTrace


def test_same_problem_never_crosses_splits() -> None:
    rows = [
        {"problem_text": "Compute  2 + 2", "trace": "a"},
        {"problem_text": " compute 2 + 2 ", "trace": "b"},
        {"problem_text": "Compute 3 + 3", "trace": "c"},
    ]
    assignments = assign_problem_splits(rows, {"train": 0.7, "test": 0.3}, seed=7)
    assert assignments[problem_group_key(rows[0])] == assignments[problem_group_key(rows[1])]


def test_leakage_assertion_fails_loudly() -> None:
    rows = [
        {"problem_text": "same", "split": "train"},
        {"problem_text": " same ", "split": "test"},
    ]
    with pytest.raises(ValueError, match="leakage"):
        assert_problem_disjoint(rows)


def test_problem_group_and_source_trace_identity_are_distinct() -> None:
    common = dict(
        problem_id="p", source_dataset="d", source_subset="s",
        source_generator="g", problem_text="same problem", final_answer_text="0",
        reference_answer=1, first_error_index=0, index_base="zero",
        final_answer_correct=False, metadata={},
    )
    first = NormalizedTrace(reasoning_steps=["wrong trace one"], **common)
    second = NormalizedTrace(reasoning_steps=["wrong trace two"], **common)
    assert problem_group_key(first.to_dict()) == problem_group_key(second.to_dict())
    assert source_trace_identity(first) != source_trace_identity(second)
