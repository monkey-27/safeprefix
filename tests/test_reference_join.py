from safeprefix.data.reference_join import (
    build_reference_index,
    extract_reference_answer,
    join_processbench_references,
    source_problem_key,
)
from safeprefix.manifests import decode_reference_answer


def test_reference_answer_extractors() -> None:
    assert extract_reference_answer("gsm8k", {"answer": "work\n#### 1,234"}) == "1234"
    assert extract_reference_answer("math", {"solution": "Thus \\boxed{7}."}) == "7"
    assert extract_reference_answer("olympiadbench", {"final_answer": ["x"]}) == "x"
    assert extract_reference_answer("omnimath", {"answer": "42"}) == "42"


def test_reference_index_rejects_conflicting_exact_problem_answers() -> None:
    index, conflicts = build_reference_index("omnimath", [
        {"problem": "Same problem", "answer": "1"},
        {"problem": "Same   problem", "answer": "2"},
    ])
    key = source_problem_key("Same problem")
    assert key not in index
    assert key in conflicts


def test_processbench_reference_join_is_exact_and_arrow_safe() -> None:
    index, conflicts = build_reference_index("math", [
        {"problem": "Compute x.", "solution": "We get \\boxed{5}."},
    ])
    assert not conflicts
    joined, report = join_processbench_references([
        {
            "problem_id": "matched",
            "source_dataset": "Qwen/ProcessBench",
            "source_subset": "math",
            "problem_text": "Compute   x.",
            "reference_answer": None,
        },
        {
            "problem_id": "unmatched",
            "source_dataset": "Qwen/ProcessBench",
            "source_subset": "math",
            "problem_text": "Different problem",
            "reference_answer": None,
        },
    ], {"math": index})
    assert decode_reference_answer(joined[0]["reference_answer"]) == "5"
    assert joined[1]["reference_answer"] is None
    assert report["matched"] == 1
    assert report["unmatched_count"] == 1
    assert report["fuzzy_assignments"] == 0
