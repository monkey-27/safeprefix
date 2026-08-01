from safeprefix.phase_a_correction import (
    classify_original_answer_failure,
    complete_boxed_regions,
    parser_development_split,
    select_blinded_audit,
)


def _row(index: int, *, model: str = "m0", prompt: str = "P0", parsed: bool = False) -> dict:
    return {
        "problem_id": f"p{index}",
        "model_name": model,
        "prompt_condition": prompt,
        "generation_seed": 7,
        "source_dataset": "Qwen/ProcessBench",
        "source_subset": ["gsm8k", "math", "olympiadbench", "omnimath"][index % 4],
        "finish_reason": "eos",
        "completion_text": r"Reasoning. \boxed{4}" if parsed else "Reasoning without an answer.",
        "completion_token_ids": list(range(10 + index)),
        "answer_parser": {"success": parsed, "method": "last_boxed" if parsed else "none"},
        "verifier_pass": parsed,
    }


def test_complete_box_detection_handles_nested_and_multiline_content() -> None:
    text = "work\n" + r"\boxed{\frac{1}{2} +" + "\n" + r"\sqrt{3}}"
    regions = complete_boxed_regions(text)
    assert len(regions) == 1
    assert regions[0][2] == r"\frac{1}{2} +" + "\n" + r"\sqrt{3}"


def test_original_failure_classifier_does_not_use_reference_correctness() -> None:
    row = _row(1)
    row["completion_text"] = "Work.\n\nFinal answer: 19"
    row["reference_answer"] = "not 19"
    result = classify_original_answer_failure(row)
    assert result["correction_bucket"] == "A_complete_answer_recoverable"
    assert result["answer_primary_category"] == "answer written after Final answer:"


def test_length_limit_is_classified_as_genuine_termination_without_answer() -> None:
    row = _row(2)
    row["finish_reason"] = "length"
    result = classify_original_answer_failure(row)
    assert result["answer_primary_category"] == "output-length limit reached"
    assert result["correction_bucket"] == "B_genuinely_ended_before_complete_answer"


def test_parser_split_is_disjoint_and_deterministic() -> None:
    rows = [_row(index) for index in range(19)]
    first = parser_development_split(rows, 17)
    second = parser_development_split(list(reversed(rows)), 17)
    assert first == second
    assert set(first[0]).isdisjoint(first[1])
    assert len(first[0]) + len(first[1]) == len(rows)


def test_blinded_audit_has_exact_model_and_prompt_margins() -> None:
    rows = []
    models = ["m0", "m1", "m2", "m3"]
    prompts = ["P0", "P1", "P2"]
    for model in models:
        for prompt in prompts:
            rows.extend(_row(len(rows) + index, model=model, prompt=prompt, parsed=index % 4 != 0) for index in range(12))
    annotations, manifest = select_blinded_audit(rows, 120, 31)
    assert len(annotations) == len(manifest) == 120
    assert all("model_name" not in row and "prompt_condition" not in row for row in annotations)
    assert {model: sum(row["model_name"] == model for row in manifest) for model in models} == {model: 30 for model in models}
    assert {prompt: sum(row["prompt_condition"] == prompt for row in manifest) for prompt in prompts} == {prompt: 40 for prompt in prompts}
