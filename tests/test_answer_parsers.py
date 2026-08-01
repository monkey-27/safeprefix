from safeprefix.parsing.answer_parsers import parse_answer_region, parse_is_acceptable_for_finish


def test_last_nested_boxed_answer_wins() -> None:
    text = r"Reasoning. \boxed{old} More. \boxed{\frac{3}{4}}"
    parsed = parse_answer_region(text)
    assert parsed.success
    assert parsed.method == "last_boxed"
    assert parsed.parsed_answer == r"\frac{3}{4}"
    assert text[parsed.char_start : parsed.char_end] == r"\boxed{\frac{3}{4}}"


def test_answer_fallback_order() -> None:
    marked = parse_answer_region("Work.\nFinal answer: 17")
    assert marked.method == "final_answer_marker"
    assert marked.parsed_answer == "17"
    isolated = parse_answer_region("Work finished.\n\n42")
    assert isolated.method == "isolated_answer_line"
    assert isolated.parsed_answer == "42"


def test_no_answer_is_explicit_failure() -> None:
    parsed = parse_answer_region("Only unfinished reasoning prose.")
    assert not parsed.success
    assert parsed.failure_reason


def test_multiline_nested_box_preserves_exact_source_region() -> None:
    text = "Work.\n" + r"\boxed{\frac{1}{2} +" + "\n" + r"\sqrt{3}}"
    parsed = parse_answer_region(text)
    assert parsed.success
    assert parsed.parsed_answer == r"\frac{1}{2} +" + "\n" + r"\sqrt{3}"
    assert text[parsed.char_start : parsed.char_end] == r"\boxed{\frac{1}{2} +" + "\n" + r"\sqrt{3}}"
    assert text[parsed.content_char_start : parsed.content_char_end] == parsed.parsed_answer
    assert parsed.confidence == "high"


def test_malformed_box_falls_back_only_when_another_rule_is_valid() -> None:
    failed = parse_answer_region(r"Work. \boxed{unfinished")
    assert not failed.success
    recovered = parse_answer_region("Work. " + r"\boxed{unfinished" + "\nFinal answer: 12")
    assert recovered.method == "final_answer_marker"
    assert recovered.parsed_answer == "12"


def test_last_complete_box_wins_even_after_an_unclosed_box() -> None:
    text = r"Intermediate \boxed{2}. Final \boxed{3}. Broken \boxed{x"
    parsed = parse_answer_region(text)
    assert parsed.method == "last_boxed"
    assert parsed.parsed_answer == "3"


def test_final_answer_marker_can_use_next_line() -> None:
    parsed = parse_answer_region("Work.\nFinal answer:\n$ a = -\\frac{3}{2} $")
    assert parsed.method == "final_answer_marker"
    assert parsed.parsed_answer == r"-\frac{3}{2}"
    assert parsed.rule_priority == 2


def test_assertive_final_prose_extracts_exact_terminal_scalar() -> None:
    text = "Work.\nTherefore, the combined amount is $50,000 per month."
    parsed = parse_answer_region(text)
    assert parsed.method == "final_assertion"
    assert parsed.parsed_answer == "$50,000"
    assert text[parsed.content_char_start : parsed.content_char_end] == "$50,000"


def test_parser_does_not_use_reference_correctness() -> None:
    text = "Work.\nAnswer: 7"
    wrong_reference = "9000"
    parsed = parse_answer_region(text, dataset_parser=lambda _: (wrong_reference, (0, 1)))
    assert parsed.parsed_answer == "7"


def test_length_limited_weak_fallback_is_not_accepted_as_completion() -> None:
    isolated = parse_answer_region("unfinished work\n$x = 4$")
    assert isolated.success
    assert not parse_is_acceptable_for_finish(isolated, "length")
    boxed = parse_answer_region(r"work \boxed{4}")
    assert parse_is_acceptable_for_finish(boxed, "length")
