"""The three preregistered visible-reasoning prompt conditions."""

from __future__ import annotations

PROMPT_CONDITIONS = {
    "P0": "Solve the problem step by step. Put your final answer in \\boxed{}.",
    "P1": (
        "Solve the problem step by step. Put each major reasoning step in a separate "
        "paragraph. Put your final answer in \\boxed{}."
    ),
    "P2": "Solve the problem in numbered reasoning steps. Put your final answer in \\boxed{}.",
}


def problem_instruction(problem: str, condition: str) -> str:
    if condition not in PROMPT_CONDITIONS:
        raise KeyError(f"unknown prompt condition: {condition}")
    return f"{PROMPT_CONDITIONS[condition]}\n\n{problem.strip()}"
