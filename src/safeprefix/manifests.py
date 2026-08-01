"""Typed row schemas and artifact validation for SafePrefix stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Iterable, Mapping


REFERENCE_ANSWER_JSON_PREFIX = "__safeprefix_reference_json__:"


def encode_reference_answer(value: Any) -> str:
    """Encode a polymorphic answer into one lossless Arrow-safe string column."""
    return REFERENCE_ANSWER_JSON_PREFIX + json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def decode_reference_answer(value: Any) -> Any:
    """Decode SafePrefix's tagged representation while accepting legacy rows."""
    if isinstance(value, str) and value.startswith(REFERENCE_ANSWER_JSON_PREFIX):
        return json.loads(value[len(REFERENCE_ANSWER_JSON_PREFIX) :])
    return value


@dataclass(frozen=True)
class NormalizedTrace:
    problem_id: str
    source_dataset: str
    source_subset: str
    source_generator: str | None
    problem_text: str
    reasoning_steps: list[str]
    final_answer_text: str
    reference_answer: Any
    first_error_index: int | None
    index_base: str | None
    final_answer_correct: bool | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.problem_id or not self.problem_text:
            raise ValueError("problem_id and problem_text must be nonempty")
        if not isinstance(self.reasoning_steps, list) or any(not isinstance(x, str) for x in self.reasoning_steps):
            raise TypeError("reasoning_steps must be a list of strings")
        if self.index_base not in {None, "zero", "one"}:
            raise ValueError("index_base must be zero, one, or null")
        if self.first_error_index is not None:
            zero = self.first_error_index - int(self.index_base == "one")
            if zero < 0 or zero >= len(self.reasoning_steps):
                raise ValueError("first_error_index is outside reasoning_steps")

    @property
    def first_error_zero_based(self) -> int | None:
        if self.first_error_index is None:
            return None
        return self.first_error_index - int(self.index_base == "one")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RolloutRecord:
    model_id: str
    problem_id: str
    trace_id: str
    checkpoint_index: int
    checkpoint_token_offset: int
    rollout_seed: int
    generated_token_count: int
    generated_text: str
    parsed_answer: Any
    verifier_pass: bool
    latency_seconds: float

    def __post_init__(self) -> None:
        if min(self.checkpoint_index, self.checkpoint_token_offset, self.generated_token_count) < 0:
            raise ValueError("checkpoint indices and token counts must be nonnegative")
        if self.latency_seconds < 0:
            raise ValueError("latency must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def require_fields(row: Mapping[str, Any], fields: set[str], artifact: str) -> None:
    missing = fields - set(row)
    if missing:
        raise ValueError(f"{artifact} row is missing fields: {sorted(missing)}")


def require_rows(rows: Iterable[Mapping[str, Any]], fields: set[str], artifact: str) -> None:
    for index, row in enumerate(rows):
        require_fields(row, fields, f"{artifact}[{index}]")


def require_columns(columns: Iterable[str], fields: set[str], artifact: str) -> None:
    missing = fields - set(columns)
    if missing:
        raise ValueError(f"{artifact} is missing columns: {sorted(missing)}")
