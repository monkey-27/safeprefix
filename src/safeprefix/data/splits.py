"""Problem-level deterministic split assignment and leakage assertions."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

from safeprefix.reproducibility import stable_hash

from .dedup import latex_whitespace_hash


def problem_group_key(row: Mapping[str, Any]) -> str:
    # The grouping key is deliberately stricter than the displayed exact hash:
    # formatting-only LaTeX and whitespace variants cannot cross partitions.
    return latex_whitespace_hash(str(row["problem_text"]))


def assign_problem_splits(
    rows: Iterable[Mapping[str, Any]],
    fractions: Mapping[str, float],
    seed: int,
) -> dict[str, str]:
    if not fractions or abs(sum(float(v) for v in fractions.values()) - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to one")
    names = list(fractions)
    cumulative: list[tuple[str, float]] = []
    total = 0.0
    for name in names:
        total += float(fractions[name])
        cumulative.append((name, total))
    assignment: dict[str, str] = {}
    for row in rows:
        group = problem_group_key(row)
        value = int(stable_hash([seed, group])[:13], 16) / float(16**13)
        chosen = next(name for name, edge in cumulative if value < edge or edge >= 1.0)
        previous = assignment.setdefault(group, chosen)
        if previous != chosen:
            raise AssertionError("one problem group received multiple splits")
    return assignment


def assert_problem_disjoint(rows: Iterable[Mapping[str, Any]]) -> None:
    observed: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        observed[problem_group_key(row)].add(str(row["split"]))
    overlaps = {key: sorted(value) for key, value in observed.items() if len(value) > 1}
    if overlaps:
        raise ValueError(f"problem-level split leakage: {overlaps}")
