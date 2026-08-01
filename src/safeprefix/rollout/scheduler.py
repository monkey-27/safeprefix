"""Checkpoint selection and nested/adaptive rollout allocation."""

from __future__ import annotations

import math
from typing import Iterable


def sentinel_candidates(clean_checkpoint_indices: Iterable[int], cap: int = 6) -> list[int]:
    """Keep root, latest safe, and evenly spaced interior sentinels.

    Index 0 is the prompt/root checkpoint. Other indices refer to states after
    visible reasoning spans. This deterministic rule never looks at rollouts.
    """
    values = sorted(set([0, *map(int, clean_checkpoint_indices)]))
    if cap < 2:
        raise ValueError("sentinel cap must be at least two")
    if len(values) <= cap:
        return values
    positions = {0, len(values) - 1}
    for slot in range(1, cap - 1):
        positions.add(round(slot * (len(values) - 1) / (cap - 1)))
    return [values[index] for index in sorted(positions)]


def nested_rollout_prefixes(outcomes: list[bool], counts: Iterable[int] = (1, 2, 4, 6, 8)) -> dict[int, list[bool]]:
    result = {}
    for count in counts:
        if count > len(outcomes):
            raise ValueError(f"requested nested k={count}, but only {len(outcomes)} outcomes exist")
        result[int(count)] = outcomes[: int(count)]
    return result


def adaptive_allocation(successes: list[int], totals: list[int], *, maximum: int = 6) -> list[int]:
    """Allocate 2 then 4 then 6 to checkpoints whose intervals can still overlap."""
    if len(successes) != len(totals) or any(total < 2 for total in totals):
        raise ValueError("adaptive allocation requires aligned initial counts >=2")
    means = [(success + 0.5) / (total + 1.0) for success, total in zip(successes, totals)]
    best = max(means)
    result = []
    for mean, total in zip(means, totals):
        uncertainty = math.sqrt(max(mean * (1 - mean), 1e-9) / (total + 1))
        if mean + 2 * uncertainty >= best - 0.05:
            result.append(min(maximum, 4 if total <= 2 else 6))
        else:
            result.append(total)
    return result
