import json
from pathlib import Path

import pytest

from safeprefix.native_data_compilation import (
    ATTEMPT_REQUIRED,
    build_execution_packs,
    initial_seed,
    rollout_seed,
    validate_exact_once,
    validate_protected_groups,
)


def _config():
    return {
        "generation": {"base_seed": 2701, "rollout_indices": [0, 1, 2, 3]},
    }


def _rows(count=7):
    return [{"problem_group_hash": f"group-{index}"} for index in range(count)]


def test_execution_packs_are_deterministic_and_exact_once():
    first = build_execution_packs(_rows(), model_key="m", pack_size=3, configuration_hash="cfg")
    second = build_execution_packs(_rows(), model_key="m", pack_size=3, configuration_hash="cfg")
    assert [value.to_dict() for value in first] == [value.to_dict() for value in second]
    assert [len(value.problem_group_hashes) for value in first] == [3, 3, 1]
    assert len({value.pack_id for value in first}) == 3


def test_rollout_seed_is_identity_bound_and_limited_to_frozen_four():
    assert rollout_seed(_config(), "m", "trace", 0) == rollout_seed(_config(), "m", "trace", 0)
    assert rollout_seed(_config(), "m", "trace", 0) != rollout_seed(_config(), "m", "trace", 1)
    assert initial_seed(_config(), "m", "group") != initial_seed(_config(), "m2", "group")
    with pytest.raises(ValueError):
        rollout_seed(_config(), "m", "trace", 4)


def test_protected_group_overlap_fails(tmp_path: Path):
    path = tmp_path / "protected.json"
    path.write_text(json.dumps({"protected_problem_groups": {"train": ["other"], "final": ["group-1"]}}))
    with pytest.raises(RuntimeError, match="overlaps protected"):
        validate_protected_groups(_rows(2), path)


def test_validate_exact_once_rejects_duplicate_attempts():
    base = {key: False for key in ATTEMPT_REQUIRED}
    base.update(attempt_key="same", model_key="m", problem_id="p", problem_group_hash="g", native_bucket="b")
    with pytest.raises(RuntimeError, match="duplicate"):
        validate_exact_once([base, dict(base)], "attempt_key", ATTEMPT_REQUIRED)
