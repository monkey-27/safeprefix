import pytest

from safeprefix.rollout.posterior import BetaPosterior, posterior_table
from safeprefix.rollout.scheduler import nested_rollout_prefixes, sentinel_candidates


def test_beta_posterior_uses_counts() -> None:
    posterior = BetaPosterior(3, 4, 0.5, 0.5)
    assert posterior.mean == pytest.approx(0.7)
    low, high = posterior.interval()
    assert 0 < low < posterior.mean < high < 1


def test_nested_k_is_ordered_prefix_not_resample() -> None:
    outcomes = [True, False, True, True, False, False, True, False]
    nested = nested_rollout_prefixes(outcomes)
    assert nested[4] == outcomes[:4]
    assert nested[8] == outcomes


def test_sentinel_rule_keeps_root_and_latest() -> None:
    selected = sentinel_candidates(range(1, 20), cap=6)
    assert selected[0] == 0
    assert selected[-1] == 19
    assert len(selected) <= 6
