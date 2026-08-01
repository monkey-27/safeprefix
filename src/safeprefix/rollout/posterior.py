"""Beta-binomial repairability summaries without hard best-boundary labels."""

from __future__ import annotations

from dataclasses import dataclass

from scipy.stats import beta


@dataclass(frozen=True)
class BetaPosterior:
    successes: int
    trials: int
    prior_alpha: float = 0.5
    prior_beta: float = 0.5

    def __post_init__(self) -> None:
        if not 0 <= self.successes <= self.trials:
            raise ValueError("require 0 <= successes <= trials")
        if self.prior_alpha <= 0 or self.prior_beta <= 0:
            raise ValueError("Beta prior parameters must be positive")

    @property
    def alpha(self) -> float:
        return self.prior_alpha + self.successes

    @property
    def beta(self) -> float:
        return self.prior_beta + self.trials - self.successes

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def interval(self, level: float = 0.95) -> tuple[float, float]:
        if not 0 < level < 1:
            raise ValueError("interval level must lie in (0, 1)")
        tail = (1 - level) / 2
        return float(beta.ppf(tail, self.alpha, self.beta)), float(beta.ppf(1 - tail, self.alpha, self.beta))


def posterior_table(rows: list[dict], prior_alpha: float = 0.5, prior_beta: float = 0.5) -> list[dict]:
    groups: dict[tuple[str, int], list[bool]] = {}
    for row in rows:
        groups.setdefault((str(row["trace_id"]), int(row["checkpoint_index"])), []).append(bool(row["verifier_pass"]))
    output = []
    for (trace_id, checkpoint), values in sorted(groups.items()):
        posterior = BetaPosterior(sum(values), len(values), prior_alpha, prior_beta)
        low, high = posterior.interval()
        output.append({
            "trace_id": trace_id, "checkpoint_index": checkpoint, "successes": sum(values),
            "trials": len(values), "posterior_mean": posterior.mean, "ci_low": low, "ci_high": high,
        })
    return output
