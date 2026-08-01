"""Deterministic analyses for the pre-registered recoverability hypotheses.

All functions are pure CPU operations over caller-supplied arrays/data frames.
The module never discovers or opens experiment artifacts, which keeps native
evaluation data outside this teacher-forced analysis surface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import kendalltau, norm, pearsonr, spearmanr


EPSILON = 1e-12
GEOMETRY_SEED = 20260728
TRAJECTORY_DELTA_BIC = 6.0
TRAJECTORY_PRIMARY_MAGNITUDE = 0.20
TRAJECTORY_RECOVERY_EXCLUSION = 0.15


def _stable_sigmoid(logits: np.ndarray | Iterable[float]) -> np.ndarray:
    """Finite sigmoid without relying on ``1 - tiny`` being representable."""
    values = np.asarray(logits, dtype=float)
    return np.exp(-np.logaddexp(0.0, -values))


def _fractional_bernoulli_loss_from_logits(
    logits: np.ndarray | Iterable[float], targets: np.ndarray | Iterable[float]
) -> np.ndarray:
    """Stable BCE for fractional binomial targets and finite linear predictors."""
    score = np.asarray(logits, dtype=float)
    target = np.asarray(targets, dtype=float)
    if score.shape != target.shape:
        raise ValueError("logits and fractional targets must align")
    return target * np.logaddexp(0.0, -score) + (1.0 - target) * np.logaddexp(
        0.0, score
    )


def _as_finite_1d(values: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite nonempty vector")
    return array


def _safe_correlation(kind: str, left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return float("nan")
    if kind == "pearson":
        return float(pearsonr(left, right).statistic)
    if kind == "spearman":
        return float(spearmanr(left, right).statistic)
    raise ValueError(kind)


def select_canonical_seed(dev_nll_by_seed: Mapping[int, float]) -> int:
    """Choose the median-dev-NLL seed without consulting test results."""
    if set(map(int, dev_nll_by_seed)) != {0, 1, 2}:
        raise ValueError("canonical-axis selection requires exactly seeds 0, 1, and 2")
    ranked = sorted(
        ((float(value), int(seed)) for seed, value in dev_nll_by_seed.items()),
        key=lambda pair: (pair[0], pair[1]),
    )
    if not all(math.isfinite(value) for value, _ in ranked):
        raise ValueError("dev NLLs must be finite")
    return ranked[1][1]


@dataclass(frozen=True)
class AffineAxis:
    """The exact frozen probe expressed in a seed-common affine feature space.

    The persisted predictor is ``learned_LayerNorm(raw_hidden) -> Linear``.
    Comparing its head weights directly across seeds is invalid because each
    seed learned a different LayerNorm affine transform.  Geometry is therefore
    defined in the common, *non-affine* normalized coordinates

    ``n(x) = (x - mean(x)) / sqrt(var(x) + eps)``.

    On construction the learned LayerNorm scale and shift are folded into the
    linear head: ``w_eff = w_head * gamma`` and
    ``b_eff = b_head + w_head @ beta``.  Consequently ``weight`` and ``bias``
    below are the effective affine parameters in the common ``n(x)`` space,
    while :meth:`logit_from_raw` remains exactly equivalent to the frozen
    predictor (up to the requested floating-point analysis precision).
    """

    weight: np.ndarray
    bias: float = 0.0
    layernorm_weight: np.ndarray | None = None
    layernorm_bias: np.ndarray | None = None
    layernorm_epsilon: float = 1e-5
    feature_space: str = "common_nonaffine_layernorm"
    raw_hidden_affine: bool = False
    seed: int | None = None

    def __post_init__(self) -> None:
        weight = np.asarray(self.weight, dtype=float)
        layernorm_weight = (
            np.ones_like(weight)
            if self.layernorm_weight is None
            else np.asarray(self.layernorm_weight, dtype=float)
        )
        layernorm_bias = (
            np.zeros_like(weight)
            if self.layernorm_bias is None
            else np.asarray(self.layernorm_bias, dtype=float)
        )
        arrays = [weight, layernorm_weight, layernorm_bias]
        if any(array.ndim != 1 for array in arrays) or len({len(array) for array in arrays}) != 1:
            raise ValueError("axis and LayerNorm parameters must be same-length vectors")
        if not len(arrays[0]) or not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("axis parameters must be finite and nonempty")
        if not math.isfinite(float(self.bias)) or self.layernorm_epsilon <= 0:
            raise ValueError("invalid affine bias or LayerNorm epsilon")
        if np.linalg.norm(arrays[0]) <= 0:
            raise ValueError("recoverability direction must be nonzero")
        # Fold seed-specific learned LayerNorm affine parameters into the head
        # so every seed direction lives in the same normalized coordinates.
        effective_weight = arrays[0] * arrays[1]
        effective_bias = float(self.bias) + float(np.dot(arrays[0], arrays[2]))
        if np.linalg.norm(effective_weight) <= 0:
            raise ValueError("effective recoverability direction must be nonzero")
        object.__setattr__(self, "weight", effective_weight)
        object.__setattr__(self, "bias", effective_bias)
        object.__setattr__(self, "layernorm_weight", arrays[1])
        object.__setattr__(self, "layernorm_bias", arrays[2])
        if self.raw_hidden_affine and (
            not np.array_equal(arrays[1], np.ones_like(arrays[1]))
            or not np.array_equal(arrays[2], np.zeros_like(arrays[2]))
        ):
            raise ValueError("raw-hidden affine axes cannot include a LayerNorm transform")

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, Any],
        *,
        layernorm_epsilon: float,
        prefix: str = "local_model",
    ) -> "AffineAxis":
        required = {
            f"{prefix}.0.weight",
            f"{prefix}.0.bias",
            f"{prefix}.1.weight",
            f"{prefix}.1.bias",
        }
        missing = required - set(state_dict)
        if missing:
            raise KeyError(f"linear-probe state is missing: {sorted(missing)}")

        def array(key: str) -> np.ndarray:
            value = state_dict[key]
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            return np.asarray(value, dtype=float)

        linear_weight = array(f"{prefix}.1.weight")
        if linear_weight.ndim == 2 and linear_weight.shape[0] == 1:
            linear_weight = linear_weight[0]
        linear_bias = array(f"{prefix}.1.bias").reshape(-1)
        if len(linear_bias) != 1:
            raise ValueError("recoverability linear head must have one output")
        return cls(
            weight=linear_weight,
            bias=float(linear_bias[0]),
            layernorm_weight=array(f"{prefix}.0.weight"),
            layernorm_bias=array(f"{prefix}.0.bias"),
            layernorm_epsilon=float(layernorm_epsilon),
        )

    @property
    def dimension(self) -> int:
        return int(len(self.weight))

    @property
    def norm(self) -> float:
        return float(np.linalg.norm(self.weight))

    def transform_raw(self, raw_hidden: np.ndarray) -> np.ndarray:
        """Map raw hidden states into the seed-common non-affine LN space."""
        raw = np.asarray(raw_hidden, dtype=float)
        if raw.shape[-1] != self.dimension:
            raise ValueError("raw hidden dimension does not match probe")
        mean = raw.mean(axis=-1, keepdims=True)
        variance = ((raw - mean) ** 2).mean(axis=-1, keepdims=True)
        return (raw - mean) / np.sqrt(variance + self.layernorm_epsilon)

    def logit_from_feature(self, feature: np.ndarray) -> np.ndarray:
        values = np.asarray(feature, dtype=float)
        if values.shape[-1] != self.dimension:
            raise ValueError("hidden dimension does not match probe feature dimension")
        return values @ self.weight + self.bias

    def logit_from_raw(self, raw_hidden: np.ndarray) -> np.ndarray:
        if self.raw_hidden_affine or self.feature_space == "provided_affine_feature":
            return self.logit_from_feature(raw_hidden)
        return self.logit_from_feature(self.transform_raw(raw_hidden))

    def logits(self, feature: np.ndarray) -> np.ndarray:
        """Compatibility alias for already transformed affine features."""
        return self.logit_from_feature(feature)

    def decompose(self, feature: np.ndarray) -> dict[str, np.ndarray]:
        values = np.asarray(feature, dtype=float)
        if values.shape[-1] != self.dimension:
            raise ValueError("feature dimension does not match probe")
        coefficient = (values @ self.weight) / (self.norm**2)
        parallel = np.expand_dims(coefficient, axis=-1) * self.weight
        perpendicular = values - parallel
        signed_distance = self.logit_from_feature(values) / self.norm
        boundary_projection = values - np.expand_dims(signed_distance, axis=-1) * (
            self.weight / self.norm
        )
        return {
            "parallel_component": parallel,
            "orthogonal_component": perpendicular,
            "signed_distance": signed_distance,
            "boundary_projection": boundary_projection,
        }

    def audit(self) -> dict[str, Any]:
        return {
            "raw_saved_representation": "raw_hidden_state",
            "fixed_transform": "non_affine_per_checkpoint_layernorm",
            "learned_layernorm_affine_folded_into_head": True,
            "effective_weight_formula": "w_eff = w_head * gamma",
            "effective_bias_formula": "b_eff = b_head + dot(w_head, beta)",
            "affine_feature_space": self.feature_space,
            "raw_hidden_affine": False,
            "projection_formula": "n_parallel=(w_eff^T n / ||w_eff||^2)w_eff in common non-affine LayerNorm space",
            "dimension": self.dimension,
            "direction_norm": self.norm,
            "layernorm_epsilon": float(self.layernorm_epsilon),
        }


def pairwise_direction_cosine(axes: Mapping[int, AffineAxis]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for left_index, left_seed in enumerate(sorted(axes)):
        for right_seed in sorted(axes)[left_index + 1 :]:
            left, right = axes[left_seed], axes[right_seed]
            if left.feature_space != right.feature_space or left.dimension != right.dimension:
                cosine = float("nan")
                valid = False
            else:
                cosine = float(np.dot(left.weight, right.weight) / (left.norm * right.norm))
                valid = True
            records.append(
                {
                    "left_seed": int(left_seed),
                    "right_seed": int(right_seed),
                    "direction_cosine": cosine,
                    "common_feature_space_valid": valid,
                }
            )
    return pd.DataFrame(records)


def _within_group_kendall(
    frame: pd.DataFrame, left: str, right: str, group: str
) -> float:
    values: list[float] = []
    for _, part in frame.groupby(group, sort=True):
        if len(part) < 2 or part[left].nunique() < 2 or part[right].nunique() < 2:
            continue
        result = kendalltau(part[left], part[right]).statistic
        if math.isfinite(float(result)):
            values.append(float(result))
    return float(np.mean(values)) if values else float("nan")


def summarize_seed_stability(
    predictions: pd.DataFrame,
    *,
    score_columns: Mapping[int, str],
    trace_column: str = "trace_id",
    axes: Mapping[int, AffineAxis] | None = None,
    success_column: str | None = None,
    trials_column: str | None = None,
    dense_score_columns: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    if set(score_columns) != {0, 1, 2}:
        raise ValueError("seed stability requires exactly seeds 0, 1, and 2")
    records: list[dict[str, Any]] = []
    seeds = sorted(score_columns)
    for index, left_seed in enumerate(seeds):
        for right_seed in seeds[index + 1 :]:
            left, right = score_columns[left_seed], score_columns[right_seed]
            if left not in predictions or right not in predictions:
                raise KeyError("seed score column is absent")
            records.append(
                {
                    "left_seed": left_seed,
                    "right_seed": right_seed,
                    "logit_correlation": _safe_correlation(
                        "pearson",
                        predictions[left].to_numpy(float),
                        predictions[right].to_numpy(float),
                    ),
                    "within_trace_ordering": _within_group_kendall(
                        predictions, left, right, trace_column
                    ),
                }
            )
    pairwise = pd.DataFrame(records)
    if axes is not None:
        pairwise = pairwise.merge(
            pairwise_direction_cosine(axes), on=["left_seed", "right_seed"], how="left"
        )
    correlations = pairwise["logit_correlation"].to_numpy(float)
    ordering = pairwise["within_trace_ordering"].to_numpy(float)
    valid_cosines = (
        pairwise["direction_cosine"].dropna().to_numpy(float)
        if "direction_cosine" in pairwise
        else np.asarray([], dtype=float)
    )
    seed_dense_metrics: dict[str, Any] = {}
    if success_column is not None or trials_column is not None:
        if not success_column or not trials_column:
            raise ValueError("both dense outcome columns are required for seed stability")
        if success_column not in predictions or trials_column not in predictions:
            raise KeyError("dense outcome column is absent")
        observed = predictions[success_column].to_numpy(float) / predictions[trials_column].to_numpy(float)
        dense_columns = dense_score_columns or score_columns
        if set(dense_columns) != set(score_columns):
            raise ValueError("dense seed score mapping must contain the same seeds")
        for seed, column in sorted(dense_columns.items()):
            seed_dense_metrics[str(seed)] = {
                "trace_weighted_k32_nll": trace_weighted_nll(
                    predictions[trace_column], predictions[success_column], predictions[trials_column], predictions[column]
                ),
                "trace_weighted_k32_brier": trace_weighted_brier(
                    predictions[trace_column], predictions[success_column], predictions[trials_column], predictions[column]
                ),
                "k32_spearman": _safe_correlation(
                    "spearman", predictions[column].to_numpy(float), observed
                ),
                "k32_within_trace_concordance": within_trace_concordance(
                    predictions.assign(_dense_observed=observed),
                    score_column=column,
                    outcome_column="_dense_observed",
                    trace_column=trace_column,
                ),
            }
    return {
        "pairwise": pairwise.to_dict(orient="records"),
        "median_logit_correlation": float(np.nanmedian(correlations)),
        "median_within_trace_ordering": float(np.nanmedian(ordering)),
        "median_direction_cosine": (
            float(np.median(valid_cosines)) if len(valid_cosines) else None
        ),
        "strong_logit_stability": bool(np.nanmedian(correlations) >= 0.95),
        "strong_direction_stability": (
            None if not len(valid_cosines) else bool(np.median(valid_cosines) >= 0.90)
        ),
        "seed_k32_metrics": seed_dense_metrics,
    }


def trace_weighted_nll(
    trace_ids: Iterable[Any], successes: Iterable[float], trials: Iterable[float], logits: Iterable[float]
) -> float:
    traces = np.asarray(list(trace_ids), dtype=object)
    success = _as_finite_1d(successes, "successes")
    total = _as_finite_1d(trials, "trials")
    score = _as_finite_1d(logits, "logits")
    if not (len(traces) == len(success) == len(total) == len(score)):
        raise ValueError("metric inputs must be aligned")
    if np.any(total <= 0) or np.any(success < 0) or np.any(success > total):
        raise ValueError("invalid binomial counts")
    outcome = success / total
    losses = _fractional_bernoulli_loss_from_logits(score, outcome)
    frame = pd.DataFrame({"trace": traces.astype(str), "loss": losses})
    return float(frame.groupby("trace", sort=True)["loss"].mean().mean())


def trace_weighted_brier(
    trace_ids: Iterable[Any], successes: Iterable[float], trials: Iterable[float], logits: Iterable[float]
) -> float:
    frame = pd.DataFrame(
        {
            "trace": list(map(str, trace_ids)),
            "success": list(successes),
            "trials": list(trials),
            "logit": list(logits),
        }
    )
    probability = _stable_sigmoid(frame["logit"].to_numpy(float))
    frame["loss"] = (probability - frame["success"] / frame["trials"]) ** 2
    return float(frame.groupby("trace", sort=True)["loss"].mean().mean())


def within_trace_concordance(
    frame: pd.DataFrame,
    *,
    score_column: str,
    outcome_column: str,
    trace_column: str = "trace_id",
) -> float:
    values: list[float] = []
    for _, part in frame.groupby(trace_column, sort=True):
        scores = part[score_column].to_numpy(float)
        outcomes = part[outcome_column].to_numpy(float)
        concordant = 0.0
        comparable = 0
        for left in range(len(part)):
            for right in range(left + 1, len(part)):
                outcome_delta = outcomes[left] - outcomes[right]
                if outcome_delta == 0:
                    continue
                score_delta = scores[left] - scores[right]
                comparable += 1
                if score_delta == 0:
                    concordant += 0.5
                elif np.sign(score_delta) == np.sign(outcome_delta):
                    concordant += 1.0
        if comparable:
            values.append(concordant / comparable)
    return float(np.mean(values)) if values else float("nan")


def compute_eta(control_nll: float, axis_nll: float, best_nll: float) -> float:
    denominator = float(control_nll) - float(best_nll)
    if abs(denominator) <= EPSILON:
        return float("nan")
    return (float(control_nll) - float(axis_nll)) / denominator


def compute_h1_metrics(
    frame: pd.DataFrame,
    *,
    model_logits: Mapping[str, str],
    trace_column: str = "trace_id",
    success_column: str = "success_count",
    trials_column: str = "num_rollouts",
    axis_name: str = "axis_only",
    control_name: str = "position_prompt_control",
    best_candidates: Sequence[str] = ("axis_plus_residual", "full_state"),
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = GEOMETRY_SEED,
) -> dict[str, Any]:
    required = {trace_column, success_column, trials_column, *model_logits.values()}
    missing = required - set(frame)
    if missing:
        raise KeyError(f"H1 frame is missing {sorted(missing)}")
    observed = frame[success_column].to_numpy(float) / frame[trials_column].to_numpy(float)
    metrics: dict[str, dict[str, float]] = {}
    for name, column in model_logits.items():
        logits = frame[column].to_numpy(float)
        metrics[name] = {
            "trace_weighted_nll": trace_weighted_nll(
                frame[trace_column], frame[success_column], frame[trials_column], logits
            ),
            "trace_weighted_brier": trace_weighted_brier(
                frame[trace_column], frame[success_column], frame[trials_column], logits
            ),
            "spearman": _safe_correlation("spearman", logits, observed),
            "within_trace_concordance": within_trace_concordance(
                frame.assign(_observed=observed),
                score_column=column,
                outcome_column="_observed",
                trace_column=trace_column,
            ),
        }
    if axis_name not in metrics or control_name not in metrics:
        raise KeyError("axis and position-plus-solvability control are required")
    available_best = [name for name in best_candidates if name in metrics]
    if not available_best:
        raise KeyError("at least one pre-registered best-model candidate is required")
    best_name = min(available_best, key=lambda name: metrics[name]["trace_weighted_nll"])
    eta = compute_eta(
        metrics[control_name]["trace_weighted_nll"],
        metrics[axis_name]["trace_weighted_nll"],
        metrics[best_name]["trace_weighted_nll"],
    )
    from .statistics import clustered_paired_bootstrap

    def row_loss(column: str) -> np.ndarray:
        target = frame[success_column].to_numpy(float) / frame[trials_column].to_numpy(float)
        return _fractional_bernoulli_loss_from_logits(
            frame[column].to_numpy(float), target
        )

    contrasts: dict[str, Any] = {}
    axis_loss = row_loss(model_logits[axis_name])
    for name in sorted(model_logits):
        if name == axis_name:
            continue
        # Positive means the axis has larger loss (the comparator improves).
        contrasts[f"axis_minus_{name}"] = clustered_paired_bootstrap(
            frame[trace_column],
            axis_loss,
            row_loss(model_logits[name]),
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        )
    residual_key = "axis_minus_axis_plus_residual"
    residual = contrasts.get(residual_key)
    return {
        "models": metrics,
        "best_model": best_name,
        "eta": float(eta),
        "paired_trace_bootstrap": contrasts,
        "axis_plus_residual_improvement": (
            None if residual is None else {
                "nll_improvement": float(residual["estimate"]),
                "ci_low": float(residual["ci_low"]),
                "ci_high": float(residual["ci_high"]),
                "ci_includes_zero": bool(residual["ci_low"] <= 0 <= residual["ci_high"]),
                "replicates": int(residual["replicates"]),
                "two_sided_sign_p": float(residual["two_sided_sign_p"]),
            }
        ),
    }


@dataclass(frozen=True)
class BinomialGLM:
    coefficients: np.ndarray
    standard_errors: np.ndarray
    covariance: np.ndarray
    log_likelihood: float
    converged: bool
    iterations: int


def fit_binomial_glm(
    design: np.ndarray,
    successes: Iterable[float],
    trials: Iterable[float],
    *,
    cluster_ids: Iterable[Any] | None = None,
    max_iterations: int = 500,
) -> BinomialGLM:
    """Fractional/binomial logistic MLE with optional cluster-robust covariance."""
    matrix = np.asarray(design, dtype=float)
    success = _as_finite_1d(successes, "successes")
    total = _as_finite_1d(trials, "trials")
    if matrix.ndim != 2 or matrix.shape[0] != len(success) or len(total) != len(success):
        raise ValueError("design and outcomes must be row-aligned")
    if not np.isfinite(matrix).all() or np.any(total <= 0) or np.any(success < 0) or np.any(success > total):
        raise ValueError("invalid GLM inputs")

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = matrix @ beta
        probability = _stable_sigmoid(eta)
        # Stable binomial log likelihood.  This remains finite under complete
        # or quasi separation where a rounded sigmoid would be exactly 0/1.
        log_likelihood = np.sum(
            -success * np.logaddexp(0.0, -eta)
            - (total - success) * np.logaddexp(0.0, eta)
        )
        gradient = matrix.T @ (success - total * probability)
        return -float(log_likelihood), -gradient

    fit = minimize(
        objective,
        np.zeros(matrix.shape[1], dtype=float),
        method="BFGS",
        jac=True,
        options={"maxiter": int(max_iterations), "gtol": 1e-8},
    )
    beta = np.asarray(fit.x, dtype=float)
    probability = _stable_sigmoid(matrix @ beta)
    weights = total * probability * (1 - probability)
    bread = np.linalg.pinv(matrix.T @ (matrix * weights[:, None]))
    if cluster_ids is None:
        covariance = bread
    else:
        clusters = np.asarray(list(map(str, cluster_ids)), dtype=object)
        if len(clusters) != len(success):
            raise ValueError("cluster IDs are not row-aligned")
        residual = success - total * probability
        meat = np.zeros_like(bread)
        for cluster in sorted(set(clusters)):
            score = matrix[clusters == cluster].T @ residual[clusters == cluster]
            meat += np.outer(score, score)
        covariance = bread @ meat @ bread
        cluster_count = len(set(clusters))
        if cluster_count > 1 and len(success) > matrix.shape[1]:
            covariance *= (cluster_count / (cluster_count - 1)) * (
                (len(success) - 1) / (len(success) - matrix.shape[1])
            )
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0))
    log_likelihood = -float(objective(beta)[0])
    return BinomialGLM(
        coefficients=beta,
        standard_errors=standard_errors,
        covariance=covariance,
        log_likelihood=log_likelihood,
        converged=bool(fit.success or np.linalg.norm(fit.jac) < 1e-5),
        iterations=int(getattr(fit, "nit", 0)),
    )


def _standardize(values: np.ndarray) -> np.ndarray:
    mean = np.mean(values)
    scale = np.std(values)
    return (values - mean) / (scale if scale > EPSILON else 1.0)


def build_h2_control_design(
    frame: pd.DataFrame,
    *,
    score_column: str = "canonical_raw_logit",
    domain_column: str = "domain",
) -> tuple[np.ndarray, list[str]]:
    numeric = {
        "axis": frame[score_column].to_numpy(float),
        "checkpoint_ordinal": frame["normalized_checkpoint_ordinal"].to_numpy(float),
        "checkpoint_token_position": frame["normalized_checkpoint_token_position"].to_numpy(float),
        "prompt_solvability": frame["prompt_solvability_smoothed"].to_numpy(float),
        "trace_token_count": frame["total_trace_token_count"].to_numpy(float),
        "checkpoint_count": frame["total_checkpoint_count"].to_numpy(float),
    }
    columns = [np.ones(len(frame), dtype=float)]
    names = ["intercept"]
    for name, values in numeric.items():
        columns.append(_standardize(values))
        names.append(name)
    domains = sorted(map(str, frame[domain_column].unique()))
    for domain in domains[1:]:
        columns.append((frame[domain_column].astype(str).to_numpy() == domain).astype(float))
        names.append(f"domain[{domain}]")
    return np.column_stack(columns), names


def within_trace_center(
    frame: pd.DataFrame, columns: Sequence[str], *, trace_column: str = "trace_id"
) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        output[f"{column}_within_trace"] = output[column] - output.groupby(trace_column)[column].transform("mean")
    return output


def compute_h2_metrics(
    frame: pd.DataFrame,
    *,
    score_column: str = "canonical_raw_logit",
    success_column: str = "success_count",
    trials_column: str = "num_rollouts",
    trace_column: str = "trace_id",
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = GEOMETRY_SEED,
) -> dict[str, Any]:
    full_design, full_names = build_h2_control_design(frame, score_column=score_column)
    full = fit_binomial_glm(
        full_design,
        frame[success_column],
        frame[trials_column],
        cluster_ids=frame[trace_column],
    )
    reduced_design = np.delete(full_design, full_names.index("axis"), axis=1)
    reduced = fit_binomial_glm(
        reduced_design,
        frame[success_column],
        frame[trials_column],
        cluster_ids=frame[trace_column],
    )
    axis_index = full_names.index("axis")
    axis_coefficient = float(full.coefficients[axis_index])
    axis_se = float(full.standard_errors[axis_index])
    centered = within_trace_center(
        frame,
        [score_column, "normalized_checkpoint_ordinal", "normalized_checkpoint_token_position"],
        trace_column=trace_column,
    )
    # A centered covariate alone does not remove trace difficulty in a nonlinear
    # logit.  Include an explicit nuisance intercept for every trace, then the
    # centered score and centered position terms.  This is a trace fixed-effect
    # binomial model (statistically equivalent to conditioning on trace for the
    # within-trajectory coefficient of interest).
    trace_categories = sorted(centered[trace_column].astype(str).unique())
    trace_index = {trace: index for index, trace in enumerate(trace_categories)}
    fixed_intercepts = np.zeros((len(centered), len(trace_categories)), dtype=float)
    for row, trace in enumerate(centered[trace_column].astype(str)):
        fixed_intercepts[row, trace_index[trace]] = 1.0
    centered_design = np.column_stack(
        [
            fixed_intercepts,
            centered[f"{score_column}_within_trace"],
            centered["normalized_checkpoint_ordinal_within_trace"],
            centered["normalized_checkpoint_token_position_within_trace"],
        ]
    )
    strict = fit_binomial_glm(
        centered_design,
        centered[success_column],
        centered[trials_column],
        cluster_ids=centered[trace_column],
    )
    strict_linear_predictor = centered_design @ strict.coefficients
    strict_nll = trace_weighted_nll(
        centered[trace_column],
        centered[success_column],
        centered[trials_column],
        strict_linear_predictor,
    )
    observed = centered[success_column] / centered[trials_column]
    centered["_outcome_centered"] = observed - observed.groupby(centered[trace_column]).transform("mean")
    pooled_centered_spearman = _safe_correlation(
        "spearman",
        centered[f"{score_column}_within_trace"].to_numpy(float),
        centered["_outcome_centered"].to_numpy(float),
    )
    # Predefined, transparent matched-bin diagnostics.  The bins are fixed
    # before inspecting outcomes and are not used to tune a model.
    position_edges = np.asarray([-np.inf, 0.25, 0.50, 0.75, np.inf])
    prompt_edges = np.asarray([-np.inf, 0.20, 0.40, 0.60, 0.80, np.inf])

    def grouped_ordering(values: pd.DataFrame, group: pd.Series) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for label, part in values.groupby(group, observed=True, sort=True):
            if part.empty:
                continue
            rows.append(
                {
                    "bin": str(label),
                    "traces": int(part[trace_column].nunique()),
                    "checkpoints": int(len(part)),
                    "within_trace_spearman": _safe_correlation(
                        "spearman",
                        part[f"{score_column}_within_trace"].to_numpy(float),
                        part["_outcome_centered"].to_numpy(float),
                    ),
                    "within_trace_concordance": within_trace_concordance(
                        part.assign(_observed=part[success_column] / part[trials_column]),
                        score_column=score_column,
                        outcome_column="_observed",
                        trace_column=trace_column,
                    ),
                }
            )
        return rows

    position_bins = pd.cut(
        centered["normalized_checkpoint_ordinal"], position_edges, right=False
    )
    prompt_bins = pd.cut(centered["prompt_solvability_smoothed"], prompt_edges, right=False)

    per_trace_rows: list[dict[str, Any]] = []
    for trace_id, part in centered.groupby(trace_column, sort=True):
        score = part[f"{score_column}_within_trace"].to_numpy(float)
        outcome_delta = part["_outcome_centered"].to_numpy(float)
        denominator = float(np.sum(score**2))
        per_trace_rows.append(
            {
                "trace_id": str(trace_id),
                "slope": float(np.sum(score * outcome_delta) / denominator)
                if denominator > EPSILON
                else 0.0,
                "spearman": _safe_correlation("spearman", score, outcome_delta),
            }
        )
    per_trace = pd.DataFrame(per_trace_rows)
    valid_slope = per_trace["slope"].to_numpy(float)
    from .statistics import clustered_bootstrap

    slope_bootstrap = clustered_bootstrap(
        per_trace["trace_id"],
        valid_slope,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    domain_results: dict[str, Any] = {}
    for domain, part in centered.groupby("domain", sort=True):
        domain_results[str(domain)] = {
            "traces": int(part[trace_column].nunique()),
            "checkpoints": int(len(part)),
            "within_trace_spearman": _safe_correlation(
                "spearman",
                part[f"{score_column}_within_trace"].to_numpy(float),
                part["_outcome_centered"].to_numpy(float),
            ),
            "within_trace_concordance": within_trace_concordance(
                part.assign(_observed=part[success_column] / part[trials_column]),
                score_column=score_column,
                outcome_column="_observed",
                trace_column=trace_column,
            ),
        }
    prompt_sensitivity: dict[str, Any] | None = None
    if "prompt_solvability_raw" in frame:
        raw_prompt_frame = frame.copy()
        raw_prompt_frame["prompt_solvability_smoothed"] = raw_prompt_frame[
            "prompt_solvability_raw"
        ]
        raw_design, raw_names = build_h2_control_design(
            raw_prompt_frame, score_column=score_column
        )
        raw_fit = fit_binomial_glm(
            raw_design,
            raw_prompt_frame[success_column],
            raw_prompt_frame[trials_column],
            cluster_ids=raw_prompt_frame[trace_column],
        )
        raw_axis = raw_names.index("axis")
        prompt_sensitivity = {
            "raw_prompt_axis_coefficient": float(raw_fit.coefficients[raw_axis]),
            "raw_prompt_axis_standard_error": float(raw_fit.standard_errors[raw_axis]),
            "jeffreys_axis_coefficient": axis_coefficient,
            "coefficient_difference_raw_minus_jeffreys": float(
                raw_fit.coefficients[raw_axis] - axis_coefficient
            ),
        }

    # Transparent five-fold group-crossfit of incremental predictive NLL.
    # This is a diagnostic control only; folds are hash-frozen and never used
    # to retrain or select the primary recoverability axis.
    group_column = "problem_group" if "problem_group" in frame else trace_column
    group_values = frame[group_column].astype(str)
    fold_ids = group_values.map(
        lambda value: int(hashlib.sha256(value.encode()).hexdigest(), 16) % 5
    ).to_numpy(int)
    full_oof = np.full(len(frame), np.nan)
    reduced_oof = np.full(len(frame), np.nan)
    for fold in range(5):
        train_mask = fold_ids != fold
        test_mask = fold_ids == fold
        if not train_mask.any() or not test_mask.any():
            continue
        full_fold = fit_binomial_glm(
            full_design[train_mask],
            frame.loc[train_mask, success_column],
            frame.loc[train_mask, trials_column],
            cluster_ids=frame.loc[train_mask, trace_column],
        )
        reduced_fold = fit_binomial_glm(
            reduced_design[train_mask],
            frame.loc[train_mask, success_column],
            frame.loc[train_mask, trials_column],
            cluster_ids=frame.loc[train_mask, trace_column],
        )
        full_oof[test_mask] = full_design[test_mask] @ full_fold.coefficients
        reduced_oof[test_mask] = reduced_design[test_mask] @ reduced_fold.coefficients
    if not np.isfinite(full_oof).all() or not np.isfinite(reduced_oof).all():
        raise RuntimeError("H2 group-crossfit failed to predict every checkpoint")
    crossfit_full_nll = trace_weighted_nll(
        frame[trace_column], frame[success_column], frame[trials_column], full_oof
    )
    crossfit_reduced_nll = trace_weighted_nll(
        frame[trace_column], frame[success_column], frame[trials_column], reduced_oof
    )
    return {
        "pooled_controlled": {
            "coefficient": axis_coefficient,
            "standard_error": axis_se,
            "ci_low": axis_coefficient - 1.96 * axis_se,
            "ci_high": axis_coefficient + 1.96 * axis_se,
            "converged": full.converged,
            "incremental_log_likelihood_over_controls": float(
                full.log_likelihood - reduced.log_likelihood
            ),
            "feature_names": full_names,
            "group_crossfit_folds": 5,
            "group_crossfit_group_column": group_column,
            "group_crossfit_full_nll": crossfit_full_nll,
            "group_crossfit_controls_only_nll": crossfit_reduced_nll,
            "group_crossfit_incremental_nll": crossfit_reduced_nll - crossfit_full_nll,
        },
        "strict_within_trace": {
            "coefficient": float(strict.coefficients[len(trace_categories)]),
            "standard_error": float(strict.standard_errors[len(trace_categories)]),
            "ci_low": float(strict.coefficients[len(trace_categories)] - 1.96 * strict.standard_errors[len(trace_categories)]),
            "ci_high": float(strict.coefficients[len(trace_categories)] + 1.96 * strict.standard_errors[len(trace_categories)]),
            "within_trace_spearman": float(per_trace["spearman"].dropna().mean()),
            "median_within_trace_spearman": float(per_trace["spearman"].dropna().median()),
            "pooled_centered_spearman": pooled_centered_spearman,
            "within_trace_concordance": within_trace_concordance(
                centered.assign(_observed=observed),
                score_column=score_column,
                outcome_column="_observed",
                trace_column=trace_column,
            ),
            "converged": strict.converged,
            "trace_weighted_nll": strict_nll,
            "trace_slope_bootstrap": slope_bootstrap,
            "trace_fixed_effect_count": int(len(trace_categories)),
        },
        "matched_checkpoint_position_bins": grouped_ordering(centered, position_bins),
        "narrow_prompt_solvability_bins": grouped_ordering(centered, prompt_bins),
        "by_domain": domain_results,
        "raw_vs_jeffreys_prompt_sensitivity": prompt_sensitivity,
    }


def _binomial_log_likelihood(success: np.ndarray, trials: np.ndarray, probability: np.ndarray) -> float:
    p = np.clip(probability, EPSILON, 1 - EPSILON)
    return float(np.sum(success * np.log(p) + (trials - success) * np.log(1 - p)))


@dataclass(frozen=True)
class TrajectoryFit:
    name: str
    bic: float
    log_likelihood: float
    parameters: int
    fitted_probabilities: tuple[float, ...]
    change_points: tuple[int, ...] = ()


def _constant_fit(success: np.ndarray, trials: np.ndarray) -> TrajectoryFit:
    probability = float(success.sum() / trials.sum())
    fitted = np.full(len(success), probability)
    ll = _binomial_log_likelihood(success, trials, fitted)
    return TrajectoryFit("constant", -2 * ll + math.log(len(success)), ll, 1, tuple(fitted))


def _linear_fit(success: np.ndarray, trials: np.ndarray) -> TrajectoryFit:
    position = np.linspace(0.0, 1.0, len(success))
    fit = fit_binomial_glm(np.column_stack([np.ones(len(success)), position]), success, trials)
    probability = _stable_sigmoid(
        np.column_stack([np.ones(len(success)), position]) @ fit.coefficients
    )
    return TrajectoryFit(
        "linear",
        -2 * fit.log_likelihood + 2 * math.log(len(success)),
        fit.log_likelihood,
        2,
        tuple(map(float, probability)),
    )


def _plateau_fit(success: np.ndarray, trials: np.ndarray, changes: tuple[int, ...]) -> TrajectoryFit:
    edges = (0, *changes, len(success))
    fitted = np.empty(len(success), dtype=float)
    for start, end in zip(edges[:-1], edges[1:], strict=True):
        fitted[start:end] = success[start:end].sum() / trials[start:end].sum()
    ll = _binomial_log_likelihood(success, trials, fitted)
    parameters = 2 * len(changes) + 1  # plateau rates plus discrete locations
    name = "one_change" if len(changes) == 1 else "two_change"
    return TrajectoryFit(
        name,
        -2 * ll + parameters * math.log(len(success)),
        ll,
        parameters,
        tuple(map(float, fitted)),
        changes,
    )


def fit_trajectory_models(successes: Iterable[float], trials: Iterable[float]) -> dict[str, TrajectoryFit]:
    success = _as_finite_1d(successes, "successes")
    total = _as_finite_1d(trials, "trials")
    if len(success) != len(total) or len(success) < 2 or np.any(total <= 0) or np.any(success < 0) or np.any(success > total):
        raise ValueError("invalid trajectory binomial counts")
    models = {"constant": _constant_fit(success, total), "linear": _linear_fit(success, total)}
    if len(success) >= 4:
        candidates = [_plateau_fit(success, total, (point,)) for point in range(2, len(success) - 1)]
        models["one_change"] = min(candidates, key=lambda fit: (fit.bic, fit.change_points))
    if len(success) >= 6:
        candidates = [
            _plateau_fit(success, total, (left, right))
            for left in range(2, len(success) - 3)
            for right in range(left + 2, len(success) - 1)
        ]
        if candidates:
            models["two_change"] = min(candidates, key=lambda fit: (fit.bic, fit.change_points))
    return models


def _plateau_levels(fit: TrajectoryFit) -> list[float]:
    fitted = np.asarray(fit.fitted_probabilities)
    edges = (0, *fit.change_points, len(fitted))
    return [float(fitted[start]) for start in edges[:-1]]


def trajectory_model_free_statistics(recoverability: Iterable[float]) -> dict[str, Any]:
    values = _as_finite_1d(recoverability, "recoverability")
    adjacent = np.diff(values)
    largest_drop_index = int(np.argmin(adjacent)) if len(adjacent) else -1
    crossings: dict[str, int] = {}
    reentries = 0
    for threshold in (0.25, 0.50, 0.75):
        above = values >= threshold
        transitions = int(np.count_nonzero(above[1:] != above[:-1])) if len(values) > 1 else 0
        crossings[f"{threshold:.2f}"] = transitions
        seen_exit = False
        for previous, current in zip(above[:-1], above[1:], strict=True):
            if previous and not current:
                seen_exit = True
            elif seen_exit and not previous and current:
                reentries += 1
    if largest_drop_index >= 0:
        post = values[largest_drop_index + 1 :]
        pre = values[largest_drop_index]
        sustained = bool(np.max(post) <= pre - TRAJECTORY_RECOVERY_EXCLUSION)
    else:
        sustained = False
    return {
        "total_change": float(values[-1] - values[0]),
        "largest_adjacent_drop": float(max(0.0, -np.min(adjacent))) if len(adjacent) else 0.0,
        "largest_adjacent_recovery": float(max(0.0, np.max(adjacent))) if len(adjacent) else 0.0,
        "threshold_crossings": crossings,
        "recoverability_reentries": int(reentries),
        "position_spearman": _safe_correlation("spearman", np.arange(len(values)), values),
        "low_sustained_after_largest_drop": sustained,
    }


def classify_trajectory(
    successes: Iterable[float],
    trials: Iterable[float],
    *,
    magnitude_threshold: float = TRAJECTORY_PRIMARY_MAGNITUDE,
    delta_bic: float = TRAJECTORY_DELTA_BIC,
    later_recovery_threshold: float = TRAJECTORY_RECOVERY_EXCLUSION,
) -> dict[str, Any]:
    success = _as_finite_1d(successes, "successes")
    total = _as_finite_1d(trials, "trials")
    if len(success) < 5:
        if len(success) != len(total):
            raise ValueError("trajectory inputs must be aligned")
        return {
            "category": "not_formally_classified",
            "formal_classification_eligible": False,
            "models": {},
            "model_free": trajectory_model_free_statistics(success / total),
            "magnitude_threshold": float(magnitude_threshold),
            "delta_bic": float(delta_bic),
        }
    models = fit_trajectory_models(success, total)
    category = "flat_or_unstructured"
    one = models.get("one_change")
    two = models.get("two_change")
    linear = models["linear"]
    constant = models["constant"]
    if two is not None:
        levels = _plateau_levels(two)
        if (
            all(models[name].bic - two.bic >= delta_bic for name in ("constant", "linear", "one_change"))
            and levels[0] - levels[1] >= magnitude_threshold
            and levels[2] - levels[1] >= magnitude_threshold
        ):
            category = "windowed_or_nonmonotonic"
    if category == "flat_or_unstructured" and one is not None:
        levels = _plateau_levels(one)
        later_recovery_supported = False
        if two is not None:
            two_levels = _plateau_levels(two)
            later_recovery_supported = bool(
                one.bic - two.bic >= delta_bic
                and two_levels[2] - two_levels[1] >= later_recovery_threshold
            )
        if (
            constant.bic - one.bic >= delta_bic
            and linear.bic - one.bic >= delta_bic
            and levels[0] - levels[1] >= magnitude_threshold
            and not later_recovery_supported
        ):
            category = "single_collapse"
    if category == "flat_or_unstructured":
        fitted = np.asarray(linear.fitted_probabilities)
        if (
            constant.bic - linear.bic >= delta_bic
            and fitted[-1] < fitted[0]
            and (one is None or linear.bic - one.bic < delta_bic)
            and fitted[0] - fitted[-1] >= magnitude_threshold
        ):
            category = "gradual_decline"
    rate = success / total
    return {
        "category": category,
        "formal_classification_eligible": bool(len(success) >= 5),
        "models": {name: asdict(fit) for name, fit in models.items()},
        "model_free": trajectory_model_free_statistics(rate),
        "magnitude_threshold": float(magnitude_threshold),
        "delta_bic": float(delta_bic),
    }


def trajectory_sensitivity(successes: Iterable[float], trials: Iterable[float]) -> dict[str, str]:
    return {
        f"magnitude_{threshold:.2f}": classify_trajectory(
            successes, trials, magnitude_threshold=threshold
        )["category"]
        for threshold in (0.15, 0.20, 0.25)
    }


def summarize_trajectory_prevalence(
    trajectories: pd.DataFrame,
    *,
    cluster_column: str = "common_trace_id",
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = GEOMETRY_SEED,
) -> dict[str, Any]:
    """Report exact H3 counts and cluster-resampled prevalence intervals.

    Four-model pooled inference resamples the shared underlying trace, not the
    4x model-expanded rows.  Short traces remain in the descriptive census but
    are excluded from the denominator of formal shape prevalence.
    """
    required = {"base_model", "trace_id", cluster_column, "category", "formal_classification_eligible"}
    if missing := required - set(trajectories):
        raise KeyError(f"trajectory prevalence table missing {sorted(missing)}")
    if trajectories.duplicated(["base_model", "trace_id"]).any():
        raise ValueError("trajectory classification must have one row per model/trace")
    eligible = trajectories.loc[trajectories["formal_classification_eligible"].astype(bool)].copy()
    categories = (
        "single_collapse",
        "gradual_decline",
        "windowed_or_nonmonotonic",
        "flat_or_unstructured",
    )
    counts = trajectories["category"].value_counts(dropna=False).to_dict()
    formal_counts = eligible["category"].value_counts().reindex(categories, fill_value=0)
    from .statistics import clustered_bootstrap

    def prevalence(sample: pd.DataFrame) -> dict[str, float]:
        denominator = max(len(sample), 1)
        return {
            category: float((sample["category"].astype(str) == category).sum() / denominator)
            for category in categories
        }

    bootstrap = clustered_bootstrap(
        eligible,
        cluster_col=cluster_column,
        statistic=prevalence,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    return {
        "all_trace_model_rows": int(len(trajectories)),
        "formal_trace_model_rows": int(len(eligible)),
        "shared_trace_clusters": int(trajectories[cluster_column].nunique()),
        "all_category_counts": {str(key): int(value) for key, value in counts.items()},
        "formal_category_counts": {key: int(formal_counts[key]) for key in categories},
        "formal_prevalence_bootstrap": bootstrap.to_dict(),
        "by_model": {
            str(model): {
                "all_rows": int(len(part)),
                "formal_rows": int(part["formal_classification_eligible"].sum()),
                "category_counts": {
                    str(key): int(value)
                    for key, value in part["category"].value_counts(dropna=False).items()
                },
            }
            for model, part in trajectories.groupby("base_model", sort=True)
        },
    }


def _stable_hash(*values: Any, seed: int = GEOMETRY_SEED) -> str:
    payload = "||".join([*(str(value) for value in values), str(seed)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _bucket(rate: float) -> str:
    if 0.35 <= rate <= 0.65:
        return "near_boundary"
    if rate >= 0.75:
        return "high"
    if rate <= 0.25:
        return "low"
    return "intermediate"


def _distance_to_bucket(rate: float, bucket: str) -> float:
    lower, upper = {
        "near_boundary": (0.35, 0.65),
        "high": (0.75, 1.0),
        "low": (0.0, 0.25),
    }[bucket]
    return max(lower - rate, 0.0, rate - upper)


def select_local_branch_parents(
    candidates: pd.DataFrame,
    *,
    model_column: str = "base_model",
    trace_column: str = "trace_id",
    checkpoint_column: str = "checkpoint_id",
    domain_column: str = "domain",
    recoverability_column: str = "dense_recoverability",
    geometry_seed: int = GEOMETRY_SEED,
    quotas: Mapping[str, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select 24/8/8 parents per model with stable hashing and substitutions."""
    quota = dict(quotas or {"near_boundary": 24, "high": 8, "low": 8})
    if set(quota) != {"near_boundary", "high", "low"} or any(value < 0 for value in quota.values()):
        raise ValueError("parent quotas must contain nonnegative near_boundary/high/low counts")
    required = {model_column, trace_column, checkpoint_column, domain_column, recoverability_column}
    missing = required - set(candidates)
    if missing:
        raise KeyError(f"parent candidates missing {sorted(missing)}")
    selected_records: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []
    for model, raw_part in candidates.groupby(model_column, sort=True):
        part = raw_part.copy()
        part["_rate"] = part[recoverability_column].astype(float)
        if np.any((part["_rate"] < 0) | (part["_rate"] > 1)):
            raise ValueError("dense recoverability must be in [0, 1]")
        part["_bucket"] = part["_rate"].map(_bucket)
        part["_hash"] = [
            _stable_hash(model, trace, checkpoint, seed=geometry_seed)
            for trace, checkpoint in zip(part[trace_column], part[checkpoint_column], strict=True)
        ]
        used_traces: set[str] = set()
        domain_counts: dict[str, int] = {str(domain): 0 for domain in part[domain_column].unique()}

        def pick(pool: pd.DataFrame, count: int, requested: str, substituted: bool) -> None:
            for _ in range(count):
                available = pool.loc[~pool[trace_column].astype(str).isin(used_traces)].copy()
                if available.empty:
                    return
                minimum_domain_count = min(domain_counts.get(str(value), 0) for value in available[domain_column])
                available = available.loc[
                    available[domain_column].astype(str).map(domain_counts).fillna(0) == minimum_domain_count
                ]
                row = available.sort_values(["_distance", "_hash"]).iloc[0]
                trace = str(row[trace_column])
                domain = str(row[domain_column])
                used_traces.add(trace)
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
                record = row.drop(labels=["_rate", "_bucket", "_hash", "_distance"]).to_dict()
                record.update(
                    {
                        "requested_category": requested,
                        "observed_category": str(row["_bucket"]),
                        "category_substitution": bool(substituted),
                        "selection_hash": str(row["_hash"]),
                        "geometry_seed": int(geometry_seed),
                    }
                )
                selected_records.append(record)

        # Allocate the scarcest exact bucket first so common candidates cannot consume
        # trace IDs needed by a scarce category.
        order = sorted(
            quota,
            key=lambda name: (
                part.loc[part["_bucket"] == name, trace_column].astype(str).nunique() / max(quota[name], 1),
                name,
            ),
        )
        selected_before: dict[str, int] = {}
        for category in order:
            selected_before[category] = len(selected_records)
            pool = part.loc[part["_bucket"] == category].copy()
            pool["_distance"] = 0.0
            pick(pool, quota[category], category, False)

        model_selected = [row for row in selected_records if str(row[model_column]) == str(model)]
        counts = {name: sum(row["requested_category"] == name for row in model_selected) for name in quota}
        # Fill each deficit from the closest remaining rate, preserving one trace/model.
        for category in ("near_boundary", "high", "low"):
            deficit = quota[category] - counts[category]
            if deficit <= 0:
                continue
            pool = part.copy()
            pool["_distance"] = pool["_rate"].map(lambda value: _distance_to_bucket(value, category))
            before = len(selected_records)
            pick(pool, deficit, category, True)
            filled = len(selected_records) - before
            audit_records.append(
                {
                    "base_model": str(model),
                    "category": category,
                    "requested": int(quota[category]),
                    "exact_available_distinct_traces": int(
                        part.loc[part["_bucket"] == category, trace_column].astype(str).nunique()
                    ),
                    "substitutions": int(filled),
                    "unfilled": int(deficit - filled),
                }
            )
        if not any(row["base_model"] == str(model) for row in audit_records):
            for category in quota:
                audit_records.append(
                    {
                        "base_model": str(model),
                        "category": category,
                        "requested": int(quota[category]),
                        "exact_available_distinct_traces": int(
                            part.loc[part["_bucket"] == category, trace_column].astype(str).nunique()
                        ),
                        "substitutions": 0,
                        "unfilled": 0,
                    }
                )
    selected = pd.DataFrame(selected_records)
    if not selected.empty and selected.duplicated([model_column, trace_column]).any():
        raise RuntimeError("parent selection violated one-parent-per-model-trace")
    return selected, pd.DataFrame(audit_records)


def _cohens_d(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or len(right) < 2:
        return float("nan")
    pooled = math.sqrt(((len(left) - 1) * np.var(left, ddof=1) + (len(right) - 1) * np.var(right, ddof=1)) / (len(left) + len(right) - 2))
    return float((np.mean(left) - np.mean(right)) / pooled) if pooled > EPSILON else float("nan")


def _stack_orthogonal_vectors(
    values: pd.Series,
    *,
    model_value: str,
    outcome_group: str,
) -> np.ndarray:
    """Stack one model's displacement vectors with an explicit shape audit."""

    vectors = [np.asarray(value) for value in values]
    shapes = {vector.shape for vector in vectors}
    if len(shapes) != 1:
        formatted = ", ".join(str(shape) for shape in sorted(shapes))
        raise ValueError(
            f"{model_value}: inconsistent {outcome_group} orthogonal vector "
            f"dimensions within model: {formatted}"
        )
    if vectors and vectors[0].ndim != 1:
        raise ValueError(
            f"{model_value}: {outcome_group} orthogonal displacements must be "
            f"one-dimensional vectors, got shape {vectors[0].shape}"
        )
    return np.stack(vectors)


def compute_branch_metrics(
    children: pd.DataFrame,
    *,
    parent_column: str = "parent_id",
    horizon_column: str = "horizon",
    score_column: str = "child_score",
    delta_score_column: str = "delta_r",
    orthogonal_norm_column: str = "orthogonal_displacement_norm",
    orthogonal_vector_column: str = "orthogonal_displacement",
    success_column: str = "child_success_count",
    trials_column: str = "child_num_rollouts",
    available_column: str = "horizon_available",
    parent_recoverability_column: str = "parent_recoverability",
    model_column: str = "base_model",
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for horizon, raw_part in children.groupby(horizon_column, sort=True):
        part = raw_part.loc[raw_part[available_column].astype(bool)].copy()
        if part.empty:
            records[str(horizon)] = {"available_children": 0, "parents": 0}
            continue
        part["_rate"] = part[success_column] / part[trials_column]
        high = part.loc[part["_rate"] >= 0.75]
        low = part.loc[part["_rate"] <= 0.25]
        axis_difference = float(high[delta_score_column].mean() - low[delta_score_column].mean()) if len(high) and len(low) else float("nan")
        # Orthogonal coordinates from different base models do not inhabit the
        # same vector space (and may have different dimensionality).  Compute
        # outcome-conditioned centroid separation inside each model, then use
        # an equal-model macro average for the pooled scalar.  A per-model call
        # has one stratum and is therefore numerically unchanged.
        orthogonal_by_model: dict[str, float] = {}
        axis_by_model: dict[str, float] = {}
        ratio_by_model: dict[str, float] = {}
        if len(high) and len(low) and orthogonal_vector_column in part:
            if model_column in part:
                model_values = sorted(part[model_column].astype(str).unique())
            else:
                model_values = ["__single_model__"]
            for model_value in model_values:
                if model_column in part:
                    high_model = high.loc[high[model_column].astype(str) == model_value]
                    low_model = low.loc[low[model_column].astype(str) == model_value]
                else:
                    high_model, low_model = high, low
                if high_model.empty or low_model.empty:
                    continue
                high_vectors = _stack_orthogonal_vectors(
                    high_model[orthogonal_vector_column],
                    model_value=model_value,
                    outcome_group="high-outcome",
                )
                low_vectors = _stack_orthogonal_vectors(
                    low_model[orthogonal_vector_column],
                    model_value=model_value,
                    outcome_group="low-outcome",
                )
                if high_vectors.shape[1:] != low_vectors.shape[1:]:
                    raise ValueError(
                        f"{model_value}: high/low orthogonal vector dimensions differ"
                    )
                orthogonal_value = float(
                    np.linalg.norm(
                        high_vectors.mean(axis=0) - low_vectors.mean(axis=0)
                    )
                )
                axis_value = float(
                    high_model[delta_score_column].mean()
                    - low_model[delta_score_column].mean()
                )
                orthogonal_by_model[model_value] = orthogonal_value
                axis_by_model[model_value] = axis_value
                if orthogonal_value > EPSILON:
                    ratio_by_model[model_value] = abs(axis_value) / orthogonal_value
        orthogonal_difference = (
            float(np.mean(list(orthogonal_by_model.values())))
            if orthogonal_by_model
            else float("nan")
        )
        stratified_ratio = (
            float(np.mean(list(ratio_by_model.values())))
            if ratio_by_model
            else None
        )
        parent_summary = part.groupby(parent_column, sort=True)["_rate"].agg(
            any_high=lambda values: bool((values >= 0.75).any()),
            all_low=lambda values: bool((values <= 0.25).all()),
        )
        centered = within_trace_center(part, [score_column], trace_column=parent_column)
        design = np.column_stack([np.ones(len(centered)), centered[f"{score_column}_within_trace"]])
        glm = fit_binomial_glm(
            design,
            centered[success_column],
            centered[trials_column],
            cluster_ids=centered[parent_column],
        )
        records[str(horizon)] = {
            "available_children": int(len(part)),
            "parents": int(part[parent_column].nunique()),
            "horizon_attrition": int(len(raw_part) - len(part)),
            "score_success_spearman": _safe_correlation(
                "spearman", part[score_column].to_numpy(float), part["_rate"].to_numpy(float)
            ),
            "within_parent_concordance": within_trace_concordance(
                part,
                score_column=score_column,
                outcome_column="_rate",
                trace_column=parent_column,
            ),
            "within_parent_binomial_coefficient": float(glm.coefficients[1]),
            "within_parent_binomial_se": float(glm.standard_errors[1]),
            "mean_delta_r_high": float(high[delta_score_column].mean()) if len(high) else None,
            "mean_delta_r_low": float(low[delta_score_column].mean()) if len(low) else None,
            "axis_outcome_separation": axis_difference,
            "axis_cohens_d": _cohens_d(
                high[delta_score_column].to_numpy(float), low[delta_score_column].to_numpy(float)
            ),
            "orthogonal_outcome_centroid_separation": orthogonal_difference,
            "orthogonal_outcome_separation": orthogonal_difference,
            "orthogonal_outcome_centroid_separation_aggregation": "equal_model_macro",
            "orthogonal_outcome_centroid_separation_by_model": orthogonal_by_model,
            "axis_outcome_separation_by_model": axis_by_model,
            "models_contributing_to_orthogonal_separation": int(len(orthogonal_by_model)),
            "axis_to_orthogonal_separation_ratio": stratified_ratio,
            "axis_to_orthogonal_separation_ratio_by_model": ratio_by_model,
            "parent_probability_any_high_child": float(parent_summary["any_high"].mean()),
            "parent_probability_all_children_low": float(parent_summary["all_low"].mean()),
        }
    reliable = [
        int(horizon)
        for horizon, metric in records.items()
        if metric.get("within_parent_binomial_coefficient", 0) > 0
        and metric.get("within_parent_binomial_se", float("inf")) > 0
        and metric["within_parent_binomial_coefficient"]
        - 1.96 * metric["within_parent_binomial_se"]
        > 0
    ]
    available_all = children.loc[children[available_column].astype(bool)].copy()
    available_all["_rate"] = (
        available_all[success_column] / available_all[trials_column]
    )
    low_parent_discovery: dict[str, Any] = {}
    if parent_recoverability_column in available_all:
        low = available_all.loc[available_all[parent_recoverability_column] <= 0.25]
        for horizon, part in low.groupby(horizon_column, sort=True):
            by_parent = part.groupby(parent_column)["_rate"].apply(
                lambda value: bool((value >= 0.75).any())
            )
            low_parent_discovery[str(horizon)] = {
                "low_recoverability_parents": int(len(by_parent)),
                "probability_any_high_recoverability_child": float(by_parent.mean())
                if len(by_parent)
                else None,
            }
    horizon_similarity: dict[str, Any] = {}
    first_horizon = min(available_all[horizon_column]) if len(available_all) else None
    if first_horizon is not None and "branch_id" in available_all:
        base = available_all.loc[
            available_all[horizon_column] == first_horizon,
            ["branch_id", "_rate"],
        ].rename(columns={"_rate": "_base_rate"})
        for horizon, later in available_all.groupby(horizon_column, sort=True):
            if horizon == first_horizon:
                continue
            paired = base.merge(
                later[["branch_id", "_rate"]], on="branch_id", validate="one_to_one"
            )
            initially_low = paired.loc[paired["_base_rate"] <= 0.25]
            horizon_similarity[str(horizon)] = {
                "paired_branches": int(len(paired)),
                "recoverability_spearman_from_first_horizon": _safe_correlation(
                    "spearman",
                    paired["_base_rate"].to_numpy(float),
                    paired["_rate"].to_numpy(float),
                ),
                "low_recoverability_persistence": float(
                    (initially_low["_rate"] <= 0.25).mean()
                ) if len(initially_low) else None,
            }
    return {
        "by_horizon": records,
        "earliest_reliable_separation_horizon": min(reliable) if reliable else None,
        "low_parent_high_child_discovery": low_parent_discovery,
        "cross_horizon_outcome_similarity": horizon_similarity,
    }


def cross_domain_transfer_summary(
    evaluations: pd.DataFrame,
    *,
    train_domain_column: str = "train_domain",
    test_domain_column: str | None = None,
    nll_column: str | None = None,
    ranking_column: str | None = None,
) -> dict[str, Any]:
    test_domain_column = test_domain_column or (
        "test_domain" if "test_domain" in evaluations else "eval_domain"
    )
    nll_column = nll_column or ("dense_nll" if "dense_nll" in evaluations else "metric")
    ranking_column = ranking_column or (
        "within_trace_ranking" if "within_trace_ranking" in evaluations else nll_column
    )
    required = {train_domain_column, test_domain_column, nll_column, ranking_column}
    if missing := required - set(evaluations):
        raise KeyError(f"cross-domain table missing {sorted(missing)}")
    nll_matrix = evaluations.pivot_table(
        index=train_domain_column, columns=test_domain_column, values=nll_column, aggfunc="mean"
    ).sort_index().sort_index(axis=1)
    ranking_matrix = evaluations.pivot_table(
        index=train_domain_column, columns=test_domain_column, values=ranking_column, aggfunc="mean"
    ).sort_index().sort_index(axis=1)
    same = evaluations[train_domain_column].astype(str) == evaluations[test_domain_column].astype(str)
    result = {
        "nll_matrix": nll_matrix.to_dict(),
        "ranking_matrix": ranking_matrix.to_dict(),
        "mean_within_domain_nll": float(evaluations.loc[same, nll_column].mean()),
        "mean_cross_domain_nll": float(evaluations.loc[~same, nll_column].mean()),
        "mean_within_domain_ranking": float(evaluations.loc[same, ranking_column].mean()),
        "mean_cross_domain_ranking": float(evaluations.loc[~same, ranking_column].mean()),
        "rows": int(len(evaluations)),
    }
    for optional in (
        "direction_cosine_to_global",
        "logit_correlation_with_global",
        "direction_cosine_across_seeds",
    ):
        if optional in evaluations:
            result[f"mean_{optional}"] = float(evaluations[optional].mean())
    # Descriptive compatibility names for a generic higher-is-better metric.
    if nll_column == "metric":
        in_domain = evaluations.loc[same, nll_column]
        cross_domain = evaluations.loc[~same, nll_column]
        worst = evaluations.sort_values(nll_column, kind="stable").iloc[0]
        result.update(
            {
                "in_domain_mean": float(in_domain.mean()),
                "cross_domain_mean": float(cross_domain.mean()),
                "transfer_gap": float(in_domain.mean() - cross_domain.mean()),
                "worst_cell": {
                    "train_domain": str(worst[train_domain_column]),
                    "eval_domain": str(worst[test_domain_column]),
                    "mean": float(worst[nll_column]),
                },
            }
        )
    return result


def compute_margin(
    feature: np.ndarray,
    axis: AffineAxis,
    *,
    calibrator_a: float,
    calibrator_b: float,
) -> np.ndarray:
    """Signed distance to calibrated p=.5 in the affine feature space."""
    if not math.isfinite(calibrator_a) or calibrator_a <= 0 or not math.isfinite(calibrator_b):
        raise ValueError("frozen calibrator must have finite positive slope")
    raw_logit = axis.logit_from_feature(feature)
    raw_threshold = -float(calibrator_b) / float(calibrator_a)
    return (raw_logit - raw_threshold) / axis.norm


def margin_probability_equivalence(
    margin: Iterable[float], calibrated_probability: Iterable[float]
) -> dict[str, Any]:
    """Audit the mathematical redundancy of margin and calibrated probability.

    Both are one-to-one transforms of the same scalar axis, so margin cannot add
    independent information unless a downstream model imposes a different
    functional restriction.  This check prevents an accidental extra claim.
    """
    distance = _as_finite_1d(margin, "margin")
    probability = _as_finite_1d(calibrated_probability, "calibrated_probability")
    if len(distance) != len(probability) or np.any((probability < 0) | (probability > 1)):
        raise ValueError("margin and calibrated probabilities in [0,1] must align")
    endpoint_count = int(np.count_nonzero((probability == 0) | (probability == 1)))
    stable_probability = np.clip(probability, EPSILON, 1 - EPSILON)
    probability_logit = np.log(stable_probability) - np.log1p(-stable_probability)
    correlation = _safe_correlation("pearson", distance, probability_logit)
    return {
        "margin_probability_logit_correlation": correlation,
        "rank_order_identical": bool(np.array_equal(np.argsort(distance, kind="stable"), np.argsort(probability, kind="stable"))),
        "incremental_information_identifiable": bool(abs(correlation) < 1 - 1e-10),
        "endpoint_probabilities_clipped_for_logit": endpoint_count,
        "probability_clip_epsilon": EPSILON,
        "interpretation": "signed margin and calibrated probability are deterministic transforms of the same axis score",
    }


def select_representative_trajectory_medoid(
    trajectories: pd.DataFrame,
    *,
    category_column: str = "trajectory_category",
    trace_column: str = "trace_id",
    feature_columns: Sequence[str] = (
        "total_change",
        "largest_adjacent_drop",
        "largest_adjacent_recovery",
        "position_spearman",
    ),
) -> pd.DataFrame:
    """Pick deterministic closest-to-category-center examples for figures."""
    records: list[pd.Series] = []
    for category, part in trajectories.groupby(category_column, sort=True):
        values = part[list(feature_columns)].to_numpy(float)
        center = np.nanmedian(values, axis=0)
        scale = np.nanmedian(np.abs(values - center), axis=0)
        scale[~np.isfinite(scale) | (scale <= EPSILON)] = 1.0
        distance = np.sqrt(np.nansum(((values - center) / scale) ** 2, axis=1))
        ranked = part.assign(
            _distance=distance,
            _hash=[_stable_hash(category, trace) for trace in part[trace_column]],
        ).sort_values(["_distance", "_hash"])
        row = ranked.iloc[0].drop(labels=["_distance", "_hash"])
        records.append(row)
    return pd.DataFrame(records).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Compatibility convenience API retained from the first isolated draft.
# These helpers are descriptive utilities.  The confirmatory H1--H4 entry
# points above implement the exact pre-registered definitions.


def select_canonical_median_seed(
    records: Sequence[Mapping[str, Any]],
    *,
    seed_key: str = "seed",
    dev_score_key: str = "dev_score",
) -> dict[str, Any]:
    if not records:
        raise ValueError("seed records cannot be empty")
    seeds = [int(record[seed_key]) for record in records]
    if len(seeds) != len(set(seeds)):
        raise ValueError("duplicate seed")
    scores = np.asarray([float(record[dev_score_key]) for record in records], dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("dev scores must be finite")
    median = float(np.median(scores))
    selected = min(
        records,
        key=lambda record: (abs(float(record[dev_score_key]) - median), int(record[seed_key])),
    )
    return {"seed": int(selected[seed_key]), "median_dev_score": median, "record": dict(selected)}


def affine_axis_audit(
    hidden: np.ndarray,
    axis: AffineAxis,
    *,
    expected_logits: Iterable[float] | None = None,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    values = np.asarray(hidden, dtype=float)
    logits = np.asarray(axis.logits(values), dtype=float)
    decomposition = axis.decompose(values)
    reconstruction = decomposition["parallel_component"] + decomposition["orthogonal_component"]
    reconstruction_error = float(np.max(np.abs(reconstruction - values)))
    logit_error = 0.0
    if expected_logits is not None:
        expected = np.asarray(list(expected_logits), dtype=float)
        if expected.shape != logits.shape:
            raise ValueError("expected logits are not aligned")
        logit_error = float(np.max(np.abs(expected - logits)))
    return {
        **axis.audit(),
        "passed": bool(reconstruction_error <= tolerance and logit_error <= tolerance),
        "maximum_reconstruction_error": reconstruction_error,
        "maximum_logit_error": logit_error,
        "observations": int(values.shape[0]) if values.ndim > 1 else 1,
    }


def seed_axis_stability(
    axes: Mapping[int, AffineAxis], *, hidden: np.ndarray
) -> dict[str, Any]:
    values = np.asarray(hidden, dtype=float)
    pairs: list[dict[str, Any]] = []
    seeds = sorted(axes)
    for index, left_seed in enumerate(seeds):
        for right_seed in seeds[index + 1 :]:
            left, right = axes[left_seed], axes[right_seed]
            if left.dimension != right.dimension:
                raise ValueError("seed axes must share a feature dimension")
            cosine = float(np.dot(left.weight, right.weight) / (left.norm * right.norm))
            left_logit, right_logit = left.logits(values), right.logits(values)
            pairs.append(
                {
                    "left_seed": left_seed,
                    "right_seed": right_seed,
                    "oriented_cosine": cosine,
                    "absolute_cosine": abs(cosine),
                    "logit_pearson": _safe_correlation("pearson", left_logit, right_logit),
                    "logit_spearman": _safe_correlation("spearman", left_logit, right_logit),
                }
            )
    return {
        "pair_count": len(pairs),
        "pairs": pairs,
        "minimum_oriented_cosine": min(row["oriented_cosine"] for row in pairs),
        "median_absolute_cosine": float(np.median([row["absolute_cosine"] for row in pairs])),
    }


def axis_energy_fraction_eta(displacements: np.ndarray, axis: AffineAxis) -> float:
    values = np.asarray(displacements, dtype=float)
    decomposition = axis.decompose(values)
    denominator = float(np.sum(values**2))
    if denominator <= EPSILON:
        return 0.0
    return float(np.sum(decomposition["parallel_component"] ** 2) / denominator)


def _unweighted_nll(score: np.ndarray, success: np.ndarray, trials: np.ndarray) -> float:
    target = success / trials
    return float(np.mean(_fractional_bernoulli_loss_from_logits(score, target)))


def h1_axis_orthogonal_diagnostics(
    hidden: np.ndarray,
    axis: AffineAxis,
    *,
    successes: Iterable[float],
    trials: Iterable[float],
    orthogonal_score: Iterable[float],
    full_score: Iterable[float],
    displacements: np.ndarray,
) -> dict[str, Any]:
    feature = np.asarray(hidden, dtype=float)
    success = _as_finite_1d(successes, "successes")
    total = _as_finite_1d(trials, "trials")
    axis_score = np.asarray(axis.logits(feature), dtype=float)
    orthogonal = _as_finite_1d(orthogonal_score, "orthogonal score")
    full = _as_finite_1d(full_score, "full score")
    if not (len(feature) == len(success) == len(total) == len(orthogonal) == len(full)):
        raise ValueError("H1 inputs must be aligned")
    observed = success / total
    axis_nll = _unweighted_nll(axis_score, success, total)
    orthogonal_nll = _unweighted_nll(orthogonal, success, total)
    full_nll = _unweighted_nll(full, success, total)
    return {
        "observations": int(len(feature)),
        "axis_observed_spearman": _safe_correlation("spearman", axis_score, observed),
        "axis": {"nll": axis_nll},
        "orthogonal": {"nll": orthogonal_nll, "nll_gain_over_axis": axis_nll - orthogonal_nll},
        "full": {"nll": full_nll, "nll_gain_over_axis": axis_nll - full_nll},
        "eta": axis_energy_fraction_eta(displacements, axis),
    }


def within_trace_metrics(
    frame: pd.DataFrame,
    *,
    trace_col: str = "trace_id",
    checkpoint_col: str = "checkpoint_ordinal",
    axis_score_col: str = "axis_score",
    success_col: str = "success_count",
    trial_col: str = "trial_count",
    control_score_cols: Sequence[str] = (),
    permutation_seed: int = GEOMETRY_SEED,
) -> dict[str, Any]:
    required = {trace_col, checkpoint_col, axis_score_col, success_col, trial_col, *control_score_cols}
    if missing := required - set(frame):
        raise KeyError(f"within-trace frame missing {sorted(missing)}")
    if frame.duplicated([trace_col, checkpoint_col]).any():
        raise ValueError("checkpoint positions must be unique within traces")
    if np.any(frame[trial_col] <= 0) or np.any(frame[success_col] < 0) or np.any(frame[success_col] > frame[trial_col]):
        raise ValueError("invalid binomial counts")
    values = frame.copy()
    values["_outcome"] = values[success_col] / values[trial_col]
    values["_axis_centered"] = values[axis_score_col] - values.groupby(trace_col)[axis_score_col].transform("mean")
    values["_outcome_centered"] = values["_outcome"] - values.groupby(trace_col)["_outcome"].transform("mean")
    denominator = float(np.sum(values["_axis_centered"] ** 2))
    slope = float(np.sum(values["_axis_centered"] * values["_outcome_centered"]) / denominator) if denominator > EPSILON else float("nan")
    per_trace_spearman = []
    adjacent = []
    for _, part in values.sort_values([trace_col, checkpoint_col]).groupby(trace_col, sort=True):
        correlation = _safe_correlation("spearman", part[axis_score_col].to_numpy(float), part["_outcome"].to_numpy(float))
        if math.isfinite(correlation):
            per_trace_spearman.append(correlation)
        score_delta = np.diff(part[axis_score_col].to_numpy(float))
        outcome_delta = np.diff(part["_outcome"].to_numpy(float))
        valid = outcome_delta != 0
        adjacent.extend((np.sign(score_delta[valid]) == np.sign(outcome_delta[valid])).astype(float))
    return {
        "traces": int(values[trace_col].nunique()),
        "checkpoints": int(len(values)),
        "within_trace_pearson": _safe_correlation("pearson", values["_axis_centered"].to_numpy(float), values["_outcome_centered"].to_numpy(float)),
        "within_trace_spearman": _safe_correlation("spearman", values["_axis_centered"].to_numpy(float), values["_outcome_centered"].to_numpy(float)),
        "trace_fixed_effect_slope": slope,
        "adjacent_direction_agreement": float(np.mean(adjacent)) if adjacent else float("nan"),
        "median_trace_spearman": float(np.median(per_trace_spearman)) if per_trace_spearman else float("nan"),
        "control_columns": list(control_score_cols),
        "permutation_seed": int(permutation_seed),
    }


@dataclass(frozen=True)
class ShapeThresholds:
    bic_improvement: float = 6.0
    minimum_probability_range: float = 0.10
    turning_point_margin: float = 0.10

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _polynomial_bic(
    positions: np.ndarray, success: np.ndarray, trials: np.ndarray, degree: int
) -> tuple[float, np.ndarray]:
    design = np.column_stack([positions**power for power in range(degree + 1)])
    fit = fit_binomial_glm(design, success, trials)
    probability = _stable_sigmoid(design @ fit.coefficients)
    bic = -2 * fit.log_likelihood + (degree + 1) * math.log(len(positions))
    return float(bic), probability


def classify_binomial_trajectory(
    positions: Iterable[float],
    successes: Iterable[float],
    trials: Iterable[float],
    *,
    thresholds: ShapeThresholds | None = None,
) -> dict[str, Any]:
    """Descriptive polynomial-shape helper retained for supplementary use."""
    limits = thresholds or ShapeThresholds()
    position = _as_finite_1d(positions, "positions")
    success = _as_finite_1d(successes, "successes")
    total = _as_finite_1d(trials, "trials")
    if not (len(position) == len(success) == len(total)) or len(position) < 3:
        raise ValueError("trajectory inputs must be aligned with at least three checkpoints")
    if np.any(success < 0) or np.any(success > total) or np.any(total <= 0):
        raise ValueError("invalid binomial counts")
    constant_bic, constant_probability = _polynomial_bic(position, success, total, 0)
    linear_bic, linear_probability = _polynomial_bic(position, success, total, 1)
    quadratic_bic, quadratic_probability = _polynomial_bic(position, success, total, 2)
    observed_range = float(np.max(success / total) - np.min(success / total))
    shape = "flat"
    if observed_range >= limits.minimum_probability_range:
        if constant_bic - quadratic_bic >= limits.bic_improvement:
            middle = float(quadratic_probability[len(quadratic_probability) // 2])
            endpoints = (float(quadratic_probability[0]) + float(quadratic_probability[-1])) / 2
            if middle - endpoints >= limits.turning_point_margin:
                shape = "inverted_u"
            elif endpoints - middle >= limits.turning_point_margin:
                shape = "u_shaped"
        if shape == "flat" and constant_bic - linear_bic >= limits.bic_improvement:
            shape = "monotone_increasing" if linear_probability[-1] > linear_probability[0] else "monotone_decreasing"
    return {
        "shape": shape,
        "thresholds": limits.to_dict(),
        "bernoulli_trials": int(total.sum()),
        "checkpoint_count": int(len(position)),
        "bic": {"constant": constant_bic, "linear": linear_bic, "quadratic": quadratic_bic},
    }


def shape_sensitivity_analysis(
    trajectories: Mapping[str, Mapping[str, Iterable[float]]],
    *,
    bic_thresholds: Sequence[float] = (6.0,),
    probability_ranges: Sequence[float] = (0.10,),
    turning_point_margins: Sequence[float] = (0.10,),
) -> dict[str, Any]:
    settings = [
        ShapeThresholds(bic, probability_range, margin)
        for bic in sorted(map(float, bic_thresholds))
        for probability_range in sorted(map(float, probability_ranges))
        for margin in sorted(map(float, turning_point_margins))
    ]
    records = []
    for trace_id in sorted(trajectories):
        trajectory = trajectories[trace_id]
        for setting in settings:
            result = classify_binomial_trajectory(
                trajectory["positions"], trajectory["successes"], trajectory["trials"], thresholds=setting
            )
            records.append({"trace_id": trace_id, **setting.to_dict(), "shape": result["shape"]})
    return {"setting_count": len(settings), "per_trace": records}


def select_h4_parents(
    frame: pd.DataFrame,
    *,
    trace_col: str = "trace_id",
    checkpoint_col: str = "checkpoint_ordinal",
    success_col: str = "success_count",
    trial_col: str = "trial_count",
) -> pd.DataFrame:
    if np.any(frame[trial_col] <= 0) or np.any(frame[success_col] < 0) or np.any(frame[success_col] > frame[trial_col]):
        raise ValueError("invalid binomial counts")
    mixed = frame.loc[(frame[success_col] > 0) & (frame[success_col] < frame[trial_col])].copy()
    if mixed.empty:
        return mixed.assign(parent_id=pd.Series(dtype=str))
    selected = mixed.sort_values([trace_col, checkpoint_col]).groupby(trace_col, sort=True).tail(1).copy()
    selected["parent_id"] = selected[trace_col].astype(str) + ":" + selected[checkpoint_col].astype(str)
    return selected.sort_values([trace_col, checkpoint_col]).reset_index(drop=True)


def branch_metrics(
    frame: pd.DataFrame,
    *,
    parent_col: str = "parent_id",
    outcome_col: str = "binary_outcome",
    parent_score_col: str = "parent_axis_score",
    child_score_col: str = "child_axis_score",
    orthogonal_col: str = "orthogonal_distance",
) -> dict[str, Any]:
    required = {parent_col, outcome_col, parent_score_col, child_score_col, orthogonal_col}
    if missing := required - set(frame):
        raise KeyError(f"branch frame missing {sorted(missing)}")
    gaps, accuracies, pair_correct, pair_total = [], [], 0.0, 0
    mixed_count = 0
    for _, part in frame.groupby(parent_col, sort=True):
        positive = part.loc[part[outcome_col] == 1, child_score_col].to_numpy(float)
        negative = part.loc[part[outcome_col] == 0, child_score_col].to_numpy(float)
        if not len(positive) or not len(negative):
            continue
        mixed_count += 1
        gaps.append(float(np.mean(positive) - np.mean(negative)))
        comparisons = [(left > right) + 0.5 * (left == right) for left in positive for right in negative]
        accuracies.append(float(np.mean(comparisons)))
        pair_correct += float(np.sum(comparisons))
        pair_total += len(comparisons)
    return {
        "parents_total": int(frame[parent_col].nunique()),
        "parents_with_both_outcomes": mixed_count,
        "mean_parent_axis_delta_gap": float(np.mean(gaps)) if gaps else float("nan"),
        "mean_parent_pairwise_axis_order_accuracy": float(np.mean(accuracies)) if accuracies else float("nan"),
        "pooled_pairwise_axis_order_accuracy": pair_correct / pair_total if pair_total else float("nan"),
    }


def margin_metrics(
    scores: Iterable[float],
    binary_outcomes: Iterable[int],
    *,
    scores_are_probabilities: bool = False,
) -> dict[str, Any]:
    score = _as_finite_1d(scores, "scores")
    outcome = np.asarray(list(binary_outcomes), dtype=int)
    if len(score) != len(outcome) or np.any(~np.isin(outcome, [0, 1])):
        raise ValueError("binary margin outcomes must be aligned")
    if scores_are_probabilities:
        if np.any((score < 0) | (score > 1)):
            raise ValueError("probabilities must lie in [0, 1]")
        score = np.clip(score, EPSILON, 1 - EPSILON)
        score = np.log(score) - np.log1p(-score)
    signed = np.where(outcome == 1, score, -score)
    positive, negative = score[outcome == 1], score[outcome == 0]
    pairs = [(left > right) + 0.5 * (left == right) for left in positive for right in negative]
    return {
        "mean_signed_margin": float(np.mean(signed)),
        "median_signed_margin": float(np.median(signed)),
        "correct_sign_fraction": float(np.mean(signed > 0)),
        "pairwise_order_accuracy": float(np.mean(pairs)) if pairs else float("nan"),
    }
