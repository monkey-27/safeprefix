"""Matched linear and position-only prefix-validity probe training."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import random
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from safeprefix.boundary_v1.models import RecoverabilityPredictor, build_predictor
from safeprefix.boundary_v1.training import position_features


Architecture = Literal["linear_probe", "position_only"]
DEFAULT_LEARNING_RATES = (1e-3, 3e-4)
DEFAULT_TRAINING_SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class PrefixValidityTrainingConfig:
    learning_rates: tuple[float, ...] = DEFAULT_LEARNING_RATES
    seeds: tuple[int, ...] = DEFAULT_TRAINING_SEEDS
    max_epochs: int = 50
    patience: int = 7
    batch_size_traces: int = 64
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    hidden_width: int = 128
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if tuple(map(float, self.learning_rates)) != DEFAULT_LEARNING_RATES:
            raise ValueError("prefix-validity learning rates must be exactly 1e-3 and 3e-4")
        if len(self.seeds) != 3 or len(set(map(int, self.seeds))) != 3:
            raise ValueError("prefix-validity training requires three distinct seeds")
        if min(self.max_epochs, self.patience, self.batch_size_traces) < 1:
            raise ValueError("training limits must be positive")
        if self.weight_decay < 0 or self.gradient_clip_norm <= 0:
            raise ValueError("optimizer controls are invalid")


@dataclass
class ProbeCandidate:
    architecture: Architecture
    learning_rate: float
    seed: int
    best_epoch: int
    dev_loss: float
    state_dict: dict[str, torch.Tensor]
    predictor_config: dict[str, Any]
    history: list[dict[str, float | int]]
    position_mean: list[float] | None = None
    position_std: list[float] | None = None

    def metadata(self) -> dict[str, Any]:
        return {
            "target": "prefix_valid_before_first_visible_error",
            "architecture": self.architecture,
            "learning_rate": float(self.learning_rate),
            "seed": int(self.seed),
            "best_epoch": int(self.best_epoch),
            "dev_loss": float(self.dev_loss),
            "predictor_config": self.predictor_config,
            "history": self.history,
            "position_mean": self.position_mean,
            "position_std": self.position_std,
            "class_balanced_loss": False,
            "trace_weighting": "mean_checkpoint_bce_within_trace_then_mean_traces",
            "native_outcomes_used": False,
            "test_used_for_selection": False,
        }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def trace_weighted_binary_bce(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Unweighted BCE per checkpoint, equal total objective mass per trace."""

    if logits.shape != targets.shape or logits.shape != mask.shape:
        raise ValueError("logits, targets, and mask must have identical shapes")
    if mask.dtype != torch.bool or bool((mask.sum(dim=1) == 0).any()):
        raise ValueError("every batch trace must have a nonempty boolean mask")
    active = targets[mask]
    if not bool(torch.isfinite(active).all()) or not set(active.detach().cpu().tolist()).issubset(
        {0.0, 1.0}
    ):
        raise ValueError("prefix-validity targets must be finite binary labels")
    checkpoint = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    valid = mask.to(checkpoint.dtype)
    return ((checkpoint * valid).sum(dim=1) / valid.sum(dim=1)).mean()


@dataclass(frozen=True)
class _TraceRows:
    trace_id: str
    rows: tuple[int, ...]


class _TraceDataset(Dataset[_TraceRows]):
    def __init__(self, values: list[_TraceRows]) -> None:
        self.values = values

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> _TraceRows:
        return self.values[index]


class PrefixValidityCorpus:
    """Full ProcessBench step rows aligned to frozen final-layer features."""

    def __init__(
        self,
        frame: pd.DataFrame,
        hidden_features: torch.Tensor | np.ndarray,
        *,
        required_splits: Sequence[str] = (
            "train",
            "architecture_dev",
            "calibration",
            "teacher_forced_test",
        ),
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        # The complete-step extraction manifest uses explicit names, while the
        # frozen production probe utilities use the aliases on the right.
        # Preserve both and materialize only exact, semantics-preserving aliases.
        for destination, source in (
            ("base_model", "model_key"),
            ("checkpoint_ordinal", "checkpoint_index"),
            ("prefix_token_count", "checkpoint_token_offset"),
            ("total_trace_token_count", "full_trace_token_count"),
        ):
            if destination not in self.frame and source in self.frame:
                self.frame[destination] = self.frame[source]
        if "total_checkpoint_count" not in self.frame and "checkpoint_ordinal" in self.frame:
            self.frame["total_checkpoint_count"] = self.frame.groupby(
                ["base_model", "trace_id"], sort=False
            )["checkpoint_ordinal"].transform("max")
        self.hidden = torch.as_tensor(hidden_features, dtype=torch.float32).clone()
        required = {
            "base_model",
            "split",
            "trace_id",
            "checkpoint_id",
            "checkpoint_ordinal",
            "prefix_token_count",
            "total_checkpoint_count",
            "total_trace_token_count",
            "prefix_valid",
        }
        missing = required - set(self.frame.columns)
        if missing:
            raise KeyError(f"prefix-validity corpus lacks columns: {sorted(missing)}")
        if self.hidden.ndim != 2 or len(self.hidden) != len(self.frame):
            raise ValueError("hidden features must be a row-aligned [checkpoint, dimension] tensor")
        if not bool(torch.isfinite(self.hidden).all()):
            raise ValueError("hidden features must be finite")
        if self.frame["base_model"].astype(str).nunique() != 1:
            raise ValueError("train one prefix-validity probe per base model")
        if bool(self.frame[["trace_id", "checkpoint_id"]].astype(str).duplicated().any()):
            raise RuntimeError("duplicate model-trace checkpoint identity")
        if not set(self.frame["prefix_valid"].astype(int).unique()).issubset({0, 1}):
            raise ValueError("prefix_valid must be binary")
        self.positions = position_features(self.frame).to(torch.float32)
        self._traces: dict[str, list[_TraceRows]] = {}
        for split, part in self.frame.groupby("split", sort=False):
            traces: list[_TraceRows] = []
            for trace_id, trace in part.groupby("trace_id", sort=True):
                ordered = trace.sort_values("checkpoint_ordinal", kind="stable")
                ordinals = ordered["checkpoint_ordinal"].to_numpy(int)
                if bool((np.diff(ordinals) <= 0).any()):
                    raise RuntimeError(f"{trace_id}: checkpoints are not strictly ordered")
                # Labels must have one 1->0 transition at most; a later valid
                # checkpoint would contradict the first-visible-error target.
                labels = ordered["prefix_valid"].to_numpy(int)
                if bool((np.diff(labels) > 0).any()):
                    raise RuntimeError(f"{trace_id}: prefix-validity labels are non-monotonic")
                traces.append(_TraceRows(str(trace_id), tuple(map(int, ordered.index))))
            self._traces[str(split)] = traces
        self.required_splits = tuple(map(str, required_splits))
        if not self.required_splits or len(set(self.required_splits)) != len(self.required_splits):
            raise ValueError("required prefix-validity splits must be distinct and nonempty")
        unexpected = set(self._traces) - set(self.required_splits)
        if unexpected:
            raise RuntimeError(
                f"corpus contains rows outside its declared split scope: {sorted(unexpected)}"
            )
        for split in self.required_splits:
            if split not in self._traces or not self._traces[split]:
                raise RuntimeError(f"prefix-validity split {split!r} is empty")
        for split in set(self.required_splits) & {"train", "architecture_dev", "calibration"}:
            if set(self.frame.loc[self.frame["split"].eq(split), "prefix_valid"].astype(int)) != {0, 1}:
                raise RuntimeError(f"prefix-validity target is single-class in {split}")

    @property
    def input_dim(self) -> int:
        return int(self.hidden.shape[1])

    def traces(self, split: str) -> list[_TraceRows]:
        values = self._traces.get(str(split), [])
        if not values:
            raise RuntimeError(f"prefix-validity split {split!r} is empty")
        return values

    def _collate(self, traces: list[_TraceRows]) -> dict[str, Any]:
        maximum = max(len(trace.rows) for trace in traces)
        batch = len(traces)
        hidden = torch.zeros((batch, maximum, self.input_dim), dtype=torch.float32)
        position = torch.zeros((batch, maximum, 4), dtype=torch.float32)
        target = torch.zeros((batch, maximum), dtype=torch.float32)
        mask = torch.zeros((batch, maximum), dtype=torch.bool)
        row_indices: list[list[int]] = []
        for batch_index, trace in enumerate(traces):
            rows = torch.tensor(trace.rows, dtype=torch.long)
            count = len(rows)
            hidden[batch_index, :count] = self.hidden[rows]
            position[batch_index, :count] = self.positions[rows]
            target[batch_index, :count] = torch.tensor(
                self.frame.loc[list(trace.rows), "prefix_valid"].to_numpy(float),
                dtype=torch.float32,
            )
            mask[batch_index, :count] = True
            row_indices.append(list(trace.rows))
        return {
            "hidden_states": hidden,
            "position_features": position,
            "target": target,
            "mask": mask,
            "row_indices": row_indices,
        }

    def loader(
        self,
        split: str,
        *,
        batch_size_traces: int,
        shuffle: bool,
        seed: int,
    ) -> DataLoader[_TraceRows]:
        values = self.traces(split)
        return DataLoader(
            _TraceDataset(values),
            batch_size=min(int(batch_size_traces), len(values)),
            shuffle=bool(shuffle),
            num_workers=0,
            collate_fn=self._collate,
            generator=torch.Generator().manual_seed(int(seed)),
        )


def _move(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _position_normalization(corpus: PrefixValidityCorpus) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [row for trace in corpus.traces("train") for row in trace.rows]
    values = corpus.positions[torch.tensor(rows, dtype=torch.long)]
    return values.mean(dim=0), values.std(dim=0, unbiased=False).clamp_min(1e-8)


@torch.inference_mode()
def predict_probe(
    model: RecoverabilityPredictor,
    corpus: PrefixValidityCorpus,
    *,
    split: str,
    batch_size_traces: int = 64,
    device: str | torch.device = "cpu",
) -> pd.DataFrame:
    resolved = torch.device(device)
    model = model.to(resolved).eval()
    records: list[dict[str, Any]] = []
    for raw in corpus.loader(
        split, batch_size_traces=batch_size_traces, shuffle=False, seed=0
    ):
        batch = _move(raw, resolved)
        logits = model(batch["hidden_states"], batch["position_features"], batch["mask"]).cpu()
        for batch_index, rows in enumerate(raw["row_indices"]):
            for local_index, row_index in enumerate(rows):
                source = corpus.frame.iloc[row_index]
                logit = float(logits[batch_index, local_index])
                records.append(
                    {
                        **source.to_dict(),
                        "raw_logit": logit,
                        "raw_probability": float(torch.sigmoid(torch.tensor(logit))),
                    }
                )
    return pd.DataFrame(records)


def _trace_weighted_dev_bce(frame: pd.DataFrame) -> float:
    logits = torch.tensor(frame["raw_logit"].to_numpy(float), dtype=torch.float64)
    target = torch.tensor(frame["prefix_valid"].to_numpy(float), dtype=torch.float64)
    losses = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none").numpy()
    return float(
        pd.DataFrame({"trace_id": frame["trace_id"].astype(str), "loss": losses})
        .groupby("trace_id", sort=False)["loss"]
        .mean()
        .mean()
    )


def _fit_candidate(
    corpus: PrefixValidityCorpus,
    *,
    architecture: Architecture,
    learning_rate: float,
    seed: int,
    config: PrefixValidityTrainingConfig,
    device: str | torch.device,
) -> ProbeCandidate:
    _set_seed(seed)
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but unavailable")
    model = build_predictor(
        architecture,
        corpus.input_dim,
        hidden_width=config.hidden_width,
        dropout=config.dropout,
    )
    position_mean: list[float] | None = None
    position_std: list[float] | None = None
    if architecture == "position_only":
        mean, std = _position_normalization(corpus)
        model.set_position_normalization(mean, std)
        position_mean, position_std = mean.tolist(), std.tolist()
    model.to(resolved)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(config.weight_decay)
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_losses: list[float] = []
        for raw in corpus.loader(
            "train",
            batch_size_traces=config.batch_size_traces,
            shuffle=True,
            seed=seed + epoch,
        ):
            batch = _move(raw, resolved)
            logits = model(batch["hidden_states"], batch["position_features"], batch["mask"])
            loss = trace_weighted_binary_bce(logits, batch["target"], batch["mask"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            train_losses.append(float(loss.detach()))
        dev = predict_probe(
            model,
            corpus,
            split="architecture_dev",
            batch_size_traces=config.batch_size_traces,
            device=resolved,
        )
        dev_loss = _trace_weighted_dev_bce(dev)
        history.append(
            {
                "epoch": epoch,
                "train_batch_mean_bce": float(np.mean(train_losses)),
                "architecture_dev_trace_weighted_bce": dev_loss,
            }
        )
        if dev_loss < best_loss - 1e-8:
            best_loss = dev_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None or not math.isfinite(best_loss):
        raise RuntimeError("prefix-validity training produced no finite dev checkpoint")
    return ProbeCandidate(
        architecture=architecture,
        learning_rate=float(learning_rate),
        seed=int(seed),
        best_epoch=best_epoch,
        dev_loss=best_loss,
        state_dict={key: value.detach().cpu() for key, value in best_state.items()},
        predictor_config=model.config.to_dict(),
        history=history,
        position_mean=position_mean,
        position_std=position_std,
    )


def fit_probe_matrix(
    corpus: PrefixValidityCorpus,
    *,
    architecture: Architecture,
    config: PrefixValidityTrainingConfig = PrefixValidityTrainingConfig(),
    device: str | torch.device = "cpu",
) -> list[ProbeCandidate]:
    candidates = [
        _fit_candidate(
            corpus,
            architecture=architecture,
            learning_rate=learning_rate,
            seed=seed,
            config=config,
            device=device,
        )
        for learning_rate in config.learning_rates
        for seed in config.seeds
    ]
    if len(candidates) != 6:
        raise RuntimeError("prefix-validity matrix must contain two LRs x three seeds")
    return candidates


def select_learning_rate_and_median_seed(
    candidates: Sequence[ProbeCandidate],
) -> tuple[ProbeCandidate, dict[str, Any]]:
    """Select LR by mean dev BCE, then seed nearest median dev BCE."""

    if not candidates or len({item.architecture for item in candidates}) != 1:
        raise ValueError("candidate selection requires one nonempty architecture")
    frame = pd.DataFrame(
        [
            {
                "learning_rate": item.learning_rate,
                "seed": item.seed,
                "dev_loss": item.dev_loss,
            }
            for item in candidates
        ]
    )
    if bool(frame[["learning_rate", "seed"]].duplicated().any()):
        raise ValueError("duplicate LR/seed candidate")
    counts = frame.groupby("learning_rate")["seed"].nunique()
    if len(counts) != 2 or set(counts.to_numpy(int)) != {3}:
        raise ValueError("selection requires two learning rates and three seeds per LR")
    summary = (
        frame.groupby("learning_rate", as_index=False)
        .agg(mean_dev_bce=("dev_loss", "mean"), std_dev_bce=("dev_loss", "std"))
        .sort_values(["mean_dev_bce", "learning_rate"], kind="stable")
    )
    learning_rate = float(summary.iloc[0]["learning_rate"])
    eligible = [item for item in candidates if item.learning_rate == learning_rate]
    median = float(np.median([item.dev_loss for item in eligible]))
    selected = min(eligible, key=lambda item: (abs(item.dev_loss - median), item.seed))
    return selected, {
        "selection_split": "architecture_dev",
        "selection_metric": "trace_weighted_unweighted_binary_cross_entropy",
        "selected_learning_rate": learning_rate,
        "selected_seed": int(selected.seed),
        "median_seed_dev_bce": median,
        "learning_rate_summary": summary.to_dict(orient="records"),
        "test_used_for_selection": False,
        "native_outcomes_used": False,
    }


def save_frozen_probe(candidate: ProbeCandidate, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save({"state_dict": candidate.state_dict, "metadata": candidate.metadata()}, temporary)
    temporary.replace(destination)


def load_frozen_probe(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> RecoverabilityPredictor:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    config = metadata["predictor_config"]
    predictor = build_predictor(
        str(config["architecture"]),
        int(config["input_dim"]),
        hidden_width=int(config["hidden_width"]),
        dropout=float(config["dropout"]),
    )
    predictor.load_state_dict(payload["state_dict"])
    return predictor.to(torch.device(device)).eval()


def training_manifest(config: PrefixValidityTrainingConfig) -> dict[str, Any]:
    return {
        **asdict(config),
        "architectures": ["linear_probe", "position_only"],
        "hidden_architecture": "LayerNorm_then_Linear",
        "position_features": [
            "checkpoint_ordinal",
            "prefix_token_count",
            "checkpoint_ordinal_fraction",
            "prefix_token_fraction",
        ],
        "loss": "unweighted_binary_cross_entropy",
        "trace_weighting": "mean_checkpoint_bce_within_trace_then_mean_traces",
        "native_outcomes_used": False,
    }
