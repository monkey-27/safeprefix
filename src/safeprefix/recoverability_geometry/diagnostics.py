"""Train-only diagnostic models for H1 and cross-domain transfer.

This module intentionally accepts arrays rather than artifact paths.  The
runner is responsible for split/integrity validation before calling it.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import random
from typing import Any, Literal

import numpy as np
import torch
from torch import nn


DiagnosticKind = Literal["axis_only", "orthogonal_linear", "orthogonal_mlp", "axis_plus_residual", "full_linear", "layernorm_linear"]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DiagnosticPredictor(nn.Module):
    def __init__(self, kind: DiagnosticKind, dimension: int) -> None:
        super().__init__()
        self.kind = kind
        if kind == "axis_only":
            self.model = nn.Linear(1, 1)
        elif kind in {"orthogonal_linear", "full_linear"}:
            self.model = nn.Linear(dimension, 1)
        elif kind == "layernorm_linear":
            self.model = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 1))
        elif kind == "orthogonal_mlp":
            self.model = nn.Sequential(
                nn.Linear(dimension, 32), nn.GELU(), nn.Dropout(0.2), nn.Linear(32, 1)
            )
        elif kind == "axis_plus_residual":
            self.axis = nn.Linear(1, 1)
            self.orthogonal = nn.Sequential(
                nn.Linear(dimension, 32), nn.GELU(), nn.Dropout(0.2), nn.Linear(32, 1, bias=False)
            )
        else:
            raise ValueError(f"unknown diagnostic kind: {kind}")

    def forward(self, axis_score: torch.Tensor, orthogonal: torch.Tensor) -> torch.Tensor:
        if self.kind == "axis_only":
            return self.model(axis_score[:, None]).squeeze(1)
        if self.kind in {"orthogonal_linear", "orthogonal_mlp", "full_linear", "layernorm_linear"}:
            return self.model(orthogonal).squeeze(1)
        return (self.axis(axis_score[:, None]) + self.orthogonal(orthogonal)).squeeze(1)


@dataclass(frozen=True)
class DiagnosticFit:
    kind: str
    seed: int
    best_epoch: int
    best_dev_nll: float
    train_trace_count: int
    dev_trace_count: int
    state_dict: dict[str, torch.Tensor]
    history: tuple[dict[str, float | int], ...]


def _trace_weights(trace_ids: np.ndarray) -> torch.Tensor:
    unique, inverse, counts = np.unique(trace_ids.astype(str), return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse] / len(unique)
    return torch.tensor(weights, dtype=torch.float32)


def _nll(
    logits: torch.Tensor,
    successes: torch.Tensor,
    trials: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    target = successes / trials
    loss = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * weights).sum()


def train_diagnostic(
    *,
    kind: DiagnosticKind,
    train_axis_score: np.ndarray,
    train_features: np.ndarray,
    train_successes: np.ndarray,
    train_trials: np.ndarray,
    train_trace_ids: np.ndarray,
    dev_axis_score: np.ndarray,
    dev_features: np.ndarray,
    dev_successes: np.ndarray,
    dev_trials: np.ndarray,
    dev_trace_ids: np.ndarray,
    seed: int,
    max_epochs: int = 50,
    patience: int = 7,
    device: str | torch.device | None = None,
    batch_size_traces: int = 64,
) -> DiagnosticFit:
    """Fit one fixed diagnostic with train labels and dev-only early stopping.

    Linear diagnostics reproduce the frozen boundary-v1 optimization protocol
    (50 epochs, patience 7, lr 1e-3, weight decay 1e-4).  The two pre-registered
    residual MLP diagnostics use their separately frozen lr/dropout/weight-decay
    specification.  CUDA is used when available solely as an execution
    acceleration; the objective, rows, and stopping rule are unchanged.
    """
    _seed_everything(seed)
    arrays = [train_axis_score, train_features, train_successes, train_trials]
    if len({len(value) for value in arrays}) != 1 or len(train_trace_ids) != len(train_axis_score):
        raise ValueError("training diagnostic arrays are not aligned")
    arrays = [dev_axis_score, dev_features, dev_successes, dev_trials]
    if len({len(value) for value in arrays}) != 1 or len(dev_trace_ids) != len(dev_axis_score):
        raise ValueError("development diagnostic arrays are not aligned")
    if np.any(train_trials <= 0) or np.any(dev_trials <= 0):
        raise ValueError("trial counts must be positive")
    dimension = int(train_features.shape[1])
    if dev_features.shape[1] != dimension:
        raise ValueError("train/dev feature dimensions differ")
    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = DiagnosticPredictor(kind, dimension).to(resolved_device)
    learning_rate = 1e-3 if kind in {"axis_only", "orthogonal_linear", "full_linear", "layernorm_linear"} else 3e-4
    weight_decay = 1e-4 if kind in {"axis_only", "orthogonal_linear", "full_linear", "layernorm_linear"} else 1e-3
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def tensors(axis: np.ndarray, features: np.ndarray, successes: np.ndarray, trials: np.ndarray, traces: np.ndarray) -> tuple[torch.Tensor, ...]:
        return (
            torch.tensor(axis, dtype=torch.float32, device=resolved_device),
            torch.tensor(features, dtype=torch.float32, device=resolved_device),
            torch.tensor(successes, dtype=torch.float32, device=resolved_device),
            torch.tensor(trials, dtype=torch.float32, device=resolved_device),
            _trace_weights(traces).to(resolved_device),
        )

    train = tensors(train_axis_score, train_features, train_successes, train_trials, train_trace_ids)
    dev = tensors(dev_axis_score, dev_features, dev_successes, dev_trials, dev_trace_ids)
    train_trace_strings = np.asarray(list(map(str, train_trace_ids)), dtype=object)
    unique_train_traces = np.asarray(sorted(set(train_trace_strings)), dtype=object)
    if batch_size_traces <= 0:
        raise ValueError("batch_size_traces must be positive")
    rows_by_trace = {
        trace: np.flatnonzero(train_trace_strings == trace) for trace in unique_train_traces
    }
    epoch_generator = np.random.default_rng(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_nll = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        permutation = epoch_generator.permutation(unique_train_traces)
        epoch_losses: list[float] = []
        for start in range(0, len(permutation), batch_size_traces):
            batch_traces = permutation[start : start + batch_size_traces]
            row_index = np.concatenate([rows_by_trace[str(trace)] for trace in batch_traces])
            index = torch.tensor(row_index, dtype=torch.long, device=resolved_device)
            batch_weights = _trace_weights(train_trace_strings[row_index]).to(resolved_device)
            logits = model(train[0][index], train[1][index])
            loss = _nll(logits, train[2][index], train[3][index], batch_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            dev_loss = float(_nll(model(dev[0], dev[1]), dev[2], dev[3], dev[4]))
        history.append({
            "epoch": epoch,
            "train_nll": float(np.mean(epoch_losses)),
            "dev_nll": dev_loss,
            "optimizer_steps": int(len(epoch_losses)),
        })
        if dev_loss < best_nll - 1e-8:
            best_nll = dev_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("diagnostic training produced no checkpoint")
    return DiagnosticFit(
        kind=kind,
        seed=seed,
        best_epoch=best_epoch,
        best_dev_nll=best_nll,
        train_trace_count=len(set(map(str, train_trace_ids))),
        dev_trace_count=len(set(map(str, dev_trace_ids))),
        state_dict={key: value.detach().cpu() for key, value in best_state.items()},
        history=tuple(history),
    )


def predict_diagnostic(
    fit: DiagnosticFit,
    *,
    axis_score: np.ndarray,
    features: np.ndarray,
    device: str | torch.device | None = None,
) -> np.ndarray:
    resolved_device = torch.device(device if device is not None else "cpu")
    model = DiagnosticPredictor(fit.kind, int(features.shape[1])).to(resolved_device)
    model.load_state_dict(fit.state_dict)
    model.eval()
    with torch.inference_mode():
        logits = model(
            torch.tensor(axis_score, dtype=torch.float32, device=resolved_device),
            torch.tensor(features, dtype=torch.float32, device=resolved_device),
        )
    return logits.cpu().numpy().astype(float)
