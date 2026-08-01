"""Trace-weighted training and prediction for recoverability probes."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import atomic_json, atomic_parquet, load_config, sha256_file, stable_hash
from .models import RecoverabilityPredictor, build_predictor


def learning_rate_slug(learning_rate: float) -> str:
    return f"lr_{learning_rate:.0e}".replace("+", "")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalized_binomial_nll(
    logits: torch.Tensor,
    success_count: torch.Tensor,
    num_rollouts: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if bool((num_rollouts[mask] <= 0).any()):
        raise ValueError("rollout counts must be positive")
    target = success_count / num_rollouts
    checkpoint_loss = nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )
    valid = mask.to(checkpoint_loss.dtype)
    per_trace = (checkpoint_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
    return per_trace.mean()


def position_features(frame: pd.DataFrame) -> torch.Tensor:
    ordinal = frame["checkpoint_ordinal"].to_numpy(dtype=np.float32)
    prefix = frame["prefix_token_count"].to_numpy(dtype=np.float32)
    checkpoint_total = frame["total_checkpoint_count"].to_numpy(dtype=np.float32)
    token_total = frame["total_trace_token_count"].to_numpy(dtype=np.float32)
    values = np.stack(
        [
            ordinal,
            prefix,
            ordinal / np.maximum(checkpoint_total, 1.0),
            prefix / np.maximum(token_total, 1.0),
        ],
        axis=1,
    )
    return torch.from_numpy(values)


@dataclass(frozen=True)
class TraceIndex:
    trace_id: str
    frame_indices: tuple[int, ...]
    feature_indices: tuple[int, ...]


class TraceDataset(Dataset[TraceIndex]):
    def __init__(self, traces: list[TraceIndex]) -> None:
        self.traces = traces

    def __len__(self) -> int:
        return len(self.traces)

    def __getitem__(self, index: int) -> TraceIndex:
        return self.traces[index]


class ModelCorpus:
    def __init__(self, artifact_root: Path, model_key: str) -> None:
        self.artifact_root = artifact_root
        self.model_key = model_key
        manifest = pd.read_parquet(artifact_root / "data/canonical_checkpoint_manifest.parquet")
        self.frame = manifest.loc[manifest["base_model"] == model_key].copy()
        self.frame.sort_values(["trace_id", "checkpoint_ordinal"], inplace=True)
        self.frame.reset_index(drop=True, inplace=True)
        payload = torch.load(
            artifact_root / f"data/features/{model_key}.pt",
            map_location="cpu",
            weights_only=False,
        )
        self.features = payload["features"].to(torch.float32)
        self.feature_metadata = {key: value for key, value in payload.items() if key != "features"}
        if self.features.ndim != 2 or len(self.features) != len(self.frame):
            raise RuntimeError(f"{model_key}: feature store and canonical manifest differ")
        if set(self.frame["num_rollouts"].astype(int)) != {4}:
            raise RuntimeError(f"{model_key}: training corpus is not exact k=4")
        expected = set(range(len(self.features)))
        observed = set(self.frame["feature_row_index"].astype(int))
        if observed != expected:
            raise RuntimeError(f"{model_key}: feature row index is not a bijection")
        self.positions = position_features(self.frame)
        self.successes = torch.tensor(self.frame["success_count"].to_numpy(), dtype=torch.float32)
        self.trials = torch.tensor(self.frame["num_rollouts"].to_numpy(), dtype=torch.float32)
        self._traces_by_split: dict[str, list[TraceIndex]] = {}
        for split, split_frame in self.frame.groupby("split", sort=False):
            traces: list[TraceIndex] = []
            for trace_id, group in split_frame.groupby("trace_id", sort=True):
                frame_indices = tuple(map(int, group.index))
                feature_indices = tuple(map(int, group["feature_row_index"]))
                if list(group["checkpoint_ordinal"].astype(int)) != list(range(len(group))):
                    raise RuntimeError(f"{model_key}/{trace_id}: checkpoint sequence is not ordered")
                traces.append(TraceIndex(str(trace_id), frame_indices, feature_indices))
            self._traces_by_split[str(split)] = traces

    @property
    def input_dim(self) -> int:
        return int(self.features.shape[1])

    def traces(self, split: str) -> list[TraceIndex]:
        values = self._traces_by_split.get(split, [])
        if not values:
            raise RuntimeError(f"{self.model_key}: split {split!r} is empty")
        return values

    def collate(self, traces: list[TraceIndex]) -> dict[str, Any]:
        maximum = max(len(trace.frame_indices) for trace in traces)
        batch = len(traces)
        hidden = torch.zeros((batch, maximum, self.input_dim), dtype=torch.float32)
        positions = torch.zeros((batch, maximum, 4), dtype=torch.float32)
        successes = torch.zeros((batch, maximum), dtype=torch.float32)
        trials = torch.ones((batch, maximum), dtype=torch.float32)
        mask = torch.zeros((batch, maximum), dtype=torch.bool)
        frame_indices: list[list[int]] = []
        for row, trace in enumerate(traces):
            count = len(trace.frame_indices)
            fidx = torch.tensor(trace.feature_indices, dtype=torch.long)
            midx = torch.tensor(trace.frame_indices, dtype=torch.long)
            hidden[row, :count] = self.features[fidx]
            positions[row, :count] = self.positions[midx]
            successes[row, :count] = self.successes[midx]
            trials[row, :count] = self.trials[midx]
            mask[row, :count] = True
            frame_indices.append(list(trace.frame_indices))
        return {
            "hidden_states": hidden,
            "position_features": positions,
            "success_count": successes,
            "num_rollouts": trials,
            "mask": mask,
            "frame_indices": frame_indices,
            "trace_ids": [trace.trace_id for trace in traces],
        }

    def loader(
        self,
        split: str,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> DataLoader[TraceIndex]:
        return DataLoader(
            TraceDataset(self.traces(split)),
            batch_size=min(batch_size, len(self.traces(split))),
            shuffle=shuffle,
            num_workers=0,
            collate_fn=self.collate,
            generator=torch.Generator().manual_seed(seed),
        )


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


@torch.inference_mode()
def predict_split(
    model: RecoverabilityPredictor,
    corpus: ModelCorpus,
    *,
    split: str,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> pd.DataFrame:
    model.eval()
    records: list[dict[str, Any]] = []
    loader = corpus.loader(split, batch_size=batch_size, shuffle=False, seed=seed)
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        logits = model(
            batch["hidden_states"], batch["position_features"], batch["mask"]
        ).cpu()
        for row, indices in enumerate(raw_batch["frame_indices"]):
            for local_index, frame_index in enumerate(indices):
                source = corpus.frame.iloc[frame_index]
                logit = float(logits[row, local_index])
                records.append(
                    {
                        "base_model": corpus.model_key,
                        "trace_id": source["trace_id"],
                        "common_trace_id": source["common_trace_id"],
                        "problem_id": source["problem_id"],
                        "problem_group": source["problem_group"],
                        "domain": source["domain"],
                        "split": source["split"],
                        "checkpoint_id": source["checkpoint_id"],
                        "checkpoint_ordinal": int(source["checkpoint_ordinal"]),
                        "checkpoint_token_offset": int(source["checkpoint_token_offset"]),
                        "prefix_token_count": int(source["prefix_token_count"]),
                        "total_trace_token_count": int(source["total_trace_token_count"]),
                        "total_checkpoint_count": int(source["total_checkpoint_count"]),
                        "success_count": int(source["success_count"]),
                        "num_rollouts": int(source["num_rollouts"]),
                        "observed_success_rate": float(source["observed_success_rate"]),
                        "first_error_zero_based_analysis_only": int(
                            source["first_error_zero_based_analysis_only"]
                        ),
                        "raw_logit": logit,
                        "raw_probability": float(torch.sigmoid(torch.tensor(logit))),
                    }
                )
    return pd.DataFrame(records)


def trace_weighted_nll(frame: pd.DataFrame, probability_column: str) -> float:
    probability = np.clip(frame[probability_column].to_numpy(float), 1e-8, 1 - 1e-8)
    observed = frame["observed_success_rate"].to_numpy(float)
    losses = -(observed * np.log(probability) + (1 - observed) * np.log(1 - probability))
    values = pd.DataFrame({"trace_id": frame["trace_id"].to_numpy(), "loss": losses})
    return float(values.groupby("trace_id", sort=False)["loss"].mean().mean())


def dev_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    overall = trace_weighted_nll(frame, "raw_probability")
    by_domain = {
        str(domain): trace_weighted_nll(group, "raw_probability")
        for domain, group in frame.groupby("domain")
    }
    return {
        "trace_weighted_binomial_nll": overall,
        "domain_macro_trace_weighted_binomial_nll": float(np.mean(list(by_domain.values()))),
        "by_domain": by_domain,
        "traces": int(frame["trace_id"].nunique()),
        "checkpoints": len(frame),
    }


def _position_normalization(corpus: ModelCorpus) -> tuple[torch.Tensor, torch.Tensor]:
    indices = [index for trace in corpus.traces("train") for index in trace.frame_indices]
    values = corpus.positions[torch.tensor(indices, dtype=torch.long)]
    return values.mean(dim=0), values.std(dim=0, unbiased=False).clamp_min(1e-8)


def train_one(
    *,
    corpus: ModelCorpus,
    config: Mapping[str, Any],
    artifact_root: Path,
    architecture: str,
    learning_rate: float,
    seed: int,
    device_name: str,
) -> dict[str, Any]:
    training = config["training"]
    identifier = learning_rate_slug(learning_rate)
    run_root = artifact_root / f"training/{corpus.model_key}/{architecture}/{identifier}/seed_{seed}"
    complete_path = run_root / "complete.json"
    resolved = {
        "base_model": corpus.model_key,
        "architecture": architecture,
        "learning_rate": float(learning_rate),
        "seed": int(seed),
        "input_dim": corpus.input_dim,
        "hidden_width": int(training["hidden_width"]),
        "dropout": float(training["dropout"]),
        "weight_decay": float(training["weight_decay"]),
        "gradient_clip_norm": float(training["gradient_clip_norm"]),
        "max_epochs": int(training["max_epochs"]),
        "patience": int(training["patience"]),
        "batch_size_traces": int(training["batch_size_traces"]),
        "objective": "trace_weighted_normalized_binomial_nll",
        "target": "absolute_success_count_over_num_rollouts",
        "input_columns": [
            "final_layer_checkpoint_hidden_state",
            *(
                [
                    "checkpoint_ordinal",
                    "prefix_token_count",
                    "checkpoint_ordinal_fraction",
                    "prefix_token_fraction",
                ]
                if architecture == "position_only"
                else []
            ),
        ],
        "native_evaluation_used": False,
        "first_error_used_as_target": False,
        "repository_commit": json.loads(
            (artifact_root / "data/environment.json").read_text()
        ).get("repository_commit"),
    }
    run_hash = stable_hash(resolved.items())
    if complete_path.is_file():
        existing = json.loads(complete_path.read_text())
        if existing.get("run_hash") != run_hash:
            raise RuntimeError(f"completed run identity differs: {run_root}")
        return existing
    run_root.mkdir(parents=True, exist_ok=True)
    resolved_path = run_root / "resolved_config.json"
    if resolved_path.is_file():
        existing = json.loads(resolved_path.read_text())
        if existing.get("run_hash") != run_hash:
            raise RuntimeError(f"partial run identity differs: {run_root}")
    else:
        atomic_json(resolved_path, {**resolved, "run_hash": run_hash})

    set_seed(seed)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but unavailable")
    torch.set_float32_matmul_precision("high")
    model = build_predictor(
        architecture,
        corpus.input_dim,
        hidden_width=int(training["hidden_width"]),
        dropout=float(training["dropout"]),
    )
    if architecture == "position_only":
        mean, std = _position_normalization(corpus)
        model.set_position_normalization(mean, std)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(training["weight_decay"]),
    )
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_dev = math.inf
    best_epoch = 0
    stale = 0
    start_epoch = 1
    max_epochs = int(training["max_epochs"])
    patience = int(training["patience"])
    batch_size = int(training["batch_size_traces"])
    resume_path = run_root / "resume.pt"
    if resume_path.is_file():
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        if resume.get("run_hash") != run_hash:
            raise RuntimeError(f"resume checkpoint identity differs: {run_root}")
        model.load_state_dict(resume["model_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        history = list(resume["history"])
        best_state = resume["best_state_dict"]
        best_dev = float(resume["best_dev"])
        best_epoch = int(resume["best_epoch"])
        stale = int(resume["stale"])
        start_epoch = int(resume["epoch"]) + 1
        torch.set_rng_state(resume["torch_rng_state"])
        if device.type == "cuda" and resume.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(resume["cuda_rng_state_all"])
        if stale >= patience:
            start_epoch = max_epochs + 1
    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        train_losses: list[float] = []
        loader = corpus.loader("train", batch_size=batch_size, shuffle=True, seed=seed + epoch)
        for raw_batch in loader:
            batch = _to_device(raw_batch, device)
            logits = model(
                batch["hidden_states"], batch["position_features"], batch["mask"]
            )
            loss = normalized_binomial_nll(
                logits,
                batch["success_count"],
                batch["num_rollouts"],
                batch["mask"],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            optimizer.step()
            train_losses.append(float(loss.detach()))
        dev_predictions = predict_split(
            model,
            corpus,
            split="architecture_dev",
            batch_size=batch_size,
            device=device,
            seed=seed,
        )
        metrics = dev_metrics(dev_predictions)
        dev_nll = float(metrics["trace_weighted_binomial_nll"])
        history.append(
            {
                "epoch": epoch,
                "train_batch_mean_nll": float(np.mean(train_losses)),
                "architecture_dev_nll": dev_nll,
                "architecture_dev_domain_macro_nll": metrics[
                    "domain_macro_trace_weighted_binomial_nll"
                ],
            }
        )
        if dev_nll < best_dev - 1e-8:
            best_dev = dev_nll
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        temporary_resume = resume_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "run_hash": run_hash,
                "epoch": epoch,
                "model_state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "best_state_dict": best_state,
                "best_dev": best_dev,
                "best_epoch": best_epoch,
                "stale": stale,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all()
                if device.type == "cuda"
                else None,
            },
            temporary_resume,
        )
        temporary_resume.replace(resume_path)
        atomic_json(run_root / "history.json", history)
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("training did not produce a finite development checkpoint")
    model.load_state_dict(best_state)
    final_dev = predict_split(
        model,
        corpus,
        split="architecture_dev",
        batch_size=batch_size,
        device=device,
        seed=seed,
    )
    final_metrics = dev_metrics(final_dev)
    final_dev["architecture"] = architecture
    final_dev["learning_rate"] = float(learning_rate)
    final_dev["training_seed"] = int(seed)
    atomic_parquet(run_root / "architecture_dev_predictions.parquet", final_dev)
    atomic_json(run_root / "history.json", history)
    checkpoint_path = run_root / "best.pt"
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in best_state.items()},
            "predictor_config": model.config.to_dict(),
            "resolved_training_config": resolved,
            "best_epoch": best_epoch,
            "best_dev_metrics": final_metrics,
            "run_hash": run_hash,
        },
        temporary,
    )
    temporary.replace(checkpoint_path)
    complete = {
        "status": "COMPLETE",
        "run_hash": run_hash,
        "base_model": corpus.model_key,
        "architecture": architecture,
        "learning_rate": float(learning_rate),
        "seed": int(seed),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "trainable_parameters": model.trainable_parameters,
        "dev_metrics": final_metrics,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "native_evaluation_used": False,
    }
    atomic_json(complete_path, complete)
    return complete


def train_model_matrix(
    *,
    config_path: Path,
    artifact_root: Path,
    model_key: str,
    device: str,
) -> dict[str, Any]:
    config = load_config(config_path)
    corpus = ModelCorpus(artifact_root, model_key)
    complete: list[dict[str, Any]] = []
    seeds = list(map(int, config["experiment"]["training_seeds"]))
    position_lr = float(config["training"]["position_only_learning_rate"])
    for seed in seeds:
        complete.append(
            train_one(
                corpus=corpus,
                config=config,
                artifact_root=artifact_root,
                architecture="position_only",
                learning_rate=position_lr,
                seed=seed,
                device_name=device,
            )
        )
    for architecture in map(str, config["training"]["architectures"]):
        for learning_rate in map(float, config["training"]["learning_rates"]):
            for seed in seeds:
                complete.append(
                    train_one(
                        corpus=corpus,
                        config=config,
                        artifact_root=artifact_root,
                        architecture=architecture,
                        learning_rate=learning_rate,
                        seed=seed,
                        device_name=device,
                    )
                )
    summary = {
        "status": "COMPLETE",
        "base_model": model_key,
        "runs": len(complete),
        "expected_runs": len(seeds)
        * (1 + len(config["training"]["architectures"]) * len(config["training"]["learning_rates"])),
        "results": complete,
        "native_evaluation_used": False,
    }
    if summary["runs"] != summary["expected_runs"]:
        raise RuntimeError(f"{model_key}: incomplete training matrix")
    atomic_json(artifact_root / f"training/{model_key}/matrix_summary.json", summary)
    return summary


def load_trained_predictor(path: Path, *, device: torch.device) -> RecoverabilityPredictor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    predictor_config = payload["predictor_config"]
    model = build_predictor(
        predictor_config["architecture"],
        int(predictor_config["input_dim"]),
        hidden_width=int(predictor_config["hidden_width"]),
        dropout=float(predictor_config["dropout"]),
    )
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()
