"""Reusable causal scoring interface for later, separately governed evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .training import load_trained_predictor


class BoundaryScorer:
    """Score an ordered checkpoint prefix with a frozen predictor and calibrator."""

    def __init__(
        self,
        *,
        base_model: str,
        predictor_checkpoint: Path,
        calibration_artifact: Path,
        device: str = "cpu",
    ) -> None:
        self.base_model = str(base_model)
        self.device = torch.device(device)
        self.predictor = load_trained_predictor(
            Path(predictor_checkpoint), device=self.device
        )
        self.calibration = json.loads(Path(calibration_artifact).read_text())
        if str(self.calibration.get("base_model")) != self.base_model:
            raise ValueError("calibration artifact belongs to a different base model")
        if self.calibration.get("fit_split") != "calibration":
            raise ValueError("calibration artifact was not fit on the frozen calibration split")
        if float(self.calibration["a"]) <= 0:
            raise ValueError("calibration slope must be positive")

    @staticmethod
    def _position_tensor(metadata: Sequence[Mapping[str, Any]]) -> torch.Tensor:
        rows: list[list[float]] = []
        for index, row in enumerate(metadata):
            ordinal = float(row.get("checkpoint_ordinal", index))
            prefix = float(row["prefix_token_count"])
            checkpoint_total = max(float(row["total_checkpoint_count"]), 1.0)
            token_total = max(float(row["total_trace_token_count"]), 1.0)
            rows.append([ordinal, prefix, ordinal / checkpoint_total, prefix / token_total])
        return torch.tensor(rows, dtype=torch.float32)

    @torch.inference_mode()
    def score(
        self,
        ordered_checkpoint_hidden_states: torch.Tensor | np.ndarray,
        checkpoint_metadata: Sequence[Mapping[str, Any]],
        *,
        tau: float | None = None,
    ) -> dict[str, Any]:
        hidden = torch.as_tensor(ordered_checkpoint_hidden_states, dtype=torch.float32)
        if hidden.ndim != 2 or not len(hidden):
            raise ValueError("ordered checkpoint hidden states must have shape [T, D]")
        if len(checkpoint_metadata) != len(hidden):
            raise ValueError("checkpoint metadata length differs from hidden-state sequence")
        if hidden.shape[1] != self.predictor.config.input_dim:
            raise ValueError(
                f"hidden dimension {hidden.shape[1]} != expected {self.predictor.config.input_dim}"
            )
        ordinals = [int(row.get("checkpoint_ordinal", index)) for index, row in enumerate(checkpoint_metadata)]
        if ordinals != list(range(len(ordinals))):
            raise ValueError("checkpoints must be ordered and zero-based")
        positions = self._position_tensor(checkpoint_metadata)
        mask = torch.ones((1, len(hidden)), dtype=torch.bool, device=self.device)
        logits = self.predictor(
            hidden.unsqueeze(0).to(self.device),
            positions.unsqueeze(0).to(self.device),
            mask,
        )[0].cpu()
        raw_probability = torch.sigmoid(logits)
        calibrated_probability = torch.sigmoid(
            float(self.calibration["a"]) * logits + float(self.calibration["b"])
        )
        checkpoints = [
            {
                **dict(metadata),
                "raw_logit": float(logit),
                "raw_probability": float(raw),
                "calibrated_repair_success_probability": float(calibrated),
            }
            for metadata, logit, raw, calibrated in zip(
                checkpoint_metadata, logits, raw_probability, calibrated_probability
            )
        ]
        decision: dict[str, Any] | None = None
        if tau is not None:
            if not 0.0 <= float(tau) <= 1.0:
                raise ValueError("tau must lie in [0, 1]")
            eligible = [
                index
                for index, probability in enumerate(calibrated_probability)
                if float(probability) >= float(tau)
            ]
            if eligible:
                selected_index = eligible[-1]
                decision = {
                    "decision": "checkpoint",
                    "selected_checkpoint_index": selected_index,
                    "selected_checkpoint": checkpoints[selected_index],
                    "tau": float(tau),
                }
            else:
                decision = {
                    "decision": "full_regeneration_fallback",
                    "selected_checkpoint_index": None,
                    "selected_checkpoint": None,
                    "tau": float(tau),
                }
        return {
            "base_model": self.base_model,
            "architecture": self.predictor.architecture,
            "checkpoints": checkpoints,
            "decision": decision,
        }
