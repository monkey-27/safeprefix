"""Small causal predictors of checkpoint recoverability."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


ARCHITECTURE_COMPLEXITY = {
    "linear_probe": 1,
    "local_mlp": 2,
    "change_aware_mlp": 3,
    "causal_gru": 4,
}


@dataclass(frozen=True)
class PredictorConfig:
    architecture: str
    input_dim: int
    hidden_width: int = 128
    dropout: float = 0.1

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


class RecoverabilityPredictor(nn.Module):
    """Return one recoverability logit for every visible checkpoint."""

    def __init__(self, config: PredictorConfig) -> None:
        super().__init__()
        self.config = config
        architecture = config.architecture
        width = int(config.hidden_width)
        dropout = float(config.dropout)
        if architecture == "position_only":
            self.register_buffer("position_mean", torch.zeros(4))
            self.register_buffer("position_std", torch.ones(4))
            self.position_model = nn.Linear(4, 1)
        elif architecture == "linear_probe":
            self.local_model = nn.Sequential(
                nn.LayerNorm(config.input_dim),
                nn.Linear(config.input_dim, 1),
            )
        elif architecture == "local_mlp":
            self.local_model = nn.Sequential(
                nn.LayerNorm(config.input_dim),
                nn.Linear(config.input_dim, width),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(width, 1),
            )
        elif architecture == "change_aware_mlp":
            projection_width = width // 2
            self.current_projection = nn.Sequential(
                nn.LayerNorm(config.input_dim),
                nn.Linear(config.input_dim, projection_width),
            )
            self.change_projection = nn.Sequential(
                nn.LayerNorm(config.input_dim),
                nn.Linear(config.input_dim, projection_width),
            )
            self.change_head = nn.Sequential(
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(width, 1),
            )
        elif architecture == "causal_gru":
            self.sequence_projection = nn.Sequential(
                nn.LayerNorm(config.input_dim),
                nn.Linear(config.input_dim, width),
                nn.GELU(),
            )
            self.gru = nn.GRU(
                width,
                width,
                num_layers=1,
                batch_first=True,
                bidirectional=False,
                dropout=0.0,
            )
            self.sequence_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, 1))
        else:
            raise ValueError(f"unknown architecture: {architecture}")

    @property
    def architecture(self) -> str:
        return self.config.architecture

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def set_position_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if self.architecture != "position_only" or mean.shape != (4,) or std.shape != (4,):
            raise ValueError("position normalization requires four features on position_only")
        self.position_mean.copy_(mean.detach())
        self.position_std.copy_(std.detach().clamp_min(1e-8))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3 or position_features.shape[:2] != hidden_states.shape[:2]:
            raise ValueError("hidden and position tensors must be batch-aligned sequences")
        if mask.shape != hidden_states.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("mask must be a boolean [batch, checkpoints] tensor")
        architecture = self.architecture
        if architecture == "position_only":
            normalized = (position_features - self.position_mean) / self.position_std
            logits = self.position_model(normalized).squeeze(-1)
        elif architecture in {"linear_probe", "local_mlp"}:
            logits = self.local_model(hidden_states).squeeze(-1)
        elif architecture == "change_aware_mlp":
            differences = torch.zeros_like(hidden_states)
            differences[:, 1:] = hidden_states[:, 1:] - hidden_states[:, :-1]
            combined = torch.cat(
                [self.current_projection(hidden_states), self.change_projection(differences)],
                dim=-1,
            )
            logits = self.change_head(combined).squeeze(-1)
        else:
            projected = self.sequence_projection(hidden_states)
            lengths = mask.sum(dim=1).clamp_min(1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                projected, lengths, batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.gru(packed)
            encoded, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=hidden_states.shape[1],
            )
            logits = self.sequence_head(encoded).squeeze(-1)
        return logits.masked_fill(~mask, 0.0)


def build_predictor(
    architecture: str,
    input_dim: int,
    *,
    hidden_width: int = 128,
    dropout: float = 0.1,
) -> RecoverabilityPredictor:
    return RecoverabilityPredictor(
        PredictorConfig(
            architecture=architecture,
            input_dim=int(input_dim),
            hidden_width=int(hidden_width),
            dropout=float(dropout),
        )
    )
