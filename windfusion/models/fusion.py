"""WindFusion-Lite model and Aetheris-inspired model registry."""

from dataclasses import dataclass

import torch
from torch import nn

from .experts import EXPERT_NAMES, AeroExpert, DriveExpert, GridExpert, ThermalExpert
from .router import SparseExpertRouter
from .temporal_encoder import GatedTemporalEncoder
from .uncertainty import BayesianUncertaintyHead, decompose_mc


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    hidden: int
    bottleneck: int
    top_k: int
    dropout: float


MODEL_REGISTRY = {
    "aetheris-wind": ModelConfig("aetheris-wind", 48, 16, 2, 0.10),
    "ra-wind": ModelConfig(
        "ra-wind", 40, 12, 2, 0.08
    ),  # Egyptian: thermal emphasis via training config
    "qinglong-wind": ModelConfig("qinglong-wind", 56, 16, 2, 0.10),  # Chinese: wake/fleet
    "vayu-wind": ModelConfig("vayu-wind", 40, 12, 1, 0.12),  # Hindu: gust/rotor
    "odin-wind": ModelConfig("odin-wind", 56, 20, 2, 0.10),  # Norse: cold drivetrain/RUL
    "aeolus-wind": ModelConfig("aeolus-wind", 48, 16, 2, 0.10),  # Greek: forecasting
    "windfusion-edge": ModelConfig("windfusion-edge", 24, 8, 1, 0.05),
    "windfusion-research": ModelConfig("windfusion-research", 96, 32, 3, 0.15),
    "windfusion-lite": ModelConfig("windfusion-lite", 48, 16, 2, 0.10),
}


class WindFusionLite(nn.Module):
    """Causal temporal encoder + sparse domain experts + uncertainty heads."""

    def __init__(
        self, input_features: int, outputs: int = 3, mode: str = "windfusion-lite"
    ) -> None:
        super().__init__()
        if mode not in MODEL_REGISTRY:
            raise ValueError(f"unknown mode {mode!r}")
        self.config, self.input_features = MODEL_REGISTRY[mode], input_features
        c = self.config
        self.edge_encoder = nn.Sequential(
            nn.Linear(input_features, c.hidden), nn.LayerNorm(c.hidden), nn.SiLU()
        )
        self.physics_encoder = nn.Sequential(nn.Linear(4, c.hidden), nn.SiLU())
        self.temporal = GatedTemporalEncoder(c.hidden, c.hidden)
        self.router = SparseExpertRouter(c.hidden, 4, c.top_k)
        kinds = (AeroExpert, DriveExpert, ThermalExpert, GridExpert)
        self.experts = nn.ModuleList(k(c.hidden, c.bottleneck) for k in kinds)
        self.head = BayesianUncertaintyHead(c.hidden, outputs, c.dropout)

    def forward(
        self, sequence: torch.Tensor, physics_features: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if sequence.ndim != 3 or sequence.shape[-1] != self.input_features:
            raise ValueError("invalid SCADA sequence shape")
        h = self.edge_encoder(torch.nan_to_num(sequence)) + self.physics_encoder(
            torch.nan_to_num(physics_features)
        ).unsqueeze(1)
        temporal = self.temporal(h)
        weights = self.router(temporal)
        # Compute only experts selected anywhere in this batch; inactive experts are skipped.
        fused = torch.zeros_like(temporal)
        for i, expert in enumerate(self.experts):
            active = weights[:, i] > 0
            if active.any():
                fused[active] += expert(temporal[active]) * weights[active, i : i + 1]
        mean, log_var = self.head(fused)
        return {"mean": mean, "log_var": log_var, "routing": weights, "representation": fused}

    @torch.no_grad()
    def predict(
        self, sequence: torch.Tensor, physics_features: torch.Tensor, samples: int = 16
    ) -> dict[str, torch.Tensor]:
        was_training = self.training
        self.train()  # activate only lightweight MC dropout
        outputs = [self(sequence, physics_features) for _ in range(samples)]
        self.train(was_training)
        result = decompose_mc(
            torch.stack([o["mean"] for o in outputs]), torch.stack([o["log_var"] for o in outputs])
        )
        result["routing"] = torch.stack([o["routing"] for o in outputs]).mean(0)
        return result

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def routing_explanation(self) -> list[str]:
        if self.router.last_weights is None:
            return []
        return [EXPERT_NAMES[i] for i in self.router.last_weights.argmax(-1).tolist()]


def create_model(
    mode: str = "windfusion-lite", input_features: int = 12, outputs: int = 3
) -> WindFusionLite:
    return WindFusionLite(input_features, outputs, mode)
