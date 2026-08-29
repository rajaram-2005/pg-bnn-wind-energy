"""Four small domain experts; each receives learned and physical features."""

import torch
from torch import nn


class CompactExpert(nn.Module):
    def __init__(self, width: int, bottleneck: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, bottleneck), nn.SiLU(), nn.Linear(bottleneck, width)
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class AeroExpert(CompactExpert):
    pass


class DriveExpert(CompactExpert):
    pass


class ThermalExpert(CompactExpert):
    pass


class GridExpert(CompactExpert):
    pass


EXPERT_NAMES = ("aero", "drive", "thermal", "grid")
