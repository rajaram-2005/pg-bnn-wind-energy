"""Compact domain experts.

Four core experts (aero, drive, thermal, grid) plus a wake expert used by the
Qinglong fleet preset. Every expert is a residual bottleneck block, so the
mixture stays cheap: the router selects ``top_k`` experts per sample and the
others are never executed.
"""

from __future__ import annotations

import torch
from torch import nn

from .registry import EXPERT_NAMES


class CompactExpert(nn.Module):
    """Residual bottleneck expert. ``depth`` stacks the bottleneck blocks."""

    def __init__(self, width: int, bottleneck: int, depth: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("expert depth must be >= 1")
        blocks = []
        for _ in range(depth):
            blocks.append(
                nn.Sequential(
                    nn.Linear(width, bottleneck),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(bottleneck, width),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for block in self.blocks:
            h = h + block(h)
        return self.norm(h)


class AeroExpert(CompactExpert):
    """Rotor aerodynamics: Cp, wake and gust response."""


class DriveExpert(CompactExpert):
    """Drivetrain: gearbox torque, bearing load-life, vibration."""


class ThermalExpert(CompactExpert):
    """Thermal network: gearbox oil and generator windings."""


class GridExpert(CompactExpert):
    """Electrical: power balance, frequency excursions."""


class WakeExpert(nn.Module):
    """Wake-aware expert: conditions on neighbour context (Qinglong preset).

    ``neighbors`` is ``(B, K, W)`` — the encoded neighbour context produced by
    :class:`~windfusion.models.base.NeighborAttention`.
    """

    def __init__(self, width: int, bottleneck: int, neighbor_width: int | None = None) -> None:
        super().__init__()
        self.context = nn.Linear(neighbor_width or width, width)
        self.block = CompactExpert(width, bottleneck)
        self.gate = nn.Linear(width * 2, width)

    def forward(self, x: torch.Tensor, neighbors: torch.Tensor | None = None) -> torch.Tensor:
        if neighbors is None:
            return self.block(x)
        nb = torch.nan_to_num(neighbors)
        if nb.ndim == 3:
            nb = nb.mean(1)
        ctx = self.context(nb)
        gate = torch.sigmoid(self.gate(torch.cat([x, ctx], dim=-1)))
        return self.block(x + gate * ctx)


def build_expert(name: str, width: int, bottleneck: int, depth: int = 1, dropout: float = 0.0) -> nn.Module:
    """Factory used by :class:`~windfusion.models.base.WindFusionBase`."""
    if name == "wake":
        return WakeExpert(width, bottleneck, neighbor_width=width)
    kinds = {
        "aero": AeroExpert,
        "drive": DriveExpert,
        "thermal": ThermalExpert,
        "grid": GridExpert,
    }
    if name not in kinds:
        raise ValueError(f"unknown expert {name!r}; known: {sorted(kinds) + ['wake']}")
    return kinds[name](width, bottleneck, depth=depth, dropout=dropout)


__all__ = [
    "EXPERT_NAMES",
    "AeroExpert",
    "CompactExpert",
    "DriveExpert",
    "GridExpert",
    "ThermalExpert",
    "WakeExpert",
    "build_expert",
]
