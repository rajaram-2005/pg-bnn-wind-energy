"""Unified modular physics residual aggregation."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PhysicsWeights:
    aero: float = 0.2
    drive: float = 0.25
    thermal: float = 0.25
    grid: float = 0.15
    consistency: float = 0.15


def physics_loss(
    residuals: dict[str, torch.Tensor], weights: PhysicsWeights | None = None
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = weights or PhysicsWeights()
    terms = {name: value.pow(2).mean() for name, value in residuals.items()}
    zero = next(iter(terms.values())).new_zeros(()) if terms else torch.tensor(0.0)
    total = sum(
        (getattr(weights, name, weights.consistency) * value for name, value in terms.items()),
        start=zero,
    )
    return total, terms
