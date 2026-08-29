"""Unified modular physics residual aggregation.

Provenance: ``soft_hinge_limits`` (wind-turbine-pg-bnn
``src/physics/constraints.py::_soft_relu_penalty``) and the modular residual
weighting introduced in WindFusion-Lite v0.1.

The limits below are **research defaults, not OEM limits**. They exist so the
model is penalised for leaving the physically plausible envelope; a deployment
must substitute asset-specific limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

# Documented research defaults (documentation values, not OEM data).
RESEARCH_LIMITS: dict[str, float] = {
    "gearbox_oil_temp_c": 80.0,
    "generator_winding_temp_c": 120.0,
    "main_bearing_temp_c": 95.0,
    "vibration_rms": 4.5,
    "rotor_speed": 2.2,
}


@dataclass(frozen=True)
class PhysicsWeights:
    """Per-residual loss weights."""

    aero: float = 0.20
    drive: float = 0.25
    thermal: float = 0.25
    grid: float = 0.15
    consistency: float = 0.15

    def as_dict(self) -> dict[str, float]:
        return {
            "aero": self.aero,
            "drive": self.drive,
            "thermal": self.thermal,
            "grid": self.grid,
            "consistency": self.consistency,
        }


def soft_hinge(x: torch.Tensor, limit: float | torch.Tensor, beta: float = 5.0) -> torch.Tensor:
    """Smooth one-sided penalty: ~0 below the limit, quadratic above it."""
    excess = x - limit
    return torch.nn.functional.softplus(beta * excess).pow(2) / (beta**2)


def limit_penalties(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Soft-hinge penalty for every channel with a documented limit."""
    out: dict[str, torch.Tensor] = {}
    for name, limit in RESEARCH_LIMITS.items():
        if name in raw:
            out[f"limit_{name}"] = soft_hinge(raw[name], limit)
    return out


def physics_loss(
    residuals: dict[str, torch.Tensor],
    weights: PhysicsWeights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Weighted sum of squared residuals.

    Unknown residual names fall back to the ``consistency`` weight so new
    architectures can add residuals without editing this function.
    """
    weights = weights or PhysicsWeights()
    terms = {name: value.pow(2).mean() for name, value in residuals.items()}
    zero = next(iter(terms.values())).new_zeros(()) if terms else torch.tensor(0.0)
    total = sum(
        (getattr(weights, name, weights.consistency) * value for name, value in terms.items()),
        start=zero,
    )
    return total, terms


@dataclass(frozen=True)
class PhysicsReport:
    """Per-sample residual summary surfaced to the verifier and reports."""

    residuals: dict[str, float] = field(default_factory=dict)
    total: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {**self.residuals, "total": self.total}


def summarise_residuals(residuals: dict[str, torch.Tensor], index: int = 0) -> PhysicsReport:
    """Extract one sample's residuals as plain floats (for reporting)."""
    values = {k: float(v.reshape(v.shape[0], -1)[index].abs().mean()) for k, v in residuals.items()}
    total = float(sum(values.values()))
    return PhysicsReport(residuals=values, total=total)


__all__ = [
    "RESEARCH_LIMITS",
    "PhysicsReport",
    "PhysicsWeights",
    "limit_penalties",
    "physics_loss",
    "soft_hinge",
    "summarise_residuals",
]
