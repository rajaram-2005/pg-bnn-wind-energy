"""Three-phase generator / grid relations.

Provenance: ``heier_cp_betz`` family (electrical balance is part of the same
upstream physics module set in wind-turbine-pg-bnn ``src/physics/``).
"""

from __future__ import annotations

import math

import torch

NOMINAL_VOLTAGE_V: float = 690.0
NOMINAL_FREQUENCY_HZ: float = 50.0


def three_phase_power(
    voltage_v: torch.Tensor, current_a: torch.Tensor, power_factor: torch.Tensor
) -> torch.Tensor:
    """P = sqrt(3) * V * I * cos(phi)."""
    return math.sqrt(3) * voltage_v * current_a * power_factor.clamp(0, 1)


def grid_residual(
    measured_power_w: torch.Tensor,
    voltage_v: torch.Tensor,
    current_a: torch.Tensor,
    power_factor: torch.Tensor,
    rated_power_w: float,
) -> torch.Tensor:
    """Normalised mismatch between measured and electrically-implied power."""
    expected = three_phase_power(voltage_v, current_a, power_factor)
    return (measured_power_w - expected) / max(rated_power_w, 1.0)


def frequency_stress(
    frequency_hz: torch.Tensor, nominal_hz: float = NOMINAL_FREQUENCY_HZ, tolerance_hz: float = 0.5
) -> torch.Tensor:
    """Normalised excursion beyond the grid tolerance band (>= 0)."""
    return torch.relu((frequency_hz - nominal_hz).abs() - tolerance_hz) / max(tolerance_hz, 1e-6)


def current_from_power(
    power_w: torch.Tensor, voltage_v: float = NOMINAL_VOLTAGE_V, power_factor: float = 0.95
) -> torch.Tensor:
    """I = P / (sqrt(3) V cos(phi))."""
    return power_w / (math.sqrt(3) * max(voltage_v, 1.0) * max(power_factor, 1e-3))


__all__ = [
    "NOMINAL_FREQUENCY_HZ",
    "NOMINAL_VOLTAGE_V",
    "current_from_power",
    "frequency_stress",
    "grid_residual",
    "three_phase_power",
]
