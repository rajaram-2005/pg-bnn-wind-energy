"""Three-phase generator/grid equations."""

import math

import torch


def three_phase_power(
    voltage_v: torch.Tensor, current_a: torch.Tensor, power_factor: torch.Tensor
) -> torch.Tensor:
    return math.sqrt(3) * voltage_v * current_a * power_factor.clamp(0, 1)


def grid_residual(
    measured_power_w: torch.Tensor,
    voltage_v: torch.Tensor,
    current_a: torch.Tensor,
    power_factor: torch.Tensor,
    rated_power_w: float,
) -> torch.Tensor:
    expected = three_phase_power(voltage_v, current_a, power_factor)
    return (measured_power_w - expected) / max(rated_power_w, 1.0)


def frequency_stress(
    frequency_hz: torch.Tensor, nominal_hz: float = 50.0, tolerance_hz: float = 0.5
) -> torch.Tensor:
    return torch.relu((frequency_hz - nominal_hz).abs() - tolerance_hz) / tolerance_hz
