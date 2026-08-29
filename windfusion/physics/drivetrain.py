"""Drivetrain conservation and degradation proxies."""

import torch


def shaft_torque(power_w: torch.Tensor, angular_speed_rad_s: torch.Tensor) -> torch.Tensor:
    return power_w / angular_speed_rad_s.abs().clamp_min(0.1)


def torque_residual(
    low_speed_torque: torch.Tensor,
    high_speed_torque: torch.Tensor,
    gear_ratio: float,
    efficiency: float = 0.96,
) -> torch.Tensor:
    expected = low_speed_torque * efficiency / max(gear_ratio, 1e-6)
    return (high_speed_torque - expected) / expected.abs().clamp_min(1.0)


def bearing_damage_rate(
    load_ratio: torch.Tensor, vibration_ratio: torch.Tensor, temperature_c: torch.Tensor
) -> torch.Tensor:
    """Dimensionless monotonic proxy inspired by L10 cubic load dependence."""
    thermal = torch.exp(((temperature_c - 70) / 40).clamp(-3, 3))
    return load_ratio.clamp_min(0).pow(3) * (1 + vibration_ratio.clamp_min(0).pow(2)) * thermal
