"""Drivetrain conservation and degradation proxies.

Provenance: ``iso281_l10`` (wind-turbine-pg-bnn
``src/physics/drivetrain.py::bearing_l10_life_hours`` and
``src/physics/constraints.py::iso_281_l10_hours``). ISO 281 is a published
standard; only the choice of using it as a damage prior comes from upstream.
"""

from __future__ import annotations

import math

import torch

DEFAULT_GEAR_RATIO: float = 97.0
DEFAULT_GEARBOX_EFFICIENCY: float = 0.97
BALL_BEARING_EXPONENT: float = 3.0
ROLLER_BEARING_EXPONENT: float = 10.0 / 3.0


def shaft_torque(power_w: torch.Tensor, angular_speed_rad_s: torch.Tensor) -> torch.Tensor:
    """T = P / omega."""
    return power_w / angular_speed_rad_s.abs().clamp_min(0.1)


def gearbox_torque_transfer(
    rotor_torque_nm: torch.Tensor,
    gear_ratio: float = DEFAULT_GEAR_RATIO,
    efficiency: float = DEFAULT_GEARBOX_EFFICIENCY,
) -> torch.Tensor:
    """T_hss = eta * T_rotor / n."""
    return efficiency * rotor_torque_nm / max(gear_ratio, 1e-6)


def high_speed_shaft_speed(
    rotor_speed_rad_s: torch.Tensor, gear_ratio: float = DEFAULT_GEAR_RATIO
) -> torch.Tensor:
    return rotor_speed_rad_s * gear_ratio


def torque_residual(
    low_speed_torque: torch.Tensor,
    high_speed_torque: torch.Tensor,
    gear_ratio: float,
    efficiency: float = 0.96,
) -> torch.Tensor:
    """Relative mismatch between measured and gearbox-implied HSS torque."""
    expected = low_speed_torque * efficiency / max(gear_ratio, 1e-6)
    return (high_speed_torque - expected) / expected.abs().clamp_min(1.0)


def bearing_l10_life_hours(
    dynamic_load_rating_n: float,
    equivalent_load_n: torch.Tensor,
    shaft_speed_rpm: torch.Tensor,
    exponent: float = ROLLER_BEARING_EXPONENT,
) -> torch.Tensor:
    """ISO 281 L10 rating life in operating hours.

    L10h = (10^6 / (60 N)) * (C / P)^p
    """
    load = equivalent_load_n.clamp_min(1.0)
    rpm = shaft_speed_rpm.clamp_min(1.0)
    return (1e6 / (60.0 * rpm)) * (dynamic_load_rating_n / load).clamp_min(0.0).pow(exponent)


def iso_281_l10_hours(
    c_rating_kn: float, p_load_kn: float, exponent: float = ROLLER_BEARING_EXPONENT, rpm: float = 1500.0
) -> float:
    """Scalar twin used by the digital twin and reports."""
    if p_load_kn <= 0 or c_rating_kn <= 0 or rpm <= 0:
        return float("inf")
    return float((c_rating_kn / p_load_kn) ** exponent * 1e6 / (rpm * 60.0))


def bearing_damage_rate(
    load_ratio: torch.Tensor, vibration_ratio: torch.Tensor, temperature_c: torch.Tensor
) -> torch.Tensor:
    """Dimensionless monotonic damage-rate proxy: cubic load law x Arrhenius-ish
    thermal acceleration x vibration amplification (inspired by ISO 281 L10)."""
    thermal = torch.exp(((temperature_c - 70) / 40).clamp(-3, 3))
    return load_ratio.clamp_min(0).pow(BALL_BEARING_EXPONENT) * (
        1 + vibration_ratio.clamp_min(0).pow(2)
    ) * thermal


def damage_to_health(damage: torch.Tensor) -> torch.Tensor:
    """Map accumulated damage in [0, inf) to a health index in [0, 1]."""
    return torch.exp(-damage.clamp_min(0.0))


def rpm_from_rad_s(omega_rad_s: torch.Tensor) -> torch.Tensor:
    return omega_rad_s * 60.0 / (2.0 * math.pi)


__all__ = [
    "BALL_BEARING_EXPONENT",
    "DEFAULT_GEAR_RATIO",
    "ROLLER_BEARING_EXPONENT",
    "bearing_damage_rate",
    "bearing_l10_life_hours",
    "damage_to_health",
    "gearbox_torque_transfer",
    "high_speed_shaft_speed",
    "iso_281_l10_hours",
    "rpm_from_rad_s",
    "shaft_torque",
    "torque_residual",
]
