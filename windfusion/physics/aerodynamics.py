"""Differentiable aerodynamic relations (SI units).

Provenance: ``heier_cp_betz`` and ``jensen_wake``
(wind-turbine-pg-bnn ``src/physics/aerodynamics.py``). Both are published
relations — Heier (1998) for the Cp surface, Betz (1919) for the 16/27 limit and
Jensen (1983) for the wake deficit — re-derived here so that the training
residuals and the synthetic generator share one implementation.
"""

from __future__ import annotations

import math

import torch

BETZ_LIMIT: float = 16.0 / 27.0
DEFAULT_AIR_DENSITY: float = 1.225
_EPS = 1e-6


def tip_speed_ratio(
    wind_speed: torch.Tensor, rotor_speed_rad_s: torch.Tensor, radius_m: float
) -> torch.Tensor:
    """lambda = omega * R / v (wind speed floored for numerical safety)."""
    return rotor_speed_rad_s * radius_m / wind_speed.clamp_min(0.5)


def power_coefficient(tsr: torch.Tensor, pitch_deg: torch.Tensor) -> torch.Tensor:
    """Stable Heier approximation, clipped to the Betz interval [0, 16/27]."""
    beta = pitch_deg.clamp(0, 45)
    lam = tsr.clamp_min(_EPS)
    inv = 1 / (lam + 0.08 * beta + 1e-4) - 0.035 / (beta.pow(3) + 1)
    lam_i = 1 / inv.clamp_min(1e-3)
    cp = 0.5176 * (116 / lam_i - 0.4 * beta - 5) * torch.exp(-21 / lam_i) + 0.0068 * lam
    return cp.clamp(0, BETZ_LIMIT)


def swept_area(radius_m: float) -> float:
    return math.pi * radius_m**2


def aerodynamic_power(
    wind_speed: torch.Tensor,
    cp: torch.Tensor,
    swept_area_m2: float,
    air_density: float = DEFAULT_AIR_DENSITY,
) -> torch.Tensor:
    """P = 0.5 * rho * A * v^3 * Cp (Watts)."""
    return 0.5 * air_density * swept_area_m2 * wind_speed.clamp_min(0).pow(3) * cp


def available_power(
    wind_speed: torch.Tensor,
    swept_area_m2: float,
    air_density: float = DEFAULT_AIR_DENSITY,
) -> torch.Tensor:
    """Total wind power through the rotor: 0.5 * rho * A * v^3."""
    return 0.5 * air_density * swept_area_m2 * wind_speed.clamp_min(0).pow(3)


def betz_power(
    wind_speed: torch.Tensor,
    swept_area_m2: float,
    air_density: float = DEFAULT_AIR_DENSITY,
) -> torch.Tensor:
    """Betz-limited extractable power (theoretical maximum)."""
    return available_power(wind_speed, swept_area_m2, air_density) * BETZ_LIMIT


def aerodynamic_residual(
    observed_power_w: torch.Tensor, predicted_power_w: torch.Tensor, rated_power_w: float
) -> torch.Tensor:
    """Normalised (observed - predicted) power mismatch."""
    return (observed_power_w - predicted_power_w) / max(rated_power_w, 1.0)


def betz_violation(
    predicted_power_w: torch.Tensor,
    wind_speed: torch.Tensor,
    swept_area_m2: float,
    air_density: float = DEFAULT_AIR_DENSITY,
) -> torch.Tensor:
    """Positive part of (predicted power - Betz limit), normalised."""
    limit = betz_power(wind_speed, swept_area_m2, air_density)
    scale = limit.clamp_min(1.0)
    return torch.relu(predicted_power_w - limit) / scale


def thrust_coefficient(cp: torch.Tensor) -> torch.Tensor:
    """Rough Ct(Cp) mapping used only for wake conditioning (0.1..0.9)."""
    return (0.6 + 0.5 * cp).clamp(0.05, 0.95)


def jensen_wake_deficit(
    distance_m: torch.Tensor,
    radius_m: float,
    ct: torch.Tensor,
    wake_decay: float = 0.075,
) -> torch.Tensor:
    """Jensen/Park velocity deficit: (1 - sqrt(1 - Ct)) / (1 + k x / R)^2."""
    ct = ct.clamp(0.0, 0.999)
    x = distance_m.clamp_min(1e-3)
    expansion = (1.0 + wake_decay * x / max(radius_m, 1e-3)).pow(2)
    return (1.0 - torch.sqrt(1.0 - ct)) / expansion


def waked_wind_speed(
    free_stream: torch.Tensor,
    distance_m: torch.Tensor,
    radius_m: float,
    ct: torch.Tensor,
    wake_decay: float = 0.075,
) -> torch.Tensor:
    """Wind speed at a downstream turbine after wake superposition."""
    deficit = jensen_wake_deficit(distance_m, radius_m, ct, wake_decay)
    return (free_stream * (1.0 - deficit)).clamp_min(0.0)


__all__ = [
    "BETZ_LIMIT",
    "DEFAULT_AIR_DENSITY",
    "aerodynamic_power",
    "aerodynamic_residual",
    "available_power",
    "betz_power",
    "betz_violation",
    "jensen_wake_deficit",
    "power_coefficient",
    "swept_area",
    "thrust_coefficient",
    "tip_speed_ratio",
    "waked_wind_speed",
]
