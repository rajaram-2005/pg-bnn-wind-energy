"""Differentiable aerodynamic relations (SI units)."""

import torch


def tip_speed_ratio(
    wind_speed: torch.Tensor, rotor_speed_rad_s: torch.Tensor, radius_m: float
) -> torch.Tensor:
    return rotor_speed_rad_s * radius_m / wind_speed.clamp_min(0.5)


def power_coefficient(tsr: torch.Tensor, pitch_deg: torch.Tensor) -> torch.Tensor:
    """Stable Heier approximation, clipped to the Betz interval."""
    beta = pitch_deg.clamp(0, 45)
    inv = 1 / (tsr + 0.08 * beta + 1e-4) - 0.035 / (beta.pow(3) + 1)
    lam_i = 1 / inv.clamp_min(1e-3)
    cp = 0.5176 * (116 / lam_i - 0.4 * beta - 5) * torch.exp(-21 / lam_i) + 0.0068 * tsr
    return cp.clamp(0, 16 / 27)


def aerodynamic_power(
    wind_speed: torch.Tensor, cp: torch.Tensor, swept_area_m2: float, air_density: float = 1.225
) -> torch.Tensor:
    return 0.5 * air_density * swept_area_m2 * wind_speed.clamp_min(0).pow(3) * cp


def aerodynamic_residual(
    observed_power_w: torch.Tensor, predicted_power_w: torch.Tensor, rated_power_w: float
) -> torch.Tensor:
    return (observed_power_w - predicted_power_w) / max(rated_power_w, 1.0)
