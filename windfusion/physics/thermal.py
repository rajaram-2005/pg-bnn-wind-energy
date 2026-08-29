"""Stable first-order lumped thermal model.

Provenance: ``lumped_thermal_rc`` (wind-turbine-pg-bnn
``src/physics/thermal.py::winding_temperature_derivative``). The RC network is
integrated semi-analytically (exact exponential for a constant source over the
step) instead of with explicit Euler, so the model stays stable at 10-minute
SCADA time steps where dt >> tau would otherwise oscillate.
"""

from __future__ import annotations

import numpy as np
import torch

DEFAULT_THERMAL_RESISTANCE_K_W: float = 0.02
DEFAULT_THERMAL_CAPACITANCE_J_K: float = 5.0e4


def thermal_step(
    temperature_c: torch.Tensor,
    ambient_c: torch.Tensor,
    heat_w: torch.Tensor,
    resistance_k_w: float,
    capacitance_j_k: float,
    dt_s: float,
) -> torch.Tensor:
    """One exact step of dT/dt = (T_amb - T)/(R C) + Q/C."""
    tau = max(resistance_k_w * capacitance_j_k, 1e-6)
    equilibrium = ambient_c + heat_w * resistance_k_w
    decay = torch.exp(
        torch.as_tensor(-dt_s / tau, dtype=temperature_c.dtype, device=temperature_c.device)
    )
    return equilibrium + (temperature_c - equilibrium) * decay


def thermal_step_numpy(
    temperature_c: float,
    ambient_c: float,
    heat_w: float,
    resistance_k_w: float,
    capacitance_j_k: float,
    dt_s: float,
) -> float:
    """NumPy/scalar twin of :func:`thermal_step` for the data generator."""
    tau = max(resistance_k_w * capacitance_j_k, 1e-6)
    equilibrium = ambient_c + heat_w * resistance_k_w
    return float(equilibrium + (temperature_c - equilibrium) * float(np.exp(-dt_s / tau)))


def generator_heat_dissipation(
    phase_current_a: torch.Tensor,
    electrical_speed_rad_s: torch.Tensor,
    winding_resistance_ohm: float = 0.02,
    iron_loss_gain: float = 0.02,
) -> torch.Tensor:
    """Copper (3 I^2 R) plus a crude iron-loss term proportional to speed."""
    copper = 3.0 * phase_current_a.pow(2) * winding_resistance_ohm
    iron = iron_loss_gain * electrical_speed_rad_s.abs().clamp_min(0.0) * 1000.0
    return copper + iron


def steady_state_temperature(
    ambient_c: torch.Tensor, heat_w: torch.Tensor, resistance_k_w: float
) -> torch.Tensor:
    """T_inf = T_ambient + R_th * Q."""
    return ambient_c + heat_w * resistance_k_w


def thermal_residual(
    observed_next_c: torch.Tensor, predicted_next_c: torch.Tensor, scale_c: float = 100.0
) -> torch.Tensor:
    """Normalised one-step temperature mismatch."""
    return (observed_next_c - predicted_next_c) / max(scale_c, 1e-6)


def second_law_penalty(
    temperature_c: torch.Tensor, ambient_c: torch.Tensor, heat_w: torch.Tensor, scale_c: float = 100.0
) -> torch.Tensor:
    """Penalise a component colder than ambient while it dissipates heat."""
    violation = torch.relu(ambient_c - temperature_c) * (heat_w > 0).to(temperature_c.dtype)
    return violation / max(scale_c, 1e-6)


__all__ = [
    "DEFAULT_THERMAL_CAPACITANCE_J_K",
    "DEFAULT_THERMAL_RESISTANCE_K_W",
    "generator_heat_dissipation",
    "second_law_penalty",
    "steady_state_temperature",
    "thermal_residual",
    "thermal_step",
    "thermal_step_numpy",
]
