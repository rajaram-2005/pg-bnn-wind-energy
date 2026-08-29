"""Compute the five physics residuals from a raw SCADA snapshot.

Every residual is dimensionless and normalised so a value of ~1.0 means "as
wrong as it could plausibly be". Residuals are evaluated on raw (SI-unit)
channels, never on normalised features, because the physical relations are only
valid in physical units.
"""

from __future__ import annotations

import torch

from ..config import TurbineConfig
from . import aerodynamics, drivetrain, electrical, thermal

RESIDUAL_NAMES: tuple[str, ...] = ("aero", "drive", "thermal", "grid", "consistency")


_CHANNEL_INDEX_CACHE: dict[str, int] | None = None


def _channel_index() -> dict[str, int]:
    """Resolve the channel vocabulary lazily (``data`` imports ``physics``)."""
    global _CHANNEL_INDEX_CACHE
    if _CHANNEL_INDEX_CACHE is None:
        from ..data.schema import CHANNEL_INDEX

        _CHANNEL_INDEX_CACHE = dict(CHANNEL_INDEX)
    return _CHANNEL_INDEX_CACHE


def _get(raw: torch.Tensor, name: str) -> torch.Tensor:
    return torch.nan_to_num(raw[..., _channel_index()[name]], nan=0.0, posinf=0.0, neginf=0.0)


RESIDUAL_CLAMP = 10.0


def _required(raw: torch.Tensor, names: tuple[str, ...]) -> torch.Tensor:
    """Per-sample validity mask: True when every required channel is present.

    A residual computed from a missing channel is not evidence of physical
    inconsistency — it is missing data. Those samples are neutralised here and
    handled separately by the verifier through ``data_completeness``.
    """
    mask = torch.ones(raw.shape[:-1], dtype=torch.bool, device=raw.device)
    index = _channel_index()
    for name in names:
        mask = mask & ~torch.isnan(raw[..., index[name]])
    return mask


def _safe(residual: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero out samples with missing inputs and bound the rest."""
    return (residual * mask.to(residual.dtype)).clamp(-RESIDUAL_CLAMP, RESIDUAL_CLAMP)


def compute_residuals(
    raw: torch.Tensor,
    turbine: TurbineConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Residuals for a batch of raw channel snapshots.

    Parameters
    ----------
    raw:
        ``(B, C)`` or ``(C,)`` tensor of SI-unit channel values (the last
        timestep of each window).
    turbine:
        Plant constants. Defaults to the research configuration.
    """
    turbine = turbine or TurbineConfig()
    if raw.ndim == 1:
        raw = raw.unsqueeze(0)
    if raw.ndim != 2:
        raise ValueError("raw must have shape (B, C) or (C,)")

    wind = _get(raw, "wind_speed").clamp_min(0.0)
    omega = _get(raw, "rotor_speed").clamp_min(0.0)
    pitch = _get(raw, "pitch_angle")
    power_kw = _get(raw, "active_power_kw")
    winding_c = _get(raw, "generator_winding_temp_c")
    oil_c = _get(raw, "gearbox_oil_temp_c")
    ambient_c = _get(raw, "ambient_temp_c")
    current_a = _get(raw, "phase_current_a").clamp_min(0.0)
    torque_nm = _get(raw, "generator_torque_nm")
    freq_hz = _get(raw, "grid_frequency_hz")

    rated_w = max(turbine.rated_power_kw, 1.0) * 1000.0
    area = aerodynamics.swept_area(turbine.rotor_radius_m)
    power_w = power_kw * 1000.0

    # ── aerodynamic ────────────────────────────────────────────────────────
    aero_mask = _required(raw, ("wind_speed", "rotor_speed", "pitch_angle", "active_power_kw"))
    tsr = aerodynamics.tip_speed_ratio(wind, omega, turbine.rotor_radius_m)
    cp = aerodynamics.power_coefficient(tsr, pitch)
    p_aero_w = aerodynamics.aerodynamic_power(wind, cp, area, turbine.air_density)
    p_elec_expected_w = p_aero_w * turbine.gearbox_efficiency * turbine.generator_efficiency
    aero = _safe((power_w - p_elec_expected_w) / rated_w, aero_mask)

    # ── drivetrain ─────────────────────────────────────────────────────────
    drive_mask = _required(raw, ("rotor_speed", "active_power_kw", "generator_torque_nm"))
    p_mech_w = power_w / max(turbine.generator_efficiency, 1e-3)
    torque_expected = drivetrain.shaft_torque(p_mech_w, omega)
    rated_torque = rated_w / max(turbine.omega_rated, 0.1)
    drive = _safe((torque_nm - torque_expected) / max(rated_torque, 1.0), drive_mask)

    # ── thermal (steady state + second law) ────────────────────────────────
    thermal_mask = _required(
        raw, ("generator_winding_temp_c", "ambient_temp_c", "phase_current_a", "rotor_speed")
    )
    # Iron loss is modelled as a linear function of rotor speed (proxy):
    # the generator electrical frequency scales with omega.
    heat_w = thermal.generator_heat_dissipation(
        current_a, omega * 100.0, turbine.winding_resistance_ohm
    )
    t_ss = thermal.steady_state_temperature(ambient_c, heat_w, turbine.thermal_resistance_k_w)
    thermal_res = (winding_c - t_ss) / 100.0
    thermal_res = thermal_res + thermal.second_law_penalty(winding_c, ambient_c, heat_w)
    thermal_res = thermal_res + 0.1 * thermal.second_law_penalty(oil_c, ambient_c, heat_w)
    thermal_res = _safe(thermal_res, thermal_mask)

    # ── grid ───────────────────────────────────────────────────────────────
    grid_mask = _required(
        raw, ("active_power_kw", "phase_current_a", "grid_frequency_hz")
    )
    voltage = torch.full_like(wind, turbine.nominal_voltage_v)
    pf = torch.full_like(wind, turbine.power_factor)
    grid = electrical.grid_residual(power_w, voltage, current_a, pf, rated_w)
    grid = _safe(
        grid + electrical.frequency_stress(freq_hz, turbine.nominal_frequency_hz), grid_mask
    )

    # ── energy conservation / Betz ─────────────────────────────────────────
    # Normalised by rated power (not by the available wind power, which is zero
    # when the wind channel is missing or the turbine is parked).
    overproduction = torch.relu(power_w - p_aero_w) / rated_w
    betz = torch.relu(power_w - aerodynamics.betz_power(wind, area, turbine.air_density)) / rated_w
    consistency_mask = _required(raw, ("wind_speed", "rotor_speed", "pitch_angle", "active_power_kw"))
    consistency = _safe(overproduction + betz, consistency_mask)

    return {
        "aero": aero,
        "drive": drive,
        "thermal": thermal_res,
        "grid": grid,
        "consistency": consistency,
    }


def physics_feature_vector(raw: torch.Tensor, turbine: TurbineConfig | None = None) -> torch.Tensor:
    """Derived physics features appended to the model input.

    Returns ``(B, 5)``: tip-speed ratio (scaled), power coefficient, power
    residual, thermal margin and data completeness.
    """
    turbine = turbine or TurbineConfig()
    if raw.ndim == 1:
        raw = raw.unsqueeze(0)
    wind = _get(raw, "wind_speed").clamp_min(0.0)
    omega = _get(raw, "rotor_speed").clamp_min(0.0)
    pitch = _get(raw, "pitch_angle")
    power_kw = _get(raw, "active_power_kw")
    winding_c = _get(raw, "generator_winding_temp_c")
    ambient_c = _get(raw, "ambient_temp_c")
    missing = torch.isnan(raw).float().mean(-1) if raw.ndim == 2 else torch.zeros(raw.shape[0])

    area = aerodynamics.swept_area(turbine.rotor_radius_m)
    tsr = aerodynamics.tip_speed_ratio(wind, omega, turbine.rotor_radius_m)
    cp = aerodynamics.power_coefficient(tsr, pitch)
    p_aero_w = aerodynamics.aerodynamic_power(wind, cp, area, turbine.air_density)
    rated_w = max(turbine.rated_power_kw, 1.0) * 1000.0
    power_residual = (power_kw * 1000.0 - p_aero_w * turbine.gearbox_efficiency) / rated_w
    thermal_margin = (winding_c - ambient_c) / 80.0
    completeness = 1.0 - missing
    return torch.stack(
        [tsr / 12.0, cp, power_residual.clamp(-2.0, 2.0), thermal_margin.clamp(-2.0, 4.0), completeness],
        dim=-1,
    )


__all__ = ["RESIDUAL_NAMES", "compute_residuals", "physics_feature_vector"]
