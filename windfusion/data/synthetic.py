"""Deterministic physics-driven synthetic fleet generator.

**This generator produces SIMULATED data. It is not field data, it is not a
substitute for SCADA from a real asset, and any metric measured on it must be
reported as synthetic.** Its purpose is reproducible regression testing,
ablations and integration tests when no licensed dataset is available.

Why a physics-driven generator rather than random noise
-------------------------------------------------------
A random generator would make the physics-residual losses meaningless, so
telemetry here is produced by the same relations the model is regularised with:

* Weibull wind with temporal coherence + turbulence,
* Heier Cp(beta, lambda) rotor power with pitch regulation above rated,
* lumped RC thermal network for gearbox oil and generator windings,
* ISO 281-style cubic load-life damage accumulation with a late-life shock,
* fault-mode specific channel signatures (taxonomy vocabulary reused from
  ``wind-turbine-pg-bnn``).

Because damage is an explicit state variable, ``health_index`` and ``rul_days``
are *known ground truth* rather than heuristic labels. That is the only reason
a supervised benchmark on this generator is well posed.

Determinism: the fleet is a pure function of :class:`SyntheticFleetConfig`
(seeded with ``numpy.random.SeedSequence``), so :func:`fleet_checksum` pins the
exact data used by an experiment.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from ..physics import aerodynamics, drivetrain, thermal
from .schema import (
    CHANNEL_INDEX,
    CHANNEL_SPECS,
    CHANNELS,
    FAULT_MODES,
    N_CHANNELS,
    RUL_SCALE_DAYS,
)


@dataclass(frozen=True)
class SyntheticFleetConfig:
    """Everything needed to reproduce a fleet byte-for-byte."""

    n_turbines: int = 36
    seq_len: int = 1440
    sample_interval_s: int = 600
    seed: int = 7
    fleet_version: str = "synthetic-v1"
    n_sites: int = 6
    missing_rate: float = 0.01
    dropout_burst_prob: float = 0.002
    dropout_burst_len: int = 12
    horizon_days: float = 30.0
    damage_rate_range: tuple[float, float] = (0.35, 1.50)
    operational_hours_per_step: float = 6.0
    fault_modes: tuple[str, ...] = FAULT_MODES
    neighbors_per_turbine: int = 3

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TurbineRun:
    """One turbine's simulated history."""

    turbine_id: int
    site_id: int
    fault_mode: str
    channels: np.ndarray  # (T, C) float32, may contain NaN (sensor dropout)
    targets: np.ndarray  # (T, 3) float32: [p_failure, health, rul/365]
    damage: np.ndarray  # (T,) float32 latent damage state
    neighbor_ids: list[int] = field(default_factory=list)
    neighbor_distance_m: list[float] = field(default_factory=list)

    @property
    def seq_len(self) -> int:
        return int(self.channels.shape[0])


# ── helpers ────────────────────────────────────────────────────────────────


def _normal_cdf_logistic(u: np.ndarray) -> np.ndarray:
    """Logistic approximation of the standard normal CDF (max abs error ~0.01)."""
    return 1.0 / (1.0 + np.exp(-1.702 * u))


def _weibull_wind(rng: np.random.Generator, seq_len: int, scale: float, rho: float) -> np.ndarray:
    """Temporally coherent Weibull(k=2) wind speed via AR(1) in normal space."""
    eps = rng.standard_normal(seq_len)
    u = np.zeros(seq_len, dtype=np.float64)
    for i in range(1, seq_len):
        u[i] = rho * u[i - 1] + np.sqrt(1.0 - rho * rho) * eps[i]
    u = np.clip(u, -3.0, 3.0)
    p = np.clip(_normal_cdf_logistic(u), 1e-6, 1 - 1e-6)
    k = 2.0
    return np.clip(scale * (-np.log(1.0 - p)) ** (1.0 / k), 0.0, 40.0)


def _np_cp(tsr: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    """Heier Cp evaluated through the shared torch implementation."""
    out = aerodynamics.power_coefficient(
        torch.from_numpy(np.asarray(tsr, dtype=np.float32)),
        torch.from_numpy(np.asarray(pitch, dtype=np.float32)),
    )
    return out.detach().cpu().numpy().astype(np.float64)


def _fault_signature(mode: str) -> dict[str, float]:
    """Per-channel end-of-life excursion for a fault mode (SI units)."""
    signatures: dict[str, dict[str, float]] = {
        "healthy": {"vibration_rms": 0.6, "main_bearing_temp_c": 4.0},
        "bearing_wear": {
            "vibration_rms": 4.5,
            "main_bearing_temp_c": 28.0,
            "gearbox_oil_temp_c": 8.0,
        },
        "gearbox_overheat": {
            "gearbox_oil_temp_c": 35.0,
            "vibration_rms": 2.2,
            "main_bearing_temp_c": 6.0,
        },
        "generator_winding": {
            "generator_winding_temp_c": 45.0,
            "grid_frequency_hz": 0.35,
            "phase_current_a": 60.0,
        },
        "blade_aero_loss": {"vibration_rms": 1.6, "active_power_kw": -180.0},
    }
    return signatures.get(mode, signatures["healthy"])


def _damage_multiplier(mode: str) -> float:
    return {
        "healthy": 0.35,
        "bearing_wear": 1.25,
        "gearbox_overheat": 1.35,
        "generator_winding": 1.15,
        "blade_aero_loss": 0.85,
    }.get(mode, 1.0)


# ── generator ──────────────────────────────────────────────────────────────


def _simulate_turbine(
    cfg: SyntheticFleetConfig, turbine_id: int, turbine_params: dict[str, float]
) -> TurbineRun:
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, turbine_id]))
    T = cfg.seq_len
    dt = float(cfg.sample_interval_s)
    turbine = turbine_params["turbine"]

    site_id = int(turbine_params["site_id"])
    ambient_base = float(turbine_params["ambient_base_c"])
    wind_scale = float(turbine_params["wind_scale"])
    turbulence = float(turbine_params["turbulence"])
    mode = str(turbine_params["fault_mode"])

    # ── environment ────────────────────────────────────────────────────────
    wind = _weibull_wind(rng, T, wind_scale, rho=0.97)
    wind = wind + rng.normal(0.0, turbulence, T)  # turbulence on top of the mean field
    wind = np.clip(wind, 0.0, 40.0)
    step = np.arange(T)
    ambient = (
        ambient_base
        + 6.0 * np.sin(2 * np.pi * step / 144.0)
        + 4.0 * np.sin(2 * np.pi * step / (144.0 * 20.0) + site_id)
        + rng.normal(0.0, 0.4, T)
    )

    # ── rotor / pitch / power from the aerodynamic model ────────────────────
    rated_kw = turbine["rated_power_kw"]
    radius = turbine["rotor_radius_m"]
    omega_rated = turbine["omega_rated"]
    tsr_opt = turbine["tsr_opt"]
    cut_in, cut_out = turbine["cut_in"], turbine["cut_out"]

    running = (wind >= cut_in) & (wind <= cut_out)
    omega = np.where(running, np.minimum(tsr_opt * wind / radius, omega_rated), 0.0)
    pitch = np.zeros(T)
    # Coarse pitch regulation: feathering grows once the aerodynamic power
    # exceeds rated. Two fixed-point passes are enough for synthetic data.
    cp_free = _np_cp(np.full(T, tsr_opt), np.zeros(T))
    p_free = 0.5 * turbine["air_density"] * np.pi * radius**2 * wind**3 * cp_free / 1000.0
    excess = np.clip(p_free / max(rated_kw, 1.0) - 1.0, 0.0, None)
    pitch = np.clip(28.0 * excess, 0.0, 30.0)
    for _ in range(2):
        cp = _np_cp(np.divide(omega * radius, np.clip(wind, 0.5, None)), pitch)
        p_aero = 0.5 * turbine["air_density"] * np.pi * radius**2 * wind**3 * cp / 1000.0
        excess = np.clip(p_aero / max(rated_kw, 1.0) - 1.0, 0.0, None)
        pitch = np.clip(pitch + 10.0 * excess, 0.0, 30.0)
    pitch = np.where(running, pitch, 90.0)
    omega = np.clip(omega + rng.normal(0.0, 0.01, T), 0.0, None)

    # ── damage accumulation (state variable → ground-truth labels) ──────────
    damage = np.zeros(T)
    rate = np.zeros(T)
    # Accelerated-life calibration: the damage clock advances
    # `operational_hours_per_step` per sample so a full degradation trajectory
    # (and therefore a usable RUL range) fits inside the simulated window,
    # while the telemetry physics still uses the real sampling interval.
    target_end_damage = float(rng.uniform(*cfg.damage_rate_range))
    mean_factor = 0.5  # long-run mean of load x thermal x vibration factors
    base_rate = (target_end_damage / max(T * mean_factor, 1.0)) * _damage_multiplier(mode)
    shock_idx = T - int(rng.integers(int(0.15 * T), int(0.5 * T)))
    d = 0.0
    for t in range(T):
        load = np.clip(p_free[t] / max(rated_kw, 1.0), 0.0, 1.2)
        load_factor = load**3 if t == 0 else load**3
        thermal_factor = float(np.exp(np.clip((45.0 + 30.0 * d - 70.0) / 40.0, -3.0, 3.0)))
        vib_factor = 1.0 + 0.5 * np.clip(d - 0.4, 0.0, None) ** 2
        shock = 1.0 + 6.0 * (max(0, t - shock_idx) / max(T - shock_idx, 1)) ** 4
        rate[t] = base_rate * load_factor * thermal_factor * vib_factor * shock
        d = min(d + rate[t], 1.05)
        damage[t] = d
    health = np.clip(1.0 - damage, 0.0, 1.0)

    # ── fault-mode channel excursions ──────────────────────────────────────
    sig = _fault_signature(mode)
    aero_loss = 0.22 * damage if mode == "blade_aero_loss" else 0.02 * damage
    cp_eff = _np_cp(np.divide(omega * radius, np.clip(wind, 0.5, None)), pitch) * (1.0 - aero_loss)
    p_aero_kw = (
        0.5 * turbine["air_density"] * np.pi * radius**2 * wind**3 * cp_eff / 1000.0
    )
    p_mech_kw = np.clip(p_aero_kw * turbine["gearbox_efficiency"], 0.0, None)
    p_elec_kw = np.clip(np.minimum(p_mech_kw * turbine["generator_efficiency"], rated_kw), 0.0, None)
    p_elec_kw = p_elec_kw + sig.get("active_power_kw", 0.0) * damage
    p_elec_kw = np.clip(p_elec_kw, 0.0, rated_kw * 1.05)
    p_elec_kw = np.where(running, p_elec_kw, 0.0)
    p_mech_kw = np.where(running, p_mech_kw, 0.0)

    torque = np.clip(
        p_mech_kw * 1000.0 / np.clip(omega, 0.15, None), 0.0, 5.0e6
    )
    current = p_elec_kw * 1000.0 / (np.sqrt(3.0) * turbine["nominal_voltage_v"] * turbine["power_factor"])
    current = np.clip(current + sig.get("phase_current_a", 0.0) * damage, 0.0, None)

    # ── thermal networks (lumped RC, semi-analytic update) ─────────────────
    winding = np.zeros(T)
    oil = np.zeros(T)
    winding[0] = ambient[0] + 10.0
    oil[0] = ambient[0] + 8.0
    copper = 3.0 * current**2 * turbine["winding_resistance_ohm"]
    iron = 0.02 * p_elec_kw * 1000.0
    for t in range(1, T):
        heat_w = copper[t - 1] + iron[t - 1]
        winding[t] = float(
            thermal.thermal_step_numpy(
                winding[t - 1],
                ambient[t - 1],
                heat_w,
                turbine["thermal_resistance_k_w"],
                turbine["thermal_capacitance_j_k"],
                dt,
            )
        )
        oil_heat = (1.0 - turbine["gearbox_efficiency"]) * p_mech_kw[t - 1] * 1000.0 + 500.0
        oil[t] = float(
            thermal.thermal_step_numpy(
                oil[t - 1],
                ambient[t - 1],
                oil_heat,
                0.0005,
                4.0e5,
                dt,
            )
        )
    winding = winding + sig.get("generator_winding_temp_c", 0.0) * damage
    oil = oil + sig.get("gearbox_oil_temp_c", 0.0) * damage
    bearing = (
        ambient
        + 12.0 * np.clip(p_mech_kw / max(rated_kw, 1.0), 0.0, 1.2)
        + sig.get("main_bearing_temp_c", 0.0) * damage
        + 0.6 * np.clip(omega, 0.0, None)
    )

    # ── vibration and grid ─────────────────────────────────────────────────
    vibration = (
        0.8
        + 0.7 * np.clip(p_mech_kw / max(rated_kw, 1.0), 0.0, 1.2)
        + turbulence * 6.0 * np.abs(rng.normal(0.0, 1.0, T))
        + sig.get("vibration_rms", 0.0) * damage
        + rng.normal(0.0, 0.08, T)
    )
    vibration = np.clip(vibration, 0.0, None)
    frequency = (
        turbine["nominal_frequency_hz"]
        + rng.normal(0.0, 0.03, T)
        + sig.get("grid_frequency_hz", 0.0) * damage * rng.normal(0.0, 1.0, T)
    )

    # ── assemble channel matrix in the canonical order ─────────────────────
    columns = {
        "wind_speed": wind,
        "rotor_speed": omega,
        "pitch_angle": pitch,
        "active_power_kw": p_elec_kw,
        "vibration_rms": vibration,
        "gearbox_oil_temp_c": oil,
        "generator_winding_temp_c": winding,
        "main_bearing_temp_c": bearing,
        "ambient_temp_c": ambient,
        "phase_current_a": current,
        "generator_torque_nm": torque,
        "grid_frequency_hz": frequency,
    }
    data = np.stack([columns[name] for name in CHANNELS], axis=1).astype(np.float32)
    for name, spec in CHANNEL_SPECS.items():
        data[:, CHANNEL_INDEX[name]] = np.clip(
            data[:, CHANNEL_INDEX[name]], spec.min_value, spec.max_value
        )

    # ── sensor dropouts (missing-data robustness) ──────────────────────────
    if cfg.missing_rate > 0:
        mask = rng.random(data.shape) < cfg.missing_rate
        data[mask] = np.nan
    if cfg.dropout_burst_prob > 0:
        n_bursts = int(rng.poisson(cfg.dropout_burst_prob * T))
        for _ in range(n_bursts):
            start = int(rng.integers(0, max(T - cfg.dropout_burst_len, 1)))
            channel = int(rng.integers(0, N_CHANNELS))
            data[start : start + cfg.dropout_burst_len, channel] = np.nan

    # ── labels from the latent damage state ────────────────────────────────
    remaining_steps = (1.0 - damage) / np.clip(rate, 1e-9, None)
    rul_days = np.clip(
        remaining_steps * cfg.operational_hours_per_step / 24.0, 0.0, 2.0 * RUL_SCALE_DAYS
    )
    p_failure = 1.0 / (1.0 + np.exp(-(cfg.horizon_days - rul_days) / 7.0))
    targets = np.stack(
        [p_failure, health, rul_days / RUL_SCALE_DAYS], axis=1
    ).astype(np.float32)

    return TurbineRun(
        turbine_id=turbine_id,
        site_id=site_id,
        fault_mode=mode,
        channels=data,
        targets=targets,
        damage=damage.astype(np.float32),
    )


def _site_layout(cfg: SyntheticFleetConfig, rng: np.random.Generator) -> list[dict]:
    """Assign turbines to sites with climate diversity (incl. cold sites)."""
    params: list[dict] = []
    for i in range(cfg.n_turbines):
        site_id = i % max(cfg.n_sites, 1)
        climate = 0.5 + 0.5 * np.sin(2 * np.pi * site_id / max(cfg.n_sites, 1))
        params.append(
            {
                "site_id": site_id,
                "ambient_base_c": float(2.0 + 20.0 * climate),
                "wind_scale": float(6.5 + 4.5 * climate),
                "turbulence": float(0.06 + 0.06 * ((site_id % 3) / 2.0)),
                "fault_mode": cfg.fault_modes[i % len(cfg.fault_modes)],
            }
        )
    return params


def generate_fleet(cfg: SyntheticFleetConfig | None = None) -> list[TurbineRun]:
    """Generate a deterministic fleet of turbine histories."""
    cfg = cfg or SyntheticFleetConfig()
    if cfg.n_turbines <= 0 or cfg.seq_len <= 0:
        raise ValueError("n_turbines and seq_len must be positive")
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 4242]))
    params = _site_layout(cfg, rng)
    turbine_constants = {
        "rated_power_kw": 2000.0,
        "rotor_radius_m": 45.0,
        "omega_rated": 1.9,  # rad/s (~18 rpm)
        "tsr_opt": 7.5,
        "cut_in": 3.0,
        "cut_out": 25.0,
        "air_density": 1.225,
        "gearbox_efficiency": 0.95,
        "generator_efficiency": 0.96,
        "nominal_voltage_v": 690.0,
        "power_factor": 0.95,
        "nominal_frequency_hz": 50.0,
        "winding_resistance_ohm": 0.002,
        "thermal_resistance_k_w": 0.006,
        "thermal_capacitance_j_k": 5.0e4,
    }
    runs = [
        _simulate_turbine(cfg, i, {**params[i], "turbine": turbine_constants})
        for i in range(cfg.n_turbines)
    ]
    _assign_neighbors(runs, cfg.neighbors_per_turbine)
    return runs


def _assign_neighbors(runs: list[TurbineRun], k: int) -> None:
    """Nearest same-site neighbours (used by the fleet/wake architecture)."""
    for run in runs:
        same_site = [r for r in runs if r.site_id == run.site_id and r.turbine_id != run.turbine_id]
        same_site.sort(key=lambda r: abs(r.turbine_id - run.turbine_id))
        chosen = same_site[:k]
        run.neighbor_ids = [r.turbine_id for r in chosen]
        run.neighbor_distance_m = [float(350.0 * (i + 1)) for i in range(len(chosen))]


def fleet_checksum(runs: list[TurbineRun]) -> str:
    """Stable SHA-256 over the fleet contents (data provenance for reports)."""
    h = hashlib.sha256()
    for run in runs:
        h.update(f"{run.turbine_id}:{run.site_id}:{run.fault_mode}".encode())
        h.update(np.nan_to_num(run.channels, nan=-999.0).astype(np.float32).tobytes())
        h.update(np.nan_to_num(run.targets, nan=-999.0).astype(np.float32).tobytes())
    return h.hexdigest()[:16]


def config_checksum(cfg: SyntheticFleetConfig) -> str:
    return hashlib.sha256(repr(sorted(cfg.to_dict().items())).encode()).hexdigest()[:16]


def fleet_summary(runs: list[TurbineRun]) -> dict:
    """Aggregate statistics recorded next to every benchmark result."""
    damage_end = np.array([r.damage[-1] for r in runs])
    rul_end = np.array([r.targets[-1, 2] * RUL_SCALE_DAYS for r in runs])
    modes: dict[str, int] = {}
    for r in runs:
        modes[r.fault_mode] = modes.get(r.fault_mode, 0) + 1
    missing = float(np.isnan(np.concatenate([r.channels for r in runs])).mean())
    return {
        "n_turbines": len(runs),
        "seq_len": int(runs[0].seq_len),
        "n_sites": len({r.site_id for r in runs}),
        "fault_modes": modes,
        "end_damage_mean": float(damage_end.mean()),
        "end_damage_max": float(damage_end.max()),
        "end_rul_days_mean": float(rul_end.mean()),
        "end_rul_days_min": float(rul_end.min()),
        "fraction_failed_or_near": float((damage_end >= 0.85).mean()),
        "missing_value_fraction": missing,
        "checksum": fleet_checksum(runs),
    }


__all__ = [
    "SyntheticFleetConfig",
    "TurbineRun",
    "generate_fleet",
    "fleet_checksum",
    "config_checksum",
    "fleet_summary",
]
