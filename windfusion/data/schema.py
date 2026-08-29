"""Channel, target and fault vocabulary shared by data, models and reports.

Provenance: ``fault_taxonomy_vocabulary`` (wind-turbine-pg-bnn
``src/faults/taxonomy.py``, ``src/faults/limits.py``), ``data_zone_layout``
(ai-machinery-etl-pipeline).

The vocabulary is deliberately narrow (12 SCADA channels) so that every model
preset shares one input contract and benchmarks stay comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

# ── SCADA channels (input order) ────────────────────────────────────────────
CHANNELS: tuple[str, ...] = (
    "wind_speed",
    "rotor_speed",
    "pitch_angle",
    "active_power_kw",
    "vibration_rms",
    "gearbox_oil_temp_c",
    "generator_winding_temp_c",
    "main_bearing_temp_c",
    "ambient_temp_c",
    "phase_current_a",
    "generator_torque_nm",
    "grid_frequency_hz",
)

CHANNEL_UNITS: dict[str, str] = {
    "wind_speed": "m/s",
    "rotor_speed": "rad/s",
    "pitch_angle": "deg",
    "active_power_kw": "kW",
    "vibration_rms": "mm/s",
    "gearbox_oil_temp_c": "degC",
    "generator_winding_temp_c": "degC",
    "main_bearing_temp_c": "degC",
    "ambient_temp_c": "degC",
    "phase_current_a": "A",
    "generator_torque_nm": "N.m",
    "grid_frequency_hz": "Hz",
}

CHANNEL_INDEX: dict[str, int] = {name: i for i, name in enumerate(CHANNELS)}
N_CHANNELS = len(CHANNELS)

# ── Derived physics features (second input tensor) ──────────────────────────
PHYSICS_FEATURES: tuple[str, ...] = (
    "tip_speed_ratio",
    "power_coefficient",
    "power_residual",
    "thermal_margin",
    "data_completeness",
)
N_PHYSICS = len(PHYSICS_FEATURES)

# ── Targets (output order) ──────────────────────────────────────────────────
TARGETS: tuple[str, ...] = (
    "failure_probability_30d",
    "health_index",
    "rul_days",
)
TARGET_INDEX: dict[str, int] = {name: i for i, name in enumerate(TARGETS)}
N_TARGETS = len(TARGETS)

# ``rul_days`` is stored normalised by RUL_SCALE_DAYS to keep the regression
# targets on a comparable numerical scale.
RUL_SCALE_DAYS = 365.0

# ── Fault-mode vocabulary (synthetic generator + reports) ───────────────────
FAULT_MODES: tuple[str, ...] = (
    "healthy",
    "bearing_wear",
    "gearbox_overheat",
    "generator_winding",
    "blade_aero_loss",
)

FAULT_SUBSYSTEM: dict[str, str] = {
    "healthy": "none",
    "bearing_wear": "main_bearing",
    "gearbox_overheat": "gearbox",
    "generator_winding": "generator",
    "blade_aero_loss": "rotor_blades",
}

SEVERITIES: tuple[str, ...] = ("LOW", "MEDIUM", "HIGH", "CRITICAL")


@dataclass(frozen=True)
class ChannelSpec:
    """Unit and plausibility range used for sanity checks and reporting."""

    name: str
    unit: str
    min_value: float
    max_value: float

    def clip(self, value: float) -> float:
        return min(max(value, self.min_value), self.max_value)


# Research-grade plausibility envelope. These are NOT OEM limits: they exist so
# the synthetic generator and the ingest sanity checks reject nonsense.
CHANNEL_SPECS: dict[str, ChannelSpec] = {
    "wind_speed": ChannelSpec("wind_speed", "m/s", 0.0, 40.0),
    "rotor_speed": ChannelSpec("rotor_speed", "rad/s", 0.0, 3.5),
    "pitch_angle": ChannelSpec("pitch_angle", "deg", -5.0, 90.0),
    "active_power_kw": ChannelSpec("active_power_kw", "kW", -50.0, 2600.0),
    "vibration_rms": ChannelSpec("vibration_rms", "mm/s", 0.0, 30.0),
    "gearbox_oil_temp_c": ChannelSpec("gearbox_oil_temp_c", "degC", -40.0, 120.0),
    "generator_winding_temp_c": ChannelSpec("generator_winding_temp_c", "degC", -40.0, 180.0),
    "main_bearing_temp_c": ChannelSpec("main_bearing_temp_c", "degC", -40.0, 130.0),
    "ambient_temp_c": ChannelSpec("ambient_temp_c", "degC", -40.0, 55.0),
    "phase_current_a": ChannelSpec("phase_current_a", "A", 0.0, 3000.0),
    "generator_torque_nm": ChannelSpec("generator_torque_nm", "N.m", 0.0, 2.0e6),
    "grid_frequency_hz": ChannelSpec("grid_frequency_hz", "Hz", 47.0, 53.0),
}


def channel_vector(values: dict[str, float]) -> list[float]:
    """Order a partial channel dict into the canonical input vector."""
    return [float(values.get(name, float("nan"))) for name in CHANNELS]


def describe_targets() -> str:
    return ", ".join(TARGETS)
