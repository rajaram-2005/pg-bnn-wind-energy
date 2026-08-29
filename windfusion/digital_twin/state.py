"""Typed turbine state for the digital twin.

Provenance: ``twin_state_vocabulary`` (TurbineDigitalTwin ``app.py``). The
upstream dashboard couples the state to a Random Forest predictor; here the
state is an explicit physical/learned hybrid that the model reads and writes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class TurbineState:
    """One snapshot of a turbine, in SI units unless stated otherwise."""

    # environment
    wind_speed: float = 0.0
    ambient_temperature: float = 15.0
    icing_risk: float = 0.0
    # rotor / drive
    rotor_speed: float = 0.0
    pitch: float = 0.0
    torque: float = 0.0
    vibration: float = 0.0
    # temperatures
    bearing_temperature: float = 20.0
    gearbox_temperature: float = 45.0
    generator_temperature: float = 40.0
    # electrical
    voltage: float = 690.0
    current: float = 0.0
    frequency: float = 50.0
    # learned / derived
    health: float = 1.0
    uncertainty: float = 1.0
    rul_days: float = 365.0
    damage: float = 0.0
    operating_hours: float = 0.0
    # bookkeeping
    site_id: int = 0
    turbine_id: int = 0
    fault_mode: str = "unknown"
    last_update_step: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def channel_vector(self) -> list[float]:
        """Project onto the SCADA channel order used by the models."""
        return [
            self.wind_speed,
            self.rotor_speed,
            self.pitch,
            self.current,  # placeholder replaced by the caller for active power
            self.vibration,
            self.gearbox_temperature,
            self.generator_temperature,
            self.bearing_temperature,
            self.ambient_temperature,
            self.current,
            self.torque,
            self.frequency,
        ]


@dataclass
class TwinHistory:
    """Rolling record of twin states so trends and drift can be reported."""

    states: list[TurbineState] = field(default_factory=list)
    max_len: int = 512

    def push(self, state: TurbineState) -> None:
        self.states.append(state)
        if len(self.states) > self.max_len:
            self.states = self.states[-self.max_len :]

    def health_trend(self) -> list[float]:
        return [s.health for s in self.states]

    def as_dict(self) -> dict:
        return {"n": len(self.states), "states": [s.to_dict() for s in self.states[-16:]]}


__all__ = ["TurbineState", "TwinHistory"]
