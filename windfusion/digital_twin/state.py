from dataclasses import asdict, dataclass


@dataclass
class TurbineState:
    wind_speed: float = 0
    rotor_speed: float = 0
    pitch: float = 0
    torque: float = 0
    vibration: float = 0
    bearing_temperature: float = 20
    generator_temperature: float = 20
    gearbox_temperature: float = 20
    voltage: float = 0
    current: float = 0
    frequency: float = 50
    health: float = 1
    uncertainty: float = 1
    rul_days: float = 0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)
