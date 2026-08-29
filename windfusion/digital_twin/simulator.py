"""Computational state fusion and counterfactual simulation."""

from dataclasses import replace

from .state import TurbineState


class DigitalTwin:
    def __init__(self, state: TurbineState | None = None):
        self.state = state or TurbineState()

    def update(
        self, observations: dict[str, float], prediction: dict[str, float] | None = None
    ) -> TurbineState:
        valid = {k: float(v) for k, v in observations.items() if hasattr(self.state, k)}
        self.state = replace(self.state, **valid)
        if prediction:
            self.state = replace(
                self.state,
                health=max(0, min(1, float(prediction.get("health", self.state.health)))),
                uncertainty=max(0, float(prediction.get("uncertainty", self.state.uncertainty))),
                rul_days=max(0, float(prediction.get("rul_days", self.state.rul_days))),
            )
        return self.state

    def simulate(self, **changes: float) -> TurbineState:
        """Side-effect-free what-if with monotonic thermal/vibration/load damage."""
        trial = replace(
            self.state, **{k: float(v) for k, v in changes.items() if hasattr(self.state, k)}
        )
        thermal = max(0, trial.bearing_temperature - 70) / 80
        vib = max(0, trial.vibration - 2.5) / 10
        load = max(0, trial.current - self.state.current) / max(abs(self.state.current), 1)
        damage = min(0.95, thermal + vib + 0.2 * load)
        return replace(
            trial,
            health=max(0, self.state.health - damage),
            rul_days=max(0, self.state.rul_days * (1 - damage)),
        )
