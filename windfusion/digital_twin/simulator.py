"""Computational digital twin: state fusion and counterfactual simulation.

The twin is *model-connected*: health and RUL come from the learned model and
are corrected by physics (accumulated damage from load, temperature and
vibration). Counterfactual ``simulate`` is side-effect free — it never mutates
the live state — and ``rollout`` applies a scenario over several steps.
"""

from __future__ import annotations

import math
from dataclasses import replace

from ..physics import thermal
from .state import TurbineState, TwinHistory


class DigitalTwin:
    """A single turbine's computational twin."""

    def __init__(self, state: TurbineState | None = None, damage_gain: float = 1.0) -> None:
        self.state = state or TurbineState()
        self.history = TwinHistory()
        self.damage_gain = float(damage_gain)
        self.history.push(self.state)

    # ── state updates ──────────────────────────────────────────────────────
    def update(
        self,
        observations: dict[str, float] | None = None,
        prediction: dict[str, float] | None = None,
        dt_hours: float = 0.0,
    ) -> TurbineState:
        """Merge sensor observations and a model prediction into the state."""
        if observations:
            valid = {
                k: float(v)
                for k, v in observations.items()
                if hasattr(self.state, k) and v is not None
            }
            self.state = replace(self.state, **valid)
        if prediction:
            self.state = replace(
                self.state,
                health=min(1.0, max(0.0, float(prediction.get("health", self.state.health)))),
                uncertainty=max(0.0, float(prediction.get("uncertainty", self.state.uncertainty))),
                rul_days=max(0.0, float(prediction.get("rul_days", self.state.rul_days))),
            )
        if dt_hours:
            damage_rate = self._damage_rate(self.state)
            self.state = replace(
                self.state,
                damage=min(1.5, self.state.damage + damage_rate * dt_hours),
                operating_hours=self.state.operating_hours + dt_hours,
            )
        # Learned health is capped by the accumulated physical damage.
        self.state = replace(
            self.state, health=min(self.state.health, math.exp(-self.state.damage))
        )
        self.history.push(self.state)
        return self.state

    def _damage_rate(self, state: TurbineState) -> float:
        """Damage per operating hour from load, vibration and temperature."""
        load_ratio = min(max(state.torque * state.rotor_speed / 2.0e6, 0.0), 2.0)
        vibration_ratio = max(state.vibration / 4.5, 0.0)
        hottest = max(state.gearbox_temperature, state.bearing_temperature)
        thermal_factor = math.exp(max(-3.0, min(3.0, (hottest - 70.0) / 40.0)))
        base = load_ratio**3 * (1.0 + vibration_ratio**2) * thermal_factor
        return self.damage_gain * base * 2.0e-4

    # ── counterfactuals ────────────────────────────────────────────────────
    def simulate(self, **changes: float) -> TurbineState:
        """Side-effect-free what-if: apply changes and re-estimate health/RUL."""
        trial = replace(
            self.state, **{k: float(v) for k, v in changes.items() if hasattr(self.state, k)}
        )
        hottest = max(trial.gearbox_temperature, trial.generator_temperature, trial.bearing_temperature)
        thermal_excess = max(0.0, hottest - 70.0) / 80.0
        vibration_excess = max(0.0, trial.vibration - 2.5) / 10.0
        load_excess = max(0.0, trial.current - self.state.current) / max(abs(self.state.current), 1.0)
        damage = min(0.95, thermal_excess + vibration_excess + 0.2 * load_excess)
        return replace(
            trial,
            health=max(0.0, self.state.health - damage),
            rul_days=max(0.0, self.state.rul_days * (1.0 - damage)),
        )

    def rollout(self, steps: int = 24, dt_hours: float = 0.5, **changes: float) -> TurbineState:
        """Apply a scenario repeatedly (Euler integration of the twin state)."""
        state = replace(
            self.state, **{k: float(v) for k, v in changes.items() if hasattr(self.state, k)}
        )
        twin = DigitalTwin(state, self.damage_gain)
        for _ in range(max(int(steps), 1)):
            state = twin.update(dt_hours=dt_hours)
        return state

    # ── reporting ──────────────────────────────────────────────────────────
    def scenario_table(self, scenarios: dict[str, dict[str, float]]) -> list[dict]:
        """Compare several what-if scenarios (used by the CLI and reports)."""
        rows = []
        baseline = self.state
        rows.append(
            {
                "scenario": "baseline",
                "health": round(baseline.health, 4),
                "rul_days": round(baseline.rul_days, 2),
            }
        )
        for name, changes in scenarios.items():
            trial = self.simulate(**changes)
            rows.append(
                {
                    "scenario": name,
                    "health": round(trial.health, 4),
                    "rul_days": round(trial.rul_days, 2),
                    "delta_health": round(trial.health - baseline.health, 4),
                }
            )
        return rows

    def to_dict(self) -> dict:
        return {"state": self.state.to_dict(), "history": self.history.as_dict()}


def steady_state_temperature(heat_w: float, ambient_c: float, resistance_k_w: float = 0.02) -> float:
    """Convenience wrapper around the lumped RC steady state."""
    return float(ambient_c + heat_w * resistance_k_w)


__all__ = ["DigitalTwin", "steady_state_temperature"]
