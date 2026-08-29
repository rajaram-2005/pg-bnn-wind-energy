"""Digital twin: counterfactuals, rollout and twin-coupled verification."""

import torch

from windfusion.digital_twin import (
    DigitalTwin,
    TurbineState,
    TwinCoupledVerifier,
    twin_from_prediction,
)
from windfusion.verification.verifier import PhysicsConfidenceFusion, Verdict


def test_counterfactual_is_connected_and_side_effect_free():
    twin = DigitalTwin(TurbineState(vibration=2.0, bearing_temperature=60, health=0.9, rul_days=100))
    hot = twin.simulate(bearing_temperature=110)
    assert hot.health < 0.9 and hot.rul_days < 100
    assert twin.state.bearing_temperature == 60


def test_scenario_only_applies_known_fields():
    twin = DigitalTwin(TurbineState(health=0.9))
    trial = twin.simulate(health=0.9, not_a_field=5)
    assert not hasattr(trial, "not_a_field")


def test_rollout_accumulates_damage():
    twin = DigitalTwin(TurbineState(gearbox_temperature=95, vibration=6.0, torque=1.2e6, rotor_speed=1.5))
    stressed = twin.rollout(steps=50, dt_hours=1.0)
    assert stressed.damage > 0.0
    assert stressed.health <= twin.state.health + 1e-9


def test_prediction_updates_twin_state():
    twin = twin_from_prediction({"health": 0.4, "uncertainty": 0.2, "rul_days": 12.0})
    assert twin.state.health == 0.4
    assert twin.state.rul_days == 12.0
    assert len(twin.history.states) >= 2


def test_observations_merge_into_state():
    twin = DigitalTwin(TurbineState())
    twin.update(observations={"wind_speed": 11.0, "vibration": 3.0})
    assert twin.state.wind_speed == 11.0
    assert twin.state.vibration == 3.0


def test_twin_coupled_verifier_uses_twin_health():
    verifier = TwinCoupledVerifier(PhysicsConfidenceFusion())
    healthy = verifier.verify_window(0.05, 0.02, 0.02, twin_health=0.98, rul_days=300)
    failing = verifier.verify_window(0.95, 0.02, 0.02, twin_health=0.05, rul_days=1.0)
    assert healthy.verdict == Verdict.NORMAL
    assert failing.verdict == Verdict.CRITICAL


def test_twin_coupled_verifier_reads_a_batched_prediction():
    prediction = {
        "mean": torch.tensor([[0.0, 0.2, 0.01]]),
        "epistemic": torch.tensor([[0.02, 0.02, 0.02]]),
        "total": torch.tensor([[0.1, 0.1, 0.1]]),
    }
    result = TwinCoupledVerifier().verify_tensor(prediction, physics_residual=0.05, rul_days=3.0)
    assert result.verdict in {Verdict.CRITICAL, Verdict.WARNING}


def test_scenario_table_reports_deltas():
    twin = DigitalTwin(TurbineState(health=0.9, rul_days=100))
    rows = twin.scenario_table({"hot": {"gearbox_temperature": 110}})
    assert rows[0]["scenario"] == "baseline"
    assert rows[1]["delta_health"] < 0
