from windfusion.digital_twin.simulator import DigitalTwin
from windfusion.digital_twin.state import TurbineState


def test_counterfactual_is_connected_and_side_effect_free():
    d = DigitalTwin(TurbineState(vibration=2, bearing_temperature=60, health=0.9, rul_days=100))
    hot = d.simulate(bearing_temperature=110)
    assert hot.health < 0.9 and hot.rul_days < 100
    assert d.state.bearing_temperature == 60
