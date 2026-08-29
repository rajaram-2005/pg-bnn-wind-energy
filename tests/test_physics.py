import torch

from windfusion.physics.aerodynamics import power_coefficient
from windfusion.physics.thermal import thermal_step


def test_betz_and_thermal_stability():
    cp = power_coefficient(torch.linspace(0.1, 20, 50), torch.zeros(50))
    assert bool(((cp >= 0) & (cp <= 16 / 27)).all())
    t = thermal_step(
        torch.tensor([100.0]), torch.tensor([20.0]), torch.tensor([0.0]), 0.1, 1000, 10
    )
    assert 20 < t.item() < 100
