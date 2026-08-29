"""Physics relations: bounds, conservation and numerical stability."""

import torch

from windfusion.physics import aerodynamics, drivetrain, electrical, thermal
from windfusion.physics.constraints import PhysicsWeights, physics_loss, soft_hinge
from windfusion.physics.residuals import RESIDUAL_NAMES, compute_residuals, physics_feature_vector


def test_betz_limit_is_respected():
    cp = aerodynamics.power_coefficient(torch.linspace(0.1, 20, 200), torch.zeros(200))
    assert bool(((cp >= 0) & (cp <= 16 / 27)).all())


def test_power_scales_with_wind_cube():
    cp = torch.tensor([0.45])
    p1 = aerodynamics.aerodynamic_power(torch.tensor([5.0]), cp, 6361.0)
    p2 = aerodynamics.aerodynamic_power(torch.tensor([10.0]), cp, 6361.0)
    assert torch.allclose(p2 / p1, torch.tensor([8.0]), rtol=1e-4)


def test_betz_violation_is_one_sided():
    wind = torch.tensor([10.0])
    area = aerodynamics.swept_area(45.0)
    limit = aerodynamics.betz_power(wind, area)
    assert aerodynamics.betz_violation(limit * 0.5, wind, area).item() == 0.0
    assert aerodynamics.betz_violation(limit * 2.0, wind, area).item() > 0.0


def test_jensen_wake_deficit_decays_with_distance():
    ct = torch.tensor([0.8])
    near = aerodynamics.jensen_wake_deficit(torch.tensor([100.0]), 45.0, ct)
    far = aerodynamics.jensen_wake_deficit(torch.tensor([2000.0]), 45.0, ct)
    assert 0.0 < far.item() < near.item() < 1.0


def test_waked_wind_speed_is_lower_than_free_stream():
    free = torch.tensor([10.0])
    waked = aerodynamics.waked_wind_speed(free, torch.tensor([500.0]), 45.0, torch.tensor([0.8]))
    assert 0.0 <= waked.item() < free.item()


def test_iso281_life_decreases_with_load_and_is_monotone_in_rpm():
    light = drivetrain.iso_281_l10_hours(1000.0, 100.0)
    heavy = drivetrain.iso_281_l10_hours(1000.0, 300.0)
    assert heavy < light
    assert drivetrain.iso_281_l10_hours(1000.0, 100.0, rpm=3000) < light


def test_thermal_step_relaxes_toward_steady_state():
    temperature = torch.tensor([100.0])
    for _ in range(200):
        temperature = thermal.thermal_step(temperature, torch.tensor([20.0]), torch.tensor([0.0]), 0.1, 1000, 10)
    assert abs(temperature.item() - 20.0) < 1.0


def test_thermal_step_matches_numpy_twin():
    torch_value = thermal.thermal_step(torch.tensor([80.0]), torch.tensor([15.0]), torch.tensor([500.0]), 0.02, 5e4, 600)
    numpy_value = thermal.thermal_step_numpy(80.0, 15.0, 500.0, 0.02, 5e4, 600)
    assert abs(float(torch_value) - numpy_value) < 1e-3


def test_second_law_penalty_is_zero_when_physical():
    penalty = thermal.second_law_penalty(torch.tensor([80.0]), torch.tensor([20.0]), torch.tensor([100.0]))
    assert penalty.item() == 0.0


def test_three_phase_power_and_frequency_stress():
    power = electrical.three_phase_power(torch.tensor([690.0]), torch.tensor([100.0]), torch.tensor([0.95]))
    assert power.item() > 0
    assert electrical.frequency_stress(torch.tensor([50.0])).item() == 0.0
    assert electrical.frequency_stress(torch.tensor([52.0])).item() > 0.0


def test_soft_hinge_is_smooth_and_one_sided():
    below = soft_hinge(torch.tensor([1.0]), 2.0)
    above = soft_hinge(torch.tensor([4.0]), 2.0)
    assert below.item() < above.item()
    assert below.item() >= 0.0


def test_physics_loss_weights_and_terms():
    residuals = {name: torch.ones(4) for name in RESIDUAL_NAMES}
    total, terms = physics_loss(residuals, PhysicsWeights(aero=1.0, drive=0.0, thermal=0.0, grid=0.0, consistency=0.0))
    assert abs(float(total) - 1.0) < 1e-6
    assert set(terms) == set(RESIDUAL_NAMES)


def test_residuals_are_finite_and_dimensionless():
    torch.manual_seed(0)
    raw = torch.stack(
        [
            torch.tensor([10.0, 1.4, 2.0, 1500.0, 2.0, 65.0, 90.0, 60.0, 20.0, 1200.0, 9.0e5, 50.0]),
            torch.tensor([5.0, 0.7, 0.0, 300.0, 1.0, 45.0, 60.0, 45.0, 10.0, 300.0, 2.0e5, 50.1]),
        ]
    )
    residuals = compute_residuals(raw)
    assert set(residuals) == set(RESIDUAL_NAMES)
    for value in residuals.values():
        assert torch.isfinite(value).all()
        assert value.abs().max().item() < 10.0


def test_physics_feature_vector_shape_and_safety():
    raw = torch.randn(3, 12)
    raw[0, 2] = float("nan")
    features = physics_feature_vector(raw)
    assert features.shape == (3, 5)
    assert torch.isfinite(features).all()
    assert (features[:, 4] <= 1.0).all()
