"""Physics relations: bounds, conservation and numerical stability."""

import torch

from windfusion.physics import aerodynamics, drivetrain, electrical, thermal
from windfusion.physics.constraints import limit_penalties, PhysicsWeights, physics_loss, soft_hinge
from windfusion.data.schema import CHANNELS, CHANNEL_INDEX
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


def test_missing_channels_do_not_create_physics_violations():
    """A dropout is missing data, not evidence of physical inconsistency."""
    raw = torch.zeros(4, len(CHANNELS), dtype=torch.float32)
    raw[0] = SANE
    raw[1] = SANE
    raw[1, CHANNEL_INDEX["wind_speed"]] = float("nan")
    raw[2] = SANE
    raw[2, CHANNEL_INDEX["grid_frequency_hz"]] = float("nan")
    raw[3] = float("nan")

    residuals = compute_residuals(raw)
    for name in ("aero", "drive", "thermal", "grid", "consistency"):
        assert torch.isfinite(residuals[name]).all(), name
        assert float(residuals[name].abs().max()) <= 10.0, name
    assert float(residuals["aero"][1]) == 0.0
    assert float(residuals["grid"][2]) == 0.0
    assert float(residuals["drive"][3]) == 0.0


def test_betz_violation_is_normalised_by_rated_power():
    """Zero wind speed must not produce a megawatt-scale 'violation'."""
    raw = torch.zeros(2, len(CHANNELS), dtype=torch.float32)
    raw[0] = SANE
    raw[1] = SANE
    raw[1, CHANNEL_INDEX["wind_speed"]] = 0.0
    raw[1, CHANNEL_INDEX["active_power_kw"]] = 2000.0
    consistency = compute_residuals(raw)["consistency"]
    assert float(consistency[1]) <= 2.0, float(consistency[1])


def test_limit_penalty_stays_bounded_for_a_clipped_sensor():
    raw = {
        "generator_winding_temp_c": torch.tensor([60.0, 121.0, 180.0]),
        "vibration_rms": torch.tensor([1.0, 4.6, 20.0]),
    }
    penalties = limit_penalties(raw)
    assert torch.isfinite(penalties["limit_generator_winding_temp_c"]).all()
    assert float(penalties["limit_generator_winding_temp_c"].max()) < 10.0

    with_missing = dict(raw)
    with_missing["generator_winding_temp_c"] = torch.tensor([60.0, float("nan"), 180.0])
    out = limit_penalties(with_missing)
    assert float(out["limit_generator_winding_temp_c"][1]) == 0.0


# A physically consistent operating point, repeated across the batch.
SANE = torch.tensor(
    [
        11.0,   # wind_speed
        1.4,    # rotor_speed
        3.0,    # pitch_angle
        1600.0, # active_power_kw
        1.2,    # vibration_rms
        62.0,   # gearbox_oil_temp_c
        85.0,   # generator_winding_temp_c
        55.0,   # main_bearing_temp_c
        18.0,   # ambient_temp_c
        1400.0, # phase_current_a
        1.1e6,  # generator_torque_nm
        50.0,   # grid_frequency_hz
    ],
    dtype=torch.float32,
)
