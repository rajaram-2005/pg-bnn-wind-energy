"""Onboarding gates: pseudo-label acceptance and promotion discipline."""

from windfusion.data.loaders import build_dataloaders, per_turbine_loaders
from windfusion.models import create_model
from windfusion.training.onboarding import OnboardingGates, gate_rmse, onboard_site, scan_site


def test_scan_accepts_confident_windows_only(tiny_config):
    bundle = build_dataloaders(tiny_config)
    model = create_model("windfusion-edge", 12, 3, 5, tiny_config.turbine)
    strict = OnboardingGates(max_epistemic=0.0, max_physics_residual=0.0, min_pseudo_labels=1)
    loose = OnboardingGates(max_epistemic=1e9, max_physics_residual=1e9, min_pseudo_labels=1)
    _, strict_report = scan_site(model, bundle.val, strict, samples=2)
    _, loose_report = scan_site(model, bundle.val, loose, samples=2)
    assert loose_report.accepted > strict_report.accepted
    assert strict_report.rejected_uncertain > 0


def test_insufficient_data_is_rejected(tiny_config):
    bundle = build_dataloaders(tiny_config)
    model = create_model("windfusion-edge", 12, 3, 5, tiny_config.turbine)
    gates = OnboardingGates(min_data_completeness=0.999, min_pseudo_labels=8)
    _, report = scan_site(model, bundle.val, gates, samples=2)
    assert report.rejected_missing + report.rejected_uncertain > 0


def test_gate_rmse_is_positive_and_finite(tiny_config):
    bundle = build_dataloaders(tiny_config)
    model = create_model("windfusion-edge", 12, 3, 5, tiny_config.turbine)
    value = gate_rmse(model, bundle.val)
    assert value >= 0 and value == value


def test_onboarding_refuses_promotion_when_gate_fails(tiny_config):
    bundle = build_dataloaders(tiny_config)
    model = create_model("windfusion-edge", 12, 3, 5, tiny_config.turbine)
    loaders = list(per_turbine_loaders(bundle.datasets.train, batch_size=8).values())[:1]
    gates = OnboardingGates(
        max_epistemic=1e9,
        max_physics_residual=1e9,
        min_pseudo_labels=10_000,  # impossible to satisfy
    )
    _, report = onboard_site(model, loaders[0], bundle.val, tiny_config, gates)
    assert report.promoted is False
    assert "confident windows" in report.reason


def test_onboarding_promotes_when_gates_pass(tiny_config):
    bundle = build_dataloaders(tiny_config)
    model = create_model("windfusion-edge", 12, 3, 5, tiny_config.turbine)
    loaders = list(per_turbine_loaders(bundle.datasets.train, batch_size=8).values())[:1]
    gates = OnboardingGates(
        max_epistemic=1e9,
        max_physics_residual=1e9,
        min_pseudo_labels=1,
        max_promotion_rmse=1e9,
        max_regression=1e9,
    )
    promoted, report = onboard_site(model, loaders[0], bundle.val, tiny_config, gates)
    assert report.promoted is True
    assert report.gate_rmse is not None
    assert "promoted" in report.reason


def test_report_serialises(tiny_config):
    bundle = build_dataloaders(tiny_config)
    model = create_model("windfusion-edge", 12, 3, 5, tiny_config.turbine)
    _, report = scan_site(model, bundle.val, OnboardingGates(), samples=2)
    payload = report.as_dict()
    assert "acceptance_rate" in payload and "observed" in payload
