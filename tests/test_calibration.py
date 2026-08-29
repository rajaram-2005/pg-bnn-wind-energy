"""Calibration: temperature scaling and conformal intervals."""

import torch

from windfusion.data.loaders import build_dataloaders
from windfusion.evaluation.metrics import expected_calibration_error, interval_coverage
from windfusion.models import create_model
from windfusion.training.calibration import (
    apply_calibration,
    calibrate,
    fit_conformal,
    fit_temperature,
)


def test_ece_is_low_for_a_well_calibrated_predictive():
    torch.manual_seed(0)
    mean = torch.zeros(2000, 1)
    target = torch.randn(2000, 1)
    good = expected_calibration_error(target.numpy(), mean.numpy(), torch.ones(2000, 1).numpy())
    bad = expected_calibration_error(target.numpy(), mean.numpy(), (torch.ones(2000, 1) * 0.2).numpy())
    assert good < bad


def test_temperature_scaling_recovers_a_known_scale():
    torch.manual_seed(0)
    log_var = torch.zeros(500, 2)
    residual = torch.randn(500, 2) * 3.0
    temperature = fit_temperature(log_var, residual)
    assert torch.allclose(temperature, torch.tensor([3.0, 3.0]), atol=0.3)


def test_conformal_quantile_achieves_requested_coverage():
    torch.manual_seed(0)
    mean = torch.zeros(4000, 1)
    log_var = torch.zeros(4000, 1)
    target = torch.randn(4000, 1)
    quantile = fit_conformal(mean, log_var, target, coverage=0.9)
    inside = (target.abs() <= quantile * log_var.exp().sqrt()).float().mean()
    assert float(inside) >= 0.88


def test_calibrate_improves_nll_on_the_calibration_split(tiny_config):
    bundle = build_dataloaders(tiny_config)
    model = create_model("windfusion-edge", 12, 3, 5, tiny_config.turbine)
    artifact = calibrate(model, bundle.val, mc_samples=2)
    assert artifact.nll_after <= artifact.nll_before + 1e-6
    assert len(artifact.temperature) == 3


def test_apply_calibration_scales_the_interval(tiny_config):
    bundle = build_dataloaders(tiny_config)
    model = create_model("windfusion-edge", 12, 3, 5, tiny_config.turbine)
    artifact = calibrate(model, bundle.val, mc_samples=2)
    prediction = model.predict(
        next(iter(bundle.test))["sequence"], next(iter(bundle.test))["physics"], samples=2
    )
    before = prediction["total"].clone()
    after = apply_calibration(prediction, artifact)["total"]
    assert after.shape == before.shape
    assert torch.isfinite(after).all()


def test_interval_coverage_reports_both_numbers():
    torch.manual_seed(0)
    mean = torch.zeros(500)
    std = torch.ones(500)
    target = torch.randn(500)
    report = interval_coverage(target.numpy(), mean.numpy(), std.numpy(), coverage=0.9)
    assert report["target_coverage"] == 0.9
    assert 0.85 <= report["empirical_coverage"] <= 0.95
