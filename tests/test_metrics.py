"""Evaluation metrics (provenance: ECE / early-warning protocol)."""

import numpy as np

from windfusion.evaluation.metrics import (
    auroc,
    calibration_summary,
    classification_metrics,
    early_warning_metrics,
    expected_asset_utilization,
    expected_calibration_error,
    first_warning_lead_time_days,
    per_target_regression,
    summarise_all,
)


def test_regression_metrics_on_a_known_error():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.0, 2.5])
    metrics = per_target_regression(
        np.stack([y_true, y_true, y_true], axis=1), np.stack([y_pred, y_pred, y_pred], axis=1)
    )
    assert abs(metrics["health_index"]["mae"] - 1 / 3) < 1e-9
    assert "mae_days" in metrics["rul_days"]


def test_auroc_is_one_for_perfect_separation_and_half_for_noise():
    y = np.array([0, 0, 1, 1])
    perfect = np.array([0.1, 0.2, 0.8, 0.9])
    assert abs(auroc(y, perfect) - 1.0) < 1e-9
    constant = np.array([0.5, 0.5, 0.5, 0.5])
    assert abs(auroc(y, constant) - 0.5) < 1e-9


def test_classification_metrics_counts():
    y = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.4, 0.6, 0.1])
    metrics = classification_metrics(y, scores, threshold=0.5)
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["false_alarm_rate"] == 0.5


def test_early_warning_protocol_matches_upstream_definition():
    y_true = np.array([5.0, 10.0, 100.0, 200.0])
    y_pred = np.array([6.0, 9.0, 300.0, 150.0])
    metrics = early_warning_metrics(y_true, y_pred, warning_horizon_days=30.0)
    assert metrics["n_at_risk"] == 2
    assert metrics["recall"] == 1.0
    assert metrics["mean_lead_time_days"] == 7.5


def test_first_warning_lead_time():
    ruls = np.array([100.0, 60.0, 20.0, 5.0])
    warned = np.array([False, True, True, True])
    assert first_warning_lead_time_days(ruls, warned) == 60.0
    assert first_warning_lead_time_days(ruls, np.array([False] * 4)) is None


def test_expected_asset_utilization_bounds():
    report = expected_asset_utilization(np.array([10.0, 200.0, 400.0]))
    assert 0.0 <= report["mean_utilization"] <= 1.0
    assert report["fraction_at_risk"] > 0


def test_calibration_summary_detects_underconfidence():
    rng = np.random.default_rng(0)
    target = rng.normal(size=500)
    mean = np.zeros(500)
    underconfident = calibration_summary(target, mean, np.ones(500) * 0.2)
    good = calibration_summary(target, mean, np.ones(500))
    assert underconfident.ece > good.ece
    assert underconfident.empirical_coverage < 0.9


def test_summarise_all_returns_every_section():
    rng = np.random.default_rng(1)
    target = np.stack(
        [rng.uniform(size=200), rng.uniform(0.4, 1.0, 200), rng.uniform(0, 1, 200)], axis=1
    )
    mean = target + rng.normal(scale=0.05, size=target.shape)
    std = np.full_like(mean, 0.1)
    bundle = summarise_all(target, mean, std)
    for section in ("regression", "classification", "early_warning", "calibration", "fleet"):
        assert section in bundle
    assert bundle["n_samples"] == 200
