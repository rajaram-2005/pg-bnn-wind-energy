"""Evaluation pipeline: model scoring, verification behaviour and baselines."""

import numpy as np

from windfusion.data.loaders import build_dataloaders
from windfusion.evaluation.baselines import build_baselines, dataset_arrays
from windfusion.evaluation.benchmark import benchmark_family, measure_model
from windfusion.evaluation.evaluate import evaluate_model, evaluate_verification
from windfusion.evaluation.metrics import summarise_all
from windfusion.models import create_model


def test_evaluate_model_returns_every_metric_section(tiny_config):
    bundle = build_dataloaders(tiny_config)
    model = create_model("windfusion-edge", 12, 3, 5, tiny_config.turbine)
    result = evaluate_model(model, bundle.test, tiny_config, mc_samples=2, keep_arrays=True)
    assert set(result.metrics) >= {"regression", "classification", "early_warning", "calibration"}
    assert result.timing["windows"] > 0
    assert result.arrays["mean"].shape[1] == 3


def test_evaluation_is_finite(tiny_config):
    bundle = build_dataloaders(tiny_config)
    model = create_model("windfusion-lite", 12, 3, 5, tiny_config.turbine)
    result = evaluate_model(model, bundle.test, tiny_config, mc_samples=2, keep_arrays=False)
    for section in ("regression", "classification", "calibration"):
        for value in result.metrics[section].values():
            if isinstance(value, float):
                assert np.isfinite(value)


def test_verification_reports_abstention_and_coverage(tiny_config):
    bundle = build_dataloaders(tiny_config)
    model = create_model("windfusion-edge", 12, 3, 5, tiny_config.turbine)
    report = evaluate_verification(model, bundle.test, tiny_config, mc_samples=2)
    assert 0.0 <= report["abstention_rate"] <= 1.0
    assert 0.0 <= report["coverage"] <= 1.0
    assert report["n"] > 0
    assert sum(report["verdict_histogram"].values()) == report["n"]


def test_baselines_fit_and_predict(tiny_config):
    bundle = build_dataloaders(tiny_config)
    train = dataset_arrays(bundle.datasets.train, limit=256)
    test = dataset_arrays(bundle.datasets.test, limit=64)
    for name, baseline in build_baselines(epochs=1).items():
        baseline.fit(train)
        mean, std = baseline.predict(test)
        assert mean.shape == test.y.shape
        assert std.shape == test.y.shape
        metrics = summarise_all(test.y, mean, std)
        assert np.isfinite(metrics["regression"]["health_index"]["mae"])


def test_benchmark_measures_a_model(tiny_config):
    report = measure_model("windfusion-edge", tiny_config, repeats=5)
    assert report.parameters > 0
    assert report.latency_ms_median > 0
    assert report.throughput_windows_per_s > 0
    assert report.energy_per_inference == "NOT MEASURED"


def test_benchmark_family_covers_edge_and_research(tiny_config):
    report = benchmark_family(tiny_config, modes=["windfusion-edge", "windfusion-research"], repeats=3)
    assert set(report["results"]) == {"windfusion-edge", "windfusion-research"}
    assert report["results"]["windfusion-edge"]["parameters"] < report["results"]["windfusion-research"]["parameters"]
    assert "energy_per_inference" in report["not_measured"]
    assert report["results"]["windfusion-edge"]["energy_per_inference"] == "NOT MEASURED"
