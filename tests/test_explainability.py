"""Explainability reports."""

import torch

from windfusion.explainability.report import (
    DISCLAIMER,
    explain,
    gradient_saliency,
    render_report,
)
from windfusion.models import create_model


def _prediction(model, batch=2, samples=2):
    return model.predict(torch.randn(batch, 24, 12), torch.randn(batch, 5), samples=samples)


def test_report_contains_routing_residuals_and_disclaimer():
    model = create_model("windfusion-lite", 12, 3, 5)
    prediction = _prediction(model)
    report = explain(
        prediction,
        {"aero": 0.1, "drive": 0.2, "thermal": 0.05, "grid": 0.01, "consistency": 0.0},
        {"health": 0.8},
    )
    assert report["routing"]["dominant_expert"] in model.expert_names
    assert set(report["physics_residuals"]) == {"aero", "drive", "thermal", "grid", "consistency"}
    assert report["recommended_action"] == DISCLAIMER
    assert 0.0 <= report["prediction"]["failure_probability"] <= 1.0


def test_gradient_saliency_prefers_the_driving_channel():
    torch.manual_seed(0)
    model = create_model("windfusion-edge", 12, 3, 5)
    sequence = torch.randn(4, 24, 12, requires_grad=False)
    saliency = gradient_saliency(model, sequence, torch.randn(4, 5))
    assert saliency.shape == (12,)
    assert torch.isfinite(saliency).all()
    assert float(saliency.sum()) > 0


def test_saliency_can_be_embedded_in_the_report():
    model = create_model("windfusion-lite", 12, 3, 5)
    prediction = _prediction(model, batch=1)
    report = explain(
        prediction,
        {"aero": 0.0},
        {"health": 0.9},
        model=model,
        sequence=torch.randn(1, 24, 12),
        physics=torch.randn(1, 5),
        top_k_features=3,
    )
    assert len(report["top_features"]) == 3
    assert {"channel", "saliency"} <= set(report["top_features"][0])


def test_render_report_is_human_readable():
    model = create_model("windfusion-edge", 12, 3, 5)
    prediction = _prediction(model, batch=1)
    report = explain(prediction, {"aero": 0.2}, {"health": 0.7}, top_features=["vibration_rms"])
    text = render_report(report)
    assert "WindFusion advisory report" in text
    assert "advisory" in text.lower()
