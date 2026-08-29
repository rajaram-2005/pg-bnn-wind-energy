import torch

from windfusion.models import create_model
from windfusion.verification.verifier import PhysicsConfidenceFusion


def test_prediction_to_verdict():
    out = create_model(input_features=6).predict(torch.randn(1, 12, 6), torch.randn(1, 4), 3)
    result = PhysicsConfidenceFusion().verify(
        float(out["mean"][0, 0].sigmoid()), float(out["epistemic"][0].mean()), 0.1, 0.8
    )
    assert result.verdict.value in {
        "NORMAL",
        "WARNING",
        "CRITICAL",
        "MODEL_UNCERTAIN",
        "INSUFFICIENT_DATA",
    }
