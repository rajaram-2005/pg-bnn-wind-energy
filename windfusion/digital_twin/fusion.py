"""Digital-twin-coupled verification.

The verifier needs a twin *health* signal; the twin needs model predictions to
stay synchronised with the asset. This module is the seam between them, and it
keeps the dependency direction clean: the twin never imports the models.
"""

from __future__ import annotations

import torch

from ..verification.verifier import PhysicsConfidenceFusion, VerificationResult
from .simulator import DigitalTwin
from .state import TurbineState


def twin_from_prediction(prediction: dict[str, float], state: TurbineState | None = None) -> DigitalTwin:
    """Build (or advance) a twin from a model prediction."""
    twin = DigitalTwin(state or TurbineState())
    twin.update(prediction=prediction)
    return twin


def twin_from_batch(prediction, index: int = 0, state: TurbineState | None = None) -> DigitalTwin:
    """Convenience wrapper for a batched torch prediction dictionary."""
    payload = {
        "health": float(prediction["mean"][index, 1]),
        "uncertainty": float(prediction["total"][index].mean()),
        "rul_days": float(prediction["mean"][index, 2]) * 365.0,
    }
    return twin_from_prediction(payload, state)


class TwinCoupledVerifier:
    """Verifier that reads twin health and writes predictions back to the twin."""

    def __init__(self, fusion: PhysicsConfidenceFusion | None = None) -> None:
        self.fusion = fusion or PhysicsConfidenceFusion()

    def verify_window(
        self,
        failure_probability: float,
        epistemic: float,
        physics_residual: float,
        twin_health: float,
        data_completeness: float = 1.0,
        rul_days: float | None = None,
        temporal_consistency: float = 1.0,
        predicted_consistency: float | None = None,
        warning_horizon_days: float = 30.0,
    ) -> VerificationResult:
        """Run the fusion check with twin state in the loop."""
        return self.fusion.verify(
            failure_probability=failure_probability,
            epistemic=epistemic,
            physics_residual=physics_residual,
            twin_health=max(0.0, min(1.0, float(twin_health))),
            temporal_consistency=temporal_consistency,
            data_completeness=data_completeness,
            predicted_consistency=predicted_consistency,
            rul_days=rul_days,
            warning_horizon_days=warning_horizon_days,
        )

    def verify_tensor(
        self,
        prediction: dict[str, torch.Tensor],
        physics_residual: float,
        index: int = 0,
        data_completeness: float = 1.0,
        rul_days: float | None = None,
    ) -> VerificationResult:
        """Verify one element of a batched prediction dictionary."""
        failure = float(torch.sigmoid(prediction["mean"][index, 0]))
        epistemic = float(prediction["epistemic"][index].mean())
        return self.verify_window(
            failure_probability=failure,
            epistemic=epistemic,
            physics_residual=physics_residual,
            twin_health=float(prediction["mean"][index, 1]),
            data_completeness=data_completeness,
            rul_days=rul_days,
            predicted_consistency=(
                float(prediction["consistency"][index].mean())
                if "consistency" in prediction
                else None
            ),
        )


__all__ = ["TwinCoupledVerifier", "twin_from_batch", "twin_from_prediction"]
