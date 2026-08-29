"""Aetheris base architecture.

Provenance: ``tiered_model_router`` and ``verify_first_self_check`` (Aetheris
``aetheris/core/model_router.py``, ``aetheris/core/proof.py``).

Aetheris supplies two ideas that survive the translation from a general
assistant to a turbine model:

1. *Tiered routing before inference* — capacity is chosen up front and recorded
   with the answer, rather than being implicit in a single monolith.
2. *Verify-first* — an answer is not returned until an independent check runs.
   Here that check is a learned self-check head that predicts the expected
   magnitude of the physics residual, plus the deterministic verifier in
   :mod:`windfusion.verification.verifier`.
"""

from __future__ import annotations

import torch

from .base import WindFusionBase
from .registry import AETHERIS

__all__ = ["AetherisWind"]


class AetherisWind(WindFusionBase):
    """Balanced base model used as the reference architecture for the family."""

    def __init__(
        self,
        input_features: int = 12,
        outputs: int = 3,
        physics_features: int = 5,
        turbine=None,
    ) -> None:
        super().__init__(AETHERIS, input_features, outputs, physics_features, turbine)

    @torch.no_grad()
    def predict_verified(
        self,
        sequence: torch.Tensor,
        physics: torch.Tensor,
        neighbors: torch.Tensor | None = None,
        samples: int = 16,
        data_completeness: float = 1.0,
        twin_health: float = 1.0,
    ) -> dict:
        """Predict and immediately run the verifier (Aetheris verify-first)."""
        from ..verification.verifier import PhysicsConfidenceFusion

        prediction = self.predict(sequence, physics, neighbors, samples=samples)
        residuals = self.expected_residuals(sequence)
        index = 0
        residual_scalar = float(
            torch.stack([v.reshape(v.shape[0], -1)[index].abs().mean() for v in residuals.values()]).mean()
        )
        failure_probability = float(torch.sigmoid(prediction["mean"][index, 0]))
        epistemic = float(prediction["epistemic"][index].mean())
        predicted_consistency = (
            float(prediction["consistency"][index].mean()) if "consistency" in prediction else residual_scalar
        )
        verdict = PhysicsConfidenceFusion().verify(
            failure_probability=failure_probability,
            epistemic=epistemic,
            physics_residual=residual_scalar,
            twin_health=twin_health,
            data_completeness=data_completeness,
            predicted_consistency=predicted_consistency,
        )
        return {
            "prediction": {k: v for k, v in prediction.items() if isinstance(v, torch.Tensor)},
            "residuals": {k: float(v.reshape(v.shape[0], -1)[index].abs().mean()) for k, v in residuals.items()},
            "failure_probability": failure_probability,
            "epistemic": epistemic,
            "verdict": verdict.verdict.value,
            "confidence": verdict.confidence,
            "reason": verdict.reason,
        }
