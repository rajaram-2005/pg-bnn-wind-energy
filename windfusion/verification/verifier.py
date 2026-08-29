"""Physics-Confidence Fusion: fail-closed verdicts for maintenance advice.

Provenance: ``verify_first_self_check`` (Aetheris ``aetheris/core/proof.py``).
A prediction is never returned as a bare number: it is combined with epistemic
uncertainty, normalised physics residuals, digital-twin health, temporal
consistency and data completeness. When the evidence does not support a claim
the verifier returns ``MODEL_UNCERTAIN`` (or ``INSUFFICIENT_DATA``) instead of a
confident-sounding guess.

This is an advisory filter, **not** a control system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..config import EvaluationConfig


class Verdict(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    MODEL_UNCERTAIN = "MODEL_UNCERTAIN"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass(frozen=True)
class VerifierThresholds:
    """Every threshold in one place so it can be audited and tuned."""

    epistemic_abstain: float = 0.45
    physics_abstain: float = 0.35
    min_data_completeness: float = 0.60
    critical_concern: float = 0.80
    warning_concern: float = 0.50
    min_confidence_for_critical: float = 0.55
    predicted_residual_abstain: float = 0.75

    @classmethod
    def from_config(cls, config: EvaluationConfig) -> "VerifierThresholds":
        return cls(
            epistemic_abstain=config.epistemic_abstain,
            physics_abstain=config.physics_abstain,
            min_data_completeness=config.min_data_completeness,
        )


@dataclass(frozen=True)
class VerificationResult:
    verdict: Verdict
    confidence: float
    physics_consistency: float
    reason: str
    concern: float = 0.0
    evidence: dict = field(default_factory=dict)
    advised_action: str = ""

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "confidence": round(self.confidence, 4),
            "physics_consistency": round(self.physics_consistency, 4),
            "reason": self.reason,
            "concern": round(self.concern, 4),
            "evidence": {k: round(float(v), 4) for k, v in self.evidence.items()},
            "advised_action": self.advised_action,
        }


ADVICE = {
    Verdict.NORMAL: "Continue routine condition monitoring.",
    Verdict.WARNING: "Schedule an inspection; review trend with a reliability engineer.",
    Verdict.CRITICAL: "Plan immediate inspection; do not actuate from this advisory.",
    Verdict.MODEL_UNCERTAIN: "Request higher-resolution telemetry and re-evaluate.",
    Verdict.INSUFFICIENT_DATA: "Restore sensor coverage before trusting any prediction.",
}


class PhysicsConfidenceFusion:
    """Combine prediction, uncertainty, physics and twin state into a verdict."""

    def __init__(self, thresholds: VerifierThresholds | None = None) -> None:
        self.thresholds = thresholds or VerifierThresholds()

    def verify(
        self,
        failure_probability: float,
        epistemic: float,
        physics_residual: float,
        twin_health: float,
        temporal_consistency: float = 1.0,
        data_completeness: float = 1.0,
        predicted_consistency: float | None = None,
        rul_days: float | None = None,
        warning_horizon_days: float = 30.0,
    ) -> VerificationResult:
        t = self.thresholds
        evidence = {
            "failure_probability": float(failure_probability),
            "epistemic": float(epistemic),
            "physics_residual": float(abs(physics_residual)),
            "twin_health": float(twin_health),
            "temporal_consistency": float(temporal_consistency),
            "data_completeness": float(data_completeness),
        }
        if predicted_consistency is not None:
            evidence["predicted_residual"] = float(predicted_consistency)
        if rul_days is not None:
            evidence["rul_days"] = float(rul_days)

        if data_completeness < t.min_data_completeness:
            return VerificationResult(
                Verdict.INSUFFICIENT_DATA,
                0.0,
                0.0,
                "missing critical sensor history",
                concern=max(failure_probability, 1 - twin_health),
                evidence=evidence,
                advised_action=ADVICE[Verdict.INSUFFICIENT_DATA],
            )

        consistency = max(0.0, 1.0 - abs(physics_residual))
        confidence = max(
            0.0,
            min(
                1.0,
                (1.0 - min(epistemic, 1.0)) * (0.5 + 0.5 * consistency) * max(temporal_consistency, 0.0),
            ),
        )
        concern = max(float(failure_probability), 1.0 - float(twin_health))
        if rul_days is not None:
            concern = max(concern, 1.0 - min(rul_days / max(warning_horizon_days, 1e-6), 1.0))

        # Fail-closed: high uncertainty, or a self-check that predicts a large
        # residual, or a strong claim contradicted by physics.
        uncertain = (
            epistemic > t.epistemic_abstain
            or consistency < t.physics_abstain
            or (predicted_consistency is not None and predicted_consistency > t.predicted_residual_abstain)
            or (failure_probability > 0.7 and consistency < 0.35 and twin_health > 0.7)
        )
        if uncertain:
            return VerificationResult(
                Verdict.MODEL_UNCERTAIN,
                confidence,
                consistency,
                "prediction requires additional high-resolution telemetry",
                concern=concern,
                evidence=evidence,
                advised_action=ADVICE[Verdict.MODEL_UNCERTAIN],
            )

        if concern >= t.critical_concern and confidence >= t.min_confidence_for_critical:
            verdict = Verdict.CRITICAL
        elif concern >= t.warning_concern:
            verdict = Verdict.WARNING
        else:
            verdict = Verdict.NORMAL
        return VerificationResult(
            verdict,
            confidence,
            consistency,
            "advisory only; qualified engineer review required",
            concern=concern,
            evidence=evidence,
            advised_action=ADVICE[verdict],
        )


__all__ = [
    "ADVICE",
    "PhysicsConfidenceFusion",
    "VerificationResult",
    "Verdict",
    "VerifierThresholds",
]
