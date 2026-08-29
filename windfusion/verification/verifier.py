"""Physics-confidence fusion and fail-closed maintenance verdicts."""

from dataclasses import dataclass
from enum import Enum


class Verdict(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    MODEL_UNCERTAIN = "MODEL_UNCERTAIN"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass(frozen=True)
class VerificationResult:
    verdict: Verdict
    confidence: float
    physics_consistency: float
    reason: str


class PhysicsConfidenceFusion:
    def verify(
        self,
        failure_probability: float,
        epistemic: float,
        physics_residual: float,
        twin_health: float,
        temporal_consistency: float = 1,
        data_completeness: float = 1,
    ) -> VerificationResult:
        if data_completeness < 0.6:
            return VerificationResult(
                Verdict.INSUFFICIENT_DATA, 0, 0, "missing critical sensor history"
            )
        consistency = max(0, 1 - abs(physics_residual))
        confidence = max(
            0, min(1, (1 - epistemic) * (0.5 + 0.5 * consistency) * temporal_consistency)
        )
        if epistemic > 0.45 or (
            failure_probability > 0.7 and consistency < 0.35 and twin_health > 0.7
        ):
            return VerificationResult(
                Verdict.MODEL_UNCERTAIN,
                confidence,
                consistency,
                "prediction requires additional high-resolution telemetry",
            )
        concern = max(failure_probability, 1 - twin_health)
        verdict = (
            Verdict.CRITICAL
            if concern >= 0.8 and confidence >= 0.55
            else Verdict.WARNING
            if concern >= 0.5
            else Verdict.NORMAL
        )
        return VerificationResult(
            verdict, confidence, consistency, "advisory only; qualified engineer review required"
        )
