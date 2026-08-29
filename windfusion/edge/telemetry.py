"""Information-preserving adaptive telemetry policy."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TelemetryDecision:
    mode: str
    keep_every: int
    risk: float


class AdaptiveTelemetryPolicy:
    def __init__(self, bypass_threshold: float = 0.65, detail_threshold: float = 0.35):
        self.bypass_threshold, self.detail_threshold = bypass_threshold, detail_threshold

    def decide(
        self,
        anomaly: float,
        epistemic: float,
        rate: float,
        physics_residual: float,
        safety: float = 0,
    ) -> TelemetryDecision:
        risk = max(
            float(safety),
            0.35 * anomaly
            + 0.25 * epistemic
            + 0.2 * min(abs(rate), 1)
            + 0.2 * min(abs(physics_residual), 1),
        )
        return (
            TelemetryDecision("raw", 1, risk)
            if risk >= self.bypass_threshold
            else (
                TelemetryDecision("detailed", 2, risk)
                if risk >= self.detail_threshold
                else TelemetryDecision("compressed", 10, risk)
            )
        )

    def encode(self, values: np.ndarray, decision: TelemetryDecision) -> dict[str, object]:
        values = np.asarray(values, dtype=np.float32)
        selected = values[:: decision.keep_every]
        return {
            "mode": decision.mode,
            "shape": values.shape,
            "samples": selected,
            "compression_ratio": float(selected.nbytes / max(values.nbytes, 1)),
        }
