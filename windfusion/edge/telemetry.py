"""Information-preserving adaptive telemetry.

Provenance: ``anomaly_bypass_telemetry`` and ``delta_deadband_quant_codec``
(AeroZip-Telemetry-Compression ``AeroZipSimulator.java``).

The upstream codec compresses with delta + deadband + quantisation and bypasses
compression when an anomaly score crosses a fixed threshold. Two changes are
made here:

1. the bypass trigger is a **multi-signal risk policy** (anomaly, epistemic
   uncertainty, rate of change, physics residual, hard safety override), not a
   single score;
2. hysteresis and a cooldown prevent flapping between modes, which is what
   makes a bypass policy usable in practice.

The Java codec is not vendored; this is a NumPy re-implementation so that
compression ratio and reconstruction fidelity can be measured in tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TelemetryDecision:
    mode: str  # raw | detailed | compressed
    keep_every: int
    risk: float
    reason: str = ""


class AdaptiveTelemetryPolicy:
    """Risk-aware compression policy with hysteresis and a bypass cooldown."""

    def __init__(
        self,
        bypass_threshold: float = 0.65,
        detail_threshold: float = 0.35,
        weights: dict[str, float] | None = None,
        cooldown_steps: int = 6,
        hysteresis: float = 0.05,
    ) -> None:
        self.bypass_threshold = float(bypass_threshold)
        self.detail_threshold = float(detail_threshold)
        self.weights = weights or {
            "anomaly": 0.35,
            "epistemic": 0.25,
            "rate": 0.20,
            "physics": 0.20,
        }
        self.cooldown_steps = int(cooldown_steps)
        self.hysteresis = float(hysteresis)
        self._cooldown = 0
        self._last_mode = "compressed"

    def risk(
        self,
        anomaly: float,
        epistemic: float,
        rate: float,
        physics_residual: float,
        safety: float = 0.0,
    ) -> float:
        """Combine the signals; a hard safety flag overrides everything."""
        score = (
            self.weights.get("anomaly", 0.25) * min(abs(anomaly), 1.0)
            + self.weights.get("epistemic", 0.25) * min(abs(epistemic), 1.0)
            + self.weights.get("rate", 0.25) * min(abs(rate), 1.0)
            + self.weights.get("physics", 0.25) * min(abs(physics_residual), 1.0)
        )
        return float(max(min(abs(safety), 1.0), min(score, 1.0)))

    def decide(
        self,
        anomaly: float,
        epistemic: float,
        rate: float,
        physics_residual: float,
        safety: float = 0.0,
    ) -> TelemetryDecision:
        risk = self.risk(anomaly, epistemic, rate, physics_residual, safety)
        if self._cooldown > 0:
            self._cooldown -= 1
            return TelemetryDecision("raw", 1, risk, "bypass cooldown active")

        # Hysteresis: once we escalate, stay escalated until risk drops clearly
        # below the threshold. Prevents mode flapping on noisy scores.
        if self._last_mode == "raw":
            bypass_at = self.bypass_threshold - self.hysteresis
            detail_at = self.detail_threshold - self.hysteresis
        else:
            bypass_at, detail_at = self.bypass_threshold, self.detail_threshold

        if risk >= bypass_at:
            self._last_mode = "raw"
            self._cooldown = self.cooldown_steps
            return TelemetryDecision("raw", 1, risk, "high risk: preserve full resolution")
        if risk >= detail_at:
            self._last_mode = "detailed"
            return TelemetryDecision("detailed", 2, risk, "moderate risk: halve sampling interval")
        self._last_mode = "compressed"
        return TelemetryDecision("compressed", 10, risk, "low risk: compress")

    # ── codec ──────────────────────────────────────────────────────────────
    def encode(self, values: np.ndarray, decision: TelemetryDecision) -> dict[str, object]:
        """Delta-code, deadband and quantise a window (or bypass it raw)."""
        values = np.asarray(values, dtype=np.float32)
        if decision.mode == "raw":
            payload = values
        else:
            payload = values[:: decision.keep_every]
            if decision.mode == "compressed":
                deltas = np.diff(payload, prepend=payload[:1])
                deltas[np.abs(deltas) < 1e-3] = 0.0
                payload = np.round(deltas / 0.01).astype(np.int32)
        return {
            "mode": decision.mode,
            "shape": values.shape,
            "keep_every": decision.keep_every,
            "samples": payload,
            "risk": decision.risk,
            "bytes_raw": int(values.nbytes),
            "bytes_encoded": int(np.asarray(payload).nbytes),
            "compression_ratio": float(np.asarray(payload).nbytes / max(values.nbytes, 1)),
        }

    def decode(self, encoded: dict[str, object]) -> np.ndarray:
        """Reconstruct an approximation of the original window."""
        payload = np.asarray(encoded["samples"])
        mode = str(encoded["mode"])
        if mode == "raw":
            return payload.astype(np.float32)
        if mode == "detailed":
            return payload.astype(np.float32)
        deltas = payload.astype(np.float32) * 0.01
        return np.cumsum(deltas).astype(np.float32)


def reconstruction_error(original: np.ndarray, decoded: np.ndarray) -> dict[str, float]:
    """Fidelity metrics for a compressed window (up to the sampling grid)."""
    original = np.asarray(original, dtype=np.float32)
    decoded = np.asarray(decoded, dtype=np.float32)
    n = min(original.shape[0], decoded.shape[0])
    if n == 0:
        return {"mae": 0.0, "max_abs": 0.0, "rmse": 0.0}
    diff = original[:n] - decoded[:n]
    return {
        "mae": float(np.abs(diff).mean()),
        "rmse": float(np.sqrt((diff**2).mean())),
        "max_abs": float(np.abs(diff).max()),
    }


def evaluate_policy(
    policy: AdaptiveTelemetryPolicy,
    windows: list[np.ndarray],
    signals: list[dict[str, float]],
) -> dict:
    """Measure a policy over a set of windows: bytes saved vs fidelity kept."""
    raw_bytes = encoded_bytes = 0
    errors: list[dict[str, float]] = []
    modes: dict[str, int] = {}
    for window, signal in zip(windows, signals):
        decision = policy.decide(**signal)
        modes[decision.mode] = modes.get(decision.mode, 0) + 1
        encoded = policy.encode(window, decision)
        decoded = policy.decode(encoded)
        errors.append(reconstruction_error(window[: len(decoded)] if decision.mode != "raw" else window, decoded))
        raw_bytes += int(encoded["bytes_raw"])
        encoded_bytes += int(encoded["bytes_encoded"])
    return {
        "windows": len(windows),
        "modes": modes,
        "raw_bytes": raw_bytes,
        "encoded_bytes": encoded_bytes,
        "bandwidth_reduction": round(1 - encoded_bytes / max(raw_bytes, 1), 4),
        "mean_mae": round(float(np.mean([e["mae"] for e in errors])), 6) if errors else 0.0,
        "mean_max_abs": round(float(np.mean([e["max_abs"] for e in errors])), 6) if errors else 0.0,
    }


__all__ = [
    "AdaptiveTelemetryPolicy",
    "TelemetryDecision",
    "evaluate_policy",
    "reconstruction_error",
]
