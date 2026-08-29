"""Edge runtime: streaming inference, drift monitoring and graceful degradation.

The runtime owns everything that a real edge deployment needs but a training
script does not:

* a fixed-length ring buffer that assembles windows from streaming samples,
* missing-sensor accounting (data completeness feeds the verifier),
* inference cadence so the CPU budget is bounded,
* covariate-drift monitoring (population stability index on the feature means),
* a telemetry decision per window,
* graceful degradation: on any inference failure the runtime falls back to the
  last known good prediction and reports it as stale rather than silent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..config import TelemetryConfig, WindFusionConfig
from ..data.schema import CHANNELS
from ..physics.residuals import physics_feature_vector
from .telemetry import AdaptiveTelemetryPolicy

STATE_OK = "ok"
STATE_WARMUP = "warmup"
STATE_STALE = "stale"
STATE_ERROR = "error"


@dataclass
class RuntimeStep:
    """What the runtime produced for one incoming sample."""

    state: str
    prediction: dict | None = None
    telemetry: dict | None = None
    drift: dict | None = None
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "state": self.state,
            "prediction": self.prediction,
            "telemetry": self.telemetry,
            "drift": self.drift,
            "note": self.note,
        }


@dataclass
class DriftMonitor:
    """Population stability index against a reference feature distribution."""

    reference: np.ndarray | None = None
    bins: int = 10
    psi_warning: float = 0.2
    history: list[float] = field(default_factory=list)

    def fit(self, features: np.ndarray) -> None:
        self.reference = np.asarray(features, dtype=np.float64)

    def _bin(self, values: np.ndarray, edges: np.ndarray) -> np.ndarray:
        counts, _ = np.histogram(values, bins=edges)
        return counts / max(counts.sum(), 1)

    def score(self, features: np.ndarray) -> float:
        if self.reference is None:
            return 0.0
        values = np.asarray(features, dtype=np.float64).ravel()
        edges = np.histogram_bin_edges(self.reference.ravel(), bins=self.bins)
        p = self._bin(values, edges)
        q = self._bin(self.reference.ravel(), edges)
        psi = float(
            np.sum((p - q) * np.log((p + 1e-6) / (q + 1e-6)))
        )
        self.history.append(psi)
        return psi

    def report(self) -> dict:
        if not self.history:
            return {"psi": 0.0, "drift": False, "n": 0}
        current = self.history[-1]
        return {
            "psi": round(current, 5),
            "drift": bool(current > self.psi_warning),
            "n": len(self.history),
            "max_psi": round(max(self.history), 5),
        }


class EdgeRuntime:
    """Streaming wrapper around any WindFusion model."""

    def __init__(
        self,
        model,
        config: WindFusionConfig | None = None,
        telemetry_config: TelemetryConfig | None = None,
        window: int = 24,
        infer_every: int = 1,
        mc_samples: int = 4,
    ) -> None:
        self.model = model.eval()
        self.config = config or WindFusionConfig()
        self.window = int(window)
        self.infer_every = max(int(infer_every), 1)
        self.mc_samples = max(int(mc_samples), 1)
        self.buffer: list[np.ndarray] = []
        telemetry_cfg = telemetry_config or self.config.telemetry
        self.policy = AdaptiveTelemetryPolicy(
            bypass_threshold=telemetry_cfg.bypass_threshold,
            detail_threshold=telemetry_cfg.detail_threshold,
            weights=dict(telemetry_cfg.weights),
            cooldown_steps=telemetry_cfg.cooldown_steps,
        )
        self.drift = DriftMonitor()
        self.counter = 0
        self.last_prediction: dict | None = None
        self.last_channels: np.ndarray | None = None

    # ── ingestion ──────────────────────────────────────────────────────────
    def push(self, sample) -> RuntimeStep:
        """Ingest one SCADA sample (dict or array in canonical channel order)."""
        if isinstance(sample, dict):
            values = np.array([float(sample.get(name, np.nan)) for name in CHANNELS], dtype=np.float32)
        else:
            values = np.asarray(sample, dtype=np.float32).reshape(-1)
            if values.shape[0] != len(CHANNELS):
                raise ValueError(f"sample must have {len(CHANNELS)} channels, got {values.shape[0]}")
        self.buffer.append(values)
        if len(self.buffer) > self.window:
            self.buffer = self.buffer[-self.window :]
        self.counter += 1

        if len(self.buffer) < self.window:
            return RuntimeStep(STATE_WARMUP, note=f"buffering {len(self.buffer)}/{self.window}")

        decision = self.policy.decide(
            anomaly=self._anomaly(values),
            epistemic=float(self.last_prediction["epistemic"]) if self.last_prediction else 0.5,
            rate=self._rate(values),
            physics_residual=float(self.last_prediction.get("residual", 0.0)) if self.last_prediction else 0.0,
        )
        telemetry = self.policy.encode(self._window(), decision)

        if self.counter % self.infer_every != 0:
            return RuntimeStep(STATE_OK, telemetry=telemetry, note="cadence skip")

        try:
            prediction = self._infer()
        except Exception as exc:  # pragma: no cover - defensive edge path
            return RuntimeStep(
                STATE_ERROR,
                prediction=self.last_prediction,
                telemetry=telemetry,
                note=f"inference failed: {type(exc).__name__}; serving last known good (stale)",
            )
        self.last_prediction = prediction
        self.drift.score(self._window().mean(0))
        return RuntimeStep(
            STATE_OK, prediction=prediction, telemetry=telemetry, drift=self.drift.report()
        )

    # ── internals ──────────────────────────────────────────────────────────
    def _window(self) -> np.ndarray:
        return np.stack(self.buffer).astype(np.float32)

    def _anomaly(self, values: np.ndarray) -> float:
        if self.last_channels is None:
            self.last_channels = values
            return 0.0
        diff = np.abs(values - self.last_channels)
        self.last_channels = values
        return float(np.nanmean(diff / (np.abs(self.last_channels) + 1.0)))

    def _rate(self, values: np.ndarray) -> float:
        if len(self.buffer) < 2:
            return 0.0
        previous = self.buffer[-2]
        return float(np.nanmean(np.abs(values - previous) / (np.abs(previous) + 1.0)))

    @torch.no_grad()
    def _infer(self) -> dict:
        window = self._window()
        raw = torch.from_numpy(np.nan_to_num(window[-1], nan=0.0)[None, :])
        physics = physics_feature_vector(raw, getattr(self.model, "turbine", None))
        prediction = self.model.predict(
            torch.from_numpy(np.nan_to_num(window)[None, ...]),
            physics,
            samples=self.mc_samples,
        )
        residual = self.model.expected_residuals(torch.from_numpy(np.nan_to_num(window)[None, ...]))
        residual_value = float(
            torch.stack([v.reshape(v.shape[0], -1).abs().mean() for v in residual.values()]).mean()
        )
        return {
            "mean": prediction["mean"][0].tolist(),
            "aleatoric": prediction["aleatoric"][0].tolist(),
            "epistemic": float(prediction["epistemic"][0].mean()),
            "total": prediction["total"][0].tolist(),
            "routing": prediction["routing"][0].tolist(),
            "residual": residual_value,
        }

    def status(self) -> dict:
        return {
            "window": len(self.buffer),
            "counter": self.counter,
            "state": STATE_STALE if self.last_prediction is None else STATE_OK,
            "drift": self.drift.report(),
            "telemetry_mode": self.policy._last_mode,
        }


__all__ = ["DriftMonitor", "EdgeRuntime", "RuntimeStep"]
