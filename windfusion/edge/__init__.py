"""Edge deployment: adaptive telemetry and the streaming runtime."""

from .runtime import DriftMonitor, EdgeRuntime, RuntimeStep
from .telemetry import (
    AdaptiveTelemetryPolicy,
    TelemetryDecision,
    evaluate_policy,
    reconstruction_error,
)

__all__ = [
    "AdaptiveTelemetryPolicy",
    "DriftMonitor",
    "EdgeRuntime",
    "RuntimeStep",
    "TelemetryDecision",
    "evaluate_policy",
    "reconstruction_error",
]
