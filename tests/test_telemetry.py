import numpy as np

from windfusion.edge.telemetry import AdaptiveTelemetryPolicy


def test_uncertainty_or_safety_preserves_resolution():
    p = AdaptiveTelemetryPolicy()
    low = p.decide(0.01, 0.01, 0.01, 0.01)
    high = p.decide(0.1, 0.9, 0.1, 0.1, safety=0.9)
    assert low.keep_every > high.keep_every
    assert p.encode(np.arange(100), high)["compression_ratio"] == 1
