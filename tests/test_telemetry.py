"""Adaptive telemetry: policy, codec and fidelity (provenance: AeroZip concepts)."""

import numpy as np

from windfusion.edge.telemetry import (
    AdaptiveTelemetryPolicy,
    evaluate_policy,
    reconstruction_error,
)


def test_low_risk_compresses_and_high_risk_preserves():
    policy = AdaptiveTelemetryPolicy()
    low = policy.decide(0.01, 0.01, 0.01, 0.01)
    high = policy.decide(0.1, 0.9, 0.1, 0.1, safety=0.9)
    assert low.mode == "compressed" and low.keep_every > 1
    assert high.mode == "raw" and high.keep_every == 1
    assert high.risk > low.risk


def test_safety_override_dominates():
    policy = AdaptiveTelemetryPolicy()
    decision = policy.decide(0.0, 0.0, 0.0, 0.0, safety=1.0)
    assert decision.risk == 1.0
    assert decision.mode == "raw"


def test_cooldown_prevents_mode_flapping():
    policy = AdaptiveTelemetryPolicy(cooldown_steps=3)
    policy.decide(0.9, 0.9, 0.9, 0.9)
    for _ in range(3):
        decision = policy.decide(0.0, 0.0, 0.0, 0.0)
        assert decision.mode == "raw" and "cooldown" in decision.reason
    assert policy.decide(0.0, 0.0, 0.0, 0.0).mode == "compressed"


def test_hysteresis_keeps_a_previously_escalated_window_escalated():
    policy = AdaptiveTelemetryPolicy(hysteresis=0.2)
    # borderline risk: from compressed it escalates one step, from raw it stays escalated
    fresh = AdaptiveTelemetryPolicy(hysteresis=0.2)
    signals = {"anomaly": 0.5, "epistemic": 0.5, "rate": 0.5, "physics_residual": 0.5}
    assert fresh.decide(**signals).mode == "detailed"
    escalated = AdaptiveTelemetryPolicy(hysteresis=0.2)
    escalated._last_mode = "raw"
    escalated._cooldown = 0
    assert escalated.decide(**signals).mode == "raw"


def test_raw_payload_is_lossless():
    policy = AdaptiveTelemetryPolicy()
    values = np.cumsum(np.random.default_rng(0).normal(0, 0.5, 64)).astype(np.float32)
    decision = policy.decide(0.9, 0.9, 0.9, 0.9)
    encoded = policy.encode(values, decision)
    decoded = policy.decode(encoded)
    assert encoded["compression_ratio"] == 1.0
    assert float(np.abs(decoded - values).max()) < 1e-6


def test_compressed_payload_reduces_bandwidth():
    policy = AdaptiveTelemetryPolicy()
    values = np.linspace(20, 25, 200).astype(np.float32)
    decision = policy.decide(0.0, 0.0, 0.0, 0.0)
    encoded = policy.encode(values, decision)
    assert encoded["mode"] == "compressed"
    assert encoded["compression_ratio"] < 0.2


def test_reconstruction_error_reports_fidelity():
    original = np.linspace(0, 1, 50).astype(np.float32)
    error = reconstruction_error(original, original + 0.1)
    assert abs(error["mae"] - 0.1) < 1e-5
    assert error["max_abs"] >= error["mae"]


def test_evaluate_policy_balances_savings_and_fidelity():
    rng = np.random.default_rng(0)
    policy = AdaptiveTelemetryPolicy()
    windows = [np.cumsum(rng.normal(0, 0.1, 64)) for _ in range(20)]
    signals = [{"anomaly": 0.05, "epistemic": 0.1, "rate": 0.05, "physics_residual": 0.05}] * 18
    signals += [{"anomaly": 0.9, "epistemic": 0.9, "rate": 0.9, "physics_residual": 0.9}] * 2
    report = evaluate_policy(policy, windows, signals)
    assert report["windows"] == 20
    assert report["bandwidth_reduction"] > 0.5
    assert report["modes"]["raw"] >= 2


def test_codec_preserves_the_operating_point():
    """The DC baseline must survive delta coding, not be integrated from zero."""
    policy = AdaptiveTelemetryPolicy()
    values = np.linspace(20, 25, 200).astype(np.float32)
    decision = policy.decide(0.0, 0.0, 0.0, 0.0)
    encoded = policy.encode(values, decision)
    decoded = policy.decode(encoded)
    assert decoded.shape[0] == len(values[:: decision.keep_every])
    # first reconstructed sample is the true first sample
    assert abs(float(decoded[0]) - float(values[0])) < 0.02
    error = reconstruction_error(values, decoded, keep_every=decision.keep_every)
    assert error["mae"] < 0.05, error
    assert error["max_abs"] < 0.5, error


def test_codec_handles_multichannel_windows():
    policy = AdaptiveTelemetryPolicy()
    window = np.random.default_rng(3).normal(0, 1, (240, 12)).astype(np.float32) * 10 + 50
    decision = policy.decide(0.0, 0.0, 0.0, 0.0)
    encoded = policy.encode(window, decision)
    decoded = policy.decode(encoded)
    assert decoded.shape[1] == 12
    error = reconstruction_error(window, decoded, keep_every=decision.keep_every)
    assert error["mae"] < 0.05, error


def test_reconstruction_error_compares_on_the_sampling_grid():
    original = np.arange(100, dtype=np.float32)
    decoded = np.arange(0, 100, 10, dtype=np.float32)
    # misaligned comparison would report a large error
    assert reconstruction_error(original, decoded, keep_every=10)["mae"] == 0.0
    assert reconstruction_error(original, decoded, keep_every=1)["mae"] > 0.0
