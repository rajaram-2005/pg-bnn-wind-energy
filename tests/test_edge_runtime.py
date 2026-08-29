"""Edge runtime: buffering, cadence, drift monitoring and degradation."""

import numpy as np

from windfusion.config import TelemetryConfig, WindFusionConfig
from windfusion.edge.runtime import EdgeRuntime
from windfusion.models import create_model


def _runtime(config, window=8, infer_every=1):
    model = create_model("windfusion-edge", 12, 3, 5, config.turbine)
    return EdgeRuntime(model, config, TelemetryConfig(), window=window, infer_every=infer_every, mc_samples=2)


def test_runtime_warms_up_before_predicting(tiny_config):
    runtime = _runtime(tiny_config, window=8)
    for i in range(7):
        step = runtime.push(np.random.default_rng(i).normal(0, 1, 12))
        assert step.state == "warmup"
    step = runtime.push(np.zeros(12, dtype=np.float32))
    assert step.state == "ok"
    assert step.prediction is not None
    assert "routing" in step.prediction


def test_inference_cadence_skips_windows(tiny_config):
    runtime = _runtime(tiny_config, window=8, infer_every=3)
    states = [runtime.push(np.zeros(12, dtype=np.float32)).state for _ in range(10)]
    assert states.count("ok") < 10


def test_missing_samples_are_tolerated(tiny_config):
    runtime = _runtime(tiny_config, window=8)
    sample = np.full(12, np.nan, dtype=np.float32)
    sample[0] = 8.0
    for _ in range(8):
        step = runtime.push(sample)
    assert step.state == "ok"
    assert all(np.isfinite(v) for v in step.prediction["mean"])


def test_drift_monitor_flags_a_shift(tiny_config):
    runtime = _runtime(tiny_config, window=8)
    rng = np.random.default_rng(0)
    for _ in range(24):
        runtime.push(rng.normal(0, 1, 12))
    assert runtime.drift.reference is None
    runtime.drift.fit(np.random.default_rng(1).normal(0, 1, 12)[None, :])
    for _ in range(8):
        runtime.push(rng.normal(20, 1, 12))  # large covariate shift
    report = runtime.drift.report()
    assert report["n"] > 0
    assert report["psi"] > 0


def test_status_reports_state(tiny_config):
    runtime = _runtime(tiny_config, window=8)
    for _ in range(9):
        runtime.push(np.zeros(12, dtype=np.float32))
    status = runtime.status()
    assert status["window"] == 8
    assert status["state"] == "ok"
    assert "drift" in status


def test_runtime_accepts_a_channel_dictionary(tiny_config):
    runtime = _runtime(tiny_config, window=8)
    payload = {"wind_speed": 9.0, "rotor_speed": 1.2, "vibration_rms": 2.0, "ambient_temp_c": 15.0}
    for _ in range(8):
        step = runtime.push(payload)
    assert step.state == "ok"
