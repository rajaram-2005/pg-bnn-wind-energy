"""Honest infrastructure benchmark.

Reports only what can be measured on the current machine: parameter count,
FP32/INT8 size, CPU latency and throughput. Predictive accuracy is
``NOT MEASURED`` here — it requires a trained checkpoint and a versioned
dataset (``windfusion train`` then ``windfusion evaluate``).
"""

from __future__ import annotations

import json
import time

import torch

from windfusion.evaluation.benchmark import NOT_MEASURED, measure_latency, quantized_size_mb
from windfusion.models import create_model


def run(mode: str = "windfusion-lite", repeats: int = 100, sequence_length: int = 24):
    """Measure one model's footprint. Kept for backwards compatibility with v0.1."""
    torch.manual_seed(7)
    model = create_model(mode).eval()
    x = torch.randn(1, sequence_length, model.input_features)
    p = torch.randn(1, model.physics_features)
    neighbors = torch.randn(1, max(model.spec.neighbors, 1), model.input_features) if model.spec.neighbors else None
    with torch.no_grad():
        for _ in range(10):
            model(x, p, neighbors)
        start = time.perf_counter()
        for _ in range(repeats):
            model(x, p, neighbors)
        elapsed = (time.perf_counter() - start) * 1000 / repeats
    params = model.parameter_count
    return {
        "mode": mode,
        "parameter_count": params,
        "model_size_mb_fp32": params * 4 / 1024**2,
        "model_size_mb_int8": quantized_size_mb(model),
        "cpu_latency_ms": elapsed,
        "accuracy": NOT_MEASURED,
        "f1": NOT_MEASURED,
        "rul_mae": NOT_MEASURED,
        "calibration": NOT_MEASURED,
        "ram_mb": NOT_MEASURED,
        "energy_per_inference": NOT_MEASURED,
        "note": "predictive metrics require a trained checkpoint and a versioned dataset",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
