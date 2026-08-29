"""Honest infrastructure benchmark; predictive metrics require supplied labels/checkpoints."""

import json
import time

import torch

from windfusion.models import create_model


def run(mode="windfusion-lite", repeats=100):
    torch.manual_seed(7)
    model = create_model(mode).eval()
    x = torch.randn(1, 24, 12)
    p = torch.randn(1, 4)
    with torch.no_grad():
        for _ in range(10):
            model(x, p)
        start = time.perf_counter()
        for _ in range(repeats):
            model(x, p)
    params = model.parameter_count
    return {
        "mode": mode,
        "parameter_count": params,
        "model_size_mb_fp32": params * 4 / 1024**2,
        "cpu_latency_ms": (time.perf_counter() - start) * 1000 / repeats,
        "accuracy": "NOT MEASURED",
        "f1": "NOT MEASURED",
        "rul_mae": "NOT MEASURED",
        "calibration": "NOT MEASURED",
        "ram_mb": "NOT MEASURED",
        "energy_per_inference": "NOT MEASURED",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
