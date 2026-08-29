"""Infrastructure benchmark: parameters, size, latency and memory.

Only quantities that can actually be measured on the current machine are
reported. Anything that needs platform tooling (energy, device RAM on the
target hardware) is returned as the string ``NOT MEASURED`` rather than a
guessed number.
"""

from __future__ import annotations

import platform
import statistics
import time
from dataclasses import asdict, dataclass

import torch

from ..config import WindFusionConfig
from ..models import MODEL_REGISTRY, create_model

NOT_MEASURED = "NOT MEASURED"


@dataclass
class Footprint:
    """Measured cost of one model on the current machine."""

    model_id: str
    parameters: int
    size_fp32_mb: float
    size_int8_mb: float | str
    latency_ms_median: float
    latency_ms_p95: float
    throughput_windows_per_s: float
    peak_rss_mb: float | str
    energy_per_inference: str = NOT_MEASURED
    device: str = "cpu"

    def as_dict(self) -> dict:
        return asdict(self)


def _peak_rss_mb() -> float | str:
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kilobytes, macOS reports bytes.
        return round(usage / 1024.0, 2) if platform.system() != "Darwin" else round(usage / 1024**2, 2)
    except Exception:  # pragma: no cover
        return NOT_MEASURED


def quantized_size_mb(model) -> float | str:
    """Dynamically quantised size (Linear layers only), or NOT MEASURED."""
    try:
        quantized = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        total = 0
        for param in quantized.state_dict().values():
            if isinstance(param, torch.Tensor):
                total += param.numel() * (1 if param.dtype == torch.qint8 else 4)
        return round(total / 1024**2, 4)
    except Exception:  # pragma: no cover - quantization is platform dependent
        return NOT_MEASURED


def measure_latency(
    model,
    sequence_length: int,
    input_features: int,
    physics_features: int,
    repeats: int = 100,
    warmup: int = 10,
    batch_size: int = 1,
    neighbors: int = 0,
) -> tuple[float, float]:
    """Return (median, p95) single-batch latency in milliseconds."""
    model.eval()
    x = torch.randn(batch_size, sequence_length, input_features)
    p = torch.randn(batch_size, physics_features)
    nb = torch.randn(batch_size, max(neighbors, 1), input_features) if neighbors else None
    with torch.no_grad():
        for _ in range(warmup):
            model(x, p, nb)
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(x, p, nb)
            samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    median = statistics.median(samples)
    p95 = samples[min(int(0.95 * len(samples)), len(samples) - 1)]
    return round(median, 4), round(p95, 4)


def measure_model(
    model_id: str,
    config: WindFusionConfig | None = None,
    repeats: int = 100,
) -> Footprint:
    """Measure one model from the registry."""
    config = config or WindFusionConfig()
    model = create_model(
        model_id,
        config.model.input_features,
        config.model.outputs,
        config.model.physics_features,
        config.turbine,
    ).eval()
    params = model.parameter_count
    median, p95 = measure_latency(
        model,
        config.model.sequence_length,
        config.model.input_features,
        config.model.physics_features,
        repeats=repeats,
        neighbors=model.spec.neighbors,
    )
    return Footprint(
        model_id=model_id,
        parameters=params,
        size_fp32_mb=round(params * 4 / 1024**2, 4),
        size_int8_mb=quantized_size_mb(model),
        latency_ms_median=median,
        latency_ms_p95=p95,
        throughput_windows_per_s=round(1000.0 / max(median, 1e-6), 2),
        peak_rss_mb=_peak_rss_mb(),
    )


def benchmark_family(
    config: WindFusionConfig | None = None,
    modes: list[str] | None = None,
    repeats: int = 50,
) -> dict:
    """Measure every registered model (or a subset) and return the table."""
    config = config or WindFusionConfig()
    modes = modes or list(MODEL_REGISTRY)
    results = {}
    for mode in modes:
        results[mode] = measure_model(mode, config, repeats=repeats).as_dict()
    return {
        "results": results,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "threads": torch.get_num_threads(),
            "cpu_count": __import__("os").cpu_count(),
        },
        "not_measured": ["energy_per_inference", "on-device RAM (target hardware)", "field accuracy"],
    }


__all__ = [
    "Footprint",
    "NOT_MEASURED",
    "benchmark_family",
    "measure_latency",
    "measure_model",
    "quantized_size_mb",
]
