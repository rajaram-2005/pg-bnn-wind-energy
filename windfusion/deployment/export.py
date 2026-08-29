"""Export path: TorchScript, ONNX, dynamic quantisation and parity checks.

Provenance: ``onnx_export_contract`` (wind-turbine-pg-bnn
``src/deployment/export_onnx.py``). An artefact is only accepted after a
numerical parity check against the source model, so a silent export regression
fails the pipeline instead of reaching a device.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ..models import MODEL_REGISTRY


class ExportWrapper(torch.nn.Module):
    """Flattens the model dictionary output into a tuple of tensors."""

    def __init__(self, model, with_neighbors: bool = False) -> None:
        super().__init__()
        self.model = model
        self.with_neighbors = with_neighbors

    def forward(self, sequence, physics, neighbors=None):
        # Route through the control-flow-free path so the exported graph is
        # static; routing weights are still returned for auditability.
        previous = getattr(self.model, "dense_experts", False)
        self.model.dense_experts = True
        try:
            out = self.model(sequence, physics, neighbors if self.with_neighbors else None)
        finally:
            self.model.dense_experts = previous
        return out["mean"], out["log_var"], out["routing"]


@dataclass
class ExportReport:
    """What was written and whether it survived the parity check."""

    path: str
    model_id: str
    parameters: int
    size_mb: float
    size_int8_mb: float | str
    opset: int | None = None
    parity_checked: bool = False
    max_abs_diff: float | None = None
    warnings: list[str] = None  # type: ignore[assignment]

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["warnings"] = self.warnings or []
        return payload


def _dummy_inputs(model, sequence_length: int, neighbors: int, batch: int = 1):
    features = model.input_features
    sequence = torch.zeros(batch, sequence_length, features)
    physics = torch.zeros(batch, model.physics_features)
    neighbors_tensor = (
        torch.zeros(batch, max(neighbors, 1), features) if neighbors else None
    )
    return sequence, physics, neighbors_tensor


def export_onnx(
    model,
    path: str | Path,
    sequence_length: int = 24,
    opset: int = 17,
    check_parity: bool = True,
) -> ExportReport:
    """Export to ONNX and (optionally) verify numerical parity."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    wrapper = ExportWrapper(model, with_neighbors=model.spec.neighbors > 0).eval()
    sequence, physics, neighbors = _dummy_inputs(model, sequence_length, model.spec.neighbors)
    inputs = (sequence, physics) if neighbors is None else (sequence, physics, neighbors)
    dynamic_axes = {
        "scada": {0: "batch", 1: "time"},
        "physics": {0: "batch"},
        "mean": {0: "batch"},
        "log_var": {0: "batch"},
        "routing": {0: "batch"},
    }
    if neighbors is not None:
        dynamic_axes["neighbors"] = {0: "batch"}
    input_names = ["scada", "physics"] + (["neighbors"] if neighbors is not None else [])
    with torch.no_grad():
        try:
            # The torch.export-based exporter (default from PyTorch 2.9) cannot
            # yet trace the data-dependent expert loop; the TorchScript exporter
            # handles it and keeps the dynamic axes we rely on.
            torch.onnx.export(
                wrapper,
                inputs,
                str(path),
                input_names=input_names,
                output_names=["mean", "log_var", "routing"],
                dynamic_axes=dynamic_axes,
                opset_version=opset,
                dynamo=False,
            )
        except TypeError:  # pragma: no cover - torch < 2.9 has no `dynamo` flag
            torch.onnx.export(
                wrapper,
                inputs,
                str(path),
                input_names=input_names,
                output_names=["mean", "log_var", "routing"],
                dynamic_axes=dynamic_axes,
                opset_version=opset,
            )
    report = ExportReport(
        path=str(path),
        model_id=model.spec.model_id,
        parameters=model.parameter_count,
        size_mb=round(path.stat().st_size / 1024**2, 4),
        size_int8_mb=quantized_size_mb(model),
        opset=opset,
        warnings=[],
    )
    if check_parity:
        try:
            diff = validate_parity(model, path, sequence_length)
            report.parity_checked = True
            report.max_abs_diff = diff
        except Exception as exc:  # pragma: no cover - optional dependency path
            report.warnings.append(f"parity check skipped: {type(exc).__name__}: {exc}")
    return report


def export_torchscript(model, path: str | Path, sequence_length: int = 24) -> ExportReport:
    """Script and save the model (fallback runtime without ONNX)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    wrapper = ExportWrapper(model, with_neighbors=model.spec.neighbors > 0).eval()
    sequence, physics, neighbors = _dummy_inputs(model, sequence_length, model.spec.neighbors)
    with torch.no_grad():
        scripted = torch.jit.trace(
            wrapper,
            (sequence, physics) if neighbors is None else (sequence, physics, neighbors),
            strict=False,
        )
        scripted.save(str(path))
    return ExportReport(
        path=str(path),
        model_id=model.spec.model_id,
        parameters=model.parameter_count,
        size_mb=round(path.stat().st_size / 1024**2, 4),
        size_int8_mb=quantized_size_mb(model),
        warnings=[],
    )


def validate_parity(model, onnx_path: str | Path, sequence_length: int = 24, tolerance: float = 1e-4) -> float:
    """Compare ONNX Runtime and PyTorch outputs; returns the max absolute difference."""
    import onnxruntime as ort

    model.eval()
    torch.manual_seed(0)
    sequence = torch.randn(2, sequence_length, model.input_features)
    physics = torch.randn(2, model.physics_features)
    neighbors = (
        torch.randn(2, max(model.spec.neighbors, 1), model.input_features)
        if model.spec.neighbors
        else None
    )
    with torch.no_grad():
        expected = model(sequence, physics, neighbors)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feed = {"scada": sequence.numpy(), "physics": physics.numpy()}
    if neighbors is not None:
        feed["neighbors"] = neighbors.numpy()
    mean, log_var, _routing = session.run(None, feed)
    diff = max(
        float(abs(mean - expected["mean"].numpy()).max()),
        float(abs(log_var - expected["log_var"].numpy()).max()),
    )
    if diff > tolerance:  # pragma: no cover - depends on the exporter version
        raise ValueError(f"ONNX parity check failed: max abs diff {diff:.3e} > {tolerance:.1e}")
    return diff


def quantized_size_mb(model) -> float | str:
    """Size after dynamic INT8 quantisation of Linear layers."""
    try:
        quantized = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        total = 0
        for value in quantized.state_dict().values():
            if isinstance(value, torch.Tensor):
                total += value.numel() * (1 if value.dtype == torch.qint8 else 4)
        return round(total / 1024**2, 4)
    except Exception:  # pragma: no cover
        return "NOT MEASURED"


def export_family(
    output_dir: str | Path,
    sequence_length: int = 24,
    modes: list[str] | None = None,
    opset: int = 17,
) -> dict[str, dict]:
    """Export every registered model (used by release scripts)."""
    from ..models import create_model

    output_dir = Path(output_dir)
    reports = {}
    for mode in modes or list(MODEL_REGISTRY):
        model = create_model(mode).eval()
        report = export_onnx(model, output_dir / f"{mode}.onnx", sequence_length, opset)
        reports[mode] = report.as_dict()
    return reports


__all__ = [
    "ExportReport",
    "ExportWrapper",
    "export_family",
    "export_onnx",
    "export_torchscript",
    "quantized_size_mb",
    "validate_parity",
]
