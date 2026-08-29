"""Export contract: ONNX parity, TorchScript and quantised size."""

import importlib.util
from pathlib import Path

import pytest

from windfusion.deployment.export import export_onnx, export_torchscript, quantized_size_mb
from windfusion.models import create_model

ORT_AVAILABLE = importlib.util.find_spec("onnxruntime") is not None
ONNX_AVAILABLE = importlib.util.find_spec("onnx") is not None


def test_torchscript_export_and_reload():
    model = create_model("windfusion-edge", 12, 3, 5)
    report = export_torchscript(model, "/tmp/windfusion_test_ts.pt", sequence_length=12)
    assert report.size_mb > 0
    assert report.model_id == "windfusion-edge"


def test_quantized_size_is_smaller_than_fp32():
    model = create_model("windfusion-lite", 12, 3, 5)
    quantized = quantized_size_mb(model)
    fp32 = model.parameter_count * 4 / 1024**2
    assert isinstance(quantized, float) and quantized < fp32


@pytest.mark.skipif(not ONNX_AVAILABLE, reason="optional ONNX dependency")
def test_onnx_export_writes_a_file(tmp_path):
    model = create_model("windfusion-edge", 12, 3, 5)
    report = export_onnx(model, tmp_path / "m.onnx", sequence_length=12, check_parity=ORT_AVAILABLE)
    assert Path(report.path).exists() and Path(report.path).stat().st_size > 0


@pytest.mark.skipif(not (ONNX_AVAILABLE and ORT_AVAILABLE), reason="optional ONNX Runtime dependency")
def test_onnx_runtime_parity(tmp_path):
    model = create_model("windfusion-lite", 12, 3, 5)
    report = export_onnx(model, tmp_path / "m.onnx", sequence_length=12, check_parity=True)
    assert report.parity_checked
    assert report.max_abs_diff < 1e-3


@pytest.mark.skipif(not (ONNX_AVAILABLE and ORT_AVAILABLE), reason="optional ONNX Runtime dependency")
def test_dense_export_path_matches_sparse_runtime(tmp_path):
    """The export graph must be numerically identical to the sparse runtime path."""
    import torch

    model = create_model("windfusion-lite", 12, 3, 5).eval()
    x, p = torch.randn(4, 24, 12), torch.randn(4, 5)
    with torch.no_grad():
        sparse = model(x, p)["mean"]
        model.dense_experts = True
        dense = model(x, p)["mean"]
        model.dense_experts = False
    assert torch.allclose(sparse, dense, atol=1e-5)


@pytest.mark.skipif(not (ONNX_AVAILABLE and ORT_AVAILABLE), reason="optional ONNX Runtime dependency")
def test_onnx_parity_for_a_fleet_aware_model(tmp_path):
    model = create_model("qinglong-wind", 12, 3, 5)
    report = export_onnx(model, tmp_path / "q.onnx", sequence_length=12, check_parity=True)
    assert report.parity_checked
