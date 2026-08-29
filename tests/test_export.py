import importlib.util

import pytest

from windfusion.models import create_model


@pytest.mark.skipif(importlib.util.find_spec("onnx") is None, reason="optional ONNX dependency")
def test_export(tmp_path):
    from windfusion.deployment.export import export_onnx

    assert export_onnx(create_model("windfusion-edge"), tmp_path / "m.onnx").exists()
