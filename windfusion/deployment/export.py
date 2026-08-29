from pathlib import Path

import torch


class ExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, sequence, physics):
        out = self.model(sequence, physics)
        return out["mean"], out["log_var"], out["routing"]


def export_onnx(model, path: str | Path, sequence_length: int = 24) -> Path:
    path = Path(path)
    model.eval()
    wrapper = ExportWrapper(model)
    torch.onnx.export(
        wrapper,
        (torch.zeros(1, sequence_length, model.input_features), torch.zeros(1, 4)),
        path,
        input_names=["scada", "physics"],
        output_names=["mean", "log_var", "routing"],
        dynamic_axes={"scada": {0: "batch", 1: "time"}},
        opset_version=17,
    )
    return path
