import torch

from windfusion.models import create_model


def test_nan_sensor_is_safely_imputed():
    x = torch.randn(2, 5, 4)
    x[0, 2, 1] = float("nan")
    o = create_model(input_features=4)(x, torch.zeros(2, 4))
    assert torch.isfinite(o["mean"]).all()
