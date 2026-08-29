import torch

from windfusion.models import create_model


def test_uncertainty_is_explicit_and_finite():
    p = create_model(input_features=4).predict(torch.randn(2, 6, 4), torch.randn(2, 5), samples=3)
    assert {"aleatoric", "epistemic", "total"} <= p.keys()
    assert all(torch.isfinite(p[k]).all() for k in ("aleatoric", "epistemic", "total"))
