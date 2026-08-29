import torch

from windfusion.models import create_model


def test_modes_and_shapes():
    for mode in (
        "windfusion-lite",
        "windfusion-edge",
        "windfusion-research",
        "aetheris-wind",
        "ra-wind",
        "qinglong-wind",
        "vayu-wind",
        "odin-wind",
        "aeolus-wind",
    ):
        m = create_model(mode, 12, 3)
        o = m(torch.randn(2, 8, 12), torch.randn(2, 4))
        assert o["mean"].shape == (2, 3)
        assert m.parameter_count > 0
