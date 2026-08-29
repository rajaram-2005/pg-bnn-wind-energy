import torch

from windfusion.models.router import SparseExpertRouter


def test_sparse_normalized_router():
    r = SparseExpertRouter(8, 4, 2)
    w = r(torch.randn(5, 8))
    assert torch.allclose(w.sum(1), torch.ones(5))
    assert (w > 0).sum(1).max() <= 2
