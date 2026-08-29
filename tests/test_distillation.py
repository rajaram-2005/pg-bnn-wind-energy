import torch

from windfusion.training.distillation import distillation_loss


def test_teacher_is_detached():
    s = {
        "mean": torch.randn(2, 1, requires_grad=True),
        "log_var": torch.randn(2, 1, requires_grad=True),
        "representation": torch.randn(2, 3, requires_grad=True),
    }
    t = {k: v.detach().clone().requires_grad_() for k, v in s.items()}
    loss = distillation_loss(s, t, torch.tensor(0.0))
    loss.backward()
    assert s["mean"].grad is not None and t["mean"].grad is None
