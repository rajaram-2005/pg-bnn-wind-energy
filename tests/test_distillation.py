"""Distillation: teacher detachment, loss terms and student training."""

import torch

from windfusion.models import create_model
from windfusion.training.distillation import DistillTrainer, distillation_loss


def test_teacher_gradients_are_detached():
    student = {
        "mean": torch.randn(2, 3, requires_grad=True),
        "log_var": torch.randn(2, 3, requires_grad=True),
        "representation": torch.randn(2, 8, requires_grad=True),
        "routing": torch.softmax(torch.randn(2, 4), -1),
    }
    teacher = {k: v.detach().clone().requires_grad_() for k, v in student.items()}
    loss = distillation_loss(student, teacher, torch.tensor(0.0))
    loss["total"].backward()
    assert student["mean"].grad is not None
    assert teacher["mean"].grad is None


def test_distillation_terms_are_all_finite():
    student = {
        "mean": torch.randn(4, 3),
        "log_var": torch.randn(4, 3),
        "representation": torch.randn(4, 8),
        "routing": torch.softmax(torch.randn(4, 4), -1),
    }
    teacher = {
        "mean": torch.randn(4, 3),
        "log_var": torch.randn(4, 3),
        "representation": torch.randn(4, 8),
        "routing": torch.softmax(torch.randn(4, 4), -1),
    }
    terms = distillation_loss(student, teacher, torch.tensor(0.1))
    for key in ("total", "prediction", "uncertainty", "feature", "router"):
        assert torch.isfinite(terms[key])


def test_projection_aligns_mismatched_representations():
    student = {
        "mean": torch.randn(3, 3),
        "log_var": torch.randn(3, 3),
        "representation": torch.randn(3, 16),
        "routing": torch.softmax(torch.randn(3, 4), -1),
    }
    teacher = {
        "mean": torch.randn(3, 3),
        "log_var": torch.randn(3, 3),
        "representation": torch.randn(3, 48),
        "routing": torch.softmax(torch.randn(3, 5), -1),
    }
    projection = torch.nn.Linear(16, 48)
    terms = distillation_loss(student, teacher, torch.tensor(0.0), projection=projection)
    assert torch.isfinite(terms["feature"])


def test_distill_trainer_runs_and_tracks_kd_terms(tiny_config):
    from windfusion.data.loaders import build_dataloaders

    config = tiny_config.with_overrides(**{"training.epochs": 1})
    bundle = build_dataloaders(config)
    teacher = create_model("windfusion-research", 12, 3, 5, config.turbine)
    student = create_model("windfusion-edge", 12, 3, 5, config.turbine)
    trainer = DistillTrainer(student, teacher, config, verbose=False)
    history = trainer.fit(bundle.train, bundle.val)
    assert "kd_total" in history.epochs[-1]
    assert trainer.projection is not None  # hidden sizes differ, so a projection is fitted
