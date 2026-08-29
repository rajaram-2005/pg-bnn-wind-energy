"""Teacher-to-student distillation.

The teacher (``windfusion-research`` by default) is trained first; the student
(``windfusion-edge``) then learns from **detached** teacher signals combined
with the supervised objective. Original systems stay external and unchanged —
only their *outputs*, never their weights, cross the boundary.

Signals
-------
``prediction``  MSE on the predictive mean (response distillation)
``uncertainty`` MSE on log-variance, so the student inherits calibration
``feature``     cosine alignment of representations after a learned projection
``physics``     teacher's physics-residual magnitude (consistency transfer)
``router``      symmetric KL between routing distributions (which expert mattered)

Provenance: knowledge distillation is prior art; the router-agreement term and
the physics-consistency term are additions made in this repository.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..config import DistillationConfig, WindFusionConfig
from .loop import Trainer
from .losses import windfusion_loss


@dataclass(frozen=True)
class DistillationWeights:
    prediction: float = 0.50
    uncertainty: float = 0.20
    feature: float = 0.15
    physics: float = 0.15


def distillation_loss(
    student: dict[str, torch.Tensor],
    teacher: dict[str, torch.Tensor],
    physics_loss_value: torch.Tensor,
    weights: DistillationWeights | None = None,
    temperature: float = 1.0,
    projection: torch.nn.Module | None = None,
    router_weight: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Distillation terms. Teacher tensors are detached inside this function."""
    weights = weights or DistillationWeights()
    t_mean = teacher["mean"].detach()
    t_log_var = teacher["log_var"].detach()
    t_repr = teacher["representation"].detach()

    pred = F.mse_loss(student["mean"] / temperature, t_mean / temperature)
    unc = F.mse_loss(student["log_var"], t_log_var)

    s_repr = student["representation"]
    if projection is not None:
        s_repr = projection(s_repr)
    if s_repr.shape == t_repr.shape:
        target = torch.ones(s_repr.shape[0], device=s_repr.device)
        feature = F.cosine_embedding_loss(s_repr, t_repr, target)
    else:  # pragma: no cover - guarded by the projection
        feature = pred.new_zeros(())

    router = pred.new_zeros(())
    if (
        "routing" in student
        and "routing" in teacher
        and router_weight > 0
        and student["routing"].shape == teacher["routing"].shape
    ):
        s_w = student["routing"].clamp_min(1e-6)
        t_w = teacher["routing"].detach().clamp_min(1e-6)
        router = 0.5 * ((t_w * (t_w.log() - s_w.log())).sum(-1) + (s_w * (s_w.log() - t_w.log())).sum(-1)).mean()

    total = (
        weights.prediction * pred
        + weights.uncertainty * unc
        + weights.feature * feature
        + weights.physics * physics_loss_value
        + router_weight * router
    )
    return {"total": total, "prediction": pred, "uncertainty": unc, "feature": feature, "router": router}


class DistillTrainer(Trainer):
    """Supervised + distillation training for a student model."""

    def __init__(
        self,
        model: torch.nn.Module,
        teacher: torch.nn.Module,
        config: WindFusionConfig | None = None,
        distill_config: DistillationConfig | None = None,
        device: str = "cpu",
        verbose: bool = True,
    ) -> None:
        super().__init__(model, config, device, verbose)
        self.teacher = teacher.to(device).eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)
        self.distill_config = distill_config or (config.distillation if config else DistillationConfig())
        self.weights = DistillationWeights(
            **{k: v for k, v in self.distill_config.weights.items() if k in ("prediction", "uncertainty", "feature", "physics")}
        )
        student_dim = model.spec.hidden
        teacher_dim = getattr(teacher.spec, "hidden", student_dim)
        self.projection = (
            torch.nn.Linear(student_dim, teacher_dim).to(device)
            if student_dim != teacher_dim
            else None
        )
        if self.projection is not None:
            self.optimizer.add_param_group({"params": list(self.projection.parameters())})
            # The scheduler captured the old parameter groups; rebuild it so
            # every group (including the projection) gets a learning rate.
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self._lr_lambda)

    @torch.no_grad()
    def teacher_outputs(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Cached-free teacher forward pass (always in eval mode)."""
        was_training = self.teacher.training
        self.teacher.eval()
        out = self.teacher(batch["sequence"], batch["physics"], batch.get("neighbors"))
        self.teacher.train(was_training)
        return out

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        from .loop import _to_device

        batch = _to_device(batch, self.device)
        self.model.train()
        outputs = self.model(batch["sequence"], batch["physics"], batch.get("neighbors"))
        supervised = windfusion_loss(outputs, batch, self.config, spec=self.spec)
        teacher = self.teacher_outputs(batch)
        teacher_physics = windfusion_loss(
            teacher, batch, self.config, spec=getattr(self.teacher, "spec", None)
        ).physics.detach()
        kd = distillation_loss(
            outputs,
            teacher,
            teacher_physics,
            self.weights,
            temperature=self.distill_config.temperature,
            projection=self.projection,
            router_weight=self.distill_config.router_agreement_weight,
        )
        total = supervised.total + kd["total"]
        terms = supervised.as_floats()
        terms.update({"kd_" + k: float(v.detach()) for k, v in kd.items()})
        terms["total"] = float(total.detach())
        return total, terms


def teacher_predictions(teacher, loader, device: str = "cpu", limit: int | None = None) -> list[dict]:
    """Materialise teacher outputs for a dataset (offline distillation)."""
    teacher = teacher.to(device).eval()
    collected = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if limit is not None and i >= limit:
                break
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            collected.append(teacher(batch["sequence"], batch["physics"], batch.get("neighbors")))
    return collected


__all__ = [
    "DistillTrainer",
    "DistillationWeights",
    "distillation_loss",
    "teacher_predictions",
]
