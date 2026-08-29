"""Teacher-agnostic distillation; original systems stay external/read-only."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DistillationWeights:
    prediction: float = 0.5
    feature: float = 0.15
    uncertainty: float = 0.2
    physics: float = 0.15


def distillation_loss(
    student: dict[str, torch.Tensor],
    teacher: dict[str, torch.Tensor],
    physics_loss: torch.Tensor,
    weights: DistillationWeights | None = None,
) -> torch.Tensor:
    weights = weights or DistillationWeights()
    pred = F.mse_loss(student["mean"], teacher["mean"].detach())
    unc = F.mse_loss(student["log_var"], teacher["log_var"].detach())
    feature = (
        F.cosine_embedding_loss(
            student["representation"],
            teacher["representation"].detach(),
            torch.ones(student["representation"].shape[0], device=pred.device),
        )
        if student["representation"].shape == teacher["representation"].shape
        else pred.new_zeros(())
    )
    return (
        weights.prediction * pred
        + weights.feature * feature
        + weights.uncertainty * unc
        + weights.physics * physics_loss
    )
