import torch

from windfusion.physics.constraints import PhysicsWeights, physics_loss


def windfusion_loss(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    residuals: dict[str, torch.Tensor],
    physics_weights: PhysicsWeights | None = None,
    uncertainty_weight: float = 0.01,
) -> dict[str, torch.Tensor]:
    physics_weights = physics_weights or PhysicsWeights()
    nll = (
        0.5
        * (
            output["log_var"] + (target - output["mean"]).pow(2) * torch.exp(-output["log_var"])
        ).mean()
    )
    p_loss, _ = physics_loss(residuals, physics_weights)
    balance = (output["routing"].mean(0) * output["routing"].mean(0).clamp_min(1e-8).log()).sum()
    total = nll + p_loss + uncertainty_weight * output["log_var"].pow(2).mean() + 0.01 * balance
    return {"total": total, "prediction": nll, "physics": p_loss, "router_balance": balance}
