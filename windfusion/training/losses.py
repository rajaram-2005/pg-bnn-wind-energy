"""Training objectives.

The total loss is

``L = NLL(heteroscedastic) + w_phys * Σ λᵢ biasᵢ rᵢ² + w_u * E[log_var²]
      + w_r * router_balance + w_m * monotone_RUL + w_f * forecast + w_c * self_check``

Provenance: ``heteroscedastic_nll``, ``soft_hinge_limits`` (wind-turbine-pg-bnn),
``iso281_l10`` (monotone RUL prior for the Odin preset).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..data.schema import CHANNELS
from ..physics.constraints import (
    RESEARCH_LIMITS,
    PhysicsWeights,
    limit_penalties,
    physics_loss,
)
from ..physics.residuals import compute_residuals


@dataclass
class LossBreakdown:
    """Every term of the objective, for logging and ablations."""

    total: torch.Tensor
    nll: torch.Tensor
    physics: torch.Tensor
    uncertainty: torch.Tensor
    router_balance: torch.Tensor
    monotone: torch.Tensor
    forecast: torch.Tensor
    consistency: torch.Tensor
    limits: torch.Tensor

    def as_floats(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach()),
            "nll": float(self.nll.detach()),
            "physics": float(self.physics.detach()),
            "uncertainty": float(self.uncertainty.detach()),
            "router_balance": float(self.router_balance.detach()),
            "monotone": float(self.monotone.detach()),
            "forecast": float(self.forecast.detach()),
            "consistency": float(self.consistency.detach()),
            "limits": float(self.limits.detach()),
        }


def heteroscedastic_nll(
    mean: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gaussian NLL with per-target weights (sum of weights is normalised)."""
    nll = 0.5 * (log_var + (target - mean).pow(2) * torch.exp(-log_var))
    if weights is None:
        return nll.mean()
    w = weights.to(nll.device).view(1, -1)
    return (nll * w).sum(-1).mean() / w.sum().clamp_min(1e-6)


def monotone_rul_penalty(
    rul: torch.Tensor, proxy: torch.Tensor, pairs: int = 64, margin: float = 0.0
) -> torch.Tensor:
    """Pairwise penalty: predicted RUL must not rise with the damage proxy.

    ``rul``   : (B,) predicted remaining useful life (any monotone unit)
    ``proxy`` : (B,) dimensionless physical damage proxy

    For random pairs (i, j) with ``proxy_i > proxy_j`` we penalise
    ``relu(rul_i - rul_j + margin)``. This is a soft ordering constraint, not a
    hard guarantee.
    """
    n = rul.shape[0]
    if n < 2:
        return rul.new_zeros(())
    pairs = max(1, min(pairs, n * (n - 1) // 2))
    idx_i = torch.randint(0, n, (pairs,), device=rul.device)
    idx_j = torch.randint(0, n, (pairs,), device=rul.device)
    proxy_i, proxy_j = proxy[idx_i], proxy[idx_j]
    rul_i, rul_j = rul[idx_i], rul[idx_j]
    direction = (proxy_i > proxy_j).to(rul.dtype)
    violation = torch.relu((rul_i - rul_j) * direction + margin * direction)
    return violation.mean()


def forecast_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE over the multi-horizon auxiliary forecast (Aeolus preset)."""
    if prediction.shape != target.shape:
        raise ValueError(f"forecast shape {tuple(prediction.shape)} != {tuple(target.shape)}")
    return torch.nn.functional.mse_loss(prediction, target)


def self_check_loss(
    predicted: torch.Tensor, residuals: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Train the physics self-check head to predict the residual magnitude."""
    observed = torch.stack(
        [v.reshape(v.shape[0], -1).abs().mean(-1) for v in residuals.values()], dim=-1
    ).mean(-1, keepdim=True)
    return torch.nn.functional.mse_loss(predicted, observed.detach())


def _bias_adjusted_weights(config, spec) -> PhysicsWeights:
    base = PhysicsWeights(**config.physics.weights)
    bias = getattr(spec, "physics_bias", None) if spec is not None else None
    if not bias:
        return base
    return PhysicsWeights(
        **{name: getattr(base, name) * float(bias.get(name, 1.0)) for name in base.as_dict()}
    )


def _raw_limits(raw: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        name: raw[..., i]
        for i, name in enumerate(CHANNELS)
        if name in RESEARCH_LIMITS and i < raw.shape[-1]
    }


def windfusion_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config,
    physics_weights: PhysicsWeights | None = None,
    spec=None,
) -> LossBreakdown:
    """Full objective for one batch."""
    target = batch["target"]
    mean, log_var = outputs["mean"], outputs["log_var"]
    zero = mean.new_zeros(())

    weight_values = [config.training.target_weights.get(name, 1.0) for name in ("failure_probability", "health_index", "rul_days")]
    target_weights = mean.new_tensor(weight_values[: mean.shape[-1]])
    nll = heteroscedastic_nll(mean, log_var, target, target_weights)

    residuals = compute_residuals(batch["raw"], config.turbine)
    weights = physics_weights or _bias_adjusted_weights(config, spec)
    phys, _ = physics_loss(residuals, weights)

    uncertainty = config.training.uncertainty_weight * log_var.pow(2).mean()
    balance = config.training.router_balance_weight * (
        outputs["routing"].mean(0).clamp_min(1e-8)
        * outputs["routing"].mean(0).clamp_min(1e-8).log()
    ).sum()

    monotone = zero
    if getattr(spec, "monotone_rul", False):
        rul = mean[..., 2]
        monotone = config.physics.monotone_rul_weight * monotone_rul_penalty(
            rul, batch["damage_proxy"].squeeze(-1)
        )

    forecast = zero
    if "forecast" in outputs and "future" in batch:
        forecast = 0.1 * forecast_loss(outputs["forecast"], batch["future"])

    consistency = zero
    if "consistency" in outputs:
        consistency = 0.1 * self_check_loss(outputs["consistency"], residuals)

    limits = zero
    if getattr(config.physics, "limit_penalty", 0.0):
        penalties = limit_penalties(_raw_limits(batch["raw"]))
        if penalties:
            limits = config.physics.limit_penalty * sum(p.mean() for p in penalties.values())

    total = nll + phys + uncertainty + balance + monotone + forecast + consistency + limits
    return LossBreakdown(
        total=total,
        nll=nll,
        physics=phys,
        uncertainty=uncertainty,
        router_balance=balance,
        monotone=monotone,
        forecast=forecast,
        consistency=consistency,
        limits=limits,
    )


__all__ = [
    "LossBreakdown",
    "forecast_loss",
    "heteroscedastic_nll",
    "monotone_rul_penalty",
    "self_check_loss",
    "windfusion_loss",
]
