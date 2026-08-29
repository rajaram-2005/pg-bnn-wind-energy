"""Heteroscedastic head and MC-dropout uncertainty decomposition.

Provenance: ``aleatoric_epistemic_split`` (wind-turbine-pg-bnn
``src/aerovigil_pg_bnn/model.py``). Bayes-by-backprop weight uncertainty is
replaced by MC dropout, which is a cheap *approximation*: it is not variational
inference and its calibration must be measured, not assumed.
"""

from __future__ import annotations

import torch
from torch import nn


class BayesianUncertaintyHead(nn.Module):
    """Dropout head producing a mean and a learned observation variance."""

    def __init__(self, width: int, outputs: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.mean = nn.Linear(width, outputs)
        self.log_var = nn.Linear(width, outputs)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.dropout(x)
        return self.mean(h), self.log_var(h).clamp(-10, 6)


def decompose_mc(means: torch.Tensor, log_vars: torch.Tensor) -> dict[str, torch.Tensor]:
    """Split MC predictive statistics into aleatoric and epistemic parts.

    ``means``    : (S, B, O) MC sample means
    ``log_vars`` : (S, B, O) MC sample log-variances
    """
    aleatoric = log_vars.exp().mean(0)
    epistemic = means.var(0, unbiased=False)
    return {
        "mean": means.mean(0),
        "log_var": log_vars.mean(0),
        "aleatoric": aleatoric.sqrt(),
        "epistemic": epistemic.sqrt(),
        "total": (aleatoric + epistemic).sqrt(),
        "std": (aleatoric + epistemic).sqrt(),
    }


def gaussian_nll(
    mean: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """Heteroscedastic Gaussian negative log-likelihood."""
    nll = 0.5 * (log_var + (target - mean).pow(2) * torch.exp(-log_var))
    if reduction == "mean":
        return nll.mean()
    if reduction == "sum":
        return nll.sum()
    return nll


def scale_log_var(log_var: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    """Apply fitted temperature scaling (``temperature`` >= 0, per output)."""
    return log_var + 2.0 * torch.log(temperature.clamp_min(1e-3))


__all__ = [
    "BayesianUncertaintyHead",
    "decompose_mc",
    "gaussian_nll",
    "scale_log_var",
]
