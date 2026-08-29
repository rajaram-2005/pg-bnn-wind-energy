"""Efficient heteroscedastic head + MC-dropout epistemic approximation."""

import torch
from torch import nn


class BayesianUncertaintyHead(nn.Module):
    def __init__(self, width: int, outputs: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.mean, self.log_var = nn.Linear(width, outputs), nn.Linear(width, outputs)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.dropout(x)
        return self.mean(h), self.log_var(h).clamp(-10, 6)


def decompose_mc(means: torch.Tensor, log_vars: torch.Tensor) -> dict[str, torch.Tensor]:
    aleatoric = log_vars.exp().mean(0)
    epistemic = means.var(0, unbiased=False)
    return {
        "mean": means.mean(0),
        "aleatoric": aleatoric.sqrt(),
        "epistemic": epistemic.sqrt(),
        "total": (aleatoric + epistemic).sqrt(),
    }
