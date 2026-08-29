"""Top-k sparse router with load balancing and recorded decisions.

Provenance: ``tiered_model_router`` (Aetheris). Every prediction carries the
routing weights it used, so the decision is auditable after the fact instead of
being an opaque inference.
"""

from __future__ import annotations

import torch
from torch import nn


class SparseExpertRouter(nn.Module):
    """Top-k straight-through routing with optional noisy exploration."""

    def __init__(
        self,
        width: int,
        experts: int = 4,
        top_k: int = 2,
        temperature: float = 1.0,
        jitter: float = 0.0,
    ) -> None:
        super().__init__()
        if not 1 <= top_k <= experts:
            raise ValueError("top_k must be between 1 and experts")
        self.gate = nn.Linear(width, experts)
        self.top_k = top_k
        self.temperature = max(temperature, 1e-3)
        self.jitter = max(jitter, 0.0)
        self.last_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        return self.forward_with_bias(x, bias)

    def forward_with_bias(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.gate(x)
        if bias is not None:
            logits = logits + bias
        if self.training and self.jitter > 0:
            logits = logits + self.jitter * torch.randn_like(logits)
        dense = torch.softmax(logits / self.temperature, dim=-1)
        if self.top_k < dense.shape[-1]:
            values, indices = dense.topk(self.top_k, dim=-1)
            sparse = torch.zeros_like(dense).scatter(-1, indices, values)
        else:
            sparse = dense
        sparse = sparse / sparse.sum(-1, keepdim=True).clamp_min(1e-8)
        self.last_weights = sparse.detach()
        return sparse


def router_balance_loss(weights: torch.Tensor) -> torch.Tensor:
    """Encourage a balanced expert load (negative entropy of the mean gate).

    Minimising ``sum(p * log p)`` pushes the average routing distribution
    toward uniform, which prevents expert collapse. The per-sample sparsity
    imposed by top-k is unaffected.
    """
    mean_gate = weights.mean(0).clamp_min(1e-8)
    return (mean_gate * mean_gate.log()).sum()


def router_agreement_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Symmetric KL between two routing distributions (distillation signal)."""
    s = student.clamp_min(1e-6)
    t = teacher.detach().clamp_min(1e-6)
    return 0.5 * ((t * (t.log() - s.log())).sum(-1) + (s * (s.log() - t.log())).sum(-1)).mean()


__all__ = ["SparseExpertRouter", "router_agreement_loss", "router_balance_loss"]
