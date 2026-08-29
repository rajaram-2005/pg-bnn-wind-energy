"""Top-k straight-through sparse routing with recorded decisions."""

import torch
from torch import nn


class SparseExpertRouter(nn.Module):
    def __init__(self, width: int, experts: int = 4, top_k: int = 2) -> None:
        super().__init__()
        if not 1 <= top_k <= experts:
            raise ValueError("top_k must be between 1 and experts")
        self.gate, self.top_k = nn.Linear(width, experts), top_k
        self.last_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dense = torch.softmax(self.gate(x), -1)
        values, indices = dense.topk(self.top_k, -1)
        sparse = torch.zeros_like(dense).scatter(-1, indices, values)
        sparse = sparse / sparse.sum(-1, keepdim=True).clamp_min(1e-8)
        self.last_weights = sparse.detach()
        return sparse
