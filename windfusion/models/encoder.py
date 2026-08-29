"""Causal multi-scale temporal encoder.

A stack of gated dilated convolutions. Causality is guaranteed by left-padding
only, so the representation at time ``t`` never sees ``t+1`` — a hard
requirement for online condition monitoring and for export to streaming
runtimes.
"""

from __future__ import annotations

import torch
from torch import nn


class GatedConvBlock(nn.Module):
    """One causal gated (GLU) dilated convolution with a residual path."""

    def __init__(self, width: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(width, width * 2, kernel_size, dilation=dilation)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, W)
        z = x.transpose(1, 2)
        if self.pad:
            z = torch.nn.functional.pad(z, (self.pad, 0))
        value, gate = self.conv(z).chunk(2, dim=1)
        out = (value * torch.sigmoid(gate)).transpose(1, 2)
        return self.norm(x + self.dropout(out))


class GatedTemporalEncoder(nn.Module):
    """Legacy v0.1 single-block encoder (kept for backwards compatibility)."""

    def __init__(self, features: int, hidden: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.proj = nn.Conv1d(features, hidden * 2, kernel_size)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("temporal input must have shape (batch, time, features)")
        z = torch.nn.functional.pad(x.transpose(1, 2), (self.kernel_size - 1, 0))
        value, gate = self.proj(z).chunk(2, dim=1)
        sequence = (value * torch.sigmoid(gate)).transpose(1, 2)
        return self.norm(sequence[:, -1])


class MultiScaleCausalEncoder(nn.Module):
    """Stacked dilated gated convolutions returning the full causal sequence."""

    def __init__(
        self,
        width: int,
        blocks: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4),
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if blocks <= 0:
            raise ValueError("encoder needs at least one block")
        dilations = tuple(dilations)
        layers = []
        for i in range(blocks):
            dilation = dilations[i % len(dilations)] if dilations else 1
            layers.append(GatedConvBlock(width, kernel_size, dilation, dropout))
        self.blocks = nn.ModuleList(layers)
        self.receptive_field = 1 + sum(
            (kernel_size - 1) * (dilations[i % len(dilations)] if dilations else 1)
            for i in range(blocks)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, W) -> (B, T, W)`` with strictly causal mixing."""
        if x.ndim != 3:
            raise ValueError("encoder input must have shape (batch, time, width)")
        h = x
        for block in self.blocks:
            h = block(h)
        return h

    def summary(self, x: torch.Tensor) -> torch.Tensor:
        """Last-timestep summary (the only causal pooling choice at inference)."""
        return self.forward(x)[:, -1]


__all__ = ["GatedConvBlock", "GatedTemporalEncoder", "MultiScaleCausalEncoder"]
