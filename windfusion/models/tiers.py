"""Capacity tiers for deployment (edge / default / research).

Tiers share the architecture and differ only in capacity and sparsity, so a
model can be swapped without changing the data contract or the export path.
"""

from __future__ import annotations

from .base import WindFusionBase
from .registry import EDGE, LITE, RESEARCH

__all__ = ["EdgeWind", "LiteWind", "ResearchWind"]


class EdgeWind(WindFusionBase):
    """Smallest tier: single expert per sample, 2 encoder blocks."""

    def __init__(self, input_features: int = 12, outputs: int = 3, physics_features: int = 5, turbine=None):
        super().__init__(EDGE, input_features, outputs, physics_features, turbine)


class LiteWind(WindFusionBase):
    """Default tier carried over from WindFusion-Lite v0.1."""

    def __init__(self, input_features: int = 12, outputs: int = 3, physics_features: int = 5, turbine=None):
        super().__init__(LITE, input_features, outputs, physics_features, turbine)


class ResearchWind(WindFusionBase):
    """Largest tier: used as the distillation teacher and for ablations."""

    def __init__(self, input_features: int = 12, outputs: int = 3, physics_features: int = 5, turbine=None):
        super().__init__(RESEARCH, input_features, outputs, physics_features, turbine)
