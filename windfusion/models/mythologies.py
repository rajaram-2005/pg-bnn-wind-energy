"""Mythology-inspired architecture presets.

**These are engineering presets, not cultural artefacts and not pretrained
models.** Each name identifies a different inductive bias in the same trunk:
what the model looks at, how wide its temporal receptive field is, which expert
is deepened and which physics residual it is pushed to respect.

===========  =============================  ========================================
Preset       Physical focus                 Architectural difference
===========  =============================  ========================================
Ra           thermal / solar loading        deeper thermal expert, ambient-modulated
                                            router bias, window statistics
Qinglong     wake and fleet coupling        cross-asset attention + wake expert,
                                            Jensen wake deficit weighting
Vayu         gust / rotor aerodynamics      long dilated receptive field (up to 16
                                            steps), top-1 routing (lowest latency)
Odin         cold-climate drivetrain, RUL   deeper drivetrain expert, ISO 281 damage
                                            feature, monotone RUL penalty
Aeolus       multi-horizon forecasting      auxiliary 6-step wind/power decoder
===========  =============================  ========================================
"""

from __future__ import annotations

import torch

from .base import WindFusionBase
from .registry import AEOLUS, ODIN, QINGLONG, RA, VAYU

__all__ = ["RaWind", "QinglongWind", "VayuWind", "OdinWind", "AeolusWind"]


class RaWind(WindFusionBase):
    """Egyptian preset (Ra): thermal and solar-load specialisation.

    The router receives an additive logit bias for the thermal expert that grows
    with the thermal margin ``(T_winding - T_ambient) / 80`` carried in the
    physics feature vector, so hot-climate samples are handled by the thermal
    expert without extra computation.
    """

    def __init__(self, input_features: int = 12, outputs: int = 3, physics_features: int = 5, turbine=None):
        super().__init__(RA, input_features, outputs, physics_features, turbine)
        self.thermal_index = self.expert_names.index("thermal")

    def expert_logit_bias(self, physics: torch.Tensor) -> torch.Tensor | None:
        if physics.shape[-1] < 4:
            return None
        thermal_margin = torch.nan_to_num(physics[:, 3]).clamp(-2.0, 4.0)
        bias = torch.zeros_like(physics[:, :1]).expand(-1, len(self.expert_names)).contiguous()
        bias[:, self.thermal_index] = 1.5 * torch.relu(thermal_margin - 0.5)
        return bias


class QinglongWind(WindFusionBase):
    """Chinese preset (Qinglong): wake-aware fleet coupling.

    Consumes ``neighbors`` — the synchronous window of the ``K`` nearest
    turbines — through cross-asset attention plus a dedicated wake expert. When
    neighbours are unavailable the model degrades to the base trunk instead of
    failing.
    """

    def __init__(self, input_features: int = 12, outputs: int = 3, physics_features: int = 5, turbine=None):
        super().__init__(QINGLONG, input_features, outputs, physics_features, turbine)

    def expected_neighbors(self) -> int:
        return self.spec.neighbors


class VayuWind(WindFusionBase):
    """Hindu preset (Vayu): gust and rotor aerodynamics.

    Longest receptive field in the family (dilations up to 16) with top-1
    routing, so gust-driven transients are captured at the lowest possible
    per-inference cost.
    """

    def __init__(self, input_features: int = 12, outputs: int = 3, physics_features: int = 5, turbine=None):
        super().__init__(VAYU, input_features, outputs, physics_features, turbine)


class OdinWind(WindFusionBase):
    """Norse preset (Odin): cold-climate drivetrain and remaining useful life.

    Adds an ISO 281-inspired damage-rate feature and enables the monotonicity
    penalty in :func:`windfusion.training.losses.monotone_rul_penalty`, so
    predicted RUL cannot increase as the physical damage proxy grows.
    """

    def __init__(self, input_features: int = 12, outputs: int = 3, physics_features: int = 5, turbine=None):
        super().__init__(ODIN, input_features, outputs, physics_features, turbine)

    def damage_proxy(self, sequence: torch.Tensor) -> torch.Tensor:
        """Dimensionless damage proxy used by the monotonicity penalty."""
        return self._damage_feature(sequence).squeeze(-1)


class AeolusWind(WindFusionBase):
    """Greek preset (Aeolus): multi-horizon forecasting auxiliary task.

    The shared encoder additionally predicts the next ``H`` wind-speed and
    power values. The auxiliary loss is only defined when the batch supplies
    ``future`` targets, so the model also runs on datasets without them.
    """

    def __init__(self, input_features: int = 12, outputs: int = 3, physics_features: int = 5, turbine=None):
        super().__init__(AEOLUS, input_features, outputs, physics_features, turbine)

    @property
    def forecast_horizon(self) -> int:
        return self.spec.forecast_horizon
