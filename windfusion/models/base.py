"""Shared WindFusion architecture: stem -> causal encoder -> sparse mixture.

All models in the family share this trunk. Presets differ by the *inductive
bias* they add on top of it (receptive field, expert specialisation, auxiliary
heads, physics damage features), and every difference is declared in
:class:`~windfusion.models.registry.ModelSpec`.
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import TurbineConfig
from ..physics import drivetrain
from ..physics.residuals import compute_residuals
from .encoder import MultiScaleCausalEncoder
from .experts import build_expert
from .registry import EXPERT_NAMES, ModelSpec
from .router import SparseExpertRouter
from .uncertainty import BayesianUncertaintyHead, decompose_mc


class NeighborAttention(nn.Module):
    """Cross-asset attention over neighbouring turbines (fleet/wake context)."""

    def __init__(self, width: int, input_width: int, key_dim: int = 32) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_width, width)
        self.query = nn.Linear(width, key_dim)
        self.key = nn.Linear(width, key_dim)
        self.value = nn.Linear(width, width)
        self.out = nn.Linear(width, width)
        self.scale = key_dim**0.5

    def forward(self, pooled: torch.Tensor, neighbors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(updated_pooled, neighbor_context)``.

        ``neighbors`` is ``(B, K, C_raw)``; returns context ``(B, K, width)``.
        """
        nb = self.input_proj(torch.nan_to_num(neighbors))  # (B, K, W)
        q = self.query(pooled).unsqueeze(1)
        k = self.key(nb)
        v = self.value(nb)
        scores = torch.softmax((q * k).sum(-1) / self.scale, dim=-1)  # (B, K)
        context = (scores.unsqueeze(-1) * v).sum(1)  # (B, W)
        updated = pooled + self.out(context)
        return updated, nb


class WindFusionBase(nn.Module):
    """Causal encoder + top-k sparse domain experts + uncertainty heads."""

    def __init__(
        self,
        spec: ModelSpec,
        input_features: int = 12,
        outputs: int = 3,
        physics_features: int = 5,
        turbine: TurbineConfig | None = None,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.input_features = int(input_features)
        self.physics_features = int(physics_features)
        self.outputs = int(outputs)
        self.turbine = turbine or TurbineConfig()

        h = spec.hidden
        self.stem = nn.Sequential(
            nn.Linear(self.input_features, h), nn.LayerNorm(h), nn.SiLU()
        )
        self.physics_proj = nn.Sequential(nn.Linear(self.physics_features, h), nn.SiLU())
        self.encoder = MultiScaleCausalEncoder(
            h, blocks=spec.encoder_blocks, dilations=spec.dilations, dropout=spec.dropout * 0.5
        )

        expert_names = tuple(EXPERT_NAMES) + tuple(spec.extra_experts)
        self.expert_names = expert_names
        self.experts = nn.ModuleList(
            [
                build_expert(
                    name,
                    h,
                    spec.bottleneck,
                    depth=int(spec.expert_depth.get(name, 1)) if name in EXPERT_NAMES else 1,
                    dropout=spec.dropout * 0.5,
                )
                for name in expert_names
            ]
        )
        self.router = SparseExpertRouter(h, len(expert_names), spec.top_k, jitter=0.01)

        # Optional architectural biases -------------------------------------------------
        self.neighbor_attention = (
            NeighborAttention(h, self.input_features) if spec.neighbors > 0 else None
        )
        self.stat_proj = nn.Linear(2 * self.input_features, h) if spec.window_stats else None
        self.damage_proj = nn.Linear(1, h) if spec.damage_feature else None
        self.self_check = nn.Linear(h, 1) if spec.self_check_head else None
        self.forecast_head = (
            nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.Linear(h, spec.forecast_horizon * 2))
            if spec.forecast_horizon
            else None
        )
        self.head = BayesianUncertaintyHead(h, outputs, spec.dropout)
        # Export flag: when True every expert is evaluated and combined with its
        # routing weight instead of skipping inactive ones. The result is
        # mathematically identical (inactive weights are exactly zero) but the
        # graph no longer contains data-dependent control flow, which ONNX and
        # other static runtimes cannot trace.
        self.dense_experts = False

    # ── hooks overridden by presets ────────────────────────────────────────
    def expert_logit_bias(self, physics: torch.Tensor) -> torch.Tensor | None:
        """Additive router-logit bias. Overridden by the Ra thermal preset."""
        return None

    # ── forward ────────────────────────────────────────────────────────────
    def encode(self, sequence: torch.Tensor, physics: torch.Tensor, neighbors=None) -> torch.Tensor:
        """Return the pooled representation ``(B, H)`` used by router and head."""
        if sequence.ndim != 3 or sequence.shape[-1] != self.input_features:
            raise ValueError(
                f"expected sequence (B, T, {self.input_features}), got {tuple(sequence.shape)}"
            )
        if physics.shape[-1] != self.physics_features:
            raise ValueError(
                f"expected physics (B, {self.physics_features}), got {tuple(physics.shape)}"
            )
        h = self.stem(torch.nan_to_num(sequence)) + self.physics_proj(
            torch.nan_to_num(physics)
        ).unsqueeze(1)
        encoded = self.encoder(h)
        pooled = encoded[:, -1]
        if self.stat_proj is not None:
            clean = torch.nan_to_num(sequence)
            stats = torch.cat([clean.std(1), clean.amax(1) - clean.amin(1)], dim=-1)
            pooled = pooled + self.stat_proj(stats)
        if self.damage_proj is not None:
            pooled = pooled + self.damage_proj(self._damage_feature(sequence))
        if self.neighbor_attention is not None and neighbors is not None:
            pooled, neighbor_context = self.neighbor_attention(pooled, neighbors)
        else:
            neighbor_context = None
        self._neighbor_context = neighbor_context
        return pooled

    def _damage_feature(self, sequence: torch.Tensor) -> torch.Tensor:
        """ISO 281-inspired dimensionless damage-rate feature (Odin preset)."""
        last = torch.nan_to_num(sequence[:, -1, :])
        idx = {name: i for i, name in enumerate(self._channel_names())}
        power_w = last[:, idx["active_power_kw"]].clamp_min(0.0) * 1000.0
        rated_w = max(self.turbine.rated_power_kw, 1.0) * 1000.0
        load_ratio = (power_w / rated_w).clamp(0, 2)
        vibration_ratio = (last[:, idx["vibration_rms"]] / 4.5).clamp(0, 4)
        temperature_c = last[:, idx["main_bearing_temp_c"]]
        rate = drivetrain.bearing_damage_rate(load_ratio, vibration_ratio, temperature_c)
        return torch.log1p(rate.clamp(0, 1e6)).unsqueeze(-1)

    def _channel_names(self) -> tuple[str, ...]:
        from ..data.schema import CHANNELS

        return CHANNELS[: self.input_features]

    def forward(
        self,
        sequence: torch.Tensor,
        physics: torch.Tensor,
        neighbors: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        pooled = self.encode(sequence, physics, neighbors)
        logits_bias = self.expert_logit_bias(physics)
        if logits_bias is not None:
            weights = self.router.forward_with_bias(pooled, logits_bias)
        else:
            weights = self.router(pooled)

        fused = torch.zeros_like(pooled)
        neighbor_context = getattr(self, "_neighbor_context", None)
        for i, expert in enumerate(self.experts):
            if not self.dense_experts:
                active = weights[:, i] > 0
                if not bool(active.any()):  # skip experts nobody selected
                    continue
                rows = pooled[active]
                context = neighbor_context[active] if neighbor_context is not None else None
                weight = weights[active, i : i + 1]
            else:
                rows, context, weight = pooled, neighbor_context, weights[:, i : i + 1]
            out = expert(rows, context) if self.expert_names[i] == "wake" else expert(rows)
            if self.dense_experts:
                fused = fused + out * weight
            else:
                fused[active] = fused[active] + out * weight
        mean, log_var = self.head(fused)
        result = {"mean": mean, "log_var": log_var, "routing": weights, "representation": fused}
        if self.self_check is not None:
            result["consistency"] = torch.nn.functional.softplus(self.self_check(pooled)) + 1e-3
        if self.forecast_head is not None:
            result["forecast"] = self.forecast_head(pooled).reshape(
                pooled.shape[0], self.spec.forecast_horizon, 2
            )
        return result

    # ── inference helpers ──────────────────────────────────────────────────
    @torch.no_grad()
    def predict(
        self,
        sequence: torch.Tensor,
        physics: torch.Tensor,
        neighbors: torch.Tensor | None = None,
        samples: int = 16,
        temperature: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """MC-dropout predictive distribution with aleatoric/epistemic split."""
        was_training = self.training
        self.train()
        outputs = [self(sequence, physics, neighbors) for _ in range(max(int(samples), 1))]
        self.train(was_training)
        result = decompose_mc(
            torch.stack([o["mean"] for o in outputs]), torch.stack([o["log_var"] for o in outputs])
        )
        if temperature is not None:
            result["log_var"] = result["log_var"] + 2.0 * torch.log(temperature.clamp_min(1e-3))
            result["aleatoric"] = result["log_var"].exp().sqrt()
            result["total"] = (result["log_var"].exp() + result["epistemic"].pow(2)).sqrt()
        result["routing"] = torch.stack([o["routing"] for o in outputs]).mean(0)
        if "consistency" in outputs[0]:
            result["consistency"] = torch.stack([o["consistency"] for o in outputs]).mean(0)
        return result

    def expected_residuals(self, sequence: torch.Tensor) -> dict[str, torch.Tensor]:
        """Physics residuals for the batch (SI-unit snapshot at the last step)."""
        return compute_residuals(sequence[:, -1, :], self.turbine)

    # ── introspection ──────────────────────────────────────────────────────
    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def physics_bias(self) -> dict[str, float]:
        return dict(self.spec.physics_bias)

    def routing_explanation(self) -> list[str]:
        if self.router.last_weights is None:
            return []
        top = self.router.last_weights.argmax(-1).tolist()
        return [self.expert_names[i] for i in top]

    def summary(self) -> dict[str, object]:
        return {
            "model_id": self.spec.model_id,
            "family": self.spec.family,
            "tradition": self.spec.tradition,
            "parameters": self.parameter_count,
            "experts": list(self.expert_names),
            "top_k": self.spec.top_k,
            "receptive_field": self.encoder.receptive_field,
            "physics_bias": self.physics_bias,
        }


__all__ = ["NeighborAttention", "WindFusionBase"]
