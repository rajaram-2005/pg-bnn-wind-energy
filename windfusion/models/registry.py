"""Model registry: one contract, many specialisations.

The registry is an explicit list of capability tiers selected before inference.
The tiers are *architectural*: each entry changes the inductive bias (temporal
receptive field, expert specialisation, auxiliary heads), not just the layer
widths.

Naming note: mythology-inspired identifiers (Ra, Qinglong, Vayu, Odin, Aeolus)
are **engineering presets**. They are not pretrained models, they do not encode
cultural content, and they carry no performance claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field

FAMILY_MYTHOLOGY = "mythology"
FAMILY_SELF_LEARNING = "self-learning"
FAMILY_TIER = "tier"

EXPERT_NAMES: tuple[str, ...] = ("aero", "drive", "thermal", "grid")

DEFAULT_PHYSICS_BIAS: dict[str, float] = {
    "aero": 1.0,
    "drive": 1.0,
    "thermal": 1.0,
    "grid": 1.0,
    "consistency": 1.0,
}


def _bias(**overrides: float) -> dict[str, float]:
    merged = dict(DEFAULT_PHYSICS_BIAS)
    merged.update(overrides)
    return merged


@dataclass(frozen=True)
class ModelSpec:
    """Declarative description of one model in the family."""

    model_id: str
    family: str
    hidden: int
    bottleneck: int
    top_k: int
    dropout: float
    tradition: str = ""
    encoder_blocks: int = 3
    dilations: tuple[int, ...] = (1, 2, 4)
    expert_depth: dict[str, int] = field(
        default_factory=lambda: {"aero": 1, "drive": 1, "thermal": 1, "grid": 1}
    )
    extra_experts: tuple[str, ...] = ()
    neighbors: int = 0
    forecast_horizon: int = 0
    monotone_rul: bool = False
    self_check_head: bool = False
    window_stats: bool = False
    damage_feature: bool = False
    self_supervised: bool = False
    physics_bias: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PHYSICS_BIAS))
    description: str = ""
    provenance: tuple[str, ...] = ()

    @property
    def n_experts(self) -> int:
        return len(EXPERT_NAMES) + len(self.extra_experts)

    def as_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "family": self.family,
            "tradition": self.tradition,
            "hidden": self.hidden,
            "bottleneck": self.bottleneck,
            "top_k": self.top_k,
            "dropout": self.dropout,
            "encoder_blocks": self.encoder_blocks,
            "dilations": list(self.dilations),
            "expert_depth": dict(self.expert_depth),
            "extra_experts": list(self.extra_experts),
            "neighbors": self.neighbors,
            "forecast_horizon": self.forecast_horizon,
            "monotone_rul": self.monotone_rul,
            "self_check_head": self.self_check_head,
            "window_stats": self.window_stats,
            "damage_feature": self.damage_feature,
            "self_supervised": self.self_supervised,
            "physics_bias": dict(self.physics_bias),
            "description": self.description,
        }


# ── Self-learning reference model (trains on its own) ──────────────────────
AUTOWIND = ModelSpec(
    model_id="windfusion-auto",
    family=FAMILY_SELF_LEARNING,
    tradition="AutoWind (self-learning)",
    hidden=64,
    bottleneck=20,
    top_k=2,
    dropout=0.10,
    encoder_blocks=4,
    dilations=(1, 2, 4, 8),
    window_stats=True,
    self_check_head=True,
    self_supervised=True,
    description=(
        "Self-learning reference model: it trains on its own. On top of the "
        "shared trunk it carries masked-channel reconstruction, next-step "
        "prediction and physics-consistency self-check objectives, plus gated "
        "pseudo-label rounds, so unlabelled SCADA windows are enough for an "
        "autonomous fit (supervised targets remain usable when present)."
    ),
    provenance=(),
)

# ── Mythology-inspired architecture presets ────────────────────────────────
RA = ModelSpec(
    model_id="ra-wind",
    family=FAMILY_MYTHOLOGY,
    tradition="Ra (Egyptian) - thermal / solar loading",
    hidden=48,
    bottleneck=16,
    top_k=2,
    dropout=0.08,
    encoder_blocks=3,
    dilations=(1, 2, 4),
    expert_depth={"aero": 1, "drive": 1, "thermal": 2, "grid": 1},
    physics_bias=_bias(thermal=1.6, drive=1.1),
    window_stats=True,
    description=(
        "Thermal-specialised preset: a deeper thermal expert, an ambient/"
        "irradiance-driven modulation gate on the expert mixture and a "
        "monotone damage-rate prior. Built for hot-climate gearbox and "
        "generator-winding degradation."
    ),
    provenance=("lumped_thermal_rc", "soft_hinge_limits"),
)

QINGLONG = ModelSpec(
    model_id="qinglong-wind",
    family=FAMILY_MYTHOLOGY,
    tradition="Qinglong (Chinese) - wake and fleet coupling",
    hidden=56,
    bottleneck=18,
    top_k=2,
    dropout=0.10,
    encoder_blocks=3,
    dilations=(1, 2, 4),
    extra_experts=("wake",),
    neighbors=3,
    window_stats=True,
    physics_bias=_bias(aero=1.5, consistency=1.2),
    description=(
        "Fleet/wake-aware preset: cross-asset attention over neighbouring "
        "turbines weighted by Jensen wake deficit, plus a dedicated wake "
        "expert. Designed for dense layouts where upstream wakes drive load."
    ),
    provenance=("jensen_wake", "heier_cp_betz"),
)

VAYU = ModelSpec(
    model_id="vayu-wind",
    family=FAMILY_MYTHOLOGY,
    tradition="Vayu (Hindu) - gust and rotor aerodynamics",
    hidden=44,
    bottleneck=14,
    top_k=1,
    dropout=0.12,
    encoder_blocks=4,
    dilations=(1, 2, 4, 8, 16),
    expert_depth={"aero": 2, "drive": 1, "thermal": 1, "grid": 1},
    physics_bias=_bias(aero=1.6, consistency=1.3),
    window_stats=True,
    description=(
        "Gust-focused preset: a long multi-scale dilated receptive field "
        "(up to 16 steps) with top-1 routing for the lowest latency, an "
        "aerodynamics-specialised expert and explicit gust statistics."
    ),
    provenance=("heier_cp_betz",),
)

ODIN = ModelSpec(
    model_id="odin-wind",
    family=FAMILY_MYTHOLOGY,
    tradition="Odin (Norse) - cold-climate drivetrain and RUL",
    hidden=56,
    bottleneck=20,
    top_k=2,
    dropout=0.10,
    encoder_blocks=3,
    dilations=(1, 2, 4),
    expert_depth={"aero": 1, "drive": 2, "thermal": 1, "grid": 1},
    monotone_rul=True,
    damage_feature=True,
    physics_bias=_bias(drive=1.6, thermal=1.2),
    description=(
        "Drivetrain/RUL preset: deeper drivetrain expert, ISO 281 load-life "
        "damage-rate feature, low-temperature lubrication margin and a "
        "monotonicity penalty so predicted RUL cannot rise as damage grows."
    ),
    provenance=("iso281_l10", "soft_hinge_limits"),
)

AEOLUS = ModelSpec(
    model_id="aeolus-wind",
    family=FAMILY_MYTHOLOGY,
    tradition="Aeolus (Greek) - multi-horizon forecasting",
    hidden=64,
    bottleneck=20,
    top_k=2,
    dropout=0.10,
    encoder_blocks=3,
    dilations=(1, 2, 4, 8),
    forecast_horizon=6,
    physics_bias=_bias(aero=1.3, consistency=1.2),
    description=(
        "Forecasting preset: an auxiliary multi-horizon decoder that predicts "
        "future wind speed and power, which regularises the shared encoder "
        "toward dynamics rather than instantaneous correlations."
    ),
    provenance=("heier_cp_betz", "heteroscedastic_nll"),
)

# ── Capacity tiers (deployment-oriented) ───────────────────────────────────
EDGE = ModelSpec(
    model_id="windfusion-edge",
    family=FAMILY_TIER,
    tradition="Edge tier",
    hidden=24,
    bottleneck=8,
    top_k=1,
    dropout=0.05,
    encoder_blocks=2,
    dilations=(1, 2),
    description="Smallest tier for constrained CPU/edge hardware.",
)

LITE = ModelSpec(
    model_id="windfusion-lite",
    family=FAMILY_TIER,
    tradition="Default tier",
    hidden=48,
    bottleneck=16,
    top_k=2,
    dropout=0.10,
    encoder_blocks=3,
    dilations=(1, 2, 4),
    description="Default balanced tier carried over from WindFusion-Lite v0.1.",
)

RESEARCH = ModelSpec(
    model_id="windfusion-research",
    family=FAMILY_TIER,
    tradition="Research tier",
    hidden=128,
    bottleneck=32,
    top_k=3,
    dropout=0.15,
    encoder_blocks=4,
    dilations=(1, 2, 4, 8),
    expert_depth={"aero": 2, "drive": 2, "thermal": 2, "grid": 2},
    self_check_head=True,
    description="Large tier used as the distillation teacher and for ablations.",
)

MODEL_REGISTRY: dict[str, ModelSpec] = {
    spec.model_id: spec
    for spec in (AUTOWIND, RA, QINGLONG, VAYU, ODIN, AEOLUS, EDGE, LITE, RESEARCH)
}

MYTHOLOGY_MODELS: tuple[str, ...] = tuple(
    spec.model_id for spec in MODEL_REGISTRY.values() if spec.family == FAMILY_MYTHOLOGY
)


def get_spec(mode: str) -> ModelSpec:
    if mode not in MODEL_REGISTRY:
        raise ValueError(f"unknown model {mode!r}; available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[mode]


def registry_table() -> str:
    """Markdown table of the model family (used by docs and the CLI)."""
    lines = [
        "| Model | Family | Tradition | Hidden | Top-k | Neighbours | Extras |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for spec in MODEL_REGISTRY.values():
        extras = []
        if spec.extra_experts:
            extras.append("experts: " + ", ".join(spec.extra_experts))
        if spec.forecast_horizon:
            extras.append(f"forecast h={spec.forecast_horizon}")
        if spec.monotone_rul:
            extras.append("monotone RUL")
        if spec.self_check_head:
            extras.append("self-check head")
        if spec.self_supervised:
            extras.append("self-learning (autonomous fit)")
        lines.append(
            f"| `{spec.model_id}` | {spec.family} | {spec.tradition} | {spec.hidden} | "
            f"{spec.top_k} | {spec.neighbors} | {', '.join(extras) or '-'} |"
        )
    return "\n".join(lines)
