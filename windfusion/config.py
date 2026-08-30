"""Versioned, offline-first configuration for WindFusion v0.2.

Provenance: ``advisory_only_safety`` (wind-turbine-pg-bnn
``configs/default.yaml::safety``).

Design rules
------------
* Every run is reproducible from one YAML file plus a seed.
* Defaults are research defaults, never OEM limits.
* The safety block is an invariant: actuation is always disabled.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SafetyConfig:
    """Advisory-only invariant carried over from the PG-BNN lineage."""

    advisory_only: bool = True
    allow_actuation: bool = False

    def __post_init__(self) -> None:
        if not self.advisory_only:
            raise ValueError("advisory_only cannot be disabled: the system has no actuation path")
        if self.allow_actuation:
            raise ValueError("allow_actuation must remain False (advisory-only system)")


@dataclass(frozen=True)
class TurbineConfig:
    """Plant constants. Research defaults for a ~2 MW geared turbine."""

    rated_power_kw: float = 2000.0
    rotor_radius_m: float = 45.0
    gear_ratio: float = 97.0
    omega_rated: float = 1.9  # rated rotor speed (rad/s, ~18 rpm)
    gearbox_efficiency: float = 0.95
    generator_efficiency: float = 0.96
    air_density: float = 1.225
    nominal_voltage_v: float = 690.0
    nominal_frequency_hz: float = 50.0
    power_factor: float = 0.95
    thermal_resistance_k_w: float = 0.006
    thermal_capacitance_j_k: float = 5.0e4
    winding_resistance_ohm: float = 0.002
    wake_decay: float = 0.075

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class PhysicsConfig:
    """Physics-residual weights and training regularisation strengths."""

    weights: dict[str, float] = field(
        default_factory=lambda: {
            "aero": 0.20,
            "drive": 0.25,
            "thermal": 0.25,
            "grid": 0.15,
            "consistency": 0.15,
        }
    )
    limit_penalty: float = 0.05
    monotone_rul_weight: float = 0.05

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelConfig:
    """Model identity and shape."""

    mode: str = "windfusion-lite"
    input_features: int = 12
    physics_features: int = 5
    sequence_length: int = 24
    outputs: int = 3
    neighbors: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DataConfig:
    """Dataset provenance. ``source`` selects the loader backend."""

    source: str = "synthetic"
    root: str = "data"
    fleet_version: str = "synthetic-v1"
    n_turbines: int = 36
    seq_len: int = 1440
    sample_interval_s: int = 600
    window: int = 24
    stride: int = 6
    missing_rate: float = 0.01
    horizon_days: float = 30.0
    seed: int = 7
    split: dict[str, float] = field(default_factory=lambda: {"train": 0.6, "val": 0.2, "test": 0.2})
    checksum: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingConfig:
    """Optimisation schedule."""

    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    warmup_fraction: float = 0.05
    patience: int = 8
    min_delta: float = 1e-4
    uncertainty_weight: float = 0.01
    router_balance_weight: float = 0.01
    target_weights: dict[str, float] = field(
        default_factory=lambda: {
            "failure_probability": 1.0,
            "health_index": 1.0,
            "rul_days": 1.0,
        }
    )
    num_workers: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TelemetryConfig:
    """Adaptive telemetry policy thresholds."""

    bypass_threshold: float = 0.65
    detail_threshold: float = 0.35
    deadband: float = 0.02
    quantum: float = 0.01
    cooldown_steps: int = 6
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "anomaly": 0.35,
            "epistemic": 0.25,
            "rate": 0.20,
            "physics": 0.20,
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationConfig:
    """Thresholds used by the verifier and the early-warning protocol."""

    mc_samples: int = 16
    warning_horizon_days: float = 30.0
    epistemic_abstain: float = 0.45
    physics_abstain: float = 0.35
    min_data_completeness: float = 0.6
    conformal_coverage: float = 0.9
    conformal: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DistillationConfig:
    """Teacher-to-student weights."""

    teacher: str = "windfusion-research"
    student: str = "windfusion-edge"
    temperature: float = 1.0
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "prediction": 0.50,
            "uncertainty": 0.20,
            "feature": 0.15,
            "physics": 0.15,
        }
    )
    router_agreement_weight: float = 0.05

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FleetConfig:
    """Fleet learning / site adaptation."""

    rounds: int = 3
    local_epochs: int = 1
    clients_per_round: int = 6
    physics_aware_aggregation: bool = True
    reptile_inner_lr: float = 5e-3
    reptile_meta_lr: float = 0.4
    reptile_inner_steps: int = 5
    reptile_meta_iterations: int = 25
    fewshot_steps: int = 25
    fewshot_lr: float = 1e-3
    adaptation_windows: int = 32

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SelfTrainingConfig:
    """Hyper-parameters for ``windfusion-auto``, the model that trains on its own.

    The autonomous objective needs no ground-truth labels: masked-channel
    reconstruction, next-step prediction, physics-consistency self-check and
    router balance carry the first round. Later rounds may add a gated
    pseudo-label term bootstrapped from the model's own confident predictions;
    set ``pseudo_label_weight`` to 0 to keep the fit purely self-supervised.
    """

    mask_rate: float = 0.15
    rounds: int = 2
    epochs: int = 10
    learning_rate: float = 1.5e-3
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    patience: int = 4
    min_delta: float = 1e-4
    mc_samples: int = 8
    reconstruction_weight: float = 1.0
    next_step_weight: float = 1.0
    physics_weight: float = 0.25
    consistency_weight: float = 0.10
    router_balance_weight: float = 0.01
    pseudo_label_weight: float = 0.50
    pseudo_label_gate: float = 0.35  # max mean epistemic for a trusted self-label

    def __post_init__(self) -> None:
        if not 0.0 < self.mask_rate <= 0.5:
            raise ValueError("mask_rate must be in (0, 0.5] — more masking stops learning")
        if self.rounds < 1:
            raise ValueError("rounds must be >= 1")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        for name in (
            "reconstruction_weight",
            "next_step_weight",
            "physics_weight",
            "consistency_weight",
            "router_balance_weight",
            "pseudo_label_weight",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 <= self.pseudo_label_gate <= 2.0:
            raise ValueError("pseudo_label_gate must be in [0, 2]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WindFusionConfig:
    """Top-level configuration object."""

    seed: int = 7
    version: str = "0.2.0"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    fleet: FleetConfig = field(default_factory=FleetConfig)
    self_training: SelfTrainingConfig = field(default_factory=SelfTrainingConfig)
    turbine: TurbineConfig = field(default_factory=TurbineConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    # ── serialisation ───────────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_yaml(), encoding="utf-8")
        return path

    # ── construction ────────────────────────────────────────────────────────
    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> WindFusionConfig:
        raw = copy.deepcopy(raw or {})
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        kwargs: dict[str, Any] = {}
        for name in _SECTION_TYPES:
            kwargs[name] = _build(_SECTION_TYPES[name], raw.get(name, {}))
        kwargs["seed"] = int(raw.get("seed", 7))
        kwargs["version"] = str(raw.get("version", "0.2.0"))
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> WindFusionConfig:
        path = Path(path)
        return cls.from_dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})

    def with_overrides(self, **overrides: Any) -> WindFusionConfig:
        """Return a copy with dotted overrides, e.g. ``training.epochs=10``."""
        data = self.to_dict()
        for key, value in overrides.items():
            parts = key.split(".")
            node = data
            for part in parts[:-1]:
                node = node.setdefault(part, {})
                if not isinstance(node, dict):
                    raise ValueError(f"cannot override {key!r}: {part} is not a section")
            node[parts[-1]] = value
        return WindFusionConfig.from_dict(data)


def _resolve(name: str) -> type:
    return _SECTION_TYPES[name]


_SECTION_TYPES: dict[str, type] = {
    "model": ModelConfig,
    "data": DataConfig,
    "training": TrainingConfig,
    "physics": PhysicsConfig,
    "telemetry": TelemetryConfig,
    "evaluation": EvaluationConfig,
    "distillation": DistillationConfig,
    "fleet": FleetConfig,
    "self_training": SelfTrainingConfig,
    "turbine": TurbineConfig,
    "safety": SafetyConfig,
}


def _build(section_type: type, raw: dict[str, Any]) -> Any:
    raw = copy.deepcopy(raw or {})
    valid = {f.name for f in fields(section_type)}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(f"unknown keys for {section_type.__name__}: {sorted(unknown)}")
    return section_type(**raw)


def default_config() -> WindFusionConfig:
    """Config bundled with the repository (``configs/default.yaml``)."""
    path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    if path.exists():
        return WindFusionConfig.from_yaml(path)
    return WindFusionConfig()


def describe(config: WindFusionConfig) -> str:
    """One-line human summary used in logs and reports."""
    return (
        f"WindFusion v{config.version} | model={config.model.mode} "
        f"| data={config.data.source}/{config.data.fleet_version} "
        f"| seed={config.seed} | epochs={config.training.epochs}"
    )
