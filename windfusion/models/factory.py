"""Model factory: build any registered model from its identifier."""

from __future__ import annotations

from ..config import TurbineConfig
from .aetheris import AetherisWind
from .mythologies import AeolusWind, OdinWind, QinglongWind, RaWind, VayuWind
from .registry import MODEL_REGISTRY, ModelSpec, get_spec
from .tiers import EdgeWind, LiteWind, ResearchWind

CONSTRUCTORS = {
    "aetheris-wind": AetherisWind,
    "ra-wind": RaWind,
    "qinglong-wind": QinglongWind,
    "vayu-wind": VayuWind,
    "odin-wind": OdinWind,
    "aeolus-wind": AeolusWind,
    "windfusion-edge": EdgeWind,
    "windfusion-lite": LiteWind,
    "windfusion-research": ResearchWind,
}


def create_model(
    mode: str = "windfusion-lite",
    input_features: int = 12,
    outputs: int = 3,
    physics_features: int = 5,
    turbine: TurbineConfig | None = None,
):
    """Instantiate a model by registry identifier."""
    spec = get_spec(mode)
    constructor = CONSTRUCTORS.get(spec.model_id)
    if constructor is None:  # pragma: no cover - registry/table must stay in sync
        raise ValueError(f"no constructor registered for {mode!r}")
    return constructor(input_features, outputs, physics_features, turbine)


def model_spec(mode: str) -> ModelSpec:
    """Convenience accessor for a model's declarative spec."""
    return get_spec(mode)


__all__ = ["CONSTRUCTORS", "MODEL_REGISTRY", "create_model", "model_spec"]
