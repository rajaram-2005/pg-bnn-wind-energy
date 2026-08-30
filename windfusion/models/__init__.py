"""WindFusion model family.

One contract — ``model(sequence, physics, neighbors=None)`` — implemented by the
self-learning AutoWind model, five mythology-inspired architecture presets and
three capacity tiers.
"""

from .base import NeighborAttention, WindFusionBase
from .encoder import GatedTemporalEncoder, MultiScaleCausalEncoder
from .experts import EXPERT_NAMES, AeroExpert, DriveExpert, GridExpert, ThermalExpert, WakeExpert
from .factory import create_model, model_spec
from .registry import (
    FAMILY_MYTHOLOGY,
    FAMILY_SELF_LEARNING,
    FAMILY_TIER,
    MODEL_REGISTRY,
    MYTHOLOGY_MODELS,
    ModelSpec,
    get_spec,
    registry_table,
)
from .router import SparseExpertRouter, router_agreement_loss, router_balance_loss
from .uncertainty import BayesianUncertaintyHead, decompose_mc, gaussian_nll
from .auto import AutoWind, SelfLearningBreakdown, self_training_from
from .mythologies import AeolusWind, OdinWind, QinglongWind, RaWind, VayuWind
from .tiers import EdgeWind, LiteWind, ResearchWind

__all__ = [
    "AeroExpert",
    "AeolusWind",
    "AutoWind",
    "BayesianUncertaintyHead",
    "DriveExpert",
    "EXPERT_NAMES",
    "EdgeWind",
    "FAMILY_MYTHOLOGY",
    "FAMILY_SELF_LEARNING",
    "FAMILY_TIER",
    "GatedTemporalEncoder",
    "GridExpert",
    "LiteWind",
    "MODEL_REGISTRY",
    "MYTHOLOGY_MODELS",
    "ModelSpec",
    "MultiScaleCausalEncoder",
    "NeighborAttention",
    "OdinWind",
    "QinglongWind",
    "RaWind",
    "ResearchWind",
    "SelfLearningBreakdown",
    "SparseExpertRouter",
    "ThermalExpert",
    "VayuWind",
    "WakeExpert",
    "WindFusionBase",
    "create_model",
    "decompose_mc",
    "gaussian_nll",
    "get_spec",
    "model_spec",
    "registry_table",
    "router_agreement_loss",
    "router_balance_loss",
    "self_training_from",
]
