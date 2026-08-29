"""Training: losses, deterministic loop, distillation, calibration, fleet learning."""

from .calibration import CalibrationArtifact, apply_calibration, calibrate
from .distillation import DistillTrainer, DistillationWeights, distillation_loss
from .fleet import (
    federated_average,
    federated_round,
    federated_train,
    fewshot_adapt,
    reptile_meta_train,
)
from .losses import LossBreakdown, windfusion_loss
from .loop import History, Trainer, load_checkpoint, save_checkpoint, set_determinism
from .onboarding import OnboardingGates, OnboardingReport, onboard_site, scan_site

__all__ = [
    "CalibrationArtifact",
    "DistillTrainer",
    "DistillationWeights",
    "History",
    "LossBreakdown",
    "OnboardingGates",
    "OnboardingReport",
    "Trainer",
    "apply_calibration",
    "calibrate",
    "distillation_loss",
    "federated_average",
    "federated_round",
    "federated_train",
    "fewshot_adapt",
    "load_checkpoint",
    "onboard_site",
    "reptile_meta_train",
    "save_checkpoint",
    "scan_site",
    "set_determinism",
    "windfusion_loss",
]
