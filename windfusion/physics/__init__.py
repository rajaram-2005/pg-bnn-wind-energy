"""Physics modules: differentiable relations, residuals and constraints.

Provenance: see ``windfusion/provenance.py`` (``heier_cp_betz``,
``jensen_wake``, ``iso281_l10``, ``lumped_thermal_rc``, ``soft_hinge_limits``).
"""

from . import aerodynamics, constraints, drivetrain, electrical, residuals, thermal
from .constraints import PhysicsWeights, physics_loss, soft_hinge
from .residuals import RESIDUAL_NAMES, compute_residuals, physics_feature_vector

__all__ = [
    "aerodynamics",
    "constraints",
    "drivetrain",
    "electrical",
    "residuals",
    "thermal",
    "PhysicsWeights",
    "physics_loss",
    "soft_hinge",
    "RESIDUAL_NAMES",
    "compute_residuals",
    "physics_feature_vector",
]
