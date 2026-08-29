"""Data layer: schema, deterministic synthetic fleet, windowed datasets."""

from .dataset import (
    DatasetBundle,
    NormalisationStats,
    TurbineWindowDataset,
    build_datasets,
    damage_proxy,
    fit_stats,
    group_split,
    load_fleet,
)
from .schema import (
    CHANNEL_INDEX,
    CHANNEL_SPECS,
    CHANNELS,
    FAULT_MODES,
    N_CHANNELS,
    N_PHYSICS,
    N_TARGETS,
    PHYSICS_FEATURES,
    RUL_SCALE_DAYS,
    TARGETS,
)
from .synthetic import (
    SyntheticFleetConfig,
    TurbineRun,
    config_checksum,
    fleet_checksum,
    fleet_summary,
    generate_fleet,
)

__all__ = [
    "CHANNEL_INDEX",
    "CHANNEL_SPECS",
    "CHANNELS",
    "DatasetBundle",
    "FAULT_MODES",
    "N_CHANNELS",
    "N_PHYSICS",
    "N_TARGETS",
    "NormalisationStats",
    "PHYSICS_FEATURES",
    "RUL_SCALE_DAYS",
    "SyntheticFleetConfig",
    "TARGETS",
    "TurbineRun",
    "TurbineWindowDataset",
    "build_datasets",
    "config_checksum",
    "damage_proxy",
    "fit_stats",
    "fleet_checksum",
    "fleet_summary",
    "generate_fleet",
    "group_split",
    "load_fleet",
]
