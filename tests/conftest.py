"""Shared fixtures: a small deterministic fleet and a tiny experiment config."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from windfusion.config import (  # noqa: E402
    DataConfig,
    TrainingConfig,
    WindFusionConfig,
)
from windfusion.data import load_fleet  # noqa: E402


@pytest.fixture(scope="session")
def tiny_runs():
    """Six short turbine runs over three sites (fast, deterministic)."""
    return load_fleet(DataConfig(n_turbines=6, seq_len=360, seed=7, fleet_version="test-v1"))


@pytest.fixture(scope="session")
def tiny_config() -> WindFusionConfig:
    return WindFusionConfig(
        seed=7,
        model=__import__("windfusion.config", fromlist=["ModelConfig"]).ModelConfig(
            mode="windfusion-lite", input_features=12, physics_features=5, sequence_length=24, outputs=3
        ),
        data=DataConfig(
            source="synthetic",
            fleet_version="test-v1",
            n_turbines=6,
            seq_len=360,
            window=24,
            stride=12,
            seed=7,
            split={"train": 0.6, "val": 0.2, "test": 0.2},
        ),
        training=TrainingConfig(
            epochs=2,
            batch_size=16,
            target_weights={"failure_probability": 1.0, "health_index": 1.0, "rul_days": 1.0},
        ),
    )
