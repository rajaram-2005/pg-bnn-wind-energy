"""Provenance registry, configuration invariants and the safety lock."""

import pytest

from windfusion import provenance
from windfusion.config import (
    DataConfig,
    SafetyConfig,
    TrainingConfig,
    WindFusionConfig,
    default_config,
)


def test_registry_is_internally_consistent():
    assert provenance.validate_provenance() == []


def test_every_concept_declares_source_and_local_path():
    for concept in provenance.PROVENANCE:
        assert concept.source_repo
        assert concept.upstream_path
        assert concept.local_path
        assert concept.reuse in provenance.REUSE_KINDS


def test_no_source_tree_was_copied():
    """Upstream repositories are referenced, never vendored."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    forbidden = ("aerovigil_pg_bnn", "AeroZipSimulator.java", "aetheris/core")
    for name in forbidden:
        assert not (root / name).exists()


def test_provenance_table_lists_every_concept():
    table = provenance.provenance_table()
    for concept in provenance.PROVENANCE:
        assert concept.concept_id in table


def test_provenance_markdown_is_generated():
    markdown = provenance.provenance_markdown()
    assert "# Provenance registry" in markdown
    assert "wind-turbine-pg-bnn" in markdown


def test_safety_invariant_cannot_be_relaxed():
    with pytest.raises(ValueError):
        SafetyConfig(advisory_only=False)
    with pytest.raises(ValueError):
        SafetyConfig(allow_actuation=True)
    with pytest.raises(ValueError):
        WindFusionConfig.from_dict({"safety": {"allow_actuation": True}})


def test_default_config_is_advisory_only():
    config = default_config()
    assert config.safety.advisory_only is True
    assert config.safety.allow_actuation is False


def test_unknown_config_keys_are_rejected():
    with pytest.raises(ValueError):
        WindFusionConfig.from_dict({"not_a_section": 1})
    with pytest.raises(ValueError):
        WindFusionConfig.from_dict({"training": {"not_a_field": 1}})


def test_dotted_overrides_and_round_trip():
    config = WindFusionConfig()
    updated = config.with_overrides(**{"training.epochs": 3, "model.mode": "odin-wind"})
    assert updated.training.epochs == 3
    assert updated.model.mode == "odin-wind"
    assert WindFusionConfig.from_dict(updated.to_dict()).model.mode == "odin-wind"


def test_config_yaml_round_trip(tmp_path):
    path = tmp_path / "config.yaml"
    config = WindFusionConfig(data=DataConfig(n_turbines=4), training=TrainingConfig(epochs=2))
    config.save(path)
    reloaded = WindFusionConfig.from_yaml(path)
    assert reloaded.data.n_turbines == 4
    assert reloaded.training.epochs == 2
