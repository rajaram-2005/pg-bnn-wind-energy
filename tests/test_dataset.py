"""Data layer: determinism, group splits, normalisation discipline."""

import numpy as np

from windfusion.config import DataConfig
from windfusion.data import (
    TurbineWindowDataset,
    build_datasets,
    fit_stats,
    fleet_checksum,
    generate_fleet,
    group_split,
)
from windfusion.data.dataset import damage_proxy
from windfusion.data.synthetic import SyntheticFleetConfig
from windfusion.data.schema import CHANNELS, FAULT_MODES, N_CHANNELS


def _split_counts(bundle):
    return {
        name: sorted({r.turbine_id for r in getattr(bundle, name).runs})
        for name in ("train", "val", "test")
    }


def test_generator_is_deterministic_for_a_given_config():
    cfg = SyntheticFleetConfig(n_turbines=4, seq_len=200, seed=3)
    a = generate_fleet(cfg)
    b = generate_fleet(cfg)
    assert fleet_checksum(a) == fleet_checksum(b)
    assert np.allclose(np.nan_to_num(a[0].channels), np.nan_to_num(b[0].channels))


def test_different_seed_changes_the_fleet():
    a = generate_fleet(SyntheticFleetConfig(n_turbines=3, seq_len=120, seed=1))
    b = generate_fleet(SyntheticFleetConfig(n_turbines=3, seq_len=120, seed=2))
    assert fleet_checksum(a) != fleet_checksum(b)


def test_channels_are_within_the_plausibility_envelope():
    runs = generate_fleet(SyntheticFleetConfig(n_turbines=3, seq_len=200, seed=5))
    data = np.concatenate([r.channels for r in runs])
    assert data.shape[1] == N_CHANNELS == len(CHANNELS)
    assert np.isfinite(data[~np.isnan(data)]).all()
    assert np.nanmin(data[:, CHANNELS.index("wind_speed")]) >= 0.0
    assert np.nanmax(data[:, CHANNELS.index("wind_speed")]) <= 40.0


def test_fault_modes_are_assigned_from_the_taxonomy():
    runs = generate_fleet(SyntheticFleetConfig(n_turbines=10, seq_len=60, seed=7))
    assert {r.fault_mode for r in runs} <= set(FAULT_MODES)


def test_group_split_never_shares_a_turbine():
    runs = generate_fleet(SyntheticFleetConfig(n_turbines=12, seq_len=60, seed=7))
    splits = group_split(runs, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=7)
    sets = {name: {r.turbine_id for r in split} for name, split in splits.items()}
    assert not (sets["train"] & sets["val"])
    assert not (sets["train"] & sets["test"])
    assert not (sets["val"] & sets["test"])
    assert sets["train"] | sets["val"] | sets["test"] == {r.turbine_id for r in runs}


def test_normalisation_is_fitted_on_train_only(bundle=None):
    runs = generate_fleet(SyntheticFleetConfig(n_turbines=9, seq_len=300, seed=11))
    bundle = build_datasets(runs, DataConfig(n_turbines=9, seq_len=300, window=24, stride=24))
    counts = _split_counts(bundle)
    assert not (set(counts["train"]) & set(counts["test"]))
    stats = fit_stats(bundle.train.runs)
    assert stats.median.shape[0] == N_CHANNELS
    assert float(np.median(stats.scale)) > 0


def test_windows_carry_every_contract_field(tiny_runs):
    bundle = build_datasets(tiny_runs, DataConfig(n_turbines=6, seq_len=360, window=24, stride=24),
                            forecast_horizon=6, neighbors=2)
    item = bundle.train[0]
    assert item["sequence"].shape == (24, N_CHANNELS)
    assert item["physics"].shape == (5,)
    assert item["target"].shape == (3,)
    assert item["future"].shape == (6, 2)
    assert item["neighbors"].shape == (2, N_CHANNELS)
    assert 0.0 <= float(item["data_completeness"]) <= 1.0


def test_dataset_is_finite_after_imputation(tiny_runs):
    bundle = build_datasets(tiny_runs, DataConfig(n_turbines=6, seq_len=360, window=24, stride=24))
    for i in range(min(len(bundle.test), 8)):
        item = bundle.test[i]
        assert bool(item["sequence"].isfinite().all())


def test_damage_proxy_is_monotone_in_vibration():
    low = np.zeros(12)
    low[4] = 1.0
    high = np.zeros(12)
    high[4] = 8.0
    assert damage_proxy(high) > damage_proxy(low)


def test_checksum_is_stable_across_dataset_rebuilds(tiny_runs):
    cfg = DataConfig(n_turbines=6, seq_len=360, window=24, stride=24)
    first = build_datasets(tiny_runs, cfg)
    second = build_datasets(tiny_runs, cfg)
    assert first.checksums() == second.checksums()


def test_metadata_records_provenance(tiny_runs):
    bundle = build_datasets(tiny_runs, DataConfig(n_turbines=6, seq_len=360, window=24, stride=24))
    meta = bundle.meta
    assert meta["dataset"] == "synthetic"
    assert "SYNTHETIC" in meta["note"]
    assert meta["fleet_checksum"]
    assert meta["channels"] == list(CHANNELS)
