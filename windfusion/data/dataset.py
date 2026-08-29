"""Windowed turbine datasets with group (turbine-level) splits.

Splitting discipline: **no turbine appears in two splits.** A random row-level
split would leak the latent damage state of a turbine into its own test set and
inflate every metric, so windows are grouped by turbine and, for the fleet
protocol, by site.

Provenance: ``data_zone_layout`` (ai-machinery-etl-pipeline),
``fault_taxonomy_vocabulary`` (wind-turbine-pg-bnn).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import DataConfig, TurbineConfig
from .schema import CHANNELS, CHANNEL_INDEX, N_CHANNELS, RUL_SCALE_DAYS
from .synthetic import SyntheticFleetConfig, TurbineRun, fleet_checksum, generate_fleet

# Scales used to build the dimensionless damage proxy (research defaults).
PROXY_SCALES = {"vibration_rms": 4.5, "gearbox_oil_temp_c": 80.0, "main_bearing_temp_c": 95.0}


@dataclass
class NormalisationStats:
    """Robust (median / IQR) statistics fitted on the training split only."""

    median: np.ndarray
    scale: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.median) / self.scale

    def transform_columns(self, x: np.ndarray, columns: list[int]) -> np.ndarray:
        # Normalise a matrix whose columns are a subset of the channels.
        cols = np.asarray(columns, dtype=int)
        return (x - self.median[cols]) / self.scale[cols]

    def to_dict(self) -> dict:
        return {"median": self.median.tolist(), "scale": self.scale.tolist()}


def fit_stats(runs: list[TurbineRun]) -> NormalisationStats:
    """Fit robust statistics on the given runs (never on validation/test)."""
    stacked = np.concatenate([r.channels for r in runs], axis=0)
    median = np.nanmedian(stacked, axis=0)
    q75 = np.nanpercentile(stacked, 75, axis=0)
    q25 = np.nanpercentile(stacked, 25, axis=0)
    iqr = np.maximum(q75 - q25, 1e-3)
    return NormalisationStats(median=median.astype(np.float32), scale=iqr.astype(np.float32))


def _forward_fill(x: np.ndarray) -> np.ndarray:
    """Fill NaNs forward, then backward, then with zero (channel-wise)."""
    out = x.copy()
    for c in range(out.shape[1]):
        col = out[:, c]
        mask = np.isnan(col)
        if not mask.any():
            continue
        idx = np.where(~mask, np.arange(len(col)), 0)
        np.maximum.accumulate(idx, out=idx)
        col = col[idx]
        col = np.where(np.isnan(col), 0.0, col)
        out[:, c] = col
    return np.nan_to_num(out, nan=0.0)


def damage_proxy(raw_last: np.ndarray) -> float:
    """Dimensionless damage proxy from a raw SI snapshot (monotonicity target)."""
    vib = raw_last[CHANNEL_INDEX["vibration_rms"]] / PROXY_SCALES["vibration_rms"]
    oil = raw_last[CHANNEL_INDEX["gearbox_oil_temp_c"]] / PROXY_SCALES["gearbox_oil_temp_c"]
    bearing = raw_last[CHANNEL_INDEX["main_bearing_temp_c"]] / PROXY_SCALES["main_bearing_temp_c"]
    return float(0.5 * max(vib, 0.0) + 0.3 * max(oil, 0.0) + 0.2 * max(bearing, 0.0))


class TurbineWindowDataset(Dataset):
    """Sliding-window dataset over a list of turbine runs."""

    def __init__(
        self,
        runs: list[TurbineRun],
        stats: NormalisationStats,
        window: int = 24,
        stride: int = 6,
        turbine: TurbineConfig | None = None,
        forecast_horizon: int = 0,
        neighbors: int = 0,
        name: str = "train",
    ) -> None:
        if window <= 0 or stride <= 0:
            raise ValueError("window and stride must be positive")
        self.runs = runs
        self.stats = stats
        self.window = int(window)
        self.stride = int(stride)
        self.turbine = turbine or TurbineConfig()
        self.forecast_horizon = int(forecast_horizon)
        self.neighbors = int(neighbors)
        self.name = name
        self.run_index = {r.turbine_id: r for r in runs}
        self.index: list[tuple[int, int]] = []
        for run_i, run in enumerate(runs):
            max_start = run.seq_len - self.window - max(self.forecast_horizon, 0)
            for start in range(0, max(0, max_start) + 1, self.stride):
                self.index.append((run_i, start))

    def __len__(self) -> int:
        return len(self.index)

    def _neighbor_window(self, run: TurbineRun, start: int) -> np.ndarray:
        k = self.neighbors
        out = np.zeros((max(k, 0), N_CHANNELS), dtype=np.float32)
        if k <= 0:
            return out
        for slot, nid in enumerate(run.neighbor_ids[:k]):
            neighbour = self.run_index.get(nid)
            if neighbour is None or start + self.window > neighbour.seq_len:
                continue
            snapshot = neighbour.channels[start + self.window - 1]
            out[slot] = self.stats.transform(_forward_fill(snapshot[None, :]))[0]
        return out

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        run_i, start = self.index[i]
        run = self.runs[run_i]
        end = start + self.window
        block = run.channels[start:end]
        missing = np.isnan(block)
        filled = _forward_fill(block)
        sequence = self.stats.transform(filled)
        raw_last = run.channels[end - 1].copy()

        from ..physics.residuals import physics_feature_vector  # lazy: physics imports data.schema

        with torch.no_grad():
            physics = physics_feature_vector(
                torch.from_numpy(raw_last[None, :].copy()), self.turbine
            )[0]

        item = {
            "sequence": torch.from_numpy(sequence.astype(np.float32)),
            "missing_mask": torch.from_numpy(missing.astype(np.float32)),
            "physics": physics.to(torch.float32),
            "target": torch.from_numpy(run.targets[end - 1].astype(np.float32)),
            "raw": torch.from_numpy(np.nan_to_num(raw_last, nan=0.0).astype(np.float32)),
            "damage_proxy": torch.tensor([damage_proxy(np.nan_to_num(raw_last, nan=0.0))], dtype=torch.float32),
            "turbine_id": torch.tensor(run.turbine_id, dtype=torch.long),
            "site_id": torch.tensor(run.site_id, dtype=torch.long),
            "data_completeness": torch.tensor([1.0 - float(missing.mean())], dtype=torch.float32),
        }
        if self.forecast_horizon:
            future = run.channels[end : end + self.forecast_horizon][
                :, [CHANNEL_INDEX["wind_speed"], CHANNEL_INDEX["active_power_kw"]]
            ]
            if future.shape[0] < self.forecast_horizon:  # pad at the end of a run
                pad = np.repeat(future[-1:], self.forecast_horizon - future.shape[0], axis=0)
                future = np.concatenate([future, pad], axis=0)
            item["future"] = torch.from_numpy(
                self.stats.transform_columns(
                    _forward_fill(future),
                    [CHANNEL_INDEX["wind_speed"], CHANNEL_INDEX["active_power_kw"]],
                ).astype(np.float32)
            )
        if self.neighbors:
            item["neighbors"] = torch.from_numpy(self._neighbor_window(run, start))
        return item

    # ── introspection ──────────────────────────────────────────────────────
    def turbine_ids(self) -> list[int]:
        return sorted({self.runs[run_i].turbine_id for run_i, _ in self.index})

    def fault_histogram(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for run_i, _ in self.index:
            mode = self.runs[run_i].fault_mode
            counts[mode] = counts.get(mode, 0) + 1
        return counts

    def checksum(self) -> str:
        h = hashlib.sha256()
        h.update(fleet_checksum(self.runs).encode())
        h.update(f"{self.window}:{self.stride}:{self.name}".encode())
        return h.hexdigest()[:16]


@dataclass
class DatasetBundle:
    """Train/validation/test datasets plus the provenance needed to reproduce."""

    train: TurbineWindowDataset
    val: TurbineWindowDataset
    test: TurbineWindowDataset
    stats: NormalisationStats
    meta: dict = field(default_factory=dict)

    def checksums(self) -> dict[str, str]:
        return {name: getattr(self, name).checksum() for name in ("train", "val", "test")}


def group_split(
    runs: list[TurbineRun], fractions: dict[str, float], seed: int = 7
) -> dict[str, list[TurbineRun]]:
    """Deterministic turbine-level split (no turbine crosses splits)."""
    total = sum(fractions.values())
    if total <= 0:
        raise ValueError("split fractions must sum to a positive number")
    ids = [r.turbine_id for r in runs]
    rng = np.random.default_rng(np.random.SeedSequence([seed, 777]))
    order = rng.permutation(ids)
    by_id = {r.turbine_id: r for r in runs}
    out: dict[str, list[TurbineRun]] = {}
    cursor = 0
    names = list(fractions)
    for i, name in enumerate(names):
        share = fractions[name] / total
        count = int(round(share * len(order))) if i < len(names) - 1 else len(order) - cursor
        chunk = order[cursor : cursor + count]
        out[name] = [by_id[int(t)] for t in chunk]
        cursor += count
    return out


def build_datasets(
    runs: list[TurbineRun],
    data_cfg: DataConfig | None = None,
    model_window: int | None = None,
    forecast_horizon: int = 0,
    neighbors: int = 0,
    turbine: TurbineConfig | None = None,
) -> DatasetBundle:
    """Split runs, fit normalisation on train only and build the three datasets."""
    cfg = data_cfg or DataConfig()
    splits = group_split(runs, cfg.split, cfg.seed)
    stats = fit_stats(splits["train"])
    window = int(model_window or cfg.window)
    common = dict(
        stats=stats,
        window=window,
        stride=cfg.stride,
        turbine=turbine,
        forecast_horizon=forecast_horizon,
        neighbors=neighbors,
    )
    bundle = DatasetBundle(
        train=TurbineWindowDataset(splits["train"], name="train", **common),
        val=TurbineWindowDataset(splits["val"], name="val", **common),
        test=TurbineWindowDataset(splits["test"], name="test", **common),
        stats=stats,
    )
    bundle.meta = {
        "dataset": "synthetic",
        "fleet_version": cfg.fleet_version,
        "fleet_checksum": fleet_checksum(runs),
        "window": window,
        "stride": cfg.stride,
        "n_turbines": {
            name: len({r.turbine_id for r in split}) for name, split in splits.items()
        },
        "n_windows": {name: len(ds) for name, ds in bundle.checksums().items()},
        "checksums": bundle.checksums(),
        "targets_rul_scale_days": RUL_SCALE_DAYS,
        "channels": list(CHANNELS),
        "warning_horizon_days": cfg.horizon_days,
        "note": "SYNTHETIC data: simulated fleet, not field measurements.",
    }
    return bundle


def load_fleet(cfg: DataConfig | None = None) -> list[TurbineRun]:
    """Generate (or later: load) the fleet described by ``cfg``."""
    cfg = cfg or DataConfig()
    if cfg.source != "synthetic":
        raise ValueError(
            f"data source {cfg.source!r} is not bundled with v0.2; "
            "implement an adapter in windfusion/data/loaders.py and record its licence"
        )
    return generate_fleet(
        SyntheticFleetConfig(
            n_turbines=cfg.n_turbines,
            seq_len=cfg.seq_len,
            sample_interval_s=cfg.sample_interval_s,
            seed=cfg.seed,
            fleet_version=cfg.fleet_version,
            missing_rate=cfg.missing_rate,
            horizon_days=cfg.horizon_days,
        )
    )


__all__ = [
    "DatasetBundle",
    "NormalisationStats",
    "TurbineWindowDataset",
    "build_datasets",
    "damage_proxy",
    "fit_stats",
    "group_split",
    "load_fleet",
]
