"""DataLoader construction with reproducibility guarantees.

Determinism notes
-----------------
* the torch generator is seeded from the config seed,
* shuffling is off for validation/test so metrics are stable,
* worker processes (if enabled) are seeded by torch itself from that generator.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import DataLoader

from ..config import DataConfig, TrainingConfig, TurbineConfig, WindFusionConfig
from .dataset import DatasetBundle, TurbineWindowDataset, build_datasets, load_fleet
from .synthetic import TurbineRun, fleet_summary


def _loader(
    dataset: TurbineWindowDataset, cfg: TrainingConfig, seed: int, shuffle: bool
) -> DataLoader:
    generator = None
    if shuffle:
        import torch

        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=cfg.num_workers,
        drop_last=False,
        pin_memory=False,
    )


@dataclass
class DataBundle:
    """Datasets, dataloaders and provenance metadata for one experiment."""

    datasets: DatasetBundle
    train: DataLoader
    val: DataLoader
    test: DataLoader
    meta: dict

    def summary(self) -> str:
        return (
            f"{self.meta['n_windows']} windows | turbines {self.meta['n_turbines']} | "
            f"fleet checksum {self.meta['fleet_checksum']}"
        )


def build_dataloaders(
    config: WindFusionConfig | None = None,
    runs: list[TurbineRun] | None = None,
    forecast_horizon: int = 0,
    neighbors: int = 0,
) -> DataBundle:
    """End-to-end data preparation: fleet -> splits -> datasets -> loaders."""
    config = config or WindFusionConfig()
    data_cfg: DataConfig = config.data
    runs = runs if runs is not None else load_fleet(data_cfg)
    bundle = build_datasets(
        runs,
        data_cfg,
        model_window=config.model.sequence_length,
        forecast_horizon=forecast_horizon,
        neighbors=neighbors,
        turbine=config.turbine,
    )
    train = _loader(bundle.train, config.training, data_cfg.seed, shuffle=True)
    val = _loader(bundle.val, config.training, data_cfg.seed, shuffle=False)
    test = _loader(bundle.test, config.training, data_cfg.seed, shuffle=False)
    meta = dict(bundle.meta)
    meta.update(
        {
            "batch_size": config.training.batch_size,
            "seed": data_cfg.seed,
            "fleet_summary": fleet_summary(runs),
        }
    )
    return DataBundle(datasets=bundle, train=train, val=val, test=test, meta=meta)


def per_turbine_loaders(
    dataset: TurbineWindowDataset, batch_size: int = 32, shuffle: bool = True
) -> dict[int, DataLoader]:
    """One loader per turbine (federated / fleet-learning clients)."""
    import torch
    from torch.utils.data import Subset

    by_turbine: dict[int, list[int]] = {}
    for i in range(len(dataset)):
        run_i, _ = dataset.index[i]
        tid = dataset.runs[run_i].turbine_id
        by_turbine.setdefault(tid, []).append(i)
    loaders = {}
    for tid, indices in by_turbine.items():
        loaders[tid] = DataLoader(
            Subset(dataset, indices), batch_size=batch_size, shuffle=shuffle
        )
    return loaders


__all__ = ["DataBundle", "build_dataloaders", "per_turbine_loaders"]
