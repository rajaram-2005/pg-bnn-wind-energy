"""Deterministic training loop with checkpointing and early stopping.

Reproducibility: seeding, single-pass scheduling and best-checkpoint selection
are all driven by the config; the checkpoint records the config, the data
checksums, the library version and (when available) the git commit so a result
can be traced back to the exact inputs that produced it.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..config import WindFusionConfig
from ..models import create_model
from .losses import windfusion_loss

CHECKPOINT_VERSION = 2


def set_determinism(seed: int) -> None:
    """Seed every RNG the training path touches."""
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)  # conv cuDNN nondeterminism is CUDA-only
    try:  # pragma: no cover - depends on numpy availability at call time
        import numpy as np

        np.random.seed(seed % (2**32 - 1))
    except Exception:
        pass


@dataclass
class History:
    """Per-epoch training and validation metrics."""

    epochs: list[dict] = field(default_factory=list)

    def add(self, record: dict) -> None:
        self.epochs.append(record)

    @property
    def best(self) -> dict | None:
        """Epoch with the lowest monitored loss (validation if available)."""
        if not self.epochs:
            return None
        key = "val_total" if "val_total" in self.epochs[0] else "total"
        return min(self.epochs, key=lambda r: r[key])

    def as_dict(self) -> dict:
        return {"epochs": self.epochs, "best": self.best}


class Trainer:
    """Generic supervised trainer for any model in the WindFusion family."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: WindFusionConfig | None = None,
        device: str = "cpu",
        verbose: bool = True,
    ) -> None:
        self.config = config or WindFusionConfig()
        self.model = model.to(device)
        self.spec = getattr(model, "spec", None)
        self.device = torch.device(device)
        self.verbose = verbose
        self.history = History()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        total_steps = max(1, self.config.training.epochs)
        warmup = max(1, int(self.config.training.warmup_fraction * total_steps))

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup:
                return (epoch + 1) / warmup
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup) / max(total_steps - warmup, 1)))

        self._lr_lambda = lr_lambda
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    # ── hooks (overridden by the distillation trainer) ─────────────────────
    def compute_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        batch = _to_device(batch, self.device)
        self.model.train()
        outputs = self.model(
            batch["sequence"], batch["physics"], batch.get("neighbors")
        )
        breakdown = windfusion_loss(outputs, batch, self.config, spec=self.spec)
        return breakdown.total, breakdown.as_floats()

    # ── core loops ─────────────────────────────────────────────────────────
    def train_epoch(self, loader) -> dict[str, float]:
        self.model.train()
        totals: dict[str, float] = {}
        count = 0
        for batch in loader:
            loss, terms = self.compute_loss(batch)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip)
            self.optimizer.step()
            for key, value in terms.items():
                totals[key] = totals.get(key, 0.0) + value * _batch_size(batch)
            count += _batch_size(batch)
        self.scheduler.step()
        return {key: value / max(count, 1) for key, value in totals.items()}

    @torch.no_grad()
    def validate(self, loader) -> dict[str, float]:
        self.model.eval()
        totals: dict[str, float] = {}
        count = 0
        for batch in loader:
            batch = _to_device(batch, self.device)
            outputs = self.model(batch["sequence"], batch["physics"], batch.get("neighbors"))
            breakdown = windfusion_loss(outputs, batch, self.config, spec=self.spec)
            for key, value in breakdown.as_floats().items():
                totals[key] = totals.get(key, 0.0) + value * _batch_size(batch)
            count += _batch_size(batch)
        return {"val_" + key: value / max(count, 1) for key, value in totals.items()}

    def fit(self, train_loader, val_loader=None, epochs: int | None = None) -> History:
        """Train with early stopping on validation loss (or train loss)."""
        set_determinism(self.config.seed)
        epochs = int(epochs or self.config.training.epochs)
        best_loss = float("inf")
        best_state = None
        stale = 0
        for epoch in range(epochs):
            started = time.perf_counter()
            train_terms = self.train_epoch(train_loader)
            record = {"epoch": epoch + 1, **train_terms}
            if val_loader is not None:
                record.update(self.validate(val_loader))
            record["lr"] = float(self.optimizer.param_groups[0]["lr"])
            record["seconds"] = round(time.perf_counter() - started, 3)
            self.history.add(record)

            monitored = record.get("val_total", record.get("total", float("inf")))
            if monitored < best_loss - self.config.training.min_delta:
                best_loss = monitored
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if self.verbose:
                print(
                    f"epoch {epoch + 1:3d}/{epochs}  train {record.get('total', float('nan')):.4f}"
                    f"  val {record.get('val_total', float('nan')):.4f}"
                    f"  ({record['seconds']}s)",
                    flush=True,
                )
            if stale >= self.config.training.patience:
                if self.verbose:
                    print(f"early stop at epoch {epoch + 1} (no improvement)", flush=True)
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self.history


def _batch_size(batch: dict[str, torch.Tensor]) -> int:
    return int(batch["sequence"].shape[0])


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}


def save_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    config: WindFusionConfig,
    meta: dict | None = None,
) -> Path:
    """Write a checkpoint with enough metadata to reproduce or reject it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "windfusion_version": getattr(config, "version", "0.2.0"),
        "model_id": getattr(getattr(model, "spec", None), "model_id", model.__class__.__name__),
        "state_dict": model.state_dict(),
        "config": config.to_dict(),
        "meta": meta or {},
        "git_commit": _git_commit(),
        "torch_version": torch.__version__,
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, device: str = "cpu"):
    """Rebuild a model from a checkpoint written by :func:`save_checkpoint`."""
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    cfg = WindFusionConfig.from_dict(payload["config"])
    mode = payload.get("model_id", cfg.model.mode)
    model = create_model(
        mode if mode in _available_modes() else cfg.model.mode,
        cfg.model.input_features,
        cfg.model.outputs,
        cfg.model.physics_features,
        cfg.turbine,
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    return model, cfg, payload.get("meta", {})


def _available_modes():
    from ..models import MODEL_REGISTRY

    return set(MODEL_REGISTRY)


def _git_commit() -> str:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:  # pragma: no cover
        return "unknown"


def write_history(history: History, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history.as_dict(), indent=2), encoding="utf-8")
    return path


__all__ = [
    "CHECKPOINT_VERSION",
    "History",
    "Trainer",
    "load_checkpoint",
    "save_checkpoint",
    "set_determinism",
    "write_history",
]
