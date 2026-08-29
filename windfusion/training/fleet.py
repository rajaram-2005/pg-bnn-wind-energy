"""Fleet learning: federated averaging and few-shot site adaptation.

Two protocols are provided, both re-implemented locally (no Flower runtime):

* :func:`federated_average` — FedAvg over per-turbine or per-site clients with
  optional physics-aware weighting (upstream ``federated.physics_aware_aggregation``).
* :func:`reptile_meta_train` / :func:`fewshot_adapt` — first-order Reptile
  meta-learning so a *new, unseen site* adapts from a handful of labelled
  windows (upstream ``src/meta/reptile.py``).

Provenance: ``federated_physics_weighted_average``, ``reptile_site_adaptation``
(wind-turbine-pg-bnn).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..config import FleetConfig, WindFusionConfig
from ..physics.residuals import compute_residuals
from .loop import _to_device


# ── federated averaging ────────────────────────────────────────────────────


def client_weight(model, loader, device: str = "cpu", physics_aware: bool = True) -> float:
    """Aggregation weight: sample count, discounted by physics inconsistency."""
    n = 0
    residual = 0.0
    model = model.to(device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, torch.device(device))
            n += int(batch["sequence"].shape[0])
            if physics_aware:
                res = compute_residuals(batch["raw"], getattr(model, "turbine", None))
                residual += float(
                    torch.stack([v.pow(2).mean() for v in res.values()]).mean()
                ) * int(batch["sequence"].shape[0])
    model.train(was_training)
    if not physics_aware or n == 0:
        return float(max(n, 1))
    consistency = 1.0 / (1.0 + residual / max(n, 1))
    return float(max(n, 1)) * float(consistency)


def federated_average(state_dicts: list[dict], weights: list[float] | None = None) -> dict:
    """Weighted parameter average (FedAvg)."""
    if not state_dicts:
        raise ValueError("no client updates to aggregate")
    weights = weights or [1.0] * len(state_dicts)
    total = float(sum(weights)) or 1.0
    averaged = {}
    for key in state_dicts[0]:
        stacked = torch.stack([sd[key].float() for sd in state_dicts])
        w = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device).view(
            (-1,) + (1,) * (stacked.dim() - 1)
        )
        averaged[key] = ((stacked * w).sum(0) / total).to(state_dicts[0][key].dtype)
    return averaged


@dataclass
class FleetRoundReport:
    round: int
    clients: int
    weights: list[float]
    mean_client_loss: float

    def as_dict(self) -> dict:
        return {
            "round": self.round,
            "clients": self.clients,
            "weights": [round(w, 4) for w in self.weights],
            "mean_client_loss": round(self.mean_client_loss, 6),
        }


def federated_round(
    model: torch.nn.Module,
    client_loaders: list,
    config: WindFusionConfig,
    device: str = "cpu",
    local_epochs: int = 1,
) -> FleetRoundReport:
    """One FedAvg round: clone, train locally on each client, aggregate."""
    from .loop import Trainer

    global_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    states, weights, losses = [], [], []
    for loader in client_loaders:
        client = copy.deepcopy(model).to(device)
        client.load_state_dict(global_state)
        trainer = Trainer(client, config, device=device, verbose=False)
        trainer.fit(loader, None, epochs=local_epochs)
        states.append({k: v.detach().clone() for k, v in client.state_dict().items()})
        weights.append(
            client_weight(client, loader, device, config.fleet.physics_aware_aggregation)
        )
        losses.append(trainer.history.epochs[-1]["total"] if trainer.history.epochs else float("nan"))
    model.load_state_dict(federated_average(states, weights))
    return FleetRoundReport(
        round=0,
        clients=len(client_loaders),
        weights=weights,
        mean_client_loss=float(sum(losses) / max(len(losses), 1)),
    )


def federated_train(
    model: torch.nn.Module,
    client_loaders: list,
    config: WindFusionConfig,
    device: str = "cpu",
    rounds: int | None = None,
) -> list[FleetRoundReport]:
    """Run several FedAvg rounds and return the per-round report."""
    fleet: FleetConfig = config.fleet
    rounds = int(rounds or fleet.rounds)
    reports = []
    for r in range(rounds):
        clients = client_loaders[: max(fleet.clients_per_round, 1)] if fleet.clients_per_round else client_loaders
        if fleet.clients_per_round:
            start = (r * fleet.clients_per_round) % max(len(client_loaders), 1)
            clients = [
                client_loaders[(start + i) % len(client_loaders)]
                for i in range(min(fleet.clients_per_round, len(client_loaders)))
            ]
        report = federated_round(model, clients, config, device, fleet.local_epochs)
        report.round = r + 1
        reports.append(report)
    return reports


# ── Reptile meta-learning and few-shot adaptation ──────────────────────────


def reptile_step(
    model: torch.nn.Module,
    loader,
    inner_lr: float,
    steps: int,
    config,
    device: str = "cpu",
    meta_lr: float | None = None,
):
    """One first-order Reptile update: adapt on a task, then interpolate back."""
    from .losses import windfusion_loss

    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=inner_lr)
    start = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.train()
    iterator = iter(loader)
    for _ in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = _to_device(batch, torch.device(device))
        outputs = model(batch["sequence"], batch["physics"], batch.get("neighbors"))
        loss = windfusion_loss(outputs, batch, config, spec=getattr(model, "spec", None)).total
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    epsilon = float(meta_lr if meta_lr is not None else getattr(config.fleet, "reptile_meta_lr", 0.4))
    # Reptile: theta <- theta + epsilon * (theta_adapted - theta)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.is_floating_point() and name in start:
                param.copy_(start[name] * (1.0 - epsilon) + param.detach() * epsilon)
    return model


def reptile_meta_train(
    model: torch.nn.Module,
    task_loaders: list,
    config: WindFusionConfig,
    device: str = "cpu",
    iterations: int | None = None,
) -> torch.nn.Module:
    """Reptile across site-tasks so later sites adapt from few windows."""
    fleet: FleetConfig = config.fleet
    iterations = int(iterations or fleet.reptile_meta_iterations)
    for _ in range(iterations):
        for loader in task_loaders:
            reptile_step(
                model,
                loader,
                fleet.reptile_inner_lr,
                fleet.reptile_inner_steps,
                config,
                device,
                meta_lr=fleet.reptile_meta_lr,
            )
    return model


def fewshot_adapt(
    model: torch.nn.Module,
    loader,
    config: WindFusionConfig,
    steps: int | None = None,
    lr: float | None = None,
    device: str = "cpu",
) -> torch.nn.Module:
    """Fine-tune a copy of ``model`` on a small support set from a new site."""
    from .losses import windfusion_loss

    fleet: FleetConfig = config.fleet
    steps = int(steps or fleet.fewshot_steps)
    lr = float(lr if lr is not None else fleet.fewshot_lr)
    adapted = copy.deepcopy(model).to(device).train()
    optimizer = torch.optim.AdamW(adapted.parameters(), lr=lr)
    iterator = iter(loader)
    for _ in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = _to_device(batch, torch.device(device))
        outputs = adapted(batch["sequence"], batch["physics"], batch.get("neighbors"))
        loss = windfusion_loss(outputs, batch, config, spec=getattr(adapted, "spec", None)).total
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return adapted


def support_loader(dataset, n_windows: int, batch_size: int = 16, seed: int = 0):
    """Deterministic small support set for few-shot adaptation."""
    from torch.utils.data import DataLoader, Subset

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:n_windows].tolist()
    return DataLoader(Subset(dataset, indices), batch_size=min(batch_size, max(len(indices), 1)), shuffle=False)


def pairwise_distance(state_a: dict, state_b: dict) -> float:
    """L2 distance between two state dicts (used to report adaptation drift)."""
    total = 0.0
    for key in state_a:
        diff = state_a[key].float() - state_b[key].float()
        total += float(diff.pow(2).sum())
    return total**0.5


__all__ = [
    "FleetRoundReport",
    "client_weight",
    "federated_average",
    "federated_round",
    "federated_train",
    "fewshot_adapt",
    "pairwise_distance",
    "reptile_meta_train",
    "reptile_step",
    "support_loader",
]
