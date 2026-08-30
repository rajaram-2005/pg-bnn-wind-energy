"""AutoWind: the self-learning model — it trains on its own.

Where every other member of the family is fitted by the shared supervised
:func:`windfusion.training.loop.Trainer`, ``windfusion-auto`` carries its own
objectives and its own loop. Nothing external is required to bring it up:

* **masked-channel reconstruction** — random SCADA channels are hidden for the
  whole window and the model must infer them from the remaining channels, which
  ties the representation to cross-channel physical structure (power follows
  wind, temperature follows load, ...) instead of to label correlations;
* **next-step prediction** — every encoder position must predict the next
  timestep of every channel; the causal encoder makes this a legitimate task
  (position ``t`` cannot see ``t+1``);
* **physics-consistency self-check** — the self-check head is regressed onto
  the model's own physics-residual magnitudes, so the model learns to know when
  its view of the window is physically plausible, and the objective rewards
  windows the lumped physics agrees with;
* **gated pseudo-label rounds** — after a self-supervised round, the model
  labels the unlabelled windows it is confident about (mean MC-dropout
  epistemic below a gate, data completeness above floor), then trains on its
  own labels for a further round. This bootstrapping is deliberately
  conservative: rejected windows stay rejected, and the round reports how many
  pseudo-labels were accepted so degradation is visible, not silent.

The class overrides ``forward`` not at all: the trunk, the routing contract,
the evaluation path, the verifier and the export path work exactly as for any
other preset. The autonomy lives beside the shared contract, not instead of
it.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
from torch import nn

from ..config import SelfTrainingConfig
from ..physics.residuals import compute_residuals
from .base import WindFusionBase
from .registry import AUTOWIND
from .uncertainty import gaussian_nll

__all__ = ["AutoWind", "SelfLearningBreakdown"]


@dataclass
class SelfLearningBreakdown:
    """Every term of the autonomous objective, for logging and ablations."""

    total: torch.Tensor
    reconstruction: torch.Tensor
    next_step: torch.Tensor
    physics: torch.Tensor
    consistency: torch.Tensor
    router_balance: torch.Tensor
    pseudo_label: torch.Tensor = None

    def as_floats(self) -> dict[str, float]:
        data = {
            "total": self.total,
            "reconstruction": self.reconstruction,
            "next_step": self.next_step,
            "physics": self.physics,
            "consistency": self.consistency,
            "router_balance": self.router_balance,
        }
        if self.pseudo_label is not None:
            data["pseudo_label"] = self.pseudo_label
        return {key: float(value.detach()) for key, value in data.items()}


def _mean_magnitude(residuals: dict[str, torch.Tensor]) -> torch.Tensor:
    """Per-sample mean |residual| across all residual channels -> (B, 1)."""
    stacked = torch.stack(
        [v.reshape(v.shape[0], -1).abs().mean(-1) for v in residuals.values()], dim=-1
    )
    return stacked.mean(-1, keepdim=True)


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}


class AutoWind(WindFusionBase):
    """Self-learning preset: masked reconstruction + next-step + physics-consistency."""

    def __init__(
        self,
        input_features: int = 12,
        outputs: int = 3,
        physics_features: int = 5,
        turbine=None,
    ) -> None:
        super().__init__(AUTOWIND, input_features, outputs, physics_features, turbine)
        h = self.spec.hidden
        # Self-supervised heads. They never touch ``forward``: the export,
        # evaluation and verifier paths see the same contract as every other
        # model in the family.
        self.reconstruction_head = nn.Linear(h, self.input_features)
        self.next_step_head = nn.Linear(h, self.input_features)

    # ── self-supervised objectives ─────────────────────────────────────────
    def masked_reconstruction(
        self,
        sequence: torch.Tensor,
        physics: torch.Tensor,
        mask_rate: float = 0.15,
        neighbors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Hide whole channels for the full window and ask the trunk to infer them.

        Masked channels are set to zero (the post-normalisation mean), so the
        only way to recover the final-step value is the cross-channel physical
        structure the encoder learned.
        """
        clean = torch.nan_to_num(sequence)
        channel_mask = torch.rand(clean.shape[0], self.input_features, device=clean.device) < mask_rate
        masked = clean * (~channel_mask).unsqueeze(1).to(clean.dtype)
        out = self.forward(masked, physics, neighbors)
        prediction = self.reconstruction_head(out["representation"])
        if not bool(channel_mask.any()):  # last step of the masked channels only
            return clean.new_zeros(())
        return (prediction - clean[:, -1])[channel_mask].pow(2).mean()

    def next_step_prediction(
        self, sequence: torch.Tensor, physics: torch.Tensor
    ) -> torch.Tensor:
        """Every position must predict the next timestep of every channel."""
        clean = torch.nan_to_num(sequence)
        if clean.shape[1] < 2:
            return clean.new_zeros(())
        encoded = self.encoder(
            self.stem(clean) + self.physics_proj(torch.nan_to_num(physics)).unsqueeze(1)
        )
        prediction = self.next_step_head(encoded[:, :-1])
        return (prediction - clean[:, 1:]).pow(2).mean()

    def self_supervised_objective(
        self,
        batch: dict[str, torch.Tensor],
        config: SelfTrainingConfig | None = None,
    ) -> SelfLearningBreakdown:
        """The full autonomous objective for one batch of (labelled or not) windows."""
        sc = config or SelfTrainingConfig()
        sequence, physics = batch["sequence"], batch["physics"]
        neighbors = batch.get("neighbors")

        reconstruction = self.masked_reconstruction(sequence, physics, sc.mask_rate, neighbors)
        next_step = self.next_step_prediction(sequence, physics)

        raw = batch.get("raw")
        snapshot = raw if raw is not None else torch.nan_to_num(sequence[:, -1, :])
        residuals = compute_residuals(snapshot, self.turbine)
        physics_term = torch.stack(
            [v.reshape(v.shape[0], -1).pow(2).mean(-1) for v in residuals.values()]
        ).mean()

        out = self.forward(sequence, physics, neighbors)
        consistency = sequence.new_zeros(())
        if "consistency" in out:
            consistency = torch.nn.functional.mse_loss(
                out["consistency"], _mean_magnitude(residuals).detach()
            )
        routing = out["routing"].mean(0).clamp_min(1e-8)
        balance = (routing * routing.log()).sum()

        total = (
            sc.reconstruction_weight * reconstruction
            + sc.next_step_weight * next_step
            + sc.physics_weight * physics_term
            + sc.consistency_weight * consistency
            + sc.router_balance_weight * balance
        )
        return SelfLearningBreakdown(
            total=total,
            reconstruction=reconstruction,
            next_step=next_step,
            physics=physics_term,
            consistency=consistency,
            router_balance=balance,
        )

    # ── self-training: gated pseudo-labels on its own predictions ─────────
    @torch.no_grad()
    def collect_pseudo_labels(
        self, loader, config: SelfTrainingConfig | None = None, device: str = "cpu"
    ) -> dict[str, torch.Tensor] | None:
        """Label the windows the model itself trusts; the gate is auditable.

        A window is accepted when its mean epistemic uncertainty is at or below
        ``pseudo_label_gate`` and its data completeness is at or above 0.6 —
        the same floor the verifier uses. Refused windows simply do not enter
        the bootstrap; nothing here silences the statistics.
        """
        sc = config or SelfTrainingConfig()
        was_training = self.training
        self.eval()
        sequences, physics, targets = [], [], []
        observed = accepted = 0
        for batch in loader:
            batch = _to_device(batch, torch.device(device))
            prediction = self.predict(
                batch["sequence"], batch["physics"], batch.get("neighbors"), samples=sc.mc_samples
            )
            epistemic = prediction["epistemic"].mean(-1)
            completeness = batch.get("data_completeness")
            keep = epistemic <= sc.pseudo_label_gate
            if completeness is not None:
                keep = keep & (completeness.reshape(-1) >= 0.6)
            for i in range(keep.shape[0]):
                observed += 1
                if bool(keep[i]):
                    accepted += 1
                    sequences.append(batch["sequence"][i].cpu())
                    physics.append(batch["physics"][i].cpu())
                    targets.append(prediction["mean"][i].cpu())
        self.train(was_training)
        if not sequences:
            self._last_pseudo_stats = {"observed": observed, "accepted": 0}
            return None
        self._last_pseudo_stats = {
            "observed": observed,
            "accepted": accepted,
            "acceptance_rate": round(accepted / max(observed, 1), 4),
        }
        return {
            "sequence": torch.stack(sequences),
            "physics": torch.stack(physics),
            "target": torch.stack(targets),
        }

    def _pseudo_loss(self, pseudo: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
        n = pseudo["sequence"].shape[0]
        step = max(1, min(32, n))
        order = torch.randperm(n)
        losses = []
        for start in range(0, n, step):
            idx = order[start : start + step].to(device)
            out = self.forward(pseudo["sequence"][idx], pseudo["physics"][idx])
            losses.append(gaussian_nll(out["mean"], out["log_var"], pseudo["target"][idx]))
        return torch.stack(losses).mean()

    # ── the loop that makes it train on its own ────────────────────────────
    def autonomous_fit(
        self,
        train_loader,
        val_loader=None,
        config=None,
        rounds: int | None = None,
        epochs: int | None = None,
        device: str = "cpu",
        verbose: bool = True,
    ) -> dict:
        """Fit the model to unlabelled windows, then to its own confident labels.

        Round 1 is pure self-supervision. Every later round re-scores the
        training set, keeps only the pseudo-labels that pass the gate, and adds
        a weighted NLL term against them. Validation (when a loader is given)
        is measured with the self-supervised objective, so early stopping never
        peeks at ground-truth labels.
        """
        sc = self_training_from(config)
        self._last_pseudo_stats: dict = {"observed": 0, "accepted": 0}
        rounds = max(1, int(rounds if rounds is not None else sc.rounds))
        epochs = max(1, int(epochs if epochs is not None else sc.epochs))
        seed = getattr(config, "seed", 7)
        try:  # runtime import: keeps models/ free of a circular training import
            from ..training.loop import set_determinism

            set_determinism(int(seed))
        except Exception:  # noqa: BLE001 - determinism is best effort
            torch.manual_seed(int(seed))

        torch_device = torch.device(device)
        self.to(torch_device)

        history: list[dict] = []
        best: dict | None = None
        best_state: dict | None = None
        n_pseudo = 0

        for round_i in range(rounds):
            pseudo = None
            if round_i > 0 and sc.pseudo_label_weight > 0:
                collected = self.collect_pseudo_labels(train_loader, sc, device=device)
                if collected is not None:
                    pseudo = {k: v.to(torch_device) for k, v in collected.items()}
                    n_pseudo = int(pseudo["sequence"].shape[0])

            optimizer = torch.optim.AdamW(
                self.parameters(), lr=sc.learning_rate, weight_decay=sc.weight_decay
            )
            warmup = max(1, int(0.1 * epochs))

            def lr_lambda(epoch: int, total=epochs, w=warmup) -> float:
                if epoch < w:
                    return (epoch + 1) / w
                return 0.5 * (1 + math.cos(math.pi * (epoch - w) / max(total - w, 1)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            stale = 0
            for epoch in range(epochs):
                started = time.perf_counter()
                self.train()
                totals: dict[str, float] = {}
                count = 0
                for batch in train_loader:
                    batch = _to_device(batch, torch_device)
                    breakdown = self.self_supervised_objective(batch, sc)
                    loss = breakdown.total
                    if pseudo is not None:
                        pseudo_term = sc.pseudo_label_weight * self._pseudo_loss(pseudo, torch_device)
                        loss = loss + pseudo_term
                        totals["pseudo_label"] = totals.get("pseudo_label", 0.0) + float(
                            pseudo_term.detach()
                        ) * int(batch["sequence"].shape[0])
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), sc.grad_clip)
                    optimizer.step()
                    for key, value in breakdown.as_floats().items():
                        totals[key] = totals.get(key, 0.0) + value * int(batch["sequence"].shape[0])
                    count += int(batch["sequence"].shape[0])
                for key in list(totals):
                    totals[key] /= max(count, 1)
                scheduler.step()

                record = {"round": round_i + 1, "epoch": epoch + 1, **totals}
                record["lr"] = float(optimizer.param_groups[0]["lr"])
                record["seconds"] = round(time.perf_counter() - started, 3)
                if val_loader is not None:
                    record.update(self.autonomous_validate(val_loader, sc, device))
                history.append(record)

                monitored = record.get("val_total", record.get("total", float("inf")))
                if best is None or monitored < best.get("monitored", float("inf")) - sc.min_delta:
                    best = {**record, "monitored": monitored}
                    best_state = {
                        k: v.detach().clone() for k, v in self.state_dict().items()
                    }
                    stale = 0
                else:
                    stale += 1
                if verbose:
                    print(
                        f"[auto] round {round_i + 1}/{rounds} epoch {epoch + 1:3d}/{epochs} "
                        f"self-sup {record.get('total', float('nan')):.4f}"
                        f"  val {record.get('val_total', float('nan')):.4f}"
                        f"  pseudo {n_pseudo}",
                        flush=True,
                    )
                if stale >= sc.patience:
                    if verbose:
                        print(f"[auto] early stop at round {round_i + 1} epoch {epoch + 1}", flush=True)
                    break

        if best_state is not None:
            self.load_state_dict(best_state)
        if best is not None:
            best = {k: v for k, v in best.items() if k != "monitored"}
        return {
            "model_id": self.spec.model_id,
            "mode": "self-supervised + gated pseudo-labels",
            "rounds": rounds,
            "epochs": epochs,
            "pseudo_labels": n_pseudo,
            "pseudo_stats": dict(self._last_pseudo_stats),
            "best": best,
            "history": history,
            "parameters": self.parameter_count,
        }

    @torch.no_grad()
    def autonomous_validate(
        self, loader, config: SelfTrainingConfig | None = None, device: str = "cpu"
    ) -> dict[str, float]:
        """Self-supervised objective on held-out windows (no labels involved)."""
        sc = config or SelfTrainingConfig()
        was_training = self.training
        self.eval()
        totals: dict[str, float] = {}
        count = 0
        for batch in loader:
            batch = _to_device(batch, torch.device(device))
            breakdown = self.self_supervised_objective(batch, sc)
            for key, value in breakdown.as_floats().items():
                totals[key] = totals.get(key, 0.0) + value * int(batch["sequence"].shape[0])
            count += int(batch["sequence"].shape[0])
        self.train(was_training)
        return {
            "val_" + key: value / max(count, 1) for key, value in totals.items()
        }

    # ── introspection ──────────────────────────────────────────────────────
    def summary(self) -> dict[str, object]:
        info = super().summary()
        info["self_supervised"] = True
        info["autonomous_objective"] = [
            "masked_reconstruction",
            "next_step_prediction",
            "physics_consistency",
            "router_balance",
            "gated_pseudo_labels",
        ]
        return info


def self_training_from(config) -> SelfTrainingConfig:
    """Accept a SelfTrainingConfig, a full WindFusionConfig, or None."""
    if config is None:
        return SelfTrainingConfig()
    if isinstance(config, SelfTrainingConfig):
        return config
    section = getattr(config, "self_training", None)
    if isinstance(section, SelfTrainingConfig):
        return section
    return SelfTrainingConfig()


__all__ = ["AutoWind", "SelfLearningBreakdown", "self_training_from"]
