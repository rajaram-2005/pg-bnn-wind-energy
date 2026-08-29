"""Self-training onboarding for a new site, with explicit promotion gates.

Provenance: ``onboarding_confidence_gates`` (wind-turbine-pg-bnn
``src/agents/hermes.py``). The upstream agent self-trains with pseudo-labels;
the risk is silent degradation, so this re-implementation makes every gate and
every accepted pseudo-label auditable and refuses to promote a site model that
fails its held-out checks.

Nothing here claims accuracy on real data: the gates are thresholds on the
*validation* statistics of the site being onboarded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..config import WindFusionConfig
from .loop import _to_device


@dataclass
class OnboardingGates:
    """Acceptance thresholds. Defaults are conservative on purpose."""

    max_epistemic: float = 0.35  # pseudo-labels above this are rejected
    max_physics_residual: float = 0.60
    min_data_completeness: float = 0.60
    min_pseudo_labels: int = 16
    max_promotion_rmse: float = 0.35  # on the held-out gate set
    max_regression: float = 0.05  # promoted model must not regress by more than this


@dataclass
class OnboardingReport:
    """Audit trail for one onboarding session."""

    site_id: int | None = None
    observed: int = 0
    accepted: int = 0
    rejected_missing: int = 0
    rejected_uncertain: int = 0
    acceptance_rate: float = 0.0
    gate_rmse: float | None = None
    baseline_rmse: float | None = None
    promoted: bool = False
    reason: str = ""
    history: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "site_id": self.site_id,
            "observed": self.observed,
            "accepted": self.accepted,
            "rejected_missing": self.rejected_missing,
            "rejected_uncertain": self.rejected_uncertain,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "gate_rmse": None if self.gate_rmse is None else round(self.gate_rmse, 6),
            "baseline_rmse": None if self.baseline_rmse is None else round(self.baseline_rmse, 6),
            "promoted": self.promoted,
            "reason": self.reason,
            "history": self.history,
        }


@torch.no_grad()
def scan_site(
    model,
    loader,
    gates: OnboardingGates | None = None,
    device: str = "cpu",
    samples: int = 8,
    site_id: int | None = None,
) -> tuple[list[dict], OnboardingReport]:
    """Collect pseudo-labels whose uncertainty and physics consistency pass."""
    gates = gates or OnboardingGates()
    model = model.to(device).eval()
    accepted: list[dict] = []
    report = OnboardingReport(site_id=site_id)
    for batch in loader:
        batch = _to_device(batch, torch.device(device))
        prediction = model.predict(
            batch["sequence"], batch["physics"], batch.get("neighbors"), samples=samples
        )
        residuals = compute_residual_magnitude(model, batch)
        epistemic = prediction["epistemic"].mean(-1)
        completeness = batch.get(
            "data_completeness", torch.ones(batch["sequence"].shape[0], device=device)
        ).reshape(-1)
        for i in range(batch["sequence"].shape[0]):
            report.observed += 1
            if float(completeness[i]) < gates.min_data_completeness:
                report.rejected_missing += 1
                continue
            if float(epistemic[i]) > gates.max_epistemic or float(residuals[i]) > gates.max_physics_residual:
                report.rejected_uncertain += 1
                continue
            accepted.append(
                {
                    "sequence": batch["sequence"][i].cpu(),
                    "physics": batch["physics"][i].cpu(),
                    "target": prediction["mean"][i].cpu(),
                    "epistemic": float(epistemic[i]),
                }
            )
            report.accepted += 1
    report.acceptance_rate = report.accepted / max(report.observed, 1)
    return accepted, report


def compute_residual_magnitude(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Mean absolute physics residual per sample, using the model's own plant constants."""
    from ..physics.residuals import compute_residuals

    residuals = compute_residuals(batch["raw"], getattr(model, "turbine", None))
    return torch.stack([v.reshape(v.shape[0], -1).abs().mean(-1) for v in residuals.values()]).mean(0)


@torch.no_grad()
def gate_rmse(model, loader, device: str = "cpu") -> float:
    """RMSE on the held-out gate set (regression targets only)."""
    model = model.to(device).eval()
    total, count = 0.0, 0
    for batch in loader:
        batch = _to_device(batch, torch.device(device))
        outputs = model(batch["sequence"], batch["physics"], batch.get("neighbors"))
        diff = outputs["mean"] - batch["target"]
        total += float(diff.pow(2).sum())
        count += int(diff.numel())
    return (total / max(count, 1)) ** 0.5


def onboard_site(
    model,
    unlabelled_loader,
    gate_loader,
    config: WindFusionConfig,
    gates: OnboardingGates | None = None,
    device: str = "cpu",
    site_id: int | None = None,
) -> tuple[torch.nn.Module, OnboardingReport]:
    """Self-train on accepted pseudo-labels, then decide whether to promote."""
    import copy

    gates = gates or OnboardingGates()
    pseudo, report = scan_site(model, unlabelled_loader, gates, device, site_id=site_id)
    baseline = gate_rmse(model, gate_loader, device)
    report.baseline_rmse = baseline

    if len(pseudo) < gates.min_pseudo_labels:
        report.promoted = False
        report.reason = (
            f"only {len(pseudo)} confident windows (< {gates.min_pseudo_labels}); "
            "keep supervised labels or lower the gate explicitly"
        )
        return model, report

    candidate = copy.deepcopy(model).train()
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=config.training.learning_rate)
    from .losses import heteroscedastic_nll

    for _ in range(max(1, config.fleet.local_epochs)):
        for item in pseudo:
            outputs = candidate(
                item["sequence"].unsqueeze(0).to(device), item["physics"].unsqueeze(0).to(device)
            )
            loss = heteroscedastic_nll(
                outputs["mean"], outputs["log_var"], item["target"].unsqueeze(0).to(device)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    candidate_rmse = gate_rmse(candidate, gate_loader, device)
    report.gate_rmse = candidate_rmse
    if candidate_rmse > gates.max_promotion_rmse:
        report.promoted = False
        report.reason = f"gate RMSE {candidate_rmse:.4f} exceeds {gates.max_promotion_rmse}"
        return model, report
    if candidate_rmse > baseline + gates.max_regression:
        report.promoted = False
        report.reason = (
            f"candidate regressed vs baseline ({candidate_rmse:.4f} > {baseline:.4f} + {gates.max_regression})"
        )
        return model, report
    report.promoted = True
    report.reason = f"promoted: gate RMSE {candidate_rmse:.4f} vs baseline {baseline:.4f}"
    report.history.append({"accepted": len(pseudo), "rmse": round(candidate_rmse, 6)})
    return candidate, report


__all__ = ["OnboardingGates", "OnboardingReport", "gate_rmse", "onboard_site", "scan_site"]
