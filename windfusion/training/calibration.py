"""Post-hoc calibration: temperature scaling and conformal intervals.

MC dropout is an *approximation* to Bayesian inference, so the raw predictive
variance is not automatically calibrated. Two cheap corrections are applied and
then **measured**, never assumed:

* temperature scaling — a per-output scalar fitted on the validation split,
* conformal intervals — an empirical residual quantile that guarantees the
  requested marginal coverage on the calibration split, distribution-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..models.uncertainty import gaussian_nll


@dataclass
class CalibrationArtifact:
    """Fitted calibration parameters plus the evidence that they helped."""

    temperature: list[float] = field(default_factory=list)
    conformal_quantile: list[float] = field(default_factory=list)
    coverage: float = 0.9
    n_calibration: int = 0
    ece_before: float | None = None
    ece_after: float | None = None
    nll_before: float | None = None
    nll_after: float | None = None

    def as_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "conformal_quantile": self.conformal_quantile,
            "coverage": self.coverage,
            "n_calibration": self.n_calibration,
            "ece_before": self.ece_before,
            "ece_after": self.ece_after,
            "nll_before": self.nll_before,
            "nll_after": self.nll_after,
        }


def _collect(model, loader, device: str = "cpu", mc_samples: int = 0):
    """Run the model over a loader and return (mean, log_var, target)."""
    means, log_vars, targets = [], [], []
    model = model.to(device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            if mc_samples:
                out = model.predict(
                    batch["sequence"], batch["physics"], batch.get("neighbors"), samples=mc_samples
                )
                means.append(out["mean"].cpu())
                log_vars.append(out["log_var"].cpu())
            else:
                out = model(batch["sequence"], batch["physics"], batch.get("neighbors"))
                means.append(out["mean"].cpu())
                log_vars.append(out["log_var"].cpu())
            targets.append(batch["target"].cpu())
    model.train(was_training)
    return torch.cat(means), torch.cat(log_vars), torch.cat(targets)


def expected_calibration_error(
    y_true: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, n_bins: int = 10
) -> float:
    """ECE over central Gaussian intervals (no SciPy dependency)."""
    errors = []
    ps = torch.linspace(1 / n_bins, 1 - 1 / n_bins, n_bins - 1)
    # Inverse normal CDF (Acklam-style rational approximation, |err| < 1e-4).
    z = _norm_ppf(0.5 + ps / 2.0)
    for p, zp in zip(ps.tolist(), z.tolist()):
        lo, hi = mean - zp * std, mean + zp * std
        covered = ((y_true >= lo) & (y_true <= hi)).float().mean()
        errors.append(abs(float(covered) - p))
    return float(sum(errors) / max(len(errors), 1))


def _norm_ppf(p: torch.Tensor) -> torch.Tensor:
    """Beasley-Springer-Moro style approximation of the standard normal inverse CDF."""
    p = p.clamp(1e-6, 1 - 1e-6)
    t = torch.sqrt(-2.0 * torch.log(1.0 - p))
    c0 = [2.515517, 0.802853, 0.010328]
    c1 = [1.432788, 0.189269, 0.001308]
    num = c0[0] + c0[1] * t + c0[2] * t.pow(2)
    den = 1.0 + c1[0] * t + c1[1] * t.pow(2) + c1[2] * t.pow(3)
    return t - num / den


def fit_temperature(log_var: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """Closed-form Gaussian temperature scaling: sigma_new² = mean(residual²)."""
    var = log_var.exp().mean(0)
    target_var = residual.pow(2).mean(0)
    scale = (target_var / var.clamp_min(1e-8)).clamp(1e-4, 1e4)
    return scale.sqrt()


def fit_conformal(mean: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor, coverage: float = 0.9) -> torch.Tensor:
    """Empirical quantile of |residual| / sigma at the requested coverage."""
    std = log_var.exp().sqrt()
    ratio = (target - mean).abs() / std.clamp_min(1e-6)
    n = ratio.shape[0]
    q = min(max(coverage, 0.0), 1.0)
    index = min(int(torch.ceil(torch.tensor(q * (n + 1))).item()), n) - 1
    return ratio.kthvalue(max(index + 1, 1), dim=0).values.clamp_min(1e-3)


def calibrate(
    model,
    loader,
    device: str = "cpu",
    coverage: float = 0.9,
    mc_samples: int = 0,
) -> CalibrationArtifact:
    """Fit temperature scaling and conformal quantiles on a held-out split."""
    mean, log_var, target = _collect(model, loader, device, mc_samples)
    std_before = log_var.exp().sqrt()
    artifact = CalibrationArtifact(coverage=coverage, n_calibration=int(mean.shape[0]))
    artifact.ece_before = expected_calibration_error(target, mean, std_before)
    artifact.nll_before = float(gaussian_nll(mean, log_var, target).mean())

    temperature = fit_temperature(log_var, target - mean)
    scaled_log_var = log_var + 2.0 * torch.log(temperature.clamp_min(1e-3))
    std_after = scaled_log_var.exp().sqrt()
    artifact.ece_after = expected_calibration_error(target, mean, std_after)
    artifact.nll_after = float(gaussian_nll(mean, scaled_log_var, target).mean())
    artifact.temperature = [round(float(v), 6) for v in temperature]
    artifact.conformal_quantile = [
        round(float(v), 6) for v in fit_conformal(mean, scaled_log_var, target, coverage)
    ]
    return artifact


def apply_calibration(prediction: dict[str, torch.Tensor], artifact: CalibrationArtifact) -> dict[str, torch.Tensor]:
    """Apply a fitted artifact to a prediction dictionary (in place on a copy)."""
    if not artifact.temperature:
        return prediction
    temperature = torch.as_tensor(artifact.temperature, dtype=prediction["log_var"].dtype)
    prediction = dict(prediction)
    prediction["log_var"] = prediction["log_var"] + 2.0 * torch.log(temperature.clamp_min(1e-3))
    prediction["aleatoric"] = prediction["log_var"].exp().sqrt()
    if "epistemic" in prediction:
        prediction["total"] = (prediction["log_var"].exp() + prediction["epistemic"].pow(2)).sqrt()
    return prediction


__all__ = [
    "CalibrationArtifact",
    "apply_calibration",
    "calibrate",
    "expected_calibration_error",
    "fit_conformal",
    "fit_temperature",
]
