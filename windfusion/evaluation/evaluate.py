"""Model evaluation: predictive metrics, verdict behaviour and timing."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from ..config import WindFusionConfig
from ..data.schema import RUL_SCALE_DAYS
from ..physics.residuals import compute_residuals
from ..training.calibration import CalibrationArtifact, apply_calibration
from ..verification.verifier import PhysicsConfidenceFusion, Verdict
from .metrics import summarise_all


@dataclass
class EvaluationResult:
    """Metrics, optional prediction arrays and provenance for one evaluation."""

    metrics: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    arrays: dict | None = None

    def as_dict(self, include_arrays: bool = False) -> dict:
        payload = {"metrics": self.metrics, "timing": self.timing, "meta": self.meta}
        if include_arrays and self.arrays:
            payload["arrays"] = {k: np.asarray(v).tolist() for k, v in self.arrays.items()}
        return payload


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    config: WindFusionConfig | None = None,
    device: str = "cpu",
    mc_samples: int = 16,
    calibration: CalibrationArtifact | None = None,
    keep_arrays: bool = True,
    max_batches: int | None = None,
) -> EvaluationResult:
    """Score a model on a loader and return the full metric bundle."""
    config = config or WindFusionConfig()
    model = model.to(device).eval()
    means, totals, aleatoric, epistemic, targets, routing, residuals, completeness = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    started = time.perf_counter()
    n_windows = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
        prediction = model.predict(
            batch["sequence"],
            batch["physics"],
            batch.get("neighbors"),
            samples=mc_samples,
        )
        if calibration is not None:
            prediction = apply_calibration(prediction, calibration)
        res = compute_residuals(batch["raw"], getattr(model, "turbine", config.turbine))
        residual_magnitude = torch.stack(
            [v.reshape(v.shape[0], -1).abs().mean(-1) for v in res.values()]
        ).mean(0)
        means.append(prediction["mean"].cpu())
        totals.append(prediction["total"].cpu())
        aleatoric.append(prediction["aleatoric"].cpu())
        epistemic.append(prediction["epistemic"].cpu())
        targets.append(batch["target"].cpu())
        routing.append(prediction["routing"].cpu())
        residuals.append(residual_magnitude.cpu())
        completeness.append(batch["data_completeness"].cpu().reshape(-1))
        n_windows += int(batch["sequence"].shape[0])
    elapsed = time.perf_counter() - started

    mean = torch.cat(means).numpy() if means else np.zeros((0, 3), dtype=np.float32)
    std = torch.cat(totals).numpy() if totals else np.zeros_like(mean)
    target = torch.cat(targets).numpy() if targets else np.zeros_like(mean)
    if mean.shape[0] == 0:
        return EvaluationResult(meta={"error": "empty loader"})

    metrics = summarise_all(
        target,
        mean,
        std,
        warning_horizon_days=config.evaluation.warning_horizon_days,
    )
    timing = {
        "windows": n_windows,
        "seconds": round(elapsed, 4),
        "windows_per_second": round(n_windows / max(elapsed, 1e-9), 2),
        "ms_per_window_mc": round(1000 * elapsed / max(n_windows, 1), 3),
        "mc_samples": int(mc_samples),
    }
    result = EvaluationResult(
        metrics=metrics,
        timing=timing,
        meta={
            "model": getattr(getattr(model, "spec", None), "model_id", model.__class__.__name__),
            "calibrated": calibration is not None,
            "device": str(device),
        },
    )
    if keep_arrays:
        result.arrays = {
            "mean": mean,
            "std": std,
            "aleatoric": torch.cat(aleatoric).numpy(),
            "epistemic": torch.cat(epistemic).numpy(),
            "target": target,
            "routing": torch.cat(routing).numpy(),
            "residual": torch.cat(residuals).numpy(),
            "completeness": torch.cat(completeness).numpy(),
        }
    return result


# ── verification behaviour ─────────────────────────────────────────────────

TRUE_CRITICAL_DAYS = 7.0


def _true_verdict(rul_days: float, horizon_days: float) -> str:
    if rul_days < TRUE_CRITICAL_DAYS:
        return Verdict.CRITICAL.value
    if rul_days < horizon_days:
        return Verdict.WARNING.value
    return Verdict.NORMAL.value


@torch.no_grad()
def evaluate_verification(
    model,
    loader,
    config: WindFusionConfig | None = None,
    device: str = "cpu",
    mc_samples: int = 8,
    use_twin: bool = True,
    max_batches: int | None = None,
) -> dict:
    """Measure fail-closed behaviour: abstention, coverage and safety errors.

    A verdict is *correct* when its severity matches the true state derived
    from the ground-truth RUL (synthetic data only). ``MODEL_UNCERTAIN`` and
    ``INSUFFICIENT_DATA`` count as abstentions: they are neither correct nor
    incorrect, but they are reported separately because a safety-critical
    advisor is allowed to say "I don't know".
    """
    from ..digital_twin.fusion import TwinCoupledVerifier, twin_from_prediction

    config = config or WindFusionConfig()
    model = model.to(device).eval()
    verifier = TwinCoupledVerifier(PhysicsConfidenceFusion()) if use_twin else None
    plain = PhysicsConfidenceFusion()
    counts = {"n": 0, "correct": 0, "abstain": 0, "false_critical": 0, "missed_critical": 0}
    verdict_histogram: dict[str, int] = {}
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
        prediction = model.predict(
            batch["sequence"], batch["physics"], batch.get("neighbors"), samples=mc_samples
        )
        res = compute_residuals(batch["raw"], getattr(model, "turbine", config.turbine))
        for i in range(batch["sequence"].shape[0]):
            residual = float(
                torch.stack([v[i].reshape(-1).abs().mean() for v in res.values()]).mean()
            )
            failure = float(torch.sigmoid(prediction["mean"][i, 0]))
            epistemic = float(prediction["epistemic"][i].mean())
            rul_days = float(batch["target"][i, 2]) * RUL_SCALE_DAYS
            completeness = float(batch["data_completeness"][i].reshape(()))
            twin = None
            if verifier is not None:
                twin = twin_from_prediction(
                    {
                        "health": float(batch["target"][i, 1]),
                        "uncertainty": epistemic,
                        "rul_days": rul_days,
                    }
                )
                result = verifier.verify_window(
                    failure_probability=failure,
                    epistemic=epistemic,
                    physics_residual=residual,
                    twin_health=twin.state.health,
                    data_completeness=completeness,
                    rul_days=rul_days,
                )
            else:
                result = plain.verify(
                    failure_probability=failure,
                    epistemic=epistemic,
                    physics_residual=residual,
                    twin_health=1.0,
                    data_completeness=completeness,
                )
            verdict = result.verdict.value
            verdict_histogram[verdict] = verdict_histogram.get(verdict, 0) + 1
            counts["n"] += 1
            truth = _true_verdict(rul_days, config.evaluation.warning_horizon_days)
            if verdict in (Verdict.MODEL_UNCERTAIN.value, Verdict.INSUFFICIENT_DATA.value):
                counts["abstain"] += 1
            elif verdict == truth:
                counts["correct"] += 1
            elif verdict == Verdict.CRITICAL.value and truth != Verdict.CRITICAL.value:
                counts["false_critical"] += 1
            elif truth == Verdict.CRITICAL.value and verdict != Verdict.CRITICAL.value:
                counts["missed_critical"] += 1
    n = max(counts["n"], 1)
    decided = n - counts["abstain"]
    return {
        "n": counts["n"],
        "verdict_histogram": verdict_histogram,
        "abstention_rate": round(counts["abstain"] / n, 4),
        "coverage": round(decided / n, 4),
        "accuracy_on_decided": round(counts["correct"] / max(decided, 1), 4),
        "false_critical_rate": round(counts["false_critical"] / n, 4),
        "missed_critical_rate": round(counts["missed_critical"] / n, 4),
        "use_twin": use_twin,
    }


__all__ = ["EvaluationResult", "evaluate_model", "evaluate_verification"]
