"""Evaluation metrics for regression, early warning, classification and calibration.

Provenance: ``ece_coverage`` and ``early_warning_lead_time``
(wind-turbine-pg-bnn ``src/eval/calibration.py``). Re-implemented here without
SciPy so the same code runs on the edge path, and extended with distribution-free
conformal coverage and CRPS.

Every function takes plain numpy arrays so baselines and neural models are
scored by exactly the same code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.schema import RUL_SCALE_DAYS, TARGETS

EPS = 1e-9


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE, RMSE and R² for one target column."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    error = y_pred - y_true
    ss_res = float(np.mean(error**2))
    ss_tot = float(np.mean((y_true - y_true.mean()) ** 2)) + EPS
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(ss_res)),
        "r2": float(1.0 - ss_res / ss_tot),
    }


def per_target_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict[str, float]]:
    """Regression metrics for every target, plus RUL expressed in days."""
    out: dict[str, dict[str, float]] = {}
    for i, name in enumerate(TARGETS[: y_true.shape[1]]):
        out[name] = regression_metrics(y_true[:, i], y_pred[:, i])
    if y_true.shape[1] > 2:
        out["rul_days"]["mae_days"] = out["rul_days"]["mae"] * RUL_SCALE_DAYS
        out["rul_days"]["rmse_days"] = out["rul_days"]["rmse"] * RUL_SCALE_DAYS
    return out


def auroc(y_true_binary: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney U); 0.5 for a constant score."""
    y = np.asarray(y_true_binary, dtype=np.float64).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    positives, negatives = y.sum(), (1 - y).sum()
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = s[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    positive_rank_sum = ranks[y > 0.5].sum()
    return float((positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def classification_metrics(
    y_true_binary: np.ndarray, scores: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Precision/recall/F1/false-alarm rate at a fixed threshold, plus AUROC."""
    y = np.asarray(y_true_binary, dtype=np.float64).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    predicted = (s >= threshold).astype(np.float64)
    tp = float(((predicted > 0.5) & (y > 0.5)).sum())
    tn = float(((predicted <= 0.5) & (y <= 0.5)).sum())
    fp = float(((predicted > 0.5) & (y <= 0.5)).sum())
    fn = float(((predicted <= 0.5) & (y > 0.5)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    n = max(tp + tn + fp + fn, 1.0)
    return {
        "auroc": auroc(y, s),
        "accuracy": (tp + tn) / n,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": fp / max(fp + tn, 1.0),
        "n_positive": int(y.sum()),
        "threshold": float(threshold),
    }


def early_warning_metrics(
    y_true_rul_days: np.ndarray,
    y_pred_rul_days: np.ndarray,
    warning_horizon_days: float = 30.0,
) -> dict[str, float]:
    """Fixed-horizon early-warning protocol (upstream headline metric)."""
    yt = np.asarray(y_true_rul_days, dtype=np.float64)
    yp = np.asarray(y_pred_rul_days, dtype=np.float64)
    metrics = classification_metrics(yt < warning_horizon_days, -(yp), threshold=-warning_horizon_days)
    return {
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "false_alarm_rate": metrics["false_alarm_rate"],
        "auroc": metrics["auroc"],
        "mean_lead_time_days": float(yt[(yt < warning_horizon_days) & (yp < warning_horizon_days)].mean())
        if ((yt < warning_horizon_days) & (yp < warning_horizon_days)).any()
        else 0.0,
        "warning_horizon_days": float(warning_horizon_days),
        "n_at_risk": int((yt < warning_horizon_days).sum()),
    }


def first_warning_lead_time_days(rul_remaining_days, warned) -> float | None:
    """Days before failure when the first warning fires along one trajectory."""
    ruls = np.asarray(rul_remaining_days, dtype=np.float64)
    flags = np.asarray(warned, dtype=bool)
    if ruls.shape != flags.shape or not flags.any():
        return None
    return float(ruls[np.argmax(flags)])


def expected_asset_utilization(
    predicted_rul_days, planning_horizon_days: float = 90.0, safety_buffer_days: float = 14.0
) -> dict[str, float]:
    """Heuristic fleet utilisation metric (upstream concept, heuristic only)."""
    ruls = np.asarray(predicted_rul_days, dtype=np.float64)
    available = np.clip((ruls - safety_buffer_days) / max(planning_horizon_days, EPS), 0.0, 1.0)
    return {
        "mean_utilization": float(available.mean()),
        "fraction_at_risk": float((ruls < planning_horizon_days + safety_buffer_days).mean()),
        "mean_rul_days": float(ruls.mean()),
    }


# ── calibration ────────────────────────────────────────────────────────────


def _norm_ppf(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    t = np.sqrt(-2.0 * np.log(1.0 - p))
    c0 = np.array([2.515517, 0.802853, 0.010328])
    c1 = np.array([1.432788, 0.189269, 0.001308])
    num = c0[0] + c0[1] * t + c0[2] * t**2
    den = 1.0 + c1[0] * t + c1[1] * t**2 + c1[2] * t**3
    return t - num / den


def expected_calibration_error(
    y_true: np.ndarray, mean: np.ndarray, std: np.ndarray, n_bins: int = 10
) -> float:
    """ECE over central predictive intervals (lower is better)."""
    y = np.asarray(y_true, dtype=np.float64)
    m = np.asarray(mean, dtype=np.float64)
    s = np.asarray(std, dtype=np.float64).clip(EPS)
    ps = np.linspace(1 / n_bins, 1 - 1 / n_bins, n_bins - 1)
    z = _norm_ppf(0.5 + ps / 2.0)
    errors = []
    for p, zp in zip(ps, z):
        lo, hi = m - zp * s, m + zp * s
        errors.append(abs(float(((y >= lo) & (y <= hi)).mean()) - float(p)))
    return float(np.mean(errors))


def gaussian_nll(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    s = np.asarray(std, dtype=np.float64).clip(EPS)
    z = (np.asarray(y_true, dtype=np.float64) - np.asarray(mean, dtype=np.float64)) / s
    return float(np.mean(0.5 * z**2 + np.log(s) + 0.5 * np.log(2 * np.pi)))


def crps_gaussian(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    """Continuous ranked probability score for a Gaussian predictive."""
    y = np.asarray(y_true, dtype=np.float64)
    m = np.asarray(mean, dtype=np.float64)
    s = np.asarray(std, dtype=np.float64).clip(EPS)
    z = (y - m) / s
    phi = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    cdf = 0.5 * (1.0 + np.vectorize(_erf_scalar)(z / np.sqrt(2.0)))
    return float(np.mean(s * (z * (2 * cdf - 1) + 2 * phi - 1 / np.sqrt(np.pi))))


def _erf_scalar(x: float) -> float:
    import math

    return math.erf(float(x))


def interval_coverage(
    y_true: np.ndarray, mean: np.ndarray, std: np.ndarray, coverage: float = 0.9
) -> dict[str, float]:
    """Empirical coverage and width of the central predictive interval."""
    y = np.asarray(y_true, dtype=np.float64)
    m = np.asarray(mean, dtype=np.float64)
    s = np.asarray(std, dtype=np.float64).clip(EPS)
    z = float(_norm_ppf(np.array([0.5 + coverage / 2.0]))[0])
    inside = (y >= m - z * s) & (y <= m + z * s)
    return {
        "target_coverage": float(coverage),
        "empirical_coverage": float(inside.mean()),
        "mean_interval_width": float((2 * z * s).mean()),
    }


@dataclass
class CalibrationSummary:
    ece: float
    nll: float
    crps: float
    coverage: float
    empirical_coverage: float
    mean_interval_width: float

    def as_dict(self) -> dict[str, float]:
        return {
            "ece": round(self.ece, 6),
            "nll": round(self.nll, 6),
            "crps": round(self.crps, 6),
            "coverage": round(self.coverage, 4),
            "empirical_coverage": round(self.empirical_coverage, 4),
            "mean_interval_width": round(self.mean_interval_width, 6),
        }


def calibration_summary(
    y_true: np.ndarray, mean: np.ndarray, std: np.ndarray, coverage: float = 0.9
) -> CalibrationSummary:
    """Aggregate calibration metrics over the scored targets."""
    interval = interval_coverage(y_true, mean, std, coverage)
    return CalibrationSummary(
        ece=expected_calibration_error(y_true, mean, std),
        nll=gaussian_nll(y_true, mean, std),
        crps=crps_gaussian(y_true, mean, std),
        coverage=interval["target_coverage"],
        empirical_coverage=interval["empirical_coverage"],
        mean_interval_width=interval["mean_interval_width"],
    )


def summarise_all(
    y_true: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    warning_horizon_days: float = 30.0,
    target_indices: tuple[int, ...] = (1, 2),
) -> dict:
    """Full metric bundle used by the benchmark tables."""
    y_true = np.asarray(y_true, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    regression = per_target_regression(y_true, mean)
    rul_true_days = y_true[:, 2] * RUL_SCALE_DAYS
    rul_pred_days = mean[:, 2] * RUL_SCALE_DAYS
    binary = (rul_true_days < warning_horizon_days).astype(float)
    failure_scores = 1.0 / (1.0 + np.exp(-mean[:, 0])) if y_true.shape[1] > 0 else mean[:, 0]
    selected = np.concatenate([y_true[:, list(target_indices)].ravel()])
    selected_mean = mean[:, list(target_indices)].ravel()
    selected_std = std[:, list(target_indices)].ravel()
    return {
        "regression": regression,
        "classification": classification_metrics(binary, failure_scores, threshold=0.5),
        "early_warning": early_warning_metrics(rul_true_days, rul_pred_days, warning_horizon_days),
        "calibration": calibration_summary(selected, selected_mean, selected_std).as_dict(),
        "fleet": expected_asset_utilization(rul_pred_days),
        "n_samples": int(y_true.shape[0]),
    }


__all__ = [
    "CalibrationSummary",
    "auroc",
    "calibration_summary",
    "classification_metrics",
    "crps_gaussian",
    "early_warning_metrics",
    "expected_asset_utilization",
    "expected_calibration_error",
    "first_warning_lead_time_days",
    "gaussian_nll",
    "interval_coverage",
    "per_target_regression",
    "regression_metrics",
    "summarise_all",
]
