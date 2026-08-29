"""Explainability reports: routing, physics residuals, saliency and provenance.

Every report states what the model looked at (expert routing), which physical
relations disagreed with the observation (residuals), what the twin thinks, and
what an engineer should do. It never presents a verdict as an instruction.
"""

from __future__ import annotations

import torch

from ..data.schema import CHANNELS
from ..models.experts import EXPERT_NAMES

DISCLAIMER = (
    "Advisory only. Do not connect this output to pitch, yaw, braking, converter "
    "or protection controls. A qualified reliability engineer must review it."
)


def gradient_saliency(model, sequence: torch.Tensor, physics: torch.Tensor, neighbors=None) -> torch.Tensor:
    """Input-gradient saliency: |d(health)/dx| averaged over the window."""
    model.eval()
    x = sequence.detach().clone().requires_grad_(True)
    outputs = model(x, physics.detach(), neighbors)
    target = outputs["mean"][:, 1].sum()
    model.zero_grad(set_to_none=True)
    target.backward()
    saliency = x.grad.abs().mean(dim=(0, 1)) if x.grad is not None else torch.zeros(sequence.shape[-1])
    return saliency.detach()


def explain(
    prediction: dict,
    physics_residuals: dict,
    twin_state: dict,
    top_features: list[str] | None = None,
    model=None,
    sequence: torch.Tensor | None = None,
    physics: torch.Tensor | None = None,
    index: int = 0,
    top_k_features: int = 5,
) -> dict:
    """Assemble a human-readable explanation for one prediction."""
    routing = prediction["routing"]
    routing_vector = (
        routing[index].tolist()
        if routing.ndim == 2
        else routing.tolist()
    )
    names = list(EXPERT_NAMES)
    if len(routing_vector) > len(names):
        names = names + [f"expert_{i}" for i in range(len(names), len(routing_vector))]
    dominant = names[max(range(len(routing_vector)), key=routing_vector.__getitem__)]

    mean = prediction["mean"][index].tolist() if prediction["mean"].ndim > 1 else prediction["mean"].tolist()
    report = {
        "prediction": {
            "failure_probability": float(torch.sigmoid(torch.tensor(mean[0]))),
            "health_index": float(mean[1]),
            "rul_days": float(mean[2]) * 365.0,
        },
        "uncertainty": {
            "aleatoric": _pick(prediction, "aleatoric", index),
            "epistemic": _pick(prediction, "epistemic", index),
            "total": _pick(prediction, "total", index),
        },
        "routing": {
            "dominant_expert": dominant,
            "expert_weights": dict(zip(names, [round(float(w), 4) for w in routing_vector])),
        },
        "physics_residuals": {k: round(float(v), 5) for k, v in physics_residuals.items()},
        "digital_twin_state": twin_state,
        "recommended_action": DISCLAIMER,
        "provenance": {
            "models": "WindFusion v0.2 (Aetheris base + mythology presets)",
            "physics": "Heier Cp / Betz, ISO 281 L10, lumped RC thermal, Jensen wake",
        },
    }
    if top_features:
        report["top_features"] = top_features
    elif model is not None and sequence is not None and physics is not None:
        saliency = gradient_saliency(model, sequence, physics)
        values = saliency[index] if saliency.ndim > 1 else saliency
        order = torch.argsort(values, descending=True)[:top_k_features].tolist()
        report["top_features"] = [
            {"channel": CHANNELS[i], "saliency": round(float(values[i]), 6)} for i in order
        ]
    return report


def _pick(prediction: dict, key: str, index: int):
    if key not in prediction:
        return None
    value = prediction[key]
    return value[index].tolist() if value.ndim > 1 else value.tolist()


def render_report(report: dict) -> str:
    """Render an explanation dict as plain text (CLI friendly)."""
    lines = ["WindFusion advisory report", "=" * 32]
    prediction = report["prediction"]
    lines.append(
        f"failure probability : {prediction['failure_probability']:.3f}\n"
        f"health index        : {prediction['health_index']:.3f}\n"
        f"RUL estimate        : {prediction['rul_days']:.1f} days"
    )
    lines.append(
        f"uncertainty         : aleatoric={report['uncertainty']['aleatoric']} "
        f"epistemic={report['uncertainty']['epistemic']}"
    )
    lines.append(f"dominant expert     : {report['routing']['dominant_expert']}")
    lines.append("expert weights      : " + ", ".join(
        f"{k}={v:.2f}" for k, v in report["routing"]["expert_weights"].items()
    ))
    lines.append("physics residuals   : " + ", ".join(
        f"{k}={v:.3f}" for k, v in report["physics_residuals"].items()
    ))
    if report.get("top_features"):
        rendered = ", ".join(
            f"{f['channel']}({f['saliency']:.3f})" if isinstance(f, dict) else str(f)
            for f in report["top_features"]
        )
        lines.append(f"top channels        : {rendered}")
    lines.append("")
    lines.append(report["recommended_action"])
    return "\n".join(lines)


__all__ = ["DISCLAIMER", "explain", "gradient_saliency", "render_report"]
