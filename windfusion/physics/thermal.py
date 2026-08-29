"""Stable first-order lumped thermal model."""

import torch


def thermal_step(
    temperature_c: torch.Tensor,
    ambient_c: torch.Tensor,
    heat_w: torch.Tensor,
    resistance_k_w: float,
    capacitance_j_k: float,
    dt_s: float,
) -> torch.Tensor:
    tau = max(resistance_k_w * capacitance_j_k, 1e-6)
    equilibrium = ambient_c + heat_w * resistance_k_w
    decay = torch.exp(
        torch.as_tensor(-dt_s / tau, dtype=temperature_c.dtype, device=temperature_c.device)
    )
    return equilibrium + (temperature_c - equilibrium) * decay


def thermal_residual(
    observed_next_c: torch.Tensor, predicted_next_c: torch.Tensor, scale_c: float = 100.0
) -> torch.Tensor:
    return (observed_next_c - predicted_next_c) / scale_c
