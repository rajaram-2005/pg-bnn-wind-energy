# Architecture

## Data flow

```
raw SCADA (T, 12) ──┐
                    ├─► missing-value mask ──► forward fill ──► robust normalisation
physics (T, 5) ─────┘                                                  │
                                                                       ▼
                                                   TurbineWindowDataset (grouped by turbine)
                                                                       │
                                                                  DataLoader
                                                                       ▼
                                                          WindFusionBase.forward
```

### 1. Input contract

| Tensor | Shape | Content |
|---|---|---|
| `sequence` | `(B, T, 12)` | normalised SCADA window, NaNs imputed |
| `physics` | `(B, 5)` | tip-speed ratio (÷12), power coefficient, power residual, thermal margin, data completeness |
| `neighbors` | `(B, K, 12)` | synchronous snapshot of the K nearest turbines (Qinglong only) |

Twelve channels, in order: `wind_speed, rotor_speed, pitch_angle, active_power_kw, vibration_rms,
gearbox_oil_temp_c, generator_winding_temp_c, main_bearing_temp_c, ambient_temp_c, phase_current_a,
generator_torque_nm, grid_frequency_hz`.

### 2. Encoder

`MultiScaleCausalEncoder` stacks gated (GLU) dilated convolutions with **left padding only**, so
the representation at time `t` never sees `t+1`. The receptive field is
`1 + Σ (k-1)·dᵢ` — 7 steps for the edge tier, 15 for the default presets, 31 for Vayu and the
research tier.

### 3. Sparse expert mixture

`SparseExpertRouter` applies a linear gate, takes `top_k` experts, renormalises and returns the
weights with the prediction. Inactive experts are skipped entirely (`weights[:, i] > 0`), so a
top-1 model executes one expert per sample. A load-balancing term (`Σ p̄ log p̄`) prevents collapse.

Experts: `aero`, `drive`, `thermal`, `grid`, plus `wake` for Qinglong. Each is a residual
bottleneck (`Linear → SiLU → Dropout → Linear`, residual + LayerNorm) with a configurable depth.

For export, `dense_experts=True` evaluates every expert and multiplies by its (possibly zero)
routing weight. The result is numerically identical but removes data-dependent control flow, which
static runtimes cannot trace (see `tests/test_export.py::test_dense_export_path_matches_sparse_runtime`).

### 4. Preset biases

| Preset | Mechanism | Where |
|---|---|---|
| Ra | additive router-logit bias for the thermal expert, scaled by the thermal margin | `mythologies.RaWind.expert_logit_bias` |
| Qinglong | cross-asset attention + wake expert + Jensen deficit weighting | `base.NeighborAttention`, `experts.WakeExpert` |
| Vayu | window statistics (std, range) projected into the pooled representation; dilated stack up to 16 | `base.stat_proj`, `registry.VAYU.dilations` |
| Odin | ISO 281 damage-rate feature + monotone RUL penalty | `base._damage_feature`, `losses.monotone_rul_penalty` |
| Aeolus | auxiliary 6-step wind/power forecast decoder | `base.forecast_head`, `losses.forecast_loss` |
| AutoWind (`windfusion-auto`) | self-supervised objectives and its own fit loop — the model that trains on its own | `models/auto.py`, `AutoWind.autonomous_fit` |
| Self-check head (research tier + AutoWind) | head predicting the expected physics-residual magnitude, used by the verifier as a consistency signal | `base.self_check`, `losses.self_check_loss` |

### 5. Uncertainty

A heteroscedastic head emits `mean` and `log_var`. MC dropout over `S` forward passes yields:

```
aleatoric = mean(exp(log_var))          # observation noise
epistemic = var(means)                  # model uncertainty
total     = sqrt(aleatoric² + epistemic²)
```

MC dropout is an **approximation**, not variational inference, so calibration is measured
(ECE, NLL, CRPS, empirical coverage) and corrected post hoc by temperature scaling plus a
distribution-free conformal quantile fitted on a held-out split.

### 6. Verification (Physics-Confidence Fusion)

```
completeness < 0.6                → INSUFFICIENT_DATA
epistemic > 0.45
  or consistency < 0.35
  or self-check predicts residual
  or (p_fail > 0.7 ∧ consistency < 0.35 ∧ twin_health > 0.7)
                                  → MODEL_UNCERTAIN
concern ≥ 0.8 ∧ confidence ≥ 0.55 → CRITICAL
concern ≥ 0.5                     → WARNING
otherwise                         → NORMAL
```

`concern = max(p_failure, 1 − twin_health, 1 − RUL/horizon)` and
`confidence = (1 − epistemic) · (0.5 + 0.5·consistency) · temporal_consistency`.

### 7. Training objective

```
L = heteroscedastic NLL
  + Σᵢ λᵢ·biasᵢ·rᵢ²                      (aero, drive, thermal, grid, consistency)
  + λ_limit · soft-hinge limit penalties
  + w_u · E[log_var²]                    (uncertainty regularisation)
  + w_r · router balance
  + w_m · monotone RUL pairs             (Odin)
  + 0.1 · forecast MSE                   (Aeolus)
  + 0.1 · self-check MSE                 (research tier, AutoWind)
```

`windfusion-auto` additionally optimises the self-supervised objective of § 8, which needs no
labels at all; the supervised terms above only enter when targets happen to exist.

Residuals are computed on **raw SI-unit** channels, never on normalised features, because the
physical relations are only valid in physical units.

**Missing-data handling.** A residual computed from a dropped-out channel is missing data, not
evidence of physical inconsistency: `compute_residuals` carries a per-sample validity mask, zeroes
the residual where a required channel is NaN, and clamps the rest to ±10 so a single corrupt sample
cannot dominate the objective. The dataset keeps NaNs in `raw` (only the normalised `data` tensor is
imputed) so this masking is possible at all; `limit_penalties` applies the same rule and normalises
each excess by a per-channel scale, so a sensor pinned at its clipping value contributes an O(1)
penalty rather than a 10³ one. Completeness is reported to the verifier separately through
`data_completeness`.

### 8. Self-learning objective (`windfusion-auto`)

The self-learning model replaces the old borrowed base architecture and trains on its own. Its
objective (implemented in `AutoWind.self_supervised_objective`, driven by `AutoWind.autonomous_fit`)
consumes unlabelled SCADA windows only:

```
L_autonomous = w_r · masked-channel reconstruction MSE   (hide whole channels, infer last step)
             + w_n · next-step prediction MSE            (each position predicts t+1; causal by construction)
             + w_p · Σᵢ rᵢ²                               (lumped-physics consistency of the snapshot)
             + w_c · self-check MSE                       (regress own residual magnitude)
             + w_b · router balance                       (Σ p̄ log p̄, same anti-collapse term)
```

Rounds after the first add a bootstrapping term: the model scores every training window with MC
dropout, keeps only windows whose mean epistemic uncertainty is at or below
`self_training.pseudo_label_gate` (and whose data completeness passes the verifier's floor), and
trains a heteroscedastic NLL against its own predictions on those. Accept/reject counts are reported
(`pseudo_stats`) — silent degradation is the failure mode of every self-training scheme, so the gate
and the numbers are part of the contract. Validation for early stopping uses the same autonomous
objective; ground-truth labels are never part of the fit.

### 9. Deployment path

`torch → ExportWrapper (dense experts) → ONNX (opset 17) → ONNX Runtime parity check → dynamic INT8`.
TorchScript remains available as a fallback. `EdgeRuntime` wraps the model with a ring buffer,
inference cadence, PSI drift monitoring and stale-fallback behaviour.

## Package map

```
windfusion/
  config.py          typed versioned configuration + safety invariant
  provenance.py      machine-readable reuse registry
  models/            registry, encoder, experts, router, uncertainty, base, presets, tiers
  physics/           aerodynamic, drivetrain, thermal, electrical relations + residuals
  data/              schema, deterministic synthetic fleet, window datasets, loaders
  training/          losses, loop, distillation, calibration, fleet, onboarding
  evaluation/        metrics, evaluate, baselines, benchmark
  digital_twin/      state, simulator, twin-coupled verification
  verification/      fail-closed verifier
  edge/              telemetry policy + codec, streaming runtime
  explainability/    routing, residuals, saliency, advisory reports
  deployment/        ONNX / TorchScript export and parity validation
```
