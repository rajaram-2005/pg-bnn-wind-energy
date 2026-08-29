# WindFusion v0.2 model card

## Intended use

Research and advisory monitoring of wind-turbine SCADA/condition data: fault-risk estimation,
health index, remaining useful life, uncertainty reporting, counterfactual digital-twin scenarios
and telemetry prioritisation.

**Out of scope:** turbine control of any kind (pitch, yaw, braking, converter, protection),
safety interlocks, warranty or commercial decisions, and any use without engineer review.

## Models

| Model | Family | Parameters | Role |
|---|---|---:|---|
| `aetheris-wind` | base | ~88k | balanced reference architecture with verify-first self-check |
| `ra-wind` | Egyptian preset | ~53k | thermal/solar-load specialisation |
| `qinglong-wind` | Chinese preset | ~92k | wake and fleet coupling (3 neighbours) |
| `vayu-wind` | Hindu preset | ~56k | gust/rotor aerodynamics, lowest latency (top-1) |
| `odin-wind` | Norse preset | ~71k | cold-climate drivetrain and RUL with monotonicity |
| `aeolus-wind` | Greek preset | ~93k | multi-horizon forecasting auxiliary task |
| `windfusion-edge` | tier | ~10k | constrained CPU/edge hardware, distillation student |
| `windfusion-lite` | tier | ~50k | default |
| `windfusion-research` | tier | ~467k | distillation teacher, ablations |

Names are engineering presets, not pretrained models and not cultural representations.

## Training and evaluation data

No weights and no real dataset ship with v0.2. The bundled data source is a **deterministic,
physics-driven synthetic fleet** (36 turbines × 1440 samples by default, 6 sites, 5 fault-mode
signatures, seeded, checksummed) in which the latent damage state — and therefore health and RUL —
is known exactly. That is the only reason a supervised benchmark on it is well posed.

Measured numbers exist only in `benchmarks/results/synthetic-v0.2.0.md` and are stamped
SYNTHETIC. `benchmarks/results/REAL_DATA_TEMPLATE.md` is the unfilled template for real data.

## Uncertainty

Aleatoric uncertainty comes from a learned heteroscedastic head; epistemic uncertainty from MC
dropout, which is an **approximation** to Bayesian inference. Calibration is measured with ECE,
NLL, CRPS and empirical coverage, and corrected with per-output temperature scaling plus a
distribution-free conformal interval fitted on a held-out split. Calibration must be re-measured
on every new site, OEM and season.

## Limitations and failure modes

- Simplified steady/lumped physics; no aeroelastic, no blade-root loads, no wake meandering.
- The synthetic fleet uses an accelerated damage clock: RUL ranges are plausible but the
  absolute time scale is a modelling choice.
- Missing or miscalibrated sensors, icing, curtailment, maintenance regime changes, unseen
  turbine types, distribution shift and weak labels can all invalidate predictions.
- Sparse routing can collapse without the balancing term; the router's choice is a correlation,
  not a causal attribution.
- Monotone RUL is a soft pairwise penalty, not a guarantee.
- No cross-farm or cross-OEM validation; no comparison against published upstream numbers is
  valid because the data differ.

## Safety

Advisory-only by construction: `SafetyConfig` raises if `advisory_only` is unset or
`allow_actuation` is set, and a test enforces it. High uncertainty or physical disagreement
yields `MODEL_UNCERTAIN`, never a confident guess. OEM protections remain authoritative.

## Deployment

Python 3.10+, PyTorch 2+, CPU-first. ONNX export with a mandatory ONNX Runtime parity check;
dynamic INT8 quantisation measured but hardware-dependent. Validate numerical parity, calibration
(before and after quantisation), latency, RAM and failure behaviour on the target device.
Energy per inference and on-device RAM are **NOT MEASURED** in this repository.
