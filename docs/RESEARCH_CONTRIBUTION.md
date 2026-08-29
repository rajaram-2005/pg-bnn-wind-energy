# Research contribution statement

## Existing ideas reused

Heteroscedastic Gaussian NLL, aleatoric/epistemic decomposition, MC dropout, physics-residual
regularisation, Heier's Cp surface, the Betz limit, Jensen's wake model, ISO 281 bearing life,
lumped RC thermal networks, soft-hinge limit penalties, expected calibration error, fixed-horizon
early-warning metrics, digital-twin what-if analysis, anomaly-triggered telemetry bypass,
knowledge distillation, federated averaging, Reptile meta-learning, self-training onboarding and
sparse mixtures of experts all have prior art. The audit in `docs/REPOSITORY_AUDIT.md` records the
local lineage; a formal literature review is still outstanding and no claim of novelty is made.

## What this repository designs

1. **Physics-Confidence Fusion** — a verification layer that combines predictive uncertainty,
   normalised physics residuals, digital-twin health, temporal consistency and data completeness,
   and *abstains* (`MODEL_UNCERTAIN`, `INSUFFICIENT_DATA`) instead of returning a confident guess.
2. **Architecture presets as inductive bias** — the mythology family is not a set of widths. Each
   preset changes what the model sees (neighbour context, window statistics, damage-rate
   features), which expert is deepened, which residual weight it is pushed to respect and which
   auxiliary head it trains.
3. **Router-agreement distillation** — in addition to response, uncertainty, feature and
   physics-consistency transfer, the student matches the teacher's *routing distribution*, so
   which expert mattered is also distilled.
4. **Physics-driven synthetic fleet with a known latent state** — health and RUL are ground truth
   rather than heuristic labels, which makes supervised benchmarking well posed and reproducible
   from a seed plus a checksum.
5. **Accelerated-life damage clock** — damage accumulates on a configurable operational clock so a
   full degradation trajectory fits in a short simulation while telemetry physics keeps the real
   sampling interval; the two clocks are documented rather than silently mixed.
6. **Adaptive telemetry as a risk policy** — multi-signal risk with hysteresis and a bypass
   cooldown, so compression escalates on evidence and does not flap.

## Hypotheses (not results)

> H1 — Routing temporal SCADA representations through sparse domain experts and verifying outputs
> against physical consistency improves the accuracy–calibration–latency trade-off versus
> equally-sized dense baselines.
>
> H2 — Distilling a large teacher into a small student with router agreement recovers more of the
> teacher's ranking quality than response distillation alone.
>
> H3 — Reptile meta-training plus a handful of labelled windows adapts a model to an unseen site
> better than zero-shot transfer or federated averaging alone.
>
> H4 — A risk-aware telemetry policy preserves event fidelity while compressing routine windows
> more aggressively than a fixed deadband policy.

Each hypothesis is tested only on the synthetic fleet; the measured outcomes are in
`benchmarks/results/synthetic-v0.2.0.md`. They are **not** evidence about real turbines.

## Ablations

`benchmarks/ablations.yaml` defines reproducible toggles (pure neural → +physics → +uncertainty →
+sparse experts → +digital twin → full → full without telemetry → full without verification, plus
preset-to-preset swaps, distillation on/off and adaptation protocol). Unrun cells must remain
`NOT MEASURED`.
