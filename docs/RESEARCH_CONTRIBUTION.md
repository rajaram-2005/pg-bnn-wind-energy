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

### What the first synthetic run supports, refutes and leaves open

Seeded fleet `bd08a45a71d0ecea`. "Supports" means *consistent with the hypothesis on this
simulation* — nothing more. The ablations needed to attribute any of it to a specific mechanism
(A–X in `benchmarks/ablations.yaml`) are still unrun.

| Hypothesis | Outcome on the synthetic fleet |
|---|---|
| H1 | **Inconclusive / partly refuted.** The presets beat `ridge` and `mlp` on health MAE and beat every baseline on interval coverage (0.94–0.98 vs 0.83–0.85) and NLL, but the `gru` baseline wins on RUL MAE (83.1 vs 96.6 days) and early-warning F1 (0.699 vs 0.553). The calibration claim holds; the accuracy claim does not, at equal-ish parameter counts, on this data. |
| H2 | **Supported.** Distillation moves the 9.7k student from 0.1033 → 0.0889 health MAE and 111.9 → 94.8 days RUL MAE, past its own 467k teacher (99.1 days). Not yet decomposed: the run compares "distilled" against "from scratch", not against response-only distillation, so the router-agreement term specifically is untested. |
| H3 | **Partly refuted.** Few-shot adaptation helps (111.9 → 104.2 days), but federated averaging gave no benefit at all (111.9 → 112.2). The Reptile meta-training arm is not separated from the few-shot arm in this run. |
| H4 | **Not tested.** The codec was verified (0.372 bandwidth reduction at 0.0022 mean reconstruction error) but no fixed-deadband comparison arm was run, so the "better than fixed deadband" half of H4 has no measurement behind it. |

Two further results are reported because they constrain the claims, not because they flatter them:
verification abstention was 0.000 on every model (see `docs/SAFETY.md`), and the `mean` baseline's
0.283 health MAE / 282-day RUL MAE is the floor any model must clear before it is interesting.

## Ablations

`benchmarks/ablations.yaml` defines reproducible toggles (pure neural → +physics → +uncertainty →
+sparse experts → +digital twin → full → full without telemetry → full without verification, plus
preset-to-preset swaps, distillation on/off and adaptation protocol). Unrun cells must remain
`NOT MEASURED`.
