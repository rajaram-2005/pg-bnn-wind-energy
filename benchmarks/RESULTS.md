# Benchmark results

Two tables, kept strictly apart:

| Table | Data | Status |
|---|---|---|
| [`results/synthetic-v0.2.0.md`](results/synthetic-v0.2.0.md) | deterministic **synthetic** fleet, seed 7, 36 turbines × 1440 samples | measured — stamped SYNTHETIC |
| [`results/REAL_DATA_TEMPLATE.md`](results/REAL_DATA_TEMPLATE.md) | licensed real SCADA/CM data | every cell `NOT MEASURED` |

> **A synthetic result is not a field result.** The synthetic fleet is a simulation in which the
> latent damage state is known exactly, so health and RUL labels are exact. That makes the
> benchmark well posed and reproducible; it says nothing about transfer to a real turbine.

## Synthetic run (latest)

<!-- SYNTHETIC_SUMMARY_START -->
Measured 2026-08-29 on the seeded synthetic fleet `bd08a45a71d0ecea` (36 turbines x 1440 samples,
6 sites, 15 epochs, ~22 min on 2 cores). Full tables:
[`results/synthetic-v0.2.0.md`](results/synthetic-v0.2.0.md).

> Scope note: this run predates the removal of the former external base model and the addition of
> the self-learning `windfusion-auto` model. `windfusion-auto` (and the benchmark's new
> `self_learning` stage) are therefore `NOT MEASURED` here — re-run the harness to populate them.

**Distillation (teacher 467k -> student 9.7k)**

| Variant | health MAE | RUL MAE (days) |
|---|---:|---:|
| `windfusion-research` (teacher) | 0.0984 | 99.07 |
| `windfusion-edge` (from scratch) | 0.1033 | 111.92 |
| `windfusion-edge` + distilled | **0.0889** | **94.80** |

**Fleet adaptation on a held-out site (RUL MAE, days)**

| zero-shot | + federated averaging | + few-shot (32 windows) |
|---:|---:|---:|
| 111.92 | 112.18 | **104.22** |

Federated averaging gave no benefit in this configuration; the result is reported as measured.

**Against the baselines — the honest reading**

| Model | params | health MAE | RUL MAE (days) | warning F1 | 90% coverage |
|---|---:|---:|---:|---:|---:|
| `gru` (baseline) | 16,131 | 0.0844 | **83.14** | **0.699** | 0.851 |
| `ridge` (baseline) | 879 | 0.1023 | 124.51 | 0.453 | 0.827 |
| `windfusion-edge` | 9,714 | 0.1033 | 111.92 | 0.418 | **0.939** |
| `vayu-wind` | 56,444 | 0.0810 | 98.01 | 0.553 | 0.973 |
| `odin-wind` | 71,006 | 0.0768 | 96.62 | 0.373 | 0.976 |
| `aeolus-wind` | 92,518 | **0.0720** | 100.36 | 0.316 | 0.967 |

The GRU baseline wins on RUL MAE and early-warning F1; the WindFusion presets win on health MAE
(most presets) and clearly on interval coverage and NLL. On this synthetic fleet there is no
single winner, and the tables are reported without a spin section.

**Verification**: abstention rate 0.000 for every model, because the test split is
in-distribution and ~99% complete (class balance: 1347 NORMAL / 44 WARNING / 268 CRITICAL, i.e. an
always-NORMAL advisor scores 0.812). Per-class recall and verdict histograms are in the full
report; the abstention path itself is exercised by unit tests, and `docs/SAFETY.md` records that
the verifier is not the deployment gate.

**Telemetry**: 256 windows, 105 compressed / 151 raw, bandwidth reduction 0.372, mean
reconstruction error 0.0022 units on the sampling grid.
<!-- SYNTHETIC_SUMMARY_END -->

## Infrastructure footprint

Run `python -m windfusion benchmark` (or `python benchmarks/run.py`) to measure parameters,
FP32/INT8 size and CPU latency **on this machine**. Those numbers are hardware-specific and are
reproduced inside the synthetic report, which also records the CPU model and thread count.

## What remains NOT MEASURED

- Any metric on real SCADA/CM data (no licensed dataset is bundled).
- Energy per inference and on-device RAM (need platform tooling).
- Cross-OEM and cross-farm generalisation.
- Accuracy after INT8 quantisation (size is measured, accuracy is not).
- Comparisons against the upstream repositories' published numbers: different data, different
  splits, not comparable.
