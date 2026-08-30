# WindFusion v0.2.0 - synthetic benchmark results

> **SYNTHETIC (simulated fleet, not field data).** Every number below was measured on a simulated fleet with a known latent damage state. It is a regression and ablation harness, not evidence of field performance.

- Generated: 2026-08-29T21:16:32Z
- Fleet checksum: `bd08a45a71d0ecea`
- Data: 36 turbines x 1440 samples, 6 sites
- Seed: 7, epochs: 15
- Machine: Linux-6.1.158+-x86_64-with-glibc2.36 (torch 2.13.0+cu130, 1 threads)
- Note: this run predates the removal of the former external base model and the
  addition of the self-learning `windfusion-auto` model; re-run the benchmark
  to regenerate every table against the current model family.

## 1. Model footprint

| Model | Parameters | FP32 MB | INT8 MB | CPU latency ms (median) | windows/s |
|---|---:|---:|---:|---:|---:|
| `ra-wind` | 53,130 | 0.203 | 0.1624 | 1.431 | 698.9 |
| `qinglong-wind` | 91,613 | 0.349 | 0.2206 | 1.627 | 614.6 |
| `vayu-wind` | 56,444 | 0.215 | 0.1817 | 1.531 | 653.2 |
| `odin-wind` | 71,006 | 0.271 | 0.2202 | 1.531 | 653.2 |
| `aeolus-wind` | 92,518 | 0.353 | 0.2867 | 1.539 | 649.8 |
| `windfusion-edge` | 9,714 | 0.037 | 0.0281 | 0.775 | 1289.5 |
| `windfusion-lite` | 50,330 | 0.192 | 0.1623 | 1.315 | 760.3 |
| `windfusion-research` | 467,211 | 1.782 | 1.5129 | 3.265 | 306.3 |

## 2. Predictive quality (synthetic test split)

| Model | Parameters | health MAE | RUL MAE (days) | warning F1 | AUROC | ECE | NLL | 90% coverage | train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mean` | 0 | 0.2829 | 282.05 | 0.317 | 0.500 | 0.1100 | 1.079 | 0.717 | 0.0 |
| `ridge` | 879 | 0.1023 | 124.51 | 0.453 | 0.949 | 0.0268 | 0.235 | 0.827 | 0.8 |
| `mlp` | 10,755 | 0.1127 | 125.81 | 0.387 | 0.964 | 0.0645 | 0.318 | 0.843 | 1.8 |
| `gru` | 16,131 | 0.0844 | 83.14 | 0.699 | 0.983 | 0.0508 | -0.118 | 0.851 | 8.6 |
| `windfusion-edge` | 9,714 | 0.1033 | 111.92 | 0.418 | 0.956 | 0.0955 | -0.062 | 0.939 | 53.5 |
| `windfusion-lite` | 50,330 | 0.0988 | 111.37 | 0.450 | 0.972 | 0.0722 | -0.148 | 0.962 | 64.3 |
| `windfusion-research` | 467,211 | 0.0984 | 99.07 | 0.379 | 0.977 | 0.1217 | -0.197 | 0.967 | 137.1 |
| `ra-wind` | 53,130 | 0.0878 | 110.19 | 0.435 | 0.978 | 0.1078 | -0.247 | 0.964 | 66.2 |
| `qinglong-wind` | 91,613 | 0.0747 | 123.28 | 0.337 | 0.980 | 0.1383 | -0.237 | 0.973 | 90.8 |
| `vayu-wind` | 56,444 | 0.0810 | 98.01 | 0.553 | 0.979 | 0.1509 | -0.313 | 0.973 | 96.0 |
| `odin-wind` | 71,006 | 0.0768 | 96.62 | 0.373 | 0.972 | 0.1620 | -0.277 | 0.976 | 69.5 |
| `aeolus-wind` | 92,518 | 0.0720 | 100.36 | 0.316 | 0.972 | 0.1623 | -0.300 | 0.967 | 74.9 |

## 3. Distillation (teacher -> student)

| Variant | Parameters | health MAE | RUL MAE (days) | ECE |
|---|---:|---:|---:|---:|
| `windfusion-research` | 467,211 | 0.0984 | 99.07 | 0.1217 |
| `windfusion-edge` | 9,714 | 0.1033 | 111.92 | 0.0955 |
| `windfusion-edge+distilled` | 9,714 | 0.0889 | 94.80 | 0.1254 |

Knowledge-distillation terms on the final epoch: {'kd_total': 0.335267, 'kd_prediction': 0.018606, 'kd_uncertainty': 0.394465, 'kd_feature': 0.017153, 'kd_router': 4.645218}

## 4. Fleet learning (held-out site)

| Protocol | RUL MAE (days) |
|---|---:|
| zero-shot (no adaptation) | 111.92 |
| + federated averaging | 112.18 |
| + few-shot (32 windows) | 104.22 |

## 5. Verification behaviour (fail-closed)

| Model | abstention rate | coverage | accuracy on decided | recall NORMAL / WARNING / CRITICAL | false critical | missed critical |
|---|---:|---:|---:|---:|---:|---:|
| `windfusion-edge` | 0.000 | 1.000 | 0.473 | 0.39 / 0.27 / 0.94 | 0.090 | 0.010 |
| `windfusion-lite` | 0.000 | 1.000 | 0.487 | 0.40 / 0.41 / 0.93 | 0.087 | 0.011 |
| `windfusion-research` | 0.000 | 1.000 | 0.364 | 0.25 / 0.11 / 0.99 | 0.109 | 0.001 |
| `ra-wind` | 0.000 | 1.000 | 0.479 | 0.42 / 0.84 / 0.71 | 0.022 | 0.046 |
| `qinglong-wind` | 0.000 | 1.000 | 0.253 | 0.11 / 0.55 / 0.93 | 0.042 | 0.011 |
| `vayu-wind` | 0.000 | 1.000 | 0.663 | 0.62 / 0.61 / 0.88 | 0.045 | 0.019 |
| `odin-wind` | 0.000 | 1.000 | 0.365 | 0.25 / 0.34 / 0.96 | 0.080 | 0.006 |
| `aeolus-wind` | 0.000 | 1.000 | 0.181 | 0.02 / 0.36 / 0.98 | 0.069 | 0.004 |

Class balance of the synthetic test split: {'NORMAL': 1347, 'WARNING': 44, 'CRITICAL': 268} (majority-class rate 0.812 — a trivial always-NORMAL advisor scores this, so compare against it, not against zero).

Verdicts actually emitted by `windfusion-edge`: {'NORMAL': 523, 'WARNING': 735, 'CRITICAL': 401}. Abstention is triggered by epistemic uncertainty, physics inconsistency or low data completeness; this split is in-distribution with ~99% complete windows, so the abstention path is exercised by the unit tests rather than by this table.

## 6. Adaptive telemetry

- windows: 256, modes: {'raw': 151, 'compressed': 105}
- bandwidth reduction: 0.372
- mean |error| on reconstruction: 0.00222

## 7. What is NOT measured here

- Real SCADA/CM data of any kind (no licensed dataset is bundled).
- Energy per inference and on-device RAM: needs platform tooling.
- Cross-OEM and cross-farm generalisation.
- Any comparison against the upstream repositories' published numbers: those numbers come from different data and are not comparable.

## 8. Fleet composition

```json
{
  "n_turbines": 36,
  "seq_len": 1440,
  "n_sites": 6,
  "fault_modes": {
    "healthy": 8,
    "bearing_wear": 7,
    "gearbox_overheat": 7,
    "generator_winding": 7,
    "blade_aero_loss": 7
  },
  "end_damage_mean": 0.6022799015045166,
  "end_damage_max": 1.0499999523162842,
  "end_rul_days_mean": 261.4234313964844,
  "end_rul_days_min": 0.0,
  "fraction_failed_or_near": 0.3611111111111111,
  "missing_value_fraction": 0.012022569444444445,
  "checksum": "bd08a45a71d0ecea"
}
```
