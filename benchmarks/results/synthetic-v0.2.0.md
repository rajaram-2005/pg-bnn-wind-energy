# WindFusion v0.2.0 - synthetic benchmark results

> **SYNTHETIC (simulated fleet, not field data).** Every number below was measured on a simulated fleet with a known latent damage state. It is a regression and ablation harness, not evidence of field performance.

- Generated: 2026-08-29T20:11:40Z
- Fleet checksum: `bd08a45a71d0ecea`
- Data: 36 turbines x 1440 samples, 6 sites
- Seed: 7, epochs: 15
- Machine: Linux-6.1.158+-x86_64-with-glibc2.36 (torch 2.13.0+cu130, 1 threads)

## 1. Model footprint

| Model | Parameters | FP32 MB | INT8 MB | CPU latency ms (median) | windows/s |
|---|---:|---:|---:|---:|---:|
| `aetheris-wind` | 87,643 | 0.334 | 0.2867 | 1.577 | 634.2 |
| `ra-wind` | 53,130 | 0.203 | 0.1624 | 1.613 | 619.9 |
| `qinglong-wind` | 91,613 | 0.349 | 0.2206 | 1.832 | 546.0 |
| `vayu-wind` | 56,444 | 0.215 | 0.1817 | 1.508 | 663.1 |
| `odin-wind` | 71,006 | 0.271 | 0.2202 | 1.601 | 624.7 |
| `aeolus-wind` | 92,518 | 0.353 | 0.2867 | 1.533 | 652.2 |
| `windfusion-edge` | 9,714 | 0.037 | 0.0281 | 0.784 | 1275.0 |
| `windfusion-lite` | 50,330 | 0.192 | 0.1623 | 1.377 | 726.2 |
| `windfusion-research` | 467,211 | 1.782 | 1.5129 | 3.181 | 314.3 |

## 2. Predictive quality (synthetic test split)

| Model | Parameters | health MAE | RUL MAE (days) | warning F1 | AUROC | ECE | NLL | 90% coverage | train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mean` | 0 | 0.2829 | 282.05 | 0.317 | 0.500 | 0.1100 | 1.079 | 0.717 | 0.0 |
| `ridge` | 879 | 0.1023 | 124.51 | 0.453 | 0.949 | 0.0268 | 0.235 | 0.827 | 0.8 |
| `mlp` | 10,755 | 0.1127 | 125.81 | 0.387 | 0.964 | 0.0645 | 0.318 | 0.843 | 1.7 |
| `gru` | 16,131 | 0.0844 | 83.14 | 0.699 | 0.983 | 0.0508 | -0.118 | 0.851 | 8.0 |
| `windfusion-edge` | 9,714 | 0.1113 | 115.28 | 0.496 | 0.951 | 0.1602 | 0.108 | 0.949 | 50.4 |
| `windfusion-lite` | 50,330 | 0.0984 | 110.99 | 0.450 | 0.972 | 0.0747 | -0.148 | 0.964 | 67.1 |
| `windfusion-research` | 467,211 | 0.0743 | 90.52 | 0.448 | 0.961 | 0.1747 | -0.344 | 0.970 | 190.5 |
| `aetheris-wind` | 87,643 | 0.0741 | 93.38 | 0.337 | 0.967 | 0.1776 | -0.304 | 0.973 | 103.2 |
| `ra-wind` | 53,130 | 0.0879 | 111.08 | 0.431 | 0.977 | 0.1060 | -0.240 | 0.962 | 68.3 |
| `qinglong-wind` | 91,613 | 0.0748 | 122.21 | 0.336 | 0.980 | 0.1411 | -0.242 | 0.976 | 95.8 |
| `vayu-wind` | 56,444 | 0.0811 | 98.01 | 0.552 | 0.979 | 0.1515 | -0.308 | 0.970 | 100.8 |
| `odin-wind` | 71,006 | 0.0789 | 96.72 | 0.373 | 0.972 | 0.1574 | -0.275 | 0.976 | 74.2 |
| `aeolus-wind` | 92,518 | 0.0771 | 105.79 | 0.433 | 0.972 | 0.1587 | -0.187 | 0.965 | 81.9 |

## 3. Distillation (teacher -> student)

| Variant | Parameters | health MAE | RUL MAE (days) | ECE |
|---|---:|---:|---:|---:|
| `windfusion-research` | 467,211 | 0.0743 | 90.52 | 0.1747 |
| `windfusion-edge` | 9,714 | 0.1113 | 115.28 | 0.1602 |
| `windfusion-edge+distilled` | 9,714 | 0.1656 | 131.19 | 0.1913 |

Knowledge-distillation terms on the final epoch: {'kd_total': 101041306.428263, 'kd_prediction': 0.017547, 'kd_uncertainty': 0.518895, 'kd_feature': 0.030241, 'kd_router': 7.142872}

## 4. Fleet learning (held-out site)

| Protocol | RUL MAE (days) |
|---|---:|
| zero-shot (no adaptation) | 115.28 |
| + federated averaging | 111.44 |
| + few-shot (32 windows) | 116.24 |

## 5. Verification behaviour (fail-closed)

| Model | abstention rate | coverage | accuracy on decided | false critical | missed critical |
|---|---:|---:|---:|---:|---:|
| `windfusion-edge` | 0.029 | 0.971 | 0.597 | 0.032 | 0.000 |
| `windfusion-lite` | 0.029 | 0.971 | 0.511 | 0.032 | 0.000 |
| `windfusion-research` | 0.029 | 0.971 | 0.494 | 0.032 | 0.000 |
| `aetheris-wind` | 0.029 | 0.971 | 0.279 | 0.032 | 0.000 |
| `ra-wind` | 0.029 | 0.971 | 0.504 | 0.032 | 0.000 |
| `qinglong-wind` | 0.029 | 0.971 | 0.277 | 0.032 | 0.000 |
| `vayu-wind` | 0.029 | 0.971 | 0.673 | 0.032 | 0.000 |
| `odin-wind` | 0.029 | 0.971 | 0.366 | 0.032 | 0.000 |
| `aeolus-wind` | 0.030 | 0.970 | 0.513 | 0.032 | 0.000 |

## 6. Adaptive telemetry

- windows: 256, modes: {'raw': 151, 'compressed': 105}
- bandwidth reduction: 0.368
- mean |error| on reconstruction: 16.46377

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
