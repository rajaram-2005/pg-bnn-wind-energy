# WindFusion v0.2.0 - synthetic benchmark results

> **SYNTHETIC (simulated fleet, not field data).** Every number below was measured on a simulated fleet with a known latent damage state. It is a regression and ablation harness, not evidence of field performance.

- Generated: 2026-08-29T20:48:04Z
- Fleet checksum: `bd08a45a71d0ecea`
- Data: 36 turbines x 1440 samples, 6 sites
- Seed: 7, epochs: 15
- Machine: Linux-6.1.158+-x86_64-with-glibc2.36 (torch 2.13.0+cu130, 1 threads)

## 1. Model footprint

| Model | Parameters | FP32 MB | INT8 MB | CPU latency ms (median) | windows/s |
|---|---:|---:|---:|---:|---:|
| `aetheris-wind` | 87,643 | 0.334 | 0.2867 | 1.493 | 669.8 |
| `ra-wind` | 53,130 | 0.203 | 0.1624 | 1.443 | 692.9 |
| `qinglong-wind` | 91,613 | 0.349 | 0.2206 | 1.781 | 561.5 |
| `vayu-wind` | 56,444 | 0.215 | 0.1817 | 1.603 | 623.9 |
| `odin-wind` | 71,006 | 0.271 | 0.2202 | 1.652 | 605.2 |
| `aeolus-wind` | 92,518 | 0.353 | 0.2867 | 1.495 | 669.0 |
| `windfusion-edge` | 9,714 | 0.037 | 0.0281 | 0.783 | 1276.5 |
| `windfusion-lite` | 50,330 | 0.192 | 0.1623 | 1.334 | 749.6 |
| `windfusion-research` | 467,211 | 1.782 | 1.5129 | 3.177 | 314.7 |

## 2. Predictive quality (synthetic test split)

| Model | Parameters | health MAE | RUL MAE (days) | warning F1 | AUROC | ECE | NLL | 90% coverage | train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mean` | 0 | 0.2829 | 282.05 | 0.317 | 0.500 | 0.1100 | 1.079 | 0.717 | 0.0 |
| `ridge` | 879 | 0.1023 | 124.51 | 0.453 | 0.949 | 0.0268 | 0.235 | 0.827 | 1.0 |
| `mlp` | 10,755 | 0.1127 | 125.81 | 0.387 | 0.964 | 0.0645 | 0.318 | 0.843 | 1.9 |
| `gru` | 16,131 | 0.0844 | 83.14 | 0.699 | 0.983 | 0.0508 | -0.118 | 0.851 | 7.8 |
| `windfusion-edge` | 9,714 | 0.1044 | 112.44 | 0.418 | 0.956 | 0.0921 | -0.061 | 0.942 | 52.0 |
| `windfusion-lite` | 50,330 | 0.0984 | 110.99 | 0.450 | 0.972 | 0.0747 | -0.148 | 0.964 | 63.9 |
| `windfusion-research` | 467,211 | 0.0983 | 98.41 | 0.376 | 0.977 | 0.1234 | -0.199 | 0.966 | 132.8 |
| `aetheris-wind` | 87,643 | 0.0942 | 109.93 | 0.342 | 0.980 | 0.1417 | -0.092 | 0.974 | 63.6 |
| `ra-wind` | 53,130 | 0.0879 | 111.08 | 0.431 | 0.977 | 0.1060 | -0.240 | 0.962 | 65.9 |
| `qinglong-wind` | 91,613 | 0.0748 | 122.21 | 0.336 | 0.980 | 0.1411 | -0.242 | 0.976 | 89.5 |
| `vayu-wind` | 56,444 | 0.0811 | 98.01 | 0.552 | 0.979 | 0.1515 | -0.308 | 0.970 | 93.1 |
| `odin-wind` | 71,006 | 0.0789 | 96.72 | 0.373 | 0.972 | 0.1574 | -0.275 | 0.976 | 67.5 |
| `aeolus-wind` | 92,518 | 0.0732 | 100.06 | 0.317 | 0.972 | 0.1605 | -0.296 | 0.966 | 73.9 |

## 3. Distillation (teacher -> student)

| Variant | Parameters | health MAE | RUL MAE (days) | ECE |
|---|---:|---:|---:|---:|
| `windfusion-research` | 467,211 | 0.0983 | 98.41 | 0.1234 |
| `windfusion-edge` | 9,714 | 0.1044 | 112.44 | 0.0921 |
| `windfusion-edge+distilled` | 9,714 | 0.0901 | 94.05 | 0.1247 |

Knowledge-distillation terms on the final epoch: {'kd_total': 0.335267, 'kd_prediction': 0.018606, 'kd_uncertainty': 0.394465, 'kd_feature': 0.017153, 'kd_router': 4.645218}

## 4. Fleet learning (held-out site)

| Protocol | RUL MAE (days) |
|---|---:|
| zero-shot (no adaptation) | 112.44 |
| + federated averaging | 111.67 |
| + few-shot (32 windows) | 106.02 |

## 5. Verification behaviour (fail-closed)

| Model | abstention rate | coverage | accuracy on decided | false critical | missed critical |
|---|---:|---:|---:|---:|---:|
| `windfusion-edge` | 0.000 | 1.000 | 0.465 | 0.077 | 0.010 |
| `windfusion-lite` | 0.000 | 1.000 | 0.494 | 0.067 | 0.011 |
| `windfusion-research` | 0.000 | 1.000 | 0.379 | 0.090 | 0.003 |
| `aetheris-wind` | 0.000 | 1.000 | 0.287 | 0.052 | 0.011 |
| `ra-wind` | 0.000 | 1.000 | 0.455 | 0.021 | 0.054 |
| `qinglong-wind` | 0.000 | 1.000 | 0.267 | 0.045 | 0.012 |
| `vayu-wind` | 0.000 | 1.000 | 0.658 | 0.043 | 0.019 |
| `odin-wind` | 0.000 | 1.000 | 0.351 | 0.073 | 0.008 |
| `aeolus-wind` | 0.000 | 1.000 | 0.180 | 0.062 | 0.004 |

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
