# Real-data benchmark template (NOT MEASURED)

This template is intentionally **empty**. No licensed wind-turbine dataset is bundled with
WindFusion v0.2, so every cell below is `NOT MEASURED`. Fill it in only after supplying data and
recording its provenance.

## Dataset provenance (fill before running)

| Field | Value |
|---|---|
| Dataset name | NOT MEASURED |
| Licence / terms of use | NOT MEASURED |
| Provider / OEM | NOT MEASURED |
| Sites / turbines | NOT MEASURED |
| Sampling interval | NOT MEASURED |
| Date range | NOT MEASURED |
| Checksum (SHA-256 of the raw archive) | NOT MEASURED |
| Split definition (grouped by turbine? by site? by time?) | NOT MEASURED |
| Label definition (what counts as a failure, and the horizon) | NOT MEASURED |
| Known censoring / maintenance interventions | NOT MEASURED |

## 1. Predictive quality

| Model | health MAE | RUL MAE (days) | warning F1 | AUROC | ECE | NLL | 90% coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| mean | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| ridge | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| mlp | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| gru | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| windfusion-edge | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| windfusion-lite | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| windfusion-research | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| aetheris-wind | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| ra-wind | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| qinglong-wind | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| vayu-wind | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| odin-wind | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| aeolus-wind | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |

## 2. Cross-site / cross-OEM generalisation

| Protocol | RUL MAE (days) | ECE | abstention rate |
|---|---:|---:|---:|
| in-distribution test split | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| held-out site (zero-shot) | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| held-out site + few-shot | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| held-out OEM (zero-shot) | NOT MEASURED | NOT MEASURED | NOT MEASURED |

## 3. Verification behaviour on real data

| Model | abstention rate | accuracy on decided | false critical | missed critical |
|---|---:|---:|---:|---:|
| windfusion-lite | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| aetheris-wind | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |

## 4. On-device measurements

| Device | Model | latency ms | peak RAM MB | energy / inference |
|---|---|---:|---:|---:|
| NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |

## Rules for filling this in

1. Record the dataset provenance table first; a metric without provenance is not usable.
2. Keep splits grouped by turbine (or site) — row-level splits leak the latent state.
3. Report the seed, config, git commit and data checksum with every run.
4. Never copy a number from the synthetic report into this file.
5. If a cell cannot be measured, leave it `NOT MEASURED`.
