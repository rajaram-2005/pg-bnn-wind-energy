# WindFusion v0.2

**Physics-confidence fusion for wind-turbine predictive maintenance** — a CPU-first, offline,
advisory-only research architecture that unifies the wind-energy work spread across the
`rajaram-2005` repositories into one maintained codebase.

> **Status: research software.** No pretrained weights, no field validation, no control path.
> Results measured in this repository are measured on a **simulated** fleet and are stamped
> `SYNTHETIC`. Anything not measured is written as `NOT MEASURED` — never estimated.

---

## What v0.2 adds over WindFusion-Lite (v0.1)

| Area | v0.1 | v0.2 |
|---|---|---|
| Model family | 8 width presets, one architecture | **9 architectures**: Aetheris base, 5 mythology-inspired presets (Ra/Qinglong/Vayu/Odin/Aeolus), 3 capacity tiers |
| Training | loss functions only | end-to-end loop: seeding, warmup+cosine schedule, early stopping, checkpoints with data checksums |
| Distillation | loss function | `DistillTrainer`: response + uncertainty + feature + physics + **router-agreement** transfer |
| Calibration | none | temperature scaling + distribution-free conformal intervals, with before/after ECE |
| Evaluation | infrastructure only | full metric suite (regression, early warning, ECE/NLL/CRPS, coverage) + 4 non-neural/neural baselines |
| Data | layout only | deterministic physics-driven synthetic fleet with **known latent damage**, group splits, checksums |
| Digital twin | state + what-if | damage-accumulating twin, rollouts, twin-coupled verification |
| Telemetry | policy + stub codec | risk policy with hysteresis/cooldown + working delta/deadband/quant codec with measured fidelity |
| Fleet | - | FedAvg over turbine clients (physics-aware weights) + Reptile meta-adaptation + few-shot site adaptation + gated onboarding |
| Edge | - | `EdgeRuntime`: ring buffer, cadence, drift monitoring (PSI), graceful degradation |
| Provenance | audit document | machine-readable registry (`windfusion provenance`), validated by tests |

## Architecture

```text
SCADA window (B, T, 12)  ──►  causal multi-scale gated TCN  ──►  pooled representation
physics features (B, 5) ──►  physics encoder ─────────────────┘        │
neighbours (B, K, 12) ──►  cross-asset attention (Qinglong)           │
                                                                      ▼
                                                          top-k sparse router
                              ┌──────────┬─────────┬─────────┬────────┴──┐
                            Aero      Drive    Thermal    Grid       Wake
                              └──────────┴─────────┴─────────┴────────────┘
                                                      │
                                       heteroscedastic head (mean + log-var)
                                       self-check head (Aetheris verify-first)
                                                      │
     digital twin ──► Physics-Confidence Fusion ──► NORMAL | WARNING | CRITICAL
                                                  | MODEL_UNCERTAIN | INSUFFICIENT_DATA
```

Only the selected experts execute per sample, and the routing decision is returned with every
prediction. A verdict is only emitted after uncertainty, physics residuals, twin health, temporal
consistency and data completeness are checked — otherwise the system abstains.

## Model family

| Model | Family | Inductive bias |
|---|---|---|
| `aetheris-wind` | base | balanced trunk + **self-check head** (Aetheris verify-first) |
| `ra-wind` | Egyptian | deeper thermal expert, ambient/thermal-modulated router bias |
| `qinglong-wind` | Chinese | cross-asset attention over neighbours + **wake expert** (Jensen deficit) |
| `vayu-wind` | Hindu | long dilated receptive field (up to 16 steps), top-1 routing, gust statistics |
| `odin-wind` | Norse | deeper drivetrain expert, ISO 281 damage feature, **monotone RUL penalty** |
| `aeolus-wind` | Greek | auxiliary multi-horizon wind/power **forecast decoder** |
| `windfusion-edge` / `-lite` / `-research` | tiers | 9.7k / 50k / 467k parameters for edge, default and teacher roles |

The mythology names identify **engineering presets** — what the architecture looks at, which
physics residual it is pushed to respect, and how much capacity it spends. They are not
pretrained models, not cultural representations, and carry no performance claim.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,export,baselines]'
pytest                       # 132 tests
```

## Usage

```python
import torch
from windfusion.models import create_model

model = create_model("aetheris-wind", input_features=12, physics_features=5)
out = model(torch.randn(2, 24, 12), torch.randn(2, 5))
prediction = model.predict(torch.randn(2, 24, 12), torch.randn(2, 5), samples=16)
print(prediction["mean"], prediction["epistemic"], prediction["routing"])
```

```bash
python -m windfusion models                    # list the family
python -m windfusion provenance --markdown     # where every reused concept comes from
python -m windfusion train --mode odin-wind --epochs 25 --out artifacts/checkpoints
python -m windfusion evaluate --checkpoint artifacts/checkpoints/odin-wind.pt
python -m windfusion benchmark --modes windfusion-edge,windfusion-research
python -m windfusion distill --teacher windfusion-research --student windfusion-edge
python -m windfusion export artifacts/edge.onnx --mode windfusion-edge
python -m windfusion simulate --gearbox-temperature 95
python -m windfusion explain --mode ra-wind --text
python -m windfusion fleet --mode windfusion-edge --rounds 3
```

## Reproducible benchmarks

```bash
python benchmarks/synthetic_benchmark.py                     # full run (~30 min, 2 cores)
python benchmarks/synthetic_benchmark.py --quick              # smoke run (~30 s)
```

The harness writes `benchmarks/results/synthetic-v0.2.0.{json,md}` with the fleet checksum, seed,
environment, per-model metrics, distillation, fleet adaptation, telemetry and verification tables.
Every table is stamped **SYNTHETIC**. `benchmarks/results/REAL_DATA_TEMPLATE.md` is the empty
template to fill once a licensed dataset is available — no real-data result is ever invented.

See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for the protocol and
[`benchmarks/RESULTS.md`](benchmarks/RESULTS.md) for the tables.

### What the latest synthetic run actually shows

Seeded fleet (`bd08a45a71d0ecea`), 36 turbines × 1440 samples, 15 epochs, all numbers SYNTHETIC:

- **Distillation works**: the 9.7k-parameter student goes from 0.1033 → **0.0889** health MAE and
  111.9 → **94.8** days RUL MAE, slightly better than its own 467k teacher (99.1 days).
- **Fleet adaptation is mixed**: few-shot adaptation on 32 windows improves a held-out site
  (111.9 → **104.2** days RUL MAE), but federated averaging gave **no** benefit in this
  configuration (111.9 → 112.2). The negative result is reported too.
- **The baselines are not pushovers.** On this fleet the GRU baseline beats every WindFusion model
  on RUL MAE (83.1 days vs 96.6 for the best preset) and on early-warning F1 (0.699 vs 0.553).
  WindFusion's margin is elsewhere: interval coverage (0.94–0.98 vs 0.85) and NLL, i.e. it knows
  better *when it does not know*. A headline that claims WindFusion wins outright would be false.
- **Verification never abstains on this split** (0.000): the test windows are in-distribution and
  ~99% complete, so nothing trips the thresholds. The abstention path is covered by unit tests
  instead, and `docs/SAFETY.md` states plainly that the verifier is not the deployment gate.

## Lineage and provenance

WindFusion v0.2 is a clean-room synthesis. No upstream source tree was copied; each reused idea is
declared in `windfusion/provenance.py` with its upstream path, licence and reuse kind
(`concept`, `equation`, `schema`, `protocol`), and the registry is enforced by tests.

| Repository | What is reused | What is deliberately not reused |
|---|---|---|
| `wind-turbine-pg-bnn` | heteroscedastic NLL, aleatoric/epistemic split, Heier Cp + Betz, ISO 281 L10, lumped RC thermal, soft-hinge limits, ECE/early-warning protocol, fault vocabulary, ONNX contract, FedAvg + Reptile protocols, Hermes onboarding gates, advisory-only safety | PINO/FNO wake operator (too costly for the edge path), Flutter/web/API/notification stack, Java/Streamlit applications |
| `TurbineDigitalTwin` | typed state vocabulary and what-if semantics | Flask dashboard, bundled Random Forest, committed `.env` |
| `AeroZip-Telemetry-Compression` | anomaly bypass, delta/deadband/quantisation | the Java simulator and its fixed thresholds |
| `Aetheris` | tiered routing registry, verify-first self-check, offline-first config | the general assistant engine, FastAPI/UI surface |
| `ai-machinery-etl-pipeline` | raw/processed/feature/synthetic/validation/benchmark data zones | n8n/Ollama/Supabase/Langflow stack, ZIP-only archive |

Full audit: [`docs/REPOSITORY_AUDIT.md`](docs/REPOSITORY_AUDIT.md).
Generated registry: [`docs/PROVENANCE.md`](docs/PROVENANCE.md).

## Safety

Advisory only. `safety.allow_actuation` is a hard invariant enforced in code
(`SafetyConfig.__post_init__` raises if it is ever set), and every report carries the disclaimer.
Never connect output to pitch, yaw, braking, converter or protection controls.

## Documentation

[Architecture](docs/ARCHITECTURE.md) · [Model card](docs/MODEL_CARD.md) ·
[Benchmarks](docs/BENCHMARKS.md) · [Deployment](docs/DEPLOYMENT.md) ·
[Provenance](docs/PROVENANCE.md) · [Audit](docs/REPOSITORY_AUDIT.md) ·
[Research contribution](docs/RESEARCH_CONTRIBUTION.md) · [Safety](docs/SAFETY.md)

## Licence

MIT. Upstream repositories retain their own licences and attribution.
