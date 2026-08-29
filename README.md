# WindFusion-Lite

**A new CPU-first, offline, physics-confidence architecture for wind-turbine predictive maintenance.** This repository does not merge or overwrite the source repositories. It treats their models as research baselines and builds a smaller independent student system.

> Status: research scaffold v0.1. No pretrained weights and no claims of field accuracy. Advisory-only; never turbine control.

## Why

Wind turbines are temporal, multi-physics systems. A row-wise black-box predictor can be confident for the wrong reason, while running every specialist model wastes edge resources. WindFusion-Lite targets useful intelligence per parameter and CPU cycle while exposing uncertainty and physical disagreement.

## Architecture

```text
SCADA → Adaptive encoder → physics features → causal gated TCN
                                             ↓
                                      top-k sparse router
                          ┌──────────┬────────┬─────────┐
                        Aero       Drive    Thermal    Grid
                          └──────────┴────────┴─────────┘
                                             ↓
                              aleatoric + epistemic head
                                             ↓
                    digital twin → Physics-Confidence Fusion
                                             ↓
                  NORMAL | WARNING | CRITICAL | MODEL_UNCERTAIN
```

Only selected experts execute for each batch. Routing is returned with every prediction. The verifier combines prediction, uncertainty, normalized physics residuals, temporal consistency and digital-twin health instead of blindly accepting a neural score.

## Model family

- `windfusion-lite`, `windfusion-edge`, `windfusion-research`
- Aetheris base: `aetheris-wind`
- Five specialist architecture presets: `ra-wind` (Egyptian), `qinglong-wind` (Chinese), `vayu-wind` (Hindu), `odin-wind` (Norse), `aeolus-wind` (Greek)

These labels identify engineering presets; they are not pretrained models or representations of cultures.

## Objective

```text
L = heteroscedastic_NLL + Σ λᵢ physics_residualᵢ²
    + λᵤ uncertainty_regularization + λᵣ router_balance
```

Distillation supports detached teacher predictions, uncertainty, compatible representations and physics consistency. Existing systems remain external and unchanged.

## Install and use

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
python -m windfusion benchmark --mode windfusion-lite
pytest
```

```python
import torch
from windfusion.models import create_model

model = create_model("aetheris-wind", input_features=12)
out = model(torch.randn(2, 24, 12), torch.randn(2, 4))
print(out["mean"], out["routing"])
```

Commands: `train`, `evaluate`, `benchmark`, `distill`, `export`, `simulate`, `explain`. Commands requiring a dataset intentionally remain explicit scaffolds rather than generating fabricated results.

## Lineage and new work

The PG-BNN supplies research baselines for uncertainty and physics; TurbineDigitalTwin supplies state/scenario concepts; AeroZip supplies anomaly bypass; Aetheris inspires offline routing and verify-first behavior; the ETL project informs data zones. New code introduces compact temporal sparse experts, adaptive multi-signal telemetry and Physics-Confidence Fusion. See [audit](docs/REPOSITORY_AUDIT.md), [contribution statement](docs/RESEARCH_CONTRIBUTION.md), and [model card](docs/MODEL_CARD.md).

## Data and reproducibility

Data zones are present but datasets are excluded from Git. Record dataset checksum, split, seed, config and model version. Default seed is 7. Benchmark tables remain `NOT MEASURED` until experiments run on versioned data. See `benchmarks/RESULTS.md` and ablations.

## Edge deployment

Export with `python -m windfusion export model.onnx --mode windfusion-edge`. Validate ONNX Runtime parity and quantization on target hardware. The local benchmark reports measured parameter count, FP32 size and CPU latency; RAM and energy remain unreported until platform tools are available.

## Limitations

The physics models are simplified; MC dropout is approximate Bayesian inference; real cross-farm/OEM validation is absent; and C++ inference is future work. Read the model card before experimentation.

## Citation

If used in research, cite this repository and commit hash. A formal paper/DOI is not yet available.

## License

MIT. Upstream concepts and repositories retain their own licenses and attribution.
