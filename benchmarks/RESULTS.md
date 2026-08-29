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
Not generated yet in this working copy — run:

```bash
python benchmarks/synthetic_benchmark.py                  # full run
python benchmarks/synthetic_benchmark.py --quick          # smoke run
```

The run writes `benchmarks/results/synthetic-v0.2.0.{json,md}` and refreshes this section.
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
