# Benchmark protocol

Two different things get measured in this repository, and they are never mixed:

1. **Infrastructure metrics** — parameters, FP32/INT8 size, CPU latency, throughput, peak RSS.
   Measured on the current machine with random inputs. Real, but hardware-specific.
2. **Predictive metrics** — MAE/RMSE per target, early-warning precision/recall/F1/AUROC, ECE,
   NLL, CRPS, empirical coverage, abstention and false-critical rates. These require data, and in
   v0.2 the only bundled data is **synthetic**.

Anything that cannot be measured (energy per inference, on-device RAM, field accuracy, real-data
metrics) is reported as the literal string `NOT MEASURED`.

## Synthetic benchmark

```bash
python benchmarks/synthetic_benchmark.py                  # 36 turbines x 1440 samples, 15 epochs
python benchmarks/synthetic_benchmark.py --quick          # 6 turbines x 400 samples, 3 epochs
```

Stages: fleet generation (checksummed) → footprint → supervised training of 4 baselines and 9
models → distillation (teacher, student from scratch, distilled student) → fleet learning
(zero-shot, federated, few-shot on a held-out site) → telemetry policy → verification behaviour.

Output: `benchmarks/results/synthetic-v0.2.0.{json,md}` with the fleet checksum, seed, environment,
thread count and total runtime. Regenerating with the same seed on the same machine reproduces the
same numbers; changing the fleet, seed or epoch budget produces a new, differently-stamped run.

### Why a synthetic fleet at all

Because health and RUL are *labels* in most datasets and *state variables* in this generator:
damage is integrated explicitly, so the ground truth is exact rather than heuristically derived.
That makes the benchmark well posed and every error meaningful — but it says nothing about
transfer to a real asset.

### Data discipline

- Splits are **grouped by turbine**: no turbine appears in two splits, so the latent damage state
  cannot leak into its own test set.
- Normalisation (median/IQR) is fitted on the training split only.
- The fleet is a pure function of its config; `fleet_checksum` pins the exact bytes.

## Real-data benchmark

Not performed. `benchmarks/results/REAL_DATA_TEMPLATE.md` is the template: supply a licensed
dataset, record its provenance (licence, checksum, site/OEM, split definition) and fill the table.
Until then every real-data cell stays `NOT MEASURED`.

## Baselines

`mean` (train mean), `ridge` (flattened-window ridge regression), `mlp` (mean-pooled window),
`gru` (recurrent reference for the TCN encoder). All are scored by the same metric code
(`windfusion.evaluation.metrics`) as the neural models, so comparisons are apples to apples.

## Comparing against the upstream repositories

Do not. The upstream numbers come from different data, different splits and different label
definitions; no cross-repository comparison is valid. WindFusion baselines exist so that this
repository's models are never compared only against themselves.
