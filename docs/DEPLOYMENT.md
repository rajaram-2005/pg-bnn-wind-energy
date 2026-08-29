# Deployment

## Export

```bash
python -m windfusion export artifacts/edge.onnx --mode windfusion-edge
python -m windfusion export artifacts/edge.pt   --mode windfusion-edge   # TorchScript fallback
```

- Opset 17, dynamic axes for batch and time, named inputs `scada`, `physics` (and `neighbors`
  for Qinglong) and outputs `mean`, `log_var`, `routing`.
- The exported graph evaluates **all** experts (`dense_experts=True`). It is numerically identical
  to the sparse runtime path (`tests/test_export.py::test_dense_export_path_matches_sparse_runtime`)
  but removes data-dependent control flow that static runtimes cannot trace. On-device sparse
  execution needs a native runtime — future work.
- A parity check against ONNX Runtime is **mandatory** by default (`--no-parity` disables it, for
  environments without `onnxruntime`). If the max absolute difference exceeds 1e-4 the export raises.
- Dynamic INT8 quantisation size is reported; accuracy after quantisation is `NOT MEASURED` and
  must be re-checked on the target device.

## Edge runtime

```python
from windfusion.edge.runtime import EdgeRuntime

runtime = EdgeRuntime(model, config, window=24, infer_every=6, mc_samples=4)
step = runtime.push(sample)        # dict or array in canonical channel order
```

- Warm-up: the first `window - 1` samples return `warmup`, never a prediction.
- Cadence: `infer_every` bounds the CPU budget; skipped windows still get a telemetry decision.
- Missing sensors: NaNs are imputed and reported through `data_completeness`, which feeds the
  verifier; low completeness produces `INSUFFICIENT_DATA` rather than a guess.
- Drift: PSI against a reference feature distribution (`DriftMonitor`), reported in `status()`.
- Degradation: on an inference exception the runtime serves the last known good prediction and
  marks it stale, with the exception type in `note`.

## Telemetry

```python
from windfusion.edge.telemetry import AdaptiveTelemetryPolicy

decision = policy.decide(anomaly=..., epistemic=..., rate=..., physics_residual=..., safety=...)
encoded = policy.encode(window, decision)
```

Modes: `raw` (bypass, full resolution), `detailed` (every 2nd sample), `compressed`
(delta + deadband + quantisation, every 10th sample). Hysteresis and a cooldown prevent flapping.
`evaluate_policy` reports bandwidth reduction alongside reconstruction error so the trade-off is
measured, not assumed.

Codec contract: in `compressed` mode the first sample is stored separately as a **DC baseline** and
the payload is the int32 quantised deltas. Baseline and deltas are kept apart so the payload keeps a
single dtype (a mixed float/int payload is silently promoted to float64 and doubles the byte count),
and so the decoder reconstructs `baseline + cumsum(deltas)` instead of integrating from zero and
losing the operating point. `reconstruction_error(original, decoded, keep_every)` compares
`decoded[k]` against `original[k * keep_every]`: a naive row-for-row comparison measures grid
misalignment, not codec error.

## Pre-deployment checklist

- [ ] Numerical parity: PyTorch vs ONNX Runtime (and vs the quantised model).
- [ ] Calibration re-measured on the target site (ECE, coverage) before and after quantisation.
- [ ] Latency, throughput and peak RSS measured on the **target** device.
- [ ] Failure behaviour reviewed: missing sensors, drift, inference exceptions, stale fallback.
- [ ] Verdict thresholds reviewed with a reliability engineer; abstention rate is acceptable.
- [ ] Advisory-only enforced: no path from model output to any actuator.
- [ ] Model version, git commit, data checksum and config recorded with the artefact
      (all written into the checkpoint by `save_checkpoint`).

## Not available in v0.2

A native C++/Rust runtime, GPU/accelerator kernels, and on-device incremental training. Energy per
inference and on-device RAM are `NOT MEASURED`.
