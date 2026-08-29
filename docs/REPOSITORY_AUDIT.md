# Repository audit and provenance

Audit date: 2026-08-30. Every repository owned by `rajaram-2005` was inspected read-only through
GitHub and shallow clones into a scratch directory outside this repository. **No source tree was
copied into WindFusion.** Reused ideas are re-implemented and declared in
`windfusion/provenance.py`; the registry is machine-readable and enforced by
`tests/test_provenance_config_safety.py`.

## Repositories inspected

| Repository | Language | Size | Relevant to wind energy |
|---|---|---|---|
| `wind-turbine-pg-bnn` | Python | ~19k LOC in `src/` | Primary research baseline |
| `TurbineDigitalTwin` | HTML/Python (Flask) | dashboard + twin logic | State/scenario semantics |
| `AeroZip-Telemetry-Compression` | Java | single simulator | Telemetry compression policy |
| `Aetheris` | Python | 194 modules | Routing tiers, verify-first, offline-first |
| `ai-machinery-etl-pipeline` | docs + ZIP | n8n/Ollama/Supabase/Langflow | Data-zone conventions |
| `pg-bnn-wind-energy` | Python | this repository | Consolidation target |
| `Personal-Productive-ai`, `Automated-Component-Management-Tracking-System` | Java | unrelated | Not used |

## Decision table

| Repository | Component | Keep | Modify | Replace | Reason |
|---|---|:---:|:---:|:---:|---|
| `wind-turbine-pg-bnn` | Bayes-by-backprop BNN, heteroscedastic NLL | ✓ | ✓ | | Preserve the uncertainty decomposition; MC dropout replaces weight uncertainty for edge cost |
| `wind-turbine-pg-bnn` | Heier Cp, Betz bound, Jensen wake, ISO 281 L10, lumped RC thermal | ✓ | ✓ | | Published relations; re-derived with provenance and shared by training and data generation |
| `wind-turbine-pg-bnn` | PINO/FNO wake operator | | | ✓ | Valuable teacher/baseline, too expensive for the default edge path |
| `wind-turbine-pg-bnn` | Row-oriented BNN, synthetic generator | | ✓ | | Replaced by a causal temporal model and a physics-driven generator with known latent damage |
| `wind-turbine-pg-bnn` | ECE, early-warning horizon, lead-time protocol | ✓ | | | Adopted verbatim as the evaluation protocol so numbers stay comparable |
| `wind-turbine-pg-bnn` | Fault taxonomy, limits vocabulary | ✓ | | | Adopted as the label/report vocabulary |
| `wind-turbine-pg-bnn` | ONNX export + quantisation contract | ✓ | ✓ | | Kept, with a mandatory runtime parity check added |
| `wind-turbine-pg-bnn` | Flower federation, Reptile, Hermes onboarding | ✓ | ✓ | | Protocols re-implemented locally without the Flower runtime, with explicit gates |
| `wind-turbine-pg-bnn` | FastAPI/Streamlit/Gradio/Flutter/k8s/notifications/media | | | ✓ | Application surface, not model architecture |
| `TurbineDigitalTwin` | State vocabulary, what-if scenarios | ✓ | ✓ | | Kept; Random Forest replaced by the learned model + accumulated damage |
| `TurbineDigitalTwin` | Flask auth/UI, committed `.env` | | | ✓ | Not an architecture; committed secrets are unsafe practice |
| `AeroZip` | Anomaly bypass, delta/deadband/quantisation | ✓ | ✓ | | Generalised from one fixed threshold to a multi-signal risk policy with hysteresis |
| `AeroZip` | Java simulator | | | ✓ | Re-implemented in NumPy so ratio and fidelity are measurable here |
| `Aetheris` | Model router + tiers, verify-first, offline-first config | ✓ | ✓ | | Recast as a sparse expert router, a physics self-check and a versioned YAML config |
| `Aetheris` | Assistant engine, FastAPI, UI, plugins (194 modules) | | | ✓ | Unrelated general-purpose application |
| `ai-machinery-etl-pipeline` | Data-zone layout, artefact provenance | ✓ | | | Adopted as directory + checksum convention |
| `ai-machinery-etl-pipeline` | n8n/Ollama/Supabase/Langflow stack, ZIP archive | | | ✓ | Opaque archive is not a maintainable research dependency |

## Findings

1. `wind-turbine-pg-bnn` is by far the strongest baseline and the only repository containing
   uncertainty, physics, calibration, active learning, federation and deployment code. Its breadth
   is also its weakness: duplicated anomaly logic, a large operational surface, and demonstrations
   that report synthetic campaigns as if they were evidence.
2. No public dataset was found in any repository. Every published number in the upstream
   repositories comes from data that is not redistributed here, so **no upstream benchmark value
   is comparable to a WindFusion number** and none is copied.
3. `TurbineDigitalTwin` ships a committed `.env` and no licence file. Neither is vendored; only
   the state vocabulary is reused.
4. `ai-machinery-etl-pipeline` is distributed mainly as a ZIP archive. Its durable contribution is
   the data-zone separation, which is adopted here as a directory convention.
5. `AeroZip`'s real contribution is the *policy* (bypass compression during anomalies), not the
   Java codec. The policy is generalised; the codec is re-implemented.

## Validation status

Every declared concept is covered by at least one test (`windfusion provenance --validate` returns
no problems). "Validated" here means *exercised by the test suite of this repository*, not
validated on real turbines.

## Known gaps

- No licensed real-world dataset; real-data metrics stay `NOT MEASURED`.
- Teacher checkpoints are produced by running this repository's own trainer; adapters for
  external teacher checkpoints are not implemented.
- Energy per inference and on-device RAM need platform tooling.
- A native (C++/Rust) on-device runtime that exploits sparse routing is future work; the exported
  graph currently evaluates every expert.
- The literature review required to claim novelty has not been performed; see
  `docs/RESEARCH_CONTRIBUTION.md`.
