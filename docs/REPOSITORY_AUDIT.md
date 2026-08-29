# Repository audit and provenance

Audit date: 2026-08-30. Repositories were inspected read-only through GitHub and shallow working copies. WindFusion-Lite is a clean-room synthesis: no source tree was copied into this repository.

| Repository | Component | Keep | Modify | Replace | Reason |
|---|---|:---:|:---:|:---:|---|
| `wind-turbine-pg-bnn` | Bayes-by-Backprop PG-BNN, heteroscedastic NLL | ✓ | ✓ | | Preserve uncertainty decomposition; use cheaper MC-dropout head for edge student |
| `wind-turbine-pg-bnn` | Heier Cp, Betz bound, drivetrain torque/L10, lumped thermal equations | ✓ | ✓ | | Valid domain priors; normalize residuals and make losses independently weighted |
| `wind-turbine-pg-bnn` | PINO/FNO wake operator | | | ✓ | Valuable teacher/research baseline but too expensive for default edge path |
| `wind-turbine-pg-bnn` | row-oriented BNN and synthetic generator | | ✓ | | Introduce causal temporal learning; retain synthetic/real labeling discipline |
| `wind-turbine-pg-bnn` | active learning, SHAP, Flower federation | ✓ | ✓ | | Keep as optional research integrations, not default runtime dependencies |
| `wind-turbine-pg-bnn` | ONNX/C++ inference and extensive tests | ✓ | ✓ | | Retain deployment contract; rebuild against the new three-output interface |
| `wind-turbine-pg-bnn` | Hermes onboarding/meta-learning | | ✓ | | Teacher/adaptation concept only; fail-closed verification remains mandatory |
| `wind-turbine-pg-bnn` | APIs, Flutter, web dashboard, media | | | ✓ | Not part of the research model core; avoid application bloat |
| `pg-bnn-wind-energy` | Empty initial repository | | ✓ | | Suitable independent destination; no legacy implementation to preserve |
| `TurbineDigitalTwin` | What-if scenarios and state vocabulary | ✓ | ✓ | | Replace dashboard-coupled Random Forest with model-connected typed state |
| `TurbineDigitalTwin` | Flask auth/UI and bundled `.env` | | | ✓ | UI is not model architecture; committed secrets/config are unsafe practice |
| `AeroZip-Telemetry-Compression` | anomaly bypass | ✓ | ✓ | | Generalize fixed thresholds to uncertainty/rate/physics/safety risk policy |
| `AeroZip-Telemetry-Compression` | Java simulator/fixed compression | | | ✓ | New NumPy policy is testable and integrates model signals; codec can remain external |
| `Aetheris` | verify-first cognition, model registry, offline-first design | ✓ | ✓ | | Recast as physics-confidence self-verification and compact model family |
| `Aetheris` | FastAPI/UI/general assistant engine | | | ✓ | Large unrelated application; not useful per edge CPU cycle |
| `ai-machinery-etl-pipeline` | ETL orchestration concept | ✓ | ✓ | | Keep data zones and adapters; n8n/Ollama/Supabase/Langflow are optional, not core |
| `ai-machinery-etl-pipeline` | ZIP-only implementation | | | ✓ | Opaque archive is not a maintainable research dependency |
| `AeroZip-Telemetry-Compression` / PG-BNN | duplicated anomaly logic | | | ✓ | One adaptive telemetry policy avoids divergent thresholds |

## Findings

The primary PG-BNN repository is the strongest research baseline: it contains uncertainty, modular physics, synthetic telemetry, calibration, active learning, federated learning and deployment code. Its breadth also creates duplication and a large operational surface. TurbineDigitalTwin contributes useful state/scenario semantics but its dashboard and Random Forest are not tightly coupled to physical state evolution. AeroZip's key contribution is fidelity-preserving anomaly bypass, not its particular Java codec. Aetheris contributes the useful architectural pattern of explicit routing and verification, but its general-purpose assistant stack is unrelated.

No public dataset was found inside the small ETL/AeroZip repositories. The PG-BNN demonstrations report synthetic campaigns; those are baselines, not field evidence. No benchmark value from any repository is copied as a WindFusion-Lite result.

## Known gaps / incomplete work

- Real multi-farm, multi-OEM evaluation data and licenses are not present here.
- Teacher checkpoint adapters and matched teacher/student feature maps require versioned checkpoints.
- ONNX export exists, but runtime parity depends on installing the optional export dependencies.
- Energy and peak-RAM measurement need platform-specific tooling.
- C++ runtime is specified as follow-up and is not falsely marked complete.
