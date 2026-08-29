# Research contribution statement

## Existing ideas reused

Bayesian predictive uncertainty, heteroscedastic Gaussian NLL, physics residual regularization, aerodynamic/drivetrain/thermal equations, digital-twin what-if analysis, anomaly-triggered telemetry bypass, knowledge distillation, and sparse mixtures of experts all have prior art. The repository audit records local lineage; a formal literature review is still required.

## Newly designed here

WindFusion-Lite combines a causal gated TCN, top-k domain routing, four compact experts, a model-connected turbine state, adaptive information-preserving telemetry, and **Physics-Confidence Fusion** in one small interface. Aetheris is the balanced base; Ra, Qinglong, Vayu, Odin and Aeolus are capacity/specialization presets, not separate giant networks.

## Experimental hypothesis

> A lightweight physics-confidence fusion architecture may improve the accuracy–calibration–latency tradeoff for wind-turbine predictive maintenance by routing temporal SCADA representations through sparse domain experts and verifying outputs against physical consistency, uncertainty and digital-twin state.

This is a hypothesis, not an established novelty or performance claim. Controlled baselines, real data and literature search are required.

## Ablations

`benchmarks/ablations.yaml` defines: pure neural; +physics; +uncertainty; +sparse experts; +digital twin; full model; full without adaptive telemetry; full without verification. Unrun cells must remain `NOT MEASURED`.
