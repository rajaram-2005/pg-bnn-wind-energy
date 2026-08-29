# WindFusion-Lite model card

## Intended use
Research and advisory monitoring of wind-turbine SCADA/condition data: fault risk, health and RUL estimation, uncertainty reporting, counterfactual scenarios and telemetry prioritization. It must not control a turbine.

## Models
`windfusion-lite` is default, `windfusion-edge` minimizes cost, and `windfusion-research` expands capacity. `aetheris-wind` is the base architecture; mythology-inspired research presets are `ra-wind` (Egyptian/thermal), `qinglong-wind` (Chinese/wake), `vayu-wind` (Hindu/gust), `odin-wind` (Norse/cold drivetrain), and `aeolus-wind` (Greek/forecast). Names are respectful identifiers, not cultural representations or performance claims.

## Training and evaluation data
No weights or training data ship in v0.1. Data directories explicitly separate raw, processed, feature, synthetic, validation and benchmark assets. No real-world accuracy has been measured. Synthetic results must be labeled synthetic.

## Uncertainty
A heteroscedastic head estimates aleatoric noise. MC dropout approximates epistemic uncertainty. Approximation quality must be checked with NLL, Brier score, ECE and coverage on shifted sites and OEMs.

## Limitations and failure modes
Missing/miscalibrated sensors, icing, curtailment, maintenance regime changes, unseen turbine types, distribution shift and weak labels can invalidate predictions. Physics approximations use simplified steady/lumped relations. Sparse routing can collapse without balancing. High uncertainty or physical disagreement yields `MODEL_UNCERTAIN`, not false confidence.

## Safety
Advisory-only. Never connect output directly to pitch, yaw, braking, converter or protection controls. Qualified engineers must review decisions. Existing OEM protections remain authoritative.

## Deployment
Python 3.10+, PyTorch 2+, CPU-first. ONNX is optional. Validate numerical parity, calibration, latency, RAM and failure behavior on target hardware before deployment.
