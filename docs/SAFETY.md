# Safety

## Invariant

**Advisory only.** WindFusion never actuates. This is not a documentation note:
`SafetyConfig.__post_init__` raises `ValueError` if `advisory_only` is disabled or
`allow_actuation` is enabled, and `tests/test_provenance_config_safety.py` asserts that behaviour.
A YAML file that tries to relax it fails to load.

## Why

Wind turbines are actuated by certified control and protection systems (pitch, yaw, converter,
brake). A learned model that is wrong in a novel regime — sensor failure, icing, curtailment, an
unseen fault mode — can be confidently wrong. The failure mode of an advisory system must be
silence, not action.

## Fail-closed behaviour

The verifier abstains when evidence is insufficient:

- `INSUFFICIENT_DATA` — data completeness below 0.6 (sensor outage, commissioning).
- `MODEL_UNCERTAIN` — epistemic uncertainty above 0.45, physics consistency below 0.35, a
  self-check predicting a large residual, or a high failure probability contradicted by physics
  while the twin reports the asset healthy.

Abstention rates are reported in every benchmark table; a model that never abstains is not a
better model, it is an uncalibrated one.

## Operational rules

1. Outputs are reviewed by a qualified reliability engineer before any maintenance decision.
2. OEM protection systems remain authoritative; WindFusion neither replaces nor overrides them.
3. Predictions on turbines, farms or OEMs outside the training distribution require re-validation
   and re-calibration.
4. Every artefact must carry its model id, git commit, config and data checksum
   (all recorded by `save_checkpoint`).
5. Metrics measured on the synthetic fleet must be labelled SYNTHETIC in any report, and must
   never be presented as field performance.

## Reporting a concern

If an output appears unsafe, do not act on it: record the artefact id, config and data checksum,
and escalate to the engineering owner.
