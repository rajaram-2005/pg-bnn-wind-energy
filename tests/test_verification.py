"""Fail-closed verification behaviour."""

from windfusion.verification.verifier import (
    PhysicsConfidenceFusion,
    VerificationResult,
    Verdict,
    VerifierThresholds,
)


def test_insufficient_data_abstains_first():
    result = PhysicsConfidenceFusion().verify(0.9, 0.1, 0.1, 0.9, data_completeness=0.3)
    assert result.verdict == Verdict.INSUFFICIENT_DATA
    assert result.confidence == 0.0


def test_high_epistemic_uncertainty_abstains():
    result = PhysicsConfidenceFusion().verify(0.9, 0.8, 0.8, 0.9)
    assert result.verdict == Verdict.MODEL_UNCERTAIN


def test_physics_disagreement_alone_can_abstain():
    thresholds = VerifierThresholds(physics_abstain=0.9)
    result = PhysicsConfidenceFusion(thresholds).verify(0.6, 0.05, 0.5, 0.9)
    assert result.verdict == Verdict.MODEL_UNCERTAIN


def test_clean_evidence_yields_severity_ordering():
    verifier = PhysicsConfidenceFusion()
    normal = verifier.verify(0.05, 0.02, 0.02, 0.98)
    warning = verifier.verify(0.6, 0.05, 0.05, 0.6)
    critical = verifier.verify(0.95, 0.02, 0.02, 0.1)
    assert normal.verdict == Verdict.NORMAL
    assert warning.verdict == Verdict.WARNING
    assert critical.verdict == Verdict.CRITICAL
    # confidence tracks evidence quality, not severity
    inconsistent = verifier.verify(0.95, 0.02, 0.5, 0.1)
    assert inconsistent.confidence < critical.confidence
    uncertain = verifier.verify(0.95, 0.3, 0.02, 0.1)
    assert uncertain.confidence < critical.confidence


def test_low_confidence_downgrades_critical_to_warning():
    verifier = PhysicsConfidenceFusion(
        VerifierThresholds(min_confidence_for_critical=0.99)
    )
    result = verifier.verify(0.95, 0.05, 0.1, 0.05)
    assert result.verdict == Verdict.WARNING


def test_rul_feeds_the_concern_score():
    verifier = PhysicsConfidenceFusion()
    soon = verifier.verify(0.1, 0.02, 0.02, 0.95, rul_days=2.0, warning_horizon_days=30.0)
    late = verifier.verify(0.1, 0.02, 0.02, 0.95, rul_days=300.0, warning_horizon_days=30.0)
    assert soon.concern > late.concern


def test_result_serialisation_and_advice_are_present():
    result = PhysicsConfidenceFusion().verify(0.1, 0.1, 0.1, 0.9)
    assert isinstance(result, VerificationResult)
    payload = result.as_dict()
    assert payload["verdict"] in {v.value for v in Verdict}
    assert payload["advised_action"]


def test_disclaimer_is_always_advisory():
    payload = PhysicsConfidenceFusion().verify(0.99, 0.0, 0.0, 0.01).as_dict()
    assert "do not actuate" in payload["advised_action"].lower()
