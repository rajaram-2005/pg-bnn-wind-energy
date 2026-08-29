from windfusion.verification.verifier import PhysicsConfidenceFusion, Verdict


def test_fail_closed_verdicts():
    v = PhysicsConfidenceFusion()
    assert v.verify(0.9, 0.8, 0.8, 0.9).verdict == Verdict.MODEL_UNCERTAIN
    assert v.verify(0.1, 0.1, 0.1, 0.9, data_completeness=0.3).verdict == Verdict.INSUFFICIENT_DATA
