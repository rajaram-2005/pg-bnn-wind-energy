from windfusion.models.experts import EXPERT_NAMES


def explain(
    prediction: dict,
    physics_residuals: dict,
    twin_state: dict,
    top_features: list[str] | None = None,
) -> dict:
    routing = prediction["routing"][0].tolist()
    dominant = EXPERT_NAMES[max(range(4), key=routing.__getitem__)]
    return {
        "prediction": prediction["mean"][0].tolist(),
        "aleatoric_uncertainty": prediction.get("aleatoric", [])[0].tolist(),
        "epistemic_uncertainty": prediction.get("epistemic", [])[0].tolist(),
        "dominant_expert": dominant,
        "expert_weights": dict(zip(EXPERT_NAMES, routing)),
        "top_features": top_features or [],
        "physics_residuals": physics_residuals,
        "digital_twin_state": twin_state,
        "recommended_action": "Review by a qualified reliability engineer; no automatic control action.",
    }
