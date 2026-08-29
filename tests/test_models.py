"""Model family contract: shapes, finite outputs, NaN robustness, sparsity."""

import torch

from windfusion.models import MODEL_REGISTRY, create_model, registry_table


def test_every_registered_model_runs():
    for mode in MODEL_REGISTRY:
        model = create_model(mode, 12, 3, 5)
        neighbors = torch.randn(2, 3, 12) if model.spec.neighbors else None
        out = model(torch.randn(2, 24, 12), torch.randn(2, 5), neighbors)
        assert out["mean"].shape == (2, 3)
        assert out["log_var"].shape == (2, 3)
        assert torch.isfinite(out["mean"]).all()
        assert model.parameter_count > 0


def test_routing_is_sparse_and_normalised():
    model = create_model("windfusion-lite", 12, 3, 5)
    out = model(torch.randn(8, 24, 12), torch.randn(8, 5))
    weights = out["routing"]
    assert torch.allclose(weights.sum(-1), torch.ones(8), atol=1e-5)
    assert (weights > 0).sum(-1).max().item() <= model.spec.top_k


def test_missing_sensors_do_not_propagate_nan():
    x = torch.randn(2, 24, 12)
    x[0, 5, 3] = float("nan")
    x[1, :, 7] = float("nan")
    out = create_model("aetheris-wind", 12, 3, 5)(x, torch.zeros(2, 5))
    assert torch.isfinite(out["mean"]).all()


def test_uncertainty_decomposition_is_consistent():
    model = create_model("windfusion-lite", 12, 3, 5)
    prediction = model.predict(torch.randn(4, 24, 12), torch.randn(4, 5), samples=8)
    for key in ("mean", "aleatoric", "epistemic", "total"):
        assert torch.isfinite(prediction[key]).all()
    assert (prediction["total"] >= prediction["aleatoric"] - 1e-6).all()


def test_mc_dropout_changes_predictions_but_mode_is_restored():
    model = create_model("windfusion-lite", 12, 3, 5).eval()
    x, p = torch.randn(2, 24, 12), torch.randn(2, 5)
    single = model(x, p)["mean"]
    mc = model.predict(x, p, samples=6)["mean"]
    assert model.training is False
    assert not torch.allclose(single, mc, atol=1e-6)


def test_mythology_presets_differ_architecturally():
    specs = {mode: MODEL_REGISTRY[mode] for mode in MODEL_REGISTRY}
    assert specs["qinglong-wind"].neighbors == 3
    assert "wake" in specs["qinglong-wind"].extra_experts
    assert specs["aeolus-wind"].forecast_horizon > 0
    assert specs["odin-wind"].monotone_rul
    assert specs["odin-wind"].damage_feature
    assert specs["vayu-wind"].dilations[-1] > specs["ra-wind"].dilations[-1]
    assert specs["ra-wind"].expert_depth["thermal"] > specs["ra-wind"].expert_depth["aero"]
    assert specs["aetheris-wind"].self_check_head


def test_aetheris_base_exposes_self_check_and_verified_prediction():
    model = create_model("aetheris-wind", 12, 3, 5)
    out = model(torch.randn(2, 24, 12), torch.randn(2, 5))
    assert "consistency" in out and (out["consistency"] > 0).all()
    result = model.predict_verified(torch.randn(1, 24, 12), torch.randn(1, 5), samples=2)
    assert result["verdict"] in {
        "NORMAL",
        "WARNING",
        "CRITICAL",
        "MODEL_UNCERTAIN",
        "INSUFFICIENT_DATA",
    }


def test_forecast_head_shape_for_aeolus():
    model = create_model("aeolus-wind", 12, 3, 5)
    out = model(torch.randn(3, 24, 12), torch.randn(3, 5))
    assert out["forecast"].shape == (3, model.spec.forecast_horizon, 2)


def test_registry_table_lists_every_model():
    table = registry_table()
    for mode in MODEL_REGISTRY:
        assert mode in table


def test_encoder_is_causal():
    from windfusion.models.encoder import MultiScaleCausalEncoder

    encoder = MultiScaleCausalEncoder(8, blocks=2, dilations=(1, 2))
    x = torch.randn(1, 16, 8)
    out = encoder(x)
    x[:, 8:] = 0.0  # future cannot affect the past
    out_perturbed = encoder(x)
    assert torch.allclose(out[:, :8], out_perturbed[:, :8], atol=1e-6)
