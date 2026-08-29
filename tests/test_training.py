"""Training loop, losses and checkpoint round-trip."""

import torch

from windfusion.config import WindFusionConfig
from windfusion.data.loaders import build_dataloaders
from windfusion.models import create_model
from windfusion.training.loop import Trainer, load_checkpoint, save_checkpoint, set_determinism
from windfusion.training.losses import (
    heteroscedastic_nll,
    monotone_rul_penalty,
    windfusion_loss,
)


def _single_batch(config):
    from windfusion.data.loaders import build_dataloaders

    bundle = build_dataloaders(config)
    return next(iter(bundle.train))


def test_loss_terms_are_finite_and_gradients_flow(tiny_config):
    model = create_model("windfusion-lite", 12, 3, 5, tiny_config.turbine)
    batch = _single_batch(tiny_config)
    outputs = model(batch["sequence"], batch["physics"])
    breakdown = windfusion_loss(outputs, batch, tiny_config, spec=model.spec)
    assert torch.isfinite(breakdown.total)
    breakdown.total.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_training_reduces_the_objective(tiny_config):
    config = tiny_config.with_overrides(**{"training.epochs": 3})
    bundle = build_dataloaders(config)
    model = create_model("windfusion-lite", 12, 3, 5, config.turbine)
    trainer = Trainer(model, config, verbose=False)
    history = trainer.fit(bundle.train, bundle.val)
    first, last = history.epochs[0]["total"], history.epochs[-1]["total"]
    assert last <= first


def test_nll_penalises_miscalibration():
    target = torch.ones(8, 1)          # observation one sigma away from the mean
    mean = torch.zeros(8, 1)
    overconfident = heteroscedastic_nll(mean, torch.full_like(mean, -6.0), target)
    calibrated = heteroscedastic_nll(mean, torch.zeros_like(mean), target)
    underconfident = heteroscedastic_nll(mean, torch.full_like(mean, 4.0), target)
    assert overconfident > calibrated
    assert underconfident > calibrated


def test_monotone_penalty_prefers_consistent_ordering():
    proxy = torch.tensor([0.1, 0.5, 0.9])
    bad = torch.tensor([100.0, 90.0, 10.0])  # RUL falls as damage grows: correct
    good = torch.tensor([10.0, 90.0, 100.0])  # RUL rises as damage grows: wrong
    torch.manual_seed(0)
    assert monotone_rul_penalty(bad, proxy) < monotone_rul_penalty(good, proxy)


def test_monotonicity_term_only_applies_to_the_odin_preset(tiny_config):
    batch = _single_batch(tiny_config)
    results = {}
    for mode in ("odin-wind", "windfusion-edge"):
        model = create_model(mode, 12, 3, 5, tiny_config.turbine)
        outputs = model(batch["sequence"], batch["physics"])
        results[mode] = windfusion_loss(outputs, batch, tiny_config, spec=model.spec)
    assert float(results["windfusion-edge"].monotone) == 0.0
    assert float(results["odin-wind"].monotone) >= 0.0
    assert torch.isfinite(results["odin-wind"].monotone)


def test_checkpoint_round_trip_preserves_predictions(tiny_config, tmp_path):
    model = create_model("windfusion-edge", 12, 3, 5, tiny_config.turbine).eval()
    path = save_checkpoint(model, tmp_path / "m.pt", tiny_config, meta={"note": "test"})
    restored, config, meta = load_checkpoint(path)
    restored.eval()  # both models must be in the same mode (MC dropout changes outputs)
    x, p = torch.randn(2, 24, 12), torch.randn(2, 5)
    with torch.no_grad():
        a = model(x, p)["mean"]
        b = restored(x, p)["mean"]
    assert torch.allclose(a, b, atol=1e-6)
    assert meta["note"] == "test"
    assert config.model.mode == tiny_config.model.mode


def test_determinism_across_two_runs(tiny_config):
    config = tiny_config.with_overrides(**{"training.epochs": 1})
    losses = []
    for _ in range(2):
        set_determinism(config.seed)  # seed before construction: init is part of the experiment
        bundle = build_dataloaders(config)
        model = create_model("windfusion-edge", 12, 3, 5, config.turbine)
        history = Trainer(model, config, verbose=False).fit(bundle.train, bundle.val)
        losses.append(history.epochs[-1]["total"])
    assert abs(losses[0] - losses[1]) < 1e-4
