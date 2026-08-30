"""The self-learning model: autonomous objective, pseudo-label gate, own loop.

These tests are the contract for ``windfusion-auto`` (AutoWind). The defining
property is that the model trains on its own: no ground-truth targets are
consumed by ``autonomous_fit``; only the self-supervised objective and (in
later rounds) the model's own gated predictions drive it.
"""

import torch

from windfusion.config import SelfTrainingConfig
from windfusion.models import MODEL_REGISTRY, create_model
from windfusion.models.auto import self_training_from


def _batch(config, seed=0):
    from windfusion.data import load_fleet
    from windfusion.data.dataset import build_datasets

    gen = torch.Generator().manual_seed(seed)
    runs = load_fleet(config.data)
    bundle = build_datasets(runs, config.data, model_window=config.model.sequence_length, turbine=config.turbine)
    ds = bundle.train
    idx = torch.randperm(len(ds), generator=gen)[:8].tolist()
    keys = ("sequence", "physics", "raw", "target", "data_completeness")
    out = {}
    for key in keys:
        vals = [ds[i][key] for i in idx if key in ds[i]]
        if vals:
            out[key] = torch.stack(vals)
    return out


# ── registry / family ──────────────────────────────────────────────────────


def test_autowind_registered_as_self_learning_family():
    spec = MODEL_REGISTRY["windfusion-auto"]
    assert spec.family == "self-learning"
    assert spec.self_supervised and spec.self_check_head
    assert spec.provenance == ()  # not imported from anywhere: the model is local
    model = create_model("windfusion-auto", 12, 3, 5)
    assert model.spec is spec


def test_registry_table_flags_autonomous_fit():
    from windfusion.models import registry_table

    table = registry_table()
    row = next(line for line in table.splitlines() if "windfusion-auto" in line)
    assert "self-learning (autonomous fit)" in row


# ── objectives ─────────────────────────────────────────────────────────────


def test_forward_contract_identical_to_family():
    model = create_model("windfusion-auto", 12, 3, 5)
    out = model(torch.randn(4, 24, 12), torch.randn(4, 5))
    assert out["mean"].shape == (4, 3)
    assert out["routing"].sum(-1).allclose(torch.ones(4), atol=1e-5)
    assert (out["consistency"] > 0).all()


def test_masked_reconstruction_zeroes_channels_and_scores_last_step():
    model = create_model("windfusion-auto", 12, 3, 5)
    x = torch.randn(8, 24, 12)
    loss = model.masked_reconstruction(x, torch.randn(8, 5), mask_rate=1.0)
    assert torch.isfinite(loss) and loss > 0  # with every channel masked it must still predict


def test_next_step_prediction_is_finite_and_differentiable():
    model = create_model("windfusion-auto", 12, 3, 5)
    loss = model.next_step_prediction(torch.randn(4, 24, 12), torch.randn(4, 5))
    loss.backward()
    grads = [p.grad for p in model.next_step_head.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_autonomous_objective_terms_and_weights(tiny_config):
    batch = _batch(tiny_config)
    model = create_model("windfusion-auto", 12, 3, 5, tiny_config.turbine)
    sc = SelfTrainingConfig()
    full = model.self_supervised_objective(batch, sc)
    floats = full.as_floats()
    for key in ("total", "reconstruction", "next_step", "physics", "consistency", "router_balance"):
        assert key in floats
        assert floats[key] == floats[key]  # finite
        if key in ("reconstruction", "next_step", "physics", "consistency"):
            assert floats[key] >= 0.0  # sum-of-squares terms
    # zeroing every other weight must leave exactly the reconstruction term.
    # torch is re-seeded before each call so the random channel mask repeats.
    zeroed = SelfTrainingConfig(
        next_step_weight=0.0,
        physics_weight=0.0,
        consistency_weight=0.0,
        router_balance_weight=0.0,
    )
    torch.manual_seed(11)
    full = model.self_supervised_objective(batch, sc)
    torch.manual_seed(11)
    only_recon = model.self_supervised_objective(batch, zeroed)
    expected = sc.reconstruction_weight * float(full.reconstruction.detach())
    assert abs(only_recon.as_floats()["total"] - expected) < 1e-5
    # the total decomposes exactly into the weighted terms
    recomposed = (
        sc.reconstruction_weight * floats["reconstruction"]
        + sc.next_step_weight * floats["next_step"]
        + sc.physics_weight * floats["physics"]
        + sc.consistency_weight * floats["consistency"]
        + sc.router_balance_weight * floats["router_balance"]
    )
    assert abs(floats["total"] - recomposed) < 1e-4


# ── pseudo-label gate ──────────────────────────────────────────────────────


def test_pseudo_label_gate_is_a_hard_filter(tiny_config):
    from torch.utils.data import DataLoader

    from windfusion.data import load_fleet
    from windfusion.data.dataset import build_datasets

    runs = load_fleet(tiny_config.data)
    bundle = build_datasets(runs, tiny_config.data, model_window=tiny_config.model.sequence_length)
    loader = DataLoader(bundle.train, batch_size=16)
    model = create_model("windfusion-auto", 12, 3, 5, tiny_config.turbine)

    everything = model.collect_pseudo_labels(
        loader, SelfTrainingConfig(pseudo_label_gate=2.0, mc_samples=2), device="cpu"
    )
    assert everything is not None
    n_all = everything["sequence"].shape[0]

    nothing = model.collect_pseudo_labels(
        loader, SelfTrainingConfig(pseudo_label_gate=0.0, mc_samples=2), device="cpu"
    )
    accepted = 0 if nothing is None else int(nothing["sequence"].shape[0])
    assert accepted == 0
    assert n_all > accepted


# ── the loop that trains on its own ────────────────────────────────────────


def test_autonomous_fit_consumes_no_labels_and_lowers_the_objective(tiny_config):
    from torch.utils.data import DataLoader

    from windfusion.data import load_fleet
    from windfusion.data.dataset import build_datasets

    runs = load_fleet(tiny_config.data)
    bundle = build_datasets(runs, tiny_config.data, model_window=tiny_config.model.sequence_length)
    train = DataLoader(bundle.train, batch_size=32, shuffle=False)

    model = create_model("windfusion-auto", 12, 3, 5, tiny_config.turbine)
    sc = SelfTrainingConfig(epochs=3, rounds=1, pseudo_label_weight=0.0)

    def objective():
        with torch.no_grad():
            vals = [float(model.self_supervised_objective(b, sc).total) for b in train]
        return sum(vals) / len(vals)

    before = objective()
    report = model.autonomous_fit(train, None, config=sc, rounds=1, epochs=3, verbose=False)
    after = objective()
    assert report["best"]["total"] <= before  # best-of tracking
    assert after < before, (before, after)
    assert len(report["history"]) == 3
    assert report["parameters"] == model.parameter_count
    # every history row records the autonomous terms, never a supervised nll
    assert "nll" not in report["history"][0]


def test_autonomous_fit_with_pseudo_label_round(tiny_config):
    from torch.utils.data import DataLoader

    from windfusion.data import load_fleet
    from windfusion.data.dataset import build_datasets

    runs = load_fleet(tiny_config.data)
    bundle = build_datasets(runs, tiny_config.data, model_window=tiny_config.model.sequence_length)
    train = DataLoader(bundle.train, batch_size=32)

    model = create_model("windfusion-auto", 12, 3, 5, tiny_config.turbine)
    report = model.autonomous_fit(
        train,
        None,
        config=SelfTrainingConfig(epochs=1, rounds=2, pseudo_label_gate=2.0, mc_samples=2),
        verbose=False,
    )
    assert report["rounds"] == 2
    assert report["pseudo_labels"] > 0
    assert report["pseudo_stats"]["accepted"] > 0
    assert report["pseudo_stats"]["acceptance_rate"] <= 1.0


def test_checkpoint_round_trip_for_autowind(tiny_config, tmp_path):
    from windfusion.training.loop import load_checkpoint, save_checkpoint

    model = create_model("windfusion-auto", 12, 3, 5, tiny_config.turbine).eval()
    path = save_checkpoint(model, tmp_path / "auto.pt", tiny_config, meta={"note": "autonomous"})
    restored, _, meta = load_checkpoint(path)
    restored.eval()
    x, p = torch.randn(2, 24, 12), torch.randn(2, 5)
    with torch.no_grad():
        assert torch.allclose(model(x, p)["mean"], restored(x, p)["mean"], atol=1e-6)
    assert meta["note"] == "autonomous"
    assert restored.spec.model_id == "windfusion-auto"


# ── config + cli ───────────────────────────────────────────────────────────


def test_self_training_config_validation():
    bad_cases = (
        {"mask_rate": 0.0},
        {"mask_rate": 0.9},
        {"rounds": 0},
        {"epochs": 0},
        {"reconstruction_weight": -1.0},
        {"pseudo_label_gate": -0.1},
    )
    for bad in bad_cases:
        try:
            SelfTrainingConfig(**bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")
    assert self_training_from(None).mask_rate == 0.15


def test_full_config_round_trip_includes_self_training(tmp_path):
    from windfusion.config import WindFusionConfig

    config = WindFusionConfig()
    config = config.with_overrides(**{"self_training.mask_rate": 0.25, "model.mode": "windfusion-auto"})
    path = config.save(tmp_path / "c.yaml")
    reloaded = WindFusionConfig.from_yaml(path)
    assert reloaded.self_training.mask_rate == 0.25
    assert reloaded.model.mode == "windfusion-auto"


def test_cli_self_train_refuses_models_without_the_objective():
    from windfusion.cli import build_parser, main

    parser = build_parser()
    args = parser.parse_args(["self-train", "--mode", "windfusion-lite", "--epochs", "1"])
    assert args.func.__name__ == "_cmd_self_train"
    # running it must refuse cleanly (exit code 2) rather than invent a loop
    rc = main(["self-train", "--mode", "windfusion-lite"])
    assert rc == 2
