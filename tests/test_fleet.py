"""Fleet learning: federated averaging, Reptile and few-shot adaptation."""

import copy

import torch

from windfusion.data.loaders import build_dataloaders, per_turbine_loaders
from windfusion.models import create_model
from windfusion.training.fleet import (
    client_weight,
    federated_average,
    fewshot_adapt,
    pairwise_distance,
    reptile_meta_train,
    support_loader,
)


def _bundle(tiny_config):
    return build_dataloaders(tiny_config)


def test_federated_average_matches_manual_weighted_mean():
    a = {"w": torch.tensor([1.0, 3.0])}
    b = {"w": torch.tensor([3.0, 7.0])}
    averaged = federated_average([a, b], [1.0, 3.0])
    expected = (1.0 * a["w"] + 3.0 * b["w"]) / 4.0
    assert torch.allclose(averaged["w"], expected)


def test_client_weight_is_positive_and_physics_discounted(tiny_config):
    bundle = _bundle(tiny_config)
    model = create_model("windfusion-edge", 12, 3, 5, tiny_config.turbine)
    loader = list(per_turbine_loaders(bundle.datasets.train, batch_size=16).values())[0]
    plain = client_weight(model, loader, physics_aware=False)
    aware = client_weight(model, loader, physics_aware=True)
    assert plain > 0
    assert 0 < aware <= plain


def test_federated_round_changes_the_global_model(tiny_config):
    bundle = _bundle(tiny_config)
    model = create_model("windfusion-edge", 12, 3, 5, tiny_config.turbine)
    before = copy.deepcopy(model.state_dict())
    loaders = list(per_turbine_loaders(bundle.datasets.train, batch_size=16).values())[:2]
    from windfusion.training.fleet import federated_round

    report = federated_round(model, loaders, tiny_config, local_epochs=1)
    assert report.clients == 2
    assert pairwise_distance(before, model.state_dict()) > 0


def test_fewshot_adaptation_moves_parameters_and_reduces_loss(tiny_config):
    from windfusion.training.losses import windfusion_loss

    bundle = _bundle(tiny_config)
    model = create_model("windfusion-edge", 12, 3, 5, tiny_config.turbine)
    support = support_loader(bundle.datasets.train, n_windows=8, batch_size=8)
    before = copy.deepcopy(model.state_dict())

    def loss_of(state):
        probe = copy.deepcopy(model)
        probe.load_state_dict(state)
        probe.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for batch in support:
                outputs = probe(batch["sequence"], batch["physics"], batch.get("neighbors"))
                total += float(windfusion_loss(outputs, batch, tiny_config, spec=probe.spec).total)
                count += 1
        return total / max(count, 1)

    initial = loss_of(before)
    adapted = fewshot_adapt(model, support, tiny_config, steps=5)
    after = loss_of(adapted.state_dict())
    assert pairwise_distance(before, adapted.state_dict()) > 0
    assert after <= initial + 1e-6


def test_reptile_changes_the_model(tiny_config):
    bundle = _bundle(tiny_config)
    model = create_model("windfusion-edge", 12, 3, 5, tiny_config.turbine)
    before = copy.deepcopy(model.state_dict())
    loaders = list(per_turbine_loaders(bundle.datasets.train, batch_size=16).values())[:2]
    reptile_meta_train(model, loaders, tiny_config, iterations=1)
    assert pairwise_distance(before, model.state_dict()) > 0


def test_support_loader_is_deterministic(tiny_config):
    bundle = _bundle(tiny_config)
    a = list(support_loader(bundle.datasets.train, 8, seed=1))
    b = list(support_loader(bundle.datasets.train, 8, seed=1))
    assert torch.allclose(a[0]["sequence"], b[0]["sequence"])
