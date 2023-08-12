import numpy as np
import pytest
import torch

from asterion.contracts.geometry import collate
from asterion.evaluation.checks import force_difference, hessian_difference


@pytest.mark.parametrize("step", [1e-4, 1e-5, 2e-6])
def test_force_central_energy(model, geometry, step):
    assert force_difference(model, geometry, step)["relative"] < 1e-4


@pytest.mark.parametrize("seed", range(4))
def test_hessian_direction(model, geometry, seed):
    direction = np.random.default_rng(seed).normal(size=geometry.positions.shape)
    direction /= np.linalg.norm(direction)
    assert hessian_difference(model, geometry, direction)["relative"] < 1e-4


def test_genuine_force_loss_double_backward(model, geometry):
    before = {key: value.detach().clone() for key, value in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    energy, force = model(collate([geometry]), create_graph=True)
    force_loss = (force - torch.ones_like(force) * 0.1).square().mean()
    force_loss.backward()
    gradient = sum(p.grad.square().sum().item() for p in model.parameters() if p.grad is not None)
    assert gradient > 1e-12
    optimizer.step()
    assert any(not torch.equal(before[k], p.detach()) for k, p in model.named_parameters())
