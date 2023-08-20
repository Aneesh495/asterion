import numpy as np
import pytest
import torch

from asterion.contracts.geometry import Geometry, collate
from asterion.evaluation.checks import assert_symmetry, symmetry_trial
from asterion.graphs.radius import NeighborCache


@pytest.mark.parametrize("seed", range(12))
def test_o3_and_additivity(model, geometry, seed):
    assert_symmetry(symmetry_trial(model, geometry, seed))


def test_singleton_and_zero_neighbor(model):
    geometry = Geometry(np.array([1]), np.zeros((1, 3)), "singleton")
    e, f = model(collate([geometry]), create_graph=True)
    assert e.shape == (1,)
    assert torch.equal(f, torch.zeros_like(f))


def test_cache_force_gradient(model, geometry):
    batch = collate([geometry])
    batch.positions.requires_grad_(True)
    cache = NeighborCache(model.config.cutoff)
    expected = model(collate([geometry]))
    actual = model(batch, edges=cache(batch))
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_cached_edges_rebind_after_coordinate_clone(model, geometry):
    batch = collate([geometry])
    cache = NeighborCache(model.config.cutoff)
    edges = cache(batch)
    energy, force = model(batch, edges=edges)
    expected_energy, expected_force = model(collate([geometry]))
    torch.testing.assert_close(energy, expected_energy)
    torch.testing.assert_close(force, expected_force)


def test_collinear(model):
    geometry = Geometry(
        np.array([1, 6, 1]),
        np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "linear",
    )
    e, f = model(collate([geometry]), create_graph=True)
    assert torch.isfinite(e).all() and torch.isfinite(f).all()
    loss = f.square().sum()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


@pytest.mark.parametrize("distance", [4.99, 5.0, 5.01, 5.49, 5.5, 5.51])
def test_cutoff_continuity(model, distance):
    geometry = Geometry(np.array([6, 6]), np.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]), "pair")
    e, f = model(collate([geometry]))
    assert torch.isfinite(e).all()
    if distance >= 5:
        assert f.abs().max() < 1e-12
    else:
        assert f.abs().max() < 1e-4
