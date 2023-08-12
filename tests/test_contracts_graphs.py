from dataclasses import replace

import numpy as np
import pytest
import torch

from asterion.contracts.errors import ContractError, ResourceError
from asterion.contracts.geometry import Chemistry, Geometry, collate
from asterion.contracts.units import ACCELERATION, KCAL_MOL_TO_EV, KINETIC_ENERGY
from asterion.graphs.radial import smooth_cutoff
from asterion.graphs.radius import NeighborCache, RadiusGraph


def test_independent_si_units():
    assert abs(KCAL_MOL_TO_EV - 0.0433641042418) < 1e-12
    assert abs(ACCELERATION - 0.009648533202185) < 1e-15
    assert abs(KINETIC_ENERGY * ACCELERATION - 1) < 1e-15


@pytest.mark.parametrize(
    "chemistry",
    [
        Chemistry(charge=1),
        Chemistry(multiplicity=3),
        Chemistry(periodic=True),
        Chemistry(elements=(2,)),
    ],
)
def test_unsupported_chemistry(geometry, chemistry):
    with pytest.raises(ContractError):
        replace(geometry, chemistry=chemistry).validate()


@pytest.mark.parametrize("seed", range(30))
def test_independent_radius_graph(seed):
    rng = np.random.default_rng(seed)
    geometries = [Geometry(np.ones(n, dtype=int), rng.normal(size=(n, 3)), str(n)) for n in (3, 7)]
    batch = collate(geometries)
    cutoff = 0.5 + rng.random() * 3
    edges = RadiusGraph(cutoff, block_size=2)(batch)
    expected = []
    for target in range(len(batch.numbers)):
        for source in range(len(batch.numbers)):
            if source != target and batch.graph[source] == batch.graph[target]:
                distance = np.linalg.norm(
                    batch.positions[source].numpy() - batch.positions[target].numpy()
                )
                if distance < cutoff:
                    expected.append((target, source))
    assert list(zip(edges.target.tolist(), edges.source.tolist())) == expected


def test_cache_skin_and_batch_invalidation(geometry):
    cache = NeighborCache(2.0, 0.4)
    cache(collate([geometry]))
    moved = replace(geometry, positions=geometry.positions + 0.05)
    cache(collate([moved]))
    assert cache.rebuilds == 1
    cache(collate([replace(geometry, positions=geometry.positions + 0.2)]))
    assert cache.rebuilds == 2
    cache(collate([geometry, geometry]))
    assert cache.rebuilds == 3


def test_resource_bound(geometry):
    with pytest.raises(ResourceError):
        RadiusGraph(5, maximum_edges=1)(collate([geometry]))


@pytest.mark.parametrize("radius", [0.0, 5.0, 5.01])
def test_cutoff_second_derivatives(radius):
    r = torch.tensor([radius], dtype=torch.float64, requires_grad=True)
    value = smooth_cutoff(r, 5)
    first = torch.autograd.grad(value.sum(), r, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), r)[0]
    assert first.abs().max() < 1e-12 and second.abs().max() < 1e-12
