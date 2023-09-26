from dataclasses import replace

import numpy as np
import pytest
import torch

from asterion.contracts.errors import ContractError
from asterion.contracts.geometry import Geometry, collate
from asterion.graphs.components import components
from asterion.graphs.radius import RadiusGraph, geometry_edges
from asterion.models.potential import ModelConfig, make_model


def test_components_include_singletons():
    batch = collate(
        [
            Geometry(
                np.array([6, 6, 6]), np.array([[0.0, 0, 0], [1.0, 0, 0], [10.0, 0, 0]]), "fixture"
            )
        ]
    )
    assert components(batch, RadiusGraph(2)(batch), 2) == ((0, 1), (2,))


def test_components_preserve_batch_boundaries(geometry):
    batch = collate([geometry, geometry])
    edges = RadiusGraph(20)(batch)
    n = len(geometry.numbers)
    assert components(batch, edges, 20) == (tuple(range(n)), tuple(range(n, 2 * n)))


def test_components_ignore_skin_only_edges():
    batch = collate([Geometry(np.array([6, 6]), np.array([[0.0, 0, 0], [5.2, 0, 0]]), "fixture")])
    assert components(batch, RadiusGraph(5.5)(batch), 5) == ((0,), (1,))


def test_components_reject_cross_graph_edges(geometry):
    batch = collate([geometry, geometry])
    edges = geometry_edges(batch, torch.tensor([0]), torch.tensor([len(geometry.numbers) + 1]))
    with pytest.raises(ContractError, match="graph boundary"):
        components(batch, edges, 5)


def test_components_reject_negative_indices(geometry):
    batch = collate([geometry])
    edges = RadiusGraph(5)(batch)
    edges = replace(edges, source=edges.source.clone())
    edges.source[0] = -1
    with pytest.raises(ContractError, match="outside atom"):
        components(batch, edges, 5)


def test_components_reject_nonfinite_distances(geometry):
    batch = collate([geometry])
    edges = RadiusGraph(5)(batch)
    edges = replace(edges, distance=edges.distance.clone())
    edges.distance[0] = float("nan")
    with pytest.raises(ContractError, match="finite"):
        components(batch, edges, 5)


def test_trained_domain_allows_separated_copy_additivity(geometry):
    composition = tuple(int(np.count_nonzero(geometry.numbers == z)) for z in (1, 6, 7, 8))
    model = make_model(ModelConfig(width=4, depth=1, radial=4, composition=composition)).double()
    separate = Geometry(
        np.tile(geometry.numbers, 2),
        np.concatenate([geometry.positions, geometry.positions + 20]),
        "separated",
    )
    first, force = model(collate([geometry]))
    second, twice = model(collate([separate]))
    assert torch.allclose(second, 2 * first, atol=1e-10, rtol=1e-10)
    assert torch.allclose(twice, force.repeat(2, 1), atol=1e-10, rtol=1e-10)


def test_trained_domain_rejects_broken_molecule_components(geometry):
    composition = tuple(int(np.count_nonzero(geometry.numbers == z)) for z in (1, 6, 7, 8))
    model = make_model(ModelConfig(width=4, depth=1, radial=4, composition=composition)).double()
    positions = geometry.positions.copy()
    positions[0] += 30
    with pytest.raises(ContractError, match="component composition"):
        model(collate([replace(geometry, positions=positions)]))
