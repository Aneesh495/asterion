import math
from dataclasses import dataclass

import torch

from asterion.contracts.errors import ContractError, ResourceError
from asterion.contracts.geometry import Batch


@dataclass
class Edges:
    source: torch.Tensor
    target: torch.Tensor
    vector: torch.Tensor
    distance: torch.Tensor
    graph: torch.Tensor

    @property
    def size(self):
        return len(self.source)


def geometry_edges(batch: Batch, source: torch.Tensor, target: torch.Tensor) -> Edges:
    vector = batch.positions[source] - batch.positions[target]
    distance = torch.linalg.vector_norm(vector, dim=-1)
    if len(source) and (distance <= 1e-8).any():
        raise ContractError("zero-distance directed edge")
    return Edges(source, target, vector, distance, batch.graph[target])


class RadiusGraph:
    """Blocked all-pairs graph with lexicographic (target,source) directed edges."""

    def __init__(self, cutoff=5.0, maximum_edges=2_000_000, block_size=128, max_atoms=8192):
        if not math.isfinite(cutoff) or cutoff <= 0 or block_size < 1 or maximum_edges < 1:
            raise ContractError("invalid radius graph limits")
        self.cutoff = cutoff
        self.maximum_edges = maximum_edges
        self.block_size = block_size
        self.max_atoms = max_atoms

    def topology(self, batch: Batch):
        batch.validate(self.max_atoms)
        positions = batch.positions.detach()
        sources = []
        targets = []
        total = 0
        for graph in range(batch.size):
            begin, end = map(int, batch.pointers[graph : graph + 2].tolist())
            for offset in range(begin, end, self.block_size):
                stop = min(end, offset + self.block_size)
                delta = positions[begin:end][None] - positions[offset:stop][:, None]
                squared = (delta**2).sum(-1)
                target, source = torch.where(squared < self.cutoff**2)
                source, target = source + begin, target + offset
                keep = source != target
                source, target = source[keep], target[keep]
                if ((positions[source] - positions[target]) ** 2).sum(-1).le(1e-16).any():
                    raise ContractError("coincident graph atoms")
                total += len(source)
                if total > self.maximum_edges:
                    raise ResourceError("radius graph exceeds maximum edges")
                sources.append(source)
                targets.append(target)
        empty = torch.empty(0, dtype=torch.long, device=positions.device)
        return (torch.cat(sources) if sources else empty, torch.cat(targets) if targets else empty)

    def __call__(self, batch: Batch):
        return geometry_edges(batch, *self.topology(batch))


class NeighborCache:
    """A topology owner, never an owner of differentiable distance tensors."""

    def __init__(self, cutoff: float, skin: float = 0.5, maximum_edges=2_000_000):
        if skin <= 0:
            raise ContractError("neighbor skin must be positive")
        self.builder = RadiusGraph(cutoff + skin, maximum_edges)
        self.cutoff = cutoff
        self.skin = skin
        self.reference: torch.Tensor | None = None
        self.numbers: torch.Tensor | None = None
        self.graph: torch.Tensor | None = None
        self.indices: tuple[torch.Tensor, torch.Tensor] | None = None
        self.rebuilds = 0

    def invalidate(self):
        self.reference = None
        self.indices = None

    def __call__(self, batch: Batch):
        batch.validate()
        valid = (
            self.reference is not None
            and self.reference.shape == batch.positions.shape
            and self.reference.device == batch.positions.device
            and self.numbers is not None
            and self.graph is not None
            and torch.equal(self.numbers, batch.numbers)
            and torch.equal(self.graph, batch.graph)
        )
        if valid:
            assert self.reference is not None
            moved = torch.linalg.vector_norm(
                batch.positions.detach() - self.reference, dim=-1
            ).max()
            # Each endpoint may move skin/2. Equality rebuilds to avoid floating boundary ambiguity.
            valid = bool(moved < self.skin / 2)
        if not valid:
            self.indices = self.builder.topology(batch)
            self.reference = batch.positions.detach().clone()
            self.numbers = batch.numbers.clone()
            self.graph = batch.graph.clone()
            self.rebuilds += 1
        assert self.indices is not None
        edges = geometry_edges(batch, *self.indices)
        active = edges.distance.detach() < self.cutoff
        return geometry_edges(batch, edges.source[active], edges.target[active])
