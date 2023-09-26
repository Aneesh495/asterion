import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

import torch
from e3nn import o3
from torch import nn

from asterion.contracts.errors import ContractError, ResourceError
from asterion.contracts.geometry import SUPPORTED_LEVELS, Batch
from asterion.graphs.components import components
from asterion.graphs.radial import RadialBasis
from asterion.graphs.radius import RadiusGraph, geometry_edges
from asterion.representations.features import ParityGate, TensorFeature, invariant_norms


@dataclass(frozen=True)
class ModelConfig:
    kind: str = "equivariant"
    width: int = 12
    depth: int = 2
    radial: int = 8
    cutoff: float = 5.0
    lmax: int = 2
    higher_order: bool = True
    smooth: bool = True
    elements: tuple[int, ...] = (1, 6, 7, 8)
    expected_neighbors: float = 8.0
    energy_scale: float = 1.0
    energy_offset: float = 0.0
    level: str = "PBE/def2-SVP"
    composition: tuple[int, ...] | None = None
    max_message_elements: int = 1 << 26

    def validate(self):
        if self.kind not in {"equivariant", "invariant", "force_head"}:
            raise ContractError("unknown original model architecture")
        if min(self.width, self.depth, self.radial) < 1 or self.lmax not in (0, 1, 2, 3):
            raise ContractError("invalid model size or angular order")
        if self.width > 64 or self.depth > 8 or self.radial > 64:
            raise ResourceError("model constructor exceeds the declared size budget")
        if not all(
            math.isfinite(value)
            for value in (
                self.expected_neighbors,
                self.cutoff,
                self.energy_scale,
                self.energy_offset,
            )
        ):
            raise ContractError("model scales and offsets must be finite")
        if self.expected_neighbors <= 0 or self.cutoff <= 0 or self.energy_scale <= 0:
            raise ContractError("invalid model scales")
        if not self.elements or not set(self.elements) <= {1, 6, 7, 8}:
            raise ContractError("unsupported elements")
        if len(set(self.elements)) != len(self.elements):
            raise ContractError("model vocabulary contains duplicate elements")
        if self.level not in SUPPORTED_LEVELS:
            raise ContractError("unsupported model label level")
        if self.composition is not None and (
            len(self.composition) != len(self.elements)
            or any(not isinstance(n, int) or n < 0 for n in self.composition)
            or not 1 <= sum(self.composition) <= 512
        ):
            raise ContractError("invalid trained composition")
        if not 1 <= self.max_message_elements <= 1 << 28:
            raise ResourceError("invalid message activation budget")
        return self

    def to_dict(self):
        return asdict(self)


def scatter_sum(values, index, size):
    output = values.new_zeros((size, *values.shape[1:]))
    return output.index_add(0, index, values)


class ScalarInteraction(nn.Module):
    def __init__(self, width: int, radial: int, normalization: float):
        super().__init__()
        self.edge = nn.Sequential(
            nn.Linear(2 * width + radial, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.update = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(), nn.Linear(width, width))
        self.normalization = normalization
        self.irreps = o3.Irreps(f"{width}x0e")

    def forward(self, features, edges, radial, envelope):
        feature = features.validate(len(features.values))
        if feature.irreps != self.irreps:
            raise ContractError("scalar interaction received incompatible irreps")
        features = feature.values
        inputs = torch.cat((features[edges.source], features[edges.target], radial), dim=-1)
        messages = self.edge(inputs) * envelope[:, None]
        aggregate = scatter_sum(messages, edges.target, len(features)) / self.normalization
        values = features + self.update(torch.cat((features, aggregate), dim=-1))
        return TensorFeature(values, self.irreps, feature.graph_identity).validate(len(values))


class EquivariantInteraction(nn.Module):
    def __init__(self, irreps, harmonics, width: int, radial: int, normalization: float):
        super().__init__()
        self.tensor_product = o3.FullyConnectedTensorProduct(
            irreps, harmonics, irreps, shared_weights=False, internal_weights=False
        )
        self.radial_weights = nn.Sequential(
            nn.Linear(radial, width), nn.SiLU(), nn.Linear(width, self.tensor_product.weight_numel)
        )
        self.mix = o3.Linear(irreps, irreps)
        self.residual = o3.Linear(irreps, irreps)
        self.gate = ParityGate(irreps, width)
        self.normalization = normalization
        self.irreps = irreps

    def forward(self, features, edges, radial, spherical, envelope):
        feature = features.validate(len(features.values))
        if feature.irreps != self.irreps:
            raise ContractError("equivariant interaction received incompatible irreps")
        features = feature.values
        weights = self.radial_weights(radial)
        messages = self.tensor_product(features[edges.source], spherical, weights)
        messages = messages * envelope[:, None]
        aggregate = scatter_sum(messages, edges.target, len(features)) / self.normalization
        values = (features + self.gate(self.mix(aggregate) + self.residual(features))) / math.sqrt(
            2
        )
        return TensorFeature(values, self.irreps, feature.graph_identity).validate(len(values))


class EnergyPotential(nn.Module, ABC):
    """Energy normalization is applied before differentiating, so force units stay physical."""

    vocabulary: torch.Tensor
    energy_scale: torch.Tensor
    energy_offset: torch.Tensor
    trained_composition: torch.Tensor
    blocks: nn.ModuleList

    @abstractmethod
    def atomic_energy(self, features, edges, radial, envelope, graph_identity) -> torch.Tensor:
        """Return one invariant scalar per atom from the declared representation path."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config.validate()
        self.graph_builder = RadiusGraph(config.cutoff)
        self.basis = RadialBasis(config.radial, config.cutoff, config.smooth)
        self.embedding = nn.Embedding(len(config.elements), config.width)
        self.register_buffer("vocabulary", torch.tensor(config.elements, dtype=torch.long))
        self.register_buffer("energy_scale", torch.tensor(config.energy_scale, dtype=torch.float64))
        self.register_buffer(
            "energy_offset", torch.tensor(config.energy_offset, dtype=torch.float64)
        )
        self.register_buffer(
            "trained_composition",
            torch.tensor(config.composition or (), dtype=torch.long),
            persistent=False,
        )

    def element_indices(self, numbers):
        matches = numbers[:, None] == self.vocabulary[None]
        if not matches.any(-1).all():
            raise ContractError("element absent from model vocabulary")
        return matches.to(torch.long).argmax(-1)

    def energies(self, batch: Batch, edges=None):
        batch.validate()
        if batch.chemistry.level != self.config.level:
            raise ContractError("model and geometry electronic structure levels differ")
        edges = (
            self.graph_builder(batch)
            if edges is None
            else geometry_edges(batch, edges.source, edges.target)
        )
        if self.config.composition is not None:
            for component in components(batch, edges, self.config.cutoff):
                numbers = batch.numbers[list(component)]
                observed = (numbers[:, None] == self.vocabulary[None]).sum(0)
                if not torch.equal(observed, self.trained_composition):
                    raise ContractError("component composition is outside the trained model domain")
        radial = self.basis(edges.distance)
        envelope = self.basis.envelope(edges.distance)
        dimensions = max(
            (
                block.tensor_product.weight_numel
                if isinstance(block, EquivariantInteraction)
                else 2 * self.config.width + self.config.radial
                for block in self.blocks
            ),
            default=1,
        )
        if edges.size * dimensions > self.config.max_message_elements:
            raise ResourceError("edge-weight activations exceed the declared message budget")
        features = self.embedding(self.element_indices(batch.numbers))
        atomic = self.atomic_energy(features, edges, radial, envelope, batch.identities)
        atomic = atomic * self.energy_scale + self.energy_offset
        energy = scatter_sum(atomic, batch.graph, batch.size)
        # Maintain a coordinate gradient for singleton and fully disconnected graphs.
        return energy + scatter_sum(batch.positions.sum(-1) * 0, batch.graph, batch.size)

    def forward(self, batch: Batch, create_graph=False, edges=None):
        with torch.enable_grad():
            if not batch.positions.requires_grad:
                batch.positions = batch.positions.detach().clone().requires_grad_(True)
            energy = self.energies(batch, edges)
            forces = -torch.autograd.grad(
                energy.sum(), batch.positions, create_graph=create_graph, retain_graph=create_graph
            )[0]
        if not torch.isfinite(energy).all() or not torch.isfinite(forces).all():
            raise ContractError("nonfinite energy or force")
        return energy, forces


class InvariantPotential(EnergyPotential):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.blocks = nn.ModuleList(
            [
                ScalarInteraction(config.width, config.radial, math.sqrt(config.expected_neighbors))
                for _ in range(config.depth)
            ]
        )
        self.readout = nn.Sequential(
            nn.Linear(config.width, config.width), nn.SiLU(), nn.Linear(config.width, 1)
        )

    def atomic_energy(self, features, edges, radial, envelope, graph_identity):
        features = TensorFeature(
            features, o3.Irreps(f"{self.config.width}x0e"), graph_identity
        ).validate(len(features))
        for block in self.blocks:
            features = block(features, edges, radial, envelope)
        return self.readout(features.values).squeeze(-1)


class EquivariantPotential(EnergyPotential):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.irreps = o3.Irreps(
            [(config.width, (order, (-1) ** order)) for order in range(config.lmax + 1)]
        )
        self.harmonics = o3.Irreps.spherical_harmonics(config.lmax)
        self.blocks = nn.ModuleList(
            [
                EquivariantInteraction(
                    self.irreps,
                    self.harmonics,
                    config.width,
                    config.radial,
                    math.sqrt(config.expected_neighbors),
                )
                for _ in range(config.depth)
            ]
        )
        self.correlation = (
            o3.FullyConnectedTensorProduct(
                self.irreps, self.irreps, o3.Irreps(f"{config.width}x0e")
            )
            if config.higher_order
            else None
        )
        invariant_size = config.width * (config.lmax + 2 + int(config.higher_order))
        self.readout = nn.Sequential(
            nn.Linear(invariant_size, config.width), nn.SiLU(), nn.Linear(config.width, 1)
        )

    def atomic_energy(self, features, edges, radial, envelope, graph_identity):
        if self.irreps.dim > features.shape[-1]:
            features = torch.cat(
                (
                    features,
                    features.new_zeros((len(features), self.irreps.dim - features.shape[-1])),
                ),
                dim=-1,
            )
        spherical = o3.spherical_harmonics(
            self.harmonics, edges.vector, normalize=True, normalization="component"
        )
        features = TensorFeature(features, self.irreps, graph_identity).validate(len(features))
        for block in self.blocks:
            features = block(features, edges, radial, spherical, envelope)
        features = features.values
        invariant = [features[:, : self.config.width], invariant_norms(features, self.irreps)]
        if self.correlation is not None:
            invariant.append(self.correlation(features, features))
        return self.readout(torch.cat(invariant, dim=-1)).squeeze(-1)


class NonconservativeAblation(InvariantPotential):
    def __init__(self, config):
        super().__init__(config)
        self.force_edge = nn.Sequential(
            nn.Linear(config.radial, config.width), nn.SiLU(), nn.Linear(config.width, 1)
        )

    def forward(self, batch, create_graph=False, edges=None):
        edges = self.graph_builder(batch) if edges is None else edges
        energy = self.energies(batch, edges)
        weights = (
            self.force_edge(self.basis(edges.distance))
            * self.basis.envelope(edges.distance)[:, None]
        )
        # Directed weights depend on target species, so forces need not satisfy pair antisymmetry.
        species = self.element_indices(batch.numbers[edges.target]).to(batch.positions.dtype) + 1
        force = scatter_sum(
            weights * edges.vector * species[:, None], edges.target, len(batch.numbers)
        )
        return energy, force


def make_model(config: ModelConfig):
    if config.kind == "invariant":
        return InvariantPotential(config)
    if config.kind == "force_head":
        return NonconservativeAblation(config)
    return EquivariantPotential(config)
