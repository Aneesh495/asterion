from dataclasses import dataclass

import torch
from e3nn import o3
from torch import nn

from asterion.contracts.errors import ContractError


@dataclass
class TensorFeature:
    values: torch.Tensor
    irreps: o3.Irreps
    graph_identity: tuple[str, ...]
    unit: str = "dimensionless"

    def validate(self, atoms: int):
        if self.values.shape != (atoms, self.irreps.dim) or not torch.isfinite(self.values).all():
            raise ContractError("irrep feature axes or finite values violated")
        if self.unit != "dimensionless" or not self.graph_identity:
            raise ContractError("tensor feature units or graph identity missing")
        return self

    def transform(self, matrix: torch.Tensor):
        if matrix.shape != (3, 3) or not torch.allclose(
            matrix.T @ matrix, torch.eye(3, dtype=matrix.dtype, device=matrix.device), atol=1e-8
        ):
            raise ContractError("feature transform must be orthogonal")
        representation = self.irreps.D_from_matrix(matrix).to(self.values)
        return TensorFeature(
            self.values @ representation.T, self.irreps, self.graph_identity, self.unit
        )


class ParityGate(nn.Module):
    """Even scalar gates multiply entire irreducible blocks, preserving O(3)."""

    def __init__(self, irreps: o3.Irreps, scalar_width: int):
        super().__init__()
        self.irreps = irreps
        self.scalar_width = scalar_width
        self.gates = nn.Linear(scalar_width, sum(mul for mul, ir in irreps if ir.l > 0))

    def forward(self, values):
        gates = torch.sigmoid(self.gates(values[:, : self.scalar_width]))
        outputs = []
        offset = 0
        gate_offset = 0
        for multiplicity, irrep in self.irreps:
            block = values[:, offset : offset + multiplicity * irrep.dim]
            if irrep.l == 0:
                block = torch.nn.functional.silu(block) if irrep.p == 1 else torch.tanh(block)
            else:
                block = block.reshape(-1, multiplicity, irrep.dim)
                block = block * gates[:, gate_offset : gate_offset + multiplicity, None]
                block = block.reshape(len(values), multiplicity * irrep.dim)
                gate_offset += multiplicity
            outputs.append(block)
            offset += multiplicity * irrep.dim
        return torch.cat(outputs, dim=-1)


def invariant_norms(values: torch.Tensor, irreps: o3.Irreps):
    outputs = []
    offset = 0
    for multiplicity, irrep in irreps:
        block = values[:, offset : offset + multiplicity * irrep.dim]
        block = block.reshape(-1, multiplicity, irrep.dim)
        outputs.append((block**2).sum(-1) / irrep.dim)
        offset += multiplicity * irrep.dim
    return torch.cat(outputs, dim=-1)
