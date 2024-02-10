import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from asterion.contracts.errors import ContractError, ResourceError

SUPPORTED_LEVELS = frozenset({"PBE/def2-SVP", "PBE+MBD", "analytic"})


@dataclass(frozen=True)
class Chemistry:
    elements: tuple[int, ...] = (1, 6, 7, 8)
    charge: int = 0
    multiplicity: int = 1
    level: str = "PBE/def2-SVP"
    periodic: bool = False

    def validate(self):
        if self.periodic or self.charge != 0 or self.multiplicity != 1:
            raise ContractError("only nonperiodic neutral singlets are supported")
        if not self.elements or len(set(self.elements)) != len(self.elements):
            raise ContractError("elements must be nonempty and unique")
        if not set(self.elements) <= {1, 6, 7, 8}:
            raise ContractError("supported elements are H,C,N,O")
        if self.level not in SUPPORTED_LEVELS:
            raise ContractError("unsupported electronic structure level")


@dataclass(frozen=True)
class Geometry:
    numbers: np.ndarray
    positions: np.ndarray
    identity: str
    chemistry: Chemistry = Chemistry()
    length_unit: str = "angstrom"

    def validate(self, maximum_atoms: int = 512, minimum_distance: float = 1e-8):
        self.chemistry.validate()
        n = len(self.numbers)
        if n == 0 or n > maximum_atoms:
            raise ResourceError(f"atom count {n} outside [1,{maximum_atoms}]")
        if self.numbers.shape != (n,) or self.numbers.dtype.kind not in "iu":
            raise ContractError("atomic numbers require integer shape [atoms]")
        if self.positions.shape != (n, 3) or self.positions.dtype.kind != "f":
            raise ContractError("positions require floating shape [atoms,3]")
        if not np.isfinite(self.positions).all():
            raise ContractError("nonfinite coordinates")
        if self.length_unit != "angstrom" or not self.identity:
            raise ContractError("geometry identity and angstrom units are required")
        if not set(self.numbers.tolist()) <= set(self.chemistry.elements):
            raise ContractError("unsupported chemistry")
        for i in range(n):
            if np.any(
                np.linalg.norm(self.positions[i + 1 :] - self.positions[i], axis=1)
                < minimum_distance
            ):
                raise ContractError("coincident atoms")
        return self

    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(np.asarray(self.numbers, dtype="<i8").tobytes())
        digest.update(np.asarray(self.positions, dtype="<f8").tobytes())
        digest.update(repr(self.chemistry).encode())
        return digest.hexdigest()


@dataclass(frozen=True)
class Label:
    geometry: Geometry
    energy: float
    forces: np.ndarray
    unit: str = "eV"

    def validate(self):
        self.geometry.validate()
        if self.unit != "eV" or not np.isfinite(self.energy):
            raise ContractError("finite eV energy required")
        if self.forces.shape != self.geometry.positions.shape or not np.isfinite(self.forces).all():
            raise ContractError("finite forces require shape [atoms,3] in eV/angstrom")
        return self


@dataclass
class Batch:
    """Flat atom axes, graph axes, and immutable external geometry identities."""

    numbers: torch.Tensor
    positions: torch.Tensor
    graph: torch.Tensor
    pointers: torch.Tensor
    identities: tuple[str, ...]
    energy: torch.Tensor | None = None
    forces: torch.Tensor | None = None
    chemistry: Chemistry = Chemistry()

    @property
    def size(self):
        return len(self.identities)

    def validate(self, max_atoms: int = 8192):
        self.chemistry.validate()
        n = len(self.numbers)
        if not n or n > max_atoms:
            raise ResourceError("empty or oversized batch")
        if self.positions.shape != (n, 3) or self.graph.shape != (n,):
            raise ContractError("invalid atom axes")
        if self.positions.dtype not in {torch.float32, torch.float64}:
            raise ContractError("positions require audited float32 or float64 precision")
        if self.pointers.shape != (self.size + 1,) or self.pointers[0].item() != 0:
            raise ContractError("invalid graph pointers")
        if self.pointers[-1].item() != n or (self.pointers.diff() <= 0).any():
            raise ContractError("empty graph or invalid terminal pointer")
        if self.graph.dtype != torch.long or self.numbers.dtype != torch.long:
            raise ContractError("index tensors must be int64")
        expected = torch.repeat_interleave(
            torch.arange(self.size, device=self.graph.device), self.pointers.diff()
        )
        if not torch.equal(self.graph, expected):
            raise ContractError("graph ordering disagrees with pointers")
        if not torch.isfinite(self.positions).all():
            raise ContractError("nonfinite batch positions")
        if self.energy is not None and (
            self.energy.shape != (self.size,) or not torch.isfinite(self.energy).all()
        ):
            raise ContractError("invalid energy labels")
        if self.forces is not None and (
            self.forces.shape != (n, 3) or not torch.isfinite(self.forces).all()
        ):
            raise ContractError("invalid force labels")
        return self


def collate(items: Sequence[Geometry | Label], dtype=torch.float64, device="cpu") -> Batch:
    if not items:
        raise ContractError("cannot collate empty input")
    labeled = isinstance(items[0], Label)
    if any(isinstance(item, Label) != labeled for item in items):
        raise ContractError("mixed labeled and unlabeled batch")
    geometries = [item.geometry if isinstance(item, Label) else item for item in items]
    for item in items:
        item.validate()
    if len({g.chemistry for g in geometries}) != 1:
        raise ContractError("different electronic structure tasks require separate batches")
    sizes = np.array([len(g.numbers) for g in geometries])
    labels = [item for item in items if isinstance(item, Label)]
    return Batch(
        torch.as_tensor(
            np.concatenate([g.numbers for g in geometries]), dtype=torch.long, device=device
        ),
        torch.as_tensor(
            np.concatenate([g.positions for g in geometries]), dtype=dtype, device=device
        ),
        torch.as_tensor(np.repeat(np.arange(len(items)), sizes), dtype=torch.long, device=device),
        torch.as_tensor(np.r_[0, sizes.cumsum()], dtype=torch.long, device=device),
        tuple(g.identity for g in geometries),
        torch.tensor([item.energy for item in labels], dtype=dtype, device=device)
        if labeled
        else None,
        torch.as_tensor(
            np.concatenate([item.forces for item in labels]), dtype=dtype, device=device
        )
        if labeled
        else None,
        geometries[0].chemistry,
    ).validate()
