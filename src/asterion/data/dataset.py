from dataclasses import dataclass
from pathlib import Path

import numpy as np

from asterion.artifacts.io import file_hash
from asterion.contracts.errors import ContractError, LabelAccessError
from asterion.contracts.geometry import Chemistry, Geometry, Label
from asterion.contracts.units import convert_labels
from asterion.data.archive import dataset_arrays, dataset_size, validate_native_arrays


@dataclass(frozen=True)
class CoordinatePool:
    numbers: np.ndarray
    positions: np.ndarray
    original_indices: np.ndarray
    dataset_hash: str
    name: str
    chemistry: Chemistry

    def __len__(self):
        return len(self.positions)

    def geometry(self, row: int) -> Geometry:
        if row < 0 or row >= len(self):
            raise ContractError("geometry row outside dataset")
        return Geometry(
            self.numbers.copy(),
            self.positions[row].copy(),
            f"{self.name}:{self.dataset_hash}:{int(row)}",
            self.chemistry,
        ).validate()


class LabelOracle:
    """Released labels are accessible only through an explicit immutable allow-list."""

    def __init__(self, pool: CoordinatePool, energies: np.ndarray, forces: np.ndarray):
        if energies.shape != (len(pool),) or forces.shape != pool.positions.shape:
            raise ContractError("label arrays do not match coordinate pool")
        if not np.isfinite(energies).all() or not np.isfinite(forces).all():
            raise ContractError("nonfinite label arrays")
        self.pool = pool
        self.__energies = energies
        self.__forces = forces

    def labels(self, rows, allowed, role: str) -> list[Label]:
        rows = [int(row) for row in rows]
        if role not in {"train", "validation", "calibration", "evaluation"}:
            raise LabelAccessError("explicit label access role required")
        if len(rows) != len(set(rows)) or not set(rows) <= set(map(int, allowed)):
            raise LabelAccessError("duplicate or unrevealed label access")
        return [
            Label(
                self.pool.geometry(row), float(self.__energies[row]), self.__forces[row].copy()
            ).validate()
            for row in rows
        ]


def load_rmd17(path: Path, name: str) -> tuple[CoordinatePool, LabelOracle]:
    dataset_size(path)
    identity = file_hash(path)
    data = dataset_arrays(path, {"nuclear_charges", "coords", "energies", "forces", "old_indices"})
    validate_native_arrays(
        data["nuclear_charges"],
        data["coords"],
        data["energies"],
        data["forces"],
        data["old_indices"],
    )
    numbers = np.array(data["nuclear_charges"], dtype=np.int64)
    coordinates = np.array(data["coords"], dtype=np.float64)
    indices = np.array(data["old_indices"], dtype=np.int64)
    energy, force = convert_labels(data["energies"].reshape(-1), data["forces"], "kcal/mol")
    if file_hash(path) != identity:
        raise ContractError("rMD17 dataset changed during loading")
    if coordinates.ndim != 3 or coordinates.shape[1:] != (len(numbers), 3):
        raise ContractError("invalid rMD17 coordinate axes")
    if len(indices) != len(coordinates) or len(set(indices.tolist())) != len(indices):
        raise ContractError("original temporal indices are not unique")
    if not np.isfinite(coordinates).all():
        raise ContractError("invalid coordinate values")
    for array in (numbers, coordinates, indices):
        array.setflags(write=False)
    pool = CoordinatePool(numbers, coordinates, indices, identity, name, Chemistry())
    pool.geometry(0)
    return pool, LabelOracle(pool, energy.astype(np.float64), force.astype(np.float64))


def load_md22(
    path: Path, name: str, expected_hash: str, level: str = "PBE+MBD"
) -> tuple[CoordinatePool, LabelOracle]:
    dataset_size(path)
    if file_hash(path) != expected_hash:
        raise ContractError("MD22 dataset changed")
    data = dataset_arrays(path, {"z", "R", "E", "F"})
    validate_native_arrays(data["z"], data["R"], data["E"], data["F"])
    numbers = np.asarray(data["z"], dtype=np.int64)
    coordinates = np.asarray(data["R"], dtype=np.float64)
    energy, force = convert_labels(data["E"].reshape(-1), data["F"], "kcal/mol")
    if file_hash(path) != expected_hash:
        raise ContractError("MD22 dataset changed during loading")
    if coordinates.ndim != 3 or coordinates.shape[1:] != (len(numbers), 3):
        raise ContractError("MD22 coordinate axes mismatch")
    for array in (numbers, coordinates):
        array.setflags(write=False)
    pool = CoordinatePool(
        numbers,
        coordinates,
        np.arange(len(coordinates)),
        expected_hash,
        name,
        Chemistry(level=level),
    )
    pool.geometry(0)
    return pool, LabelOracle(pool, energy, force)
