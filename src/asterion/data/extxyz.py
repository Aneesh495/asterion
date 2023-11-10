"""Bounded parsing of the exported reference schema, without importing ASE models."""

import shlex
from pathlib import Path

import numpy as np

from asterion.contracts.errors import ContractError, ResourceError
from asterion.contracts.units import SYMBOLS


def parse_properties(description: str, predictions=True):
    parts = description.split(":")
    if len(parts) % 3:
        raise ContractError("malformed XYZ property declaration")
    result = {}
    types = {}
    offset = 0
    for index in range(0, len(parts), 3):
        name, kind, width = parts[index : index + 3]
        if name in result or kind not in {"S", "R", "I"}:
            raise ContractError("duplicate or unsupported XYZ property")
        try:
            columns = int(width)
        except ValueError as failure:
            raise ContractError("invalid XYZ property width") from failure
        if not 1 <= columns <= 9:
            raise ResourceError("XYZ property width exceeds schema budget")
        result[name] = (offset, offset + columns)
        types[name] = kind
        offset += columns
    required = [("species", 1), ("pos", 3), ("REF_forces", 3)]
    if predictions:
        required.append(("PRED_forces", 3))
    for name, size in required:
        if name not in result or result[name][1] - result[name][0] != size:
            raise ContractError(f"required XYZ property is absent or malformed: {name}")
        if types[name] != ("S" if name == "species" else "R"):
            raise ContractError("XYZ property has incorrect scalar type")
    return result, offset


def reference_arrays(path: Path, maximum_frames=10000, maximum_atoms=512, predictions=True):
    if path.stat().st_size > 256 * 1024**2:
        raise ResourceError("XYZ file budget exceeded")
    reverse_symbols = {name: z for z, name in SYMBOLS.items()}
    geometries = []
    energies: list[float] = []
    targets: list[float] = []
    identities = []
    predicted_forces = []
    target_forces = []
    numbers = []
    pointers = [0]
    with path.open() as stream:
        while True:
            line = stream.readline()
            if not line:
                break
            try:
                atoms = int(line)
            except ValueError as failure:
                raise ContractError("invalid XYZ atom count") from failure
            if not 1 <= atoms <= maximum_atoms or len(targets) >= maximum_frames:
                raise ResourceError("XYZ atom/frame budget exceeded")
            header = stream.readline(65537)
            if len(header) > 65536:
                raise ResourceError("XYZ header budget exceeded")
            try:
                entries = [part.split("=", 1) for part in shlex.split(header)]
                fields = dict(entries)
                if len(fields) != len(entries):
                    raise ContractError("duplicate XYZ header keys")
                if fields.get("pbc", "F F F").split() != ["F", "F", "F"]:
                    raise ContractError("periodic reference geometries are unsupported")
                properties, columns = parse_properties(fields["Properties"], predictions)
                if predictions:
                    energies.append(float(fields["PRED_energy"]))
                targets.append(float(fields["REF_energy"]))
                identities.append(fields["asterion_id"])
            except (KeyError, ValueError) as failure:
                raise ContractError("XYZ energy/identity metadata is missing") from failure
            coordinates: list[list[float]] = []
            predicted: list[list[float]] = []
            reference: list[list[float]] = []
            atomic_numbers: list[int] = []
            for _ in range(atoms):
                atom_line = stream.readline(8193)
                if len(atom_line) > 8192:
                    raise ResourceError("XYZ atom row budget exceeded")
                values = atom_line.split()
                if len(values) != columns:
                    raise ContractError("XYZ atom row has wrong column count")
                try:
                    first, _ = properties["species"]
                    atomic_numbers.append(reverse_symbols[values[first]])
                    vectors = [("pos", coordinates), ("REF_forces", reference)]
                    if predictions:
                        vectors.append(("PRED_forces", predicted))
                    for name, destination in vectors:
                        begin, end = properties[name]
                        destination.append([float(v) for v in values[begin:end]])
                except (KeyError, ValueError) as failure:
                    raise ContractError("unsupported XYZ element or numeric value") from failure
            geometries.extend(coordinates)
            predicted_forces.extend(predicted)
            target_forces.extend(reference)
            numbers.extend(atomic_numbers)
            pointers.append(pointers[-1] + atoms)
    if not targets or len(set(identities)) != len(identities):
        raise ContractError("XYZ evaluation is empty or repeats geometry identities")
    arrays = {
        "target_energy": np.asarray(targets, dtype=float),
        "target_force": np.asarray(target_forces, dtype=float),
        "positions": np.asarray(geometries, dtype=float),
        "numbers": np.asarray(numbers, dtype=np.int64),
        "pointers": np.asarray(pointers, dtype=np.int64),
        "identities": np.asarray(identities),
    }
    if predictions:
        arrays.update(
            energy=np.asarray(energies, dtype=float),
            force=np.asarray(predicted_forces, dtype=float),
        )
    if any(not np.isfinite(value).all() for key, value in arrays.items() if key != "identities"):
        raise ContractError("nonfinite XYZ reference output")
    return arrays
