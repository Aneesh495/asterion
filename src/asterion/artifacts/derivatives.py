"""Check retained numerical observations without calling the campaign's check functions."""

from dataclasses import replace

import numpy as np
import torch

from asterion.artifacts.io import object_hash, read_json, safe_member
from asterion.contracts.errors import IntegrityError
from asterion.contracts.geometry import Chemistry, Geometry, collate
from asterion.training.identity import state_fingerprint


def require(condition, message):
    if not condition:
        raise IntegrityError(message)


def geometry_from_trace(trace):
    numbers, positions = np.asarray(trace["numbers"]), np.asarray(trace["positions"], dtype=float)
    require(numbers.dtype.kind in "iu", "derivative trace atomic numbers are not integers")
    specification = dict(trace["chemistry"])
    specification["elements"] = tuple(specification["elements"])
    chemistry = Chemistry(**specification)
    return Geometry(numbers, positions, trace["identity"], chemistry).validate()


def vector(trace, name, shape):
    value = np.asarray(trace[name], dtype=float)
    require(
        value.shape == shape and np.isfinite(value).all(), f"malformed derivative vector: {name}"
    )
    return value


def agreement(row, expected):
    require(
        all(
            np.isfinite(row[key]) and np.isclose(row[key], value, atol=1e-12, rtol=1e-9)
            for key, value in expected.items()
        ),
        "derivative summary disagrees with retained physical observations",
    )


def force_observation(row):
    trace = row["trace"]
    geometry = geometry_from_trace(trace["geometry"])
    shape = geometry.positions.shape
    step = row["step"]
    require(np.isfinite(step) and step > 0, "invalid finite-difference step")
    positive, negative = vector(trace, "plus_energy", shape), vector(trace, "minus_energy", shape)
    force = vector(trace, "force", shape)
    finite = -(positive - negative) / (2 * step)
    difference = float(np.sqrt(np.sum((finite - force) ** 2)))
    scale = max(float(np.sqrt(np.sum(finite**2))), float(np.sqrt(np.sum(force**2))), 1e-8)
    agreement(
        row, {"absolute": difference, "relative": difference / scale, "components": force.size}
    )
    require(
        difference / scale < 1e-4,
        "raw energy differences do not reproduce the negative coordinate gradient",
    )
    return geometry, finite


def hessian_observation(row):
    trace = row["trace"]
    geometry = geometry_from_trace(trace["geometry"])
    shape = geometry.positions.shape
    step = row["step"]
    require(np.isfinite(step) and step > 0, "invalid Hessian difference step")
    direction = vector(trace, "direction", shape)
    require(
        np.isclose(np.sum(direction**2), 1.0, atol=1e-12),
        "Hessian probe direction is not normalized",
    )
    product = vector(trace, "product", shape)
    finite = -(vector(trace, "plus_force", shape) - vector(trace, "minus_force", shape)) / (
        2 * step
    )
    difference = float(np.sqrt(np.sum((finite - product) ** 2)))
    scale = max(float(np.sqrt(np.sum(finite**2))), float(np.sqrt(np.sum(product**2))), 1e-8)
    agreement(row, {"absolute": difference, "relative": difference / scale})
    require(
        difference / scale < 1e-4,
        "raw directional force differences do not reproduce the Hessian product",
    )
    return geometry, direction


def symmetry_observation(row):
    trace = row["trace"]
    geometry = geometry_from_trace(trace["geometry"])
    atoms = len(geometry.numbers)
    shape = geometry.positions.shape
    matrix = vector(trace, "matrix", (3, 3))
    require(
        np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-12, rtol=0),
        "symmetry transform is not orthogonal",
    )
    determinant = float(np.linalg.det(matrix))
    require(
        np.isclose(determinant, -1 if row["reflection"] else 1, atol=1e-12),
        "symmetry reflection label changed",
    )
    permutation = np.asarray(trace["permutation"])
    require(
        permutation.dtype.kind in "iu"
        and permutation.shape == (atoms,)
        and sorted(permutation.tolist()) == list(range(atoms)),
        "invalid symmetry atom permutation",
    )
    require(
        np.array_equal(geometry.numbers[permutation], geometry.numbers),
        "symmetry permutation changes chemical species",
    )
    vector(trace, "translation", (3,))
    force = vector(trace, "force", shape)
    transformed = vector(trace, "transformed_force", shape)
    disconnected = vector(trace, "disconnected_force", (2 * atoms, 3))
    batched = vector(trace, "batch_force", (2 * atoms, 3))
    batch_energy = vector(trace, "batch_energy", (2,))
    energy, changed, double = (
        trace[key] for key in ("energy", "transformed_energy", "disconnected_energy")
    )
    require(all(np.isfinite(x) for x in (energy, changed, double)), "nonfinite symmetry energy")
    center = geometry.positions.mean(0)
    norm = max(float(np.sqrt(np.sum(force**2))), 1e-8)
    torque = np.add.reduce(np.cross(geometry.positions - center, force))
    expected = {
        "energy_error": abs(energy - changed),
        "force_relative": float(np.sqrt(np.sum((transformed - force[permutation] @ matrix.T) ** 2)))
        / norm,
        "net_force": float(np.sqrt(np.sum(np.add.reduce(force) ** 2))) / norm,
        "torque": float(np.sqrt(np.sum(torque**2)))
        / max(norm * float(np.sqrt(np.sum((geometry.positions - center) ** 2))), 1e-8),
        "additivity": abs(double - 2 * energy),
        "disconnected_force": float(
            np.sqrt(np.sum((disconnected - np.concatenate([force, force])) ** 2))
        )
        / norm,
        "batch_force": float(np.sqrt(np.sum((batched - np.concatenate([force, force])) ** 2)))
        / norm,
        "batch_energy": float(np.max(np.abs(batch_energy - energy))),
    }
    agreement(row, expected)
    require(
        all(value < (1e-7) for value in expected.values()),
        "raw symmetry, force/torque or additivity identity failed",
    )
    return geometry, matrix, permutation


def model_observation(model, geometry, force, energy=None):
    actual_energy, actual_force = model(collate([geometry]))
    require(
        np.allclose(actual_force.detach().numpy(), force, atol=1e-9, rtol=1e-7),
        "derivative force observation does not belong to saved weights",
    )
    if energy is not None:
        require(
            np.isclose(actual_energy.item(), energy, atol=1e-9, rtol=1e-8),
            "derivative energy observation does not belong to saved weights",
        )


def replay_probe(model, family, row):
    trace = row["trace"]
    geometry = geometry_from_trace(trace["geometry"])
    if family == "symmetry":
        matrix, permutation = np.asarray(trace["matrix"]), np.asarray(trace["permutation"])
        changed = replace(
            geometry,
            numbers=geometry.numbers[permutation],
            positions=geometry.positions[permutation] @ matrix.T + np.asarray(trace["translation"]),
        )
        model_observation(model, geometry, trace["force"], trace["energy"])
        model_observation(model, changed, trace["transformed_force"], trace["transformed_energy"])
        copy = replace(geometry, identity="replay-copy", positions=geometry.positions + [100, 0, 0])
        disconnected = replace(
            geometry,
            identity="replay-disconnected",
            numbers=np.concatenate([geometry.numbers, copy.numbers]),
            positions=np.concatenate([geometry.positions, copy.positions]),
        )
        model_observation(
            model, disconnected, trace["disconnected_force"], trace["disconnected_energy"]
        )
        energy, force = model(collate([copy, geometry]))
        require(
            np.allclose(energy.detach().numpy(), trace["batch_energy"], atol=1e-9, rtol=1e-8)
            and np.allclose(force.detach().numpy(), trace["batch_force"], atol=1e-9, rtol=1e-7),
            "batched symmetry observation uses different weights",
        )
    elif family == "force":
        model_observation(model, geometry, trace["force"])
        # One source-selected coordinate per retained probe bounds verification work.
        coordinate = int(object_hash(geometry.identity), 16) % geometry.positions.size
        atom, axis = divmod(coordinate, 3)
        for sign, name in ((1, "plus_energy"), (-1, "minus_energy")):
            positions = geometry.positions.copy()
            positions[atom, axis] += sign * row["step"]
            energy = model.energies(collate([replace(geometry, positions=positions)])).item()
            require(
                np.isclose(energy, trace[name][atom][axis], atol=1e-9, rtol=1e-8),
                "finite-difference energy belongs to another model",
            )
    else:
        direction = np.asarray(trace["direction"])
        for sign, name in ((1, "plus_force"), (-1, "minus_force")):
            changed = replace(
                geometry, positions=geometry.positions + sign * row["step"] * direction
            )
            model_observation(model, changed, trace[name])
        batch = collate([geometry])
        _, force = model(batch, create_graph=True)
        product = -torch.autograd.grad((force * torch.as_tensor(direction)).sum(), batch.positions)[
            0
        ]
        require(
            np.allclose(product.detach().numpy(), trace["product"], atol=1e-9, rtol=1e-7),
            "retained Hessian product uses different weights",
        )


def verify_derivative_campaign(root, report_path, source, models, datasets):
    report = read_json(safe_member(root, report_path))
    require(
        report["format"] == "asterion-raw-derivatives-v1"
        and report["source"]["digest"] == source["digest"]
        and report["dtype"] == "float64",
        "derivative execution source or precision changed",
    )
    require(report["checkpoint"] in models, "derivative campaign uses another trained checkpoint")
    model = models[report["checkpoint"]]
    require(
        state_fingerprint(model) == report["executed_state"],
        "derivative executed parameter state changed",
    )
    require(
        len(report["base_geometries"]) == len(set(report["base_geometries"])) >= 8,
        "derivative source geometry population is incomplete",
    )
    bases = {}
    for identity in report["base_geometries"]:
        molecule, digest, row = identity.split(":")
        path = datasets[f"rmd17_{molecule}.npz"]
        from asterion.artifacts.io import file_hash

        require(file_hash(path) == digest, "derivative base geometry dataset changed")
        with np.load(path, allow_pickle=False) as author:
            bases[identity] = (author["nuclear_charges"].copy(), author["coords"][int(row)].copy())
    checkers = {
        "symmetry": symmetry_observation,
        "force": force_observation,
        "hessian": hessian_observation,
    }
    counts = {}
    for family, checker in checkers.items():
        rows = report[family]
        require(
            [row["seed"] for row in rows] == list(range(len(rows))),
            "derivative trial identities are missing or repeated",
        )
        transforms = set()
        for row in rows:
            checker(row)
            raw = row["trace"]["geometry"]
            identity = report["base_geometries"][row["seed"] % len(bases)]
            require(raw["identity"] == identity, "derivative probe geometry identity changed")
            numbers, positions = bases[identity]
            if family == "force":
                require(row["step"] == 1e-5, "force difference step contract changed")
                positions = (
                    positions
                    + np.random.default_rng(row["seed"] + 1900).normal(size=positions.shape) * 0.001
                )
            elif family == "hessian":
                require(row["step"] == 1e-4, "Hessian difference step contract changed")
                direction = np.random.default_rng(row["seed"] + 3900).normal(size=positions.shape)
                direction /= np.sqrt(np.sum(direction**2))
                require(
                    np.allclose(direction, row["trace"]["direction"], atol=1e-12, rtol=0),
                    "Hessian direction differs from frozen seed",
                )
            else:
                require(
                    row["reflection"] == bool(row["seed"] % 2),
                    "symmetry reflection population changed",
                )
                transforms.add(
                    object_hash(
                        [row["trace"][key] for key in ("matrix", "permutation", "translation")]
                    )
                )
            require(
                np.array_equal(numbers, raw["numbers"])
                and np.allclose(positions, raw["positions"], atol=1e-12, rtol=0),
                "derivative inputs do not match author geometries and declared perturbations",
            )
        if family == "symmetry":
            require(len(transforms) == len(rows), "symmetry trials repeat the same transformation")
        selected = sorted(
            rows, key=lambda row: object_hash([source["digest"], family, row["seed"]])
        )[:8]
        require(len(selected) == 8, "derivative saved-weight probe count shortened")
        for row in selected:
            replay_probe(model, family, row)
        counts[family] = len(rows)
    return counts
