from dataclasses import asdict, replace

import numpy as np
import torch

from asterion.contracts.geometry import Geometry, collate


def force_difference(model, geometry: Geometry, step=1e-5, retain_trace=False):
    n = len(geometry.numbers)
    finite = np.empty((n, 3))
    positive, negative = np.empty((n, 3)), np.empty((n, 3))
    for atom in range(n):
        for axis in range(3):
            plus, minus = geometry.positions.copy(), geometry.positions.copy()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            with torch.no_grad():
                ep = model.energies(collate([replace(geometry, positions=plus)])).item()
                em = model.energies(collate([replace(geometry, positions=minus)])).item()
            finite[atom, axis] = -(ep - em) / (2 * step)
            positive[atom, axis], negative[atom, axis] = ep, em
    _, force = model(collate([geometry]))
    actual = force.detach().numpy()
    absolute = float(np.linalg.norm(actual - finite))
    scale = max(float(np.linalg.norm(actual)), float(np.linalg.norm(finite)), 1e-8)
    result = {
        "absolute": float(absolute),
        "relative": float(absolute / scale),
        "step": step,
        "components": 3 * n,
        "finite": finite,
        "autograd": actual,
    }
    if retain_trace:
        result["trace"] = {
            "geometry": geometry_observation(geometry),
            "plus_energy": positive.tolist(),
            "minus_energy": negative.tolist(),
            "force": actual.tolist(),
        }
    return result


def hessian_difference(
    model, geometry: Geometry, direction: np.ndarray, step=1e-4, retain_trace=False
):
    batch = collate([geometry])
    _, force = model(batch, create_graph=True)
    vector = torch.as_tensor(direction, dtype=torch.float64)
    product = -torch.autograd.grad((force * vector).sum(), batch.positions)[0].detach().numpy()
    plus = replace(geometry, positions=geometry.positions + step * direction)
    minus = replace(geometry, positions=geometry.positions - step * direction)
    _, fp = model(collate([plus]))
    _, fm = model(collate([minus]))
    finite = -(fp.detach().numpy() - fm.detach().numpy()) / (2 * step)
    absolute = float(np.linalg.norm(product - finite))
    scale = max(float(np.linalg.norm(product)), float(np.linalg.norm(finite)), 1e-8)
    result = {
        "absolute": float(absolute),
        "relative": float(absolute / scale),
        "step": step,
        "product": product,
        "finite": finite,
    }
    if retain_trace:
        result["trace"] = {
            "geometry": geometry_observation(geometry),
            "direction": direction.tolist(),
            "product": product.tolist(),
            "plus_force": fp.detach().numpy().tolist(),
            "minus_force": fm.detach().numpy().tolist(),
        }
    return result


def geometry_observation(geometry):
    return {
        "numbers": geometry.numbers.tolist(),
        "positions": geometry.positions.tolist(),
        "identity": geometry.identity,
        "chemistry": asdict(geometry.chemistry),
    }


def random_orthogonal(rng: np.random.Generator, reflection=False):
    matrix, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(matrix) < 0:
        matrix[:, 0] *= -1
    if reflection:
        matrix[:, 0] *= -1
    return matrix


def symmetry_trial(model, geometry: Geometry, seed: int, retain_trace=False):
    rng = np.random.default_rng(seed)
    matrix = random_orthogonal(rng, bool(seed % 2))
    permutation = np.arange(len(geometry.numbers))
    for species in np.unique(geometry.numbers):
        selected = np.flatnonzero(geometry.numbers == species)
        permutation[selected] = rng.permutation(selected)
    translation = rng.normal(size=3)
    transformed = replace(
        geometry,
        numbers=geometry.numbers[permutation],
        positions=geometry.positions[permutation] @ matrix.T + translation,
    )
    energy, force = model(collate([geometry]))
    changed_energy, changed_force = model(collate([transformed]))
    actual_force = force.detach().numpy()
    transformed_force = changed_force.detach().numpy()
    target_force = actual_force[permutation] @ matrix.T
    center = geometry.positions.mean(axis=0)
    net = actual_force.sum(axis=0)
    torque = np.cross(geometry.positions - center, actual_force).sum(axis=0)
    double = replace(
        geometry, positions=geometry.positions + np.array([100, 0, 0]), identity="copy"
    )
    disconnected = Geometry(
        np.r_[geometry.numbers, double.numbers],
        np.r_[geometry.positions, double.positions],
        "disconnected",
        geometry.chemistry,
    )
    de, df = model(collate([disconnected]))
    be, bf = model(collate([double, geometry]))
    scale = max(float(np.linalg.norm(actual_force)), 1e-8)
    result = {
        "seed": seed,
        "reflection": bool(seed % 2),
        "energy_error": abs((energy - changed_energy).item()),
        "force_relative": float(float(np.linalg.norm(transformed_force - target_force)) / scale),
        "net_force": float(float(np.linalg.norm(net)) / scale),
        "torque": float(
            float(np.linalg.norm(torque))
            / max(scale * float(np.linalg.norm(center - geometry.positions)), 1e-8)
        ),
        "additivity": abs(de.item() - 2 * energy.item()),
        "disconnected_force": float(
            np.linalg.norm(df.detach().numpy() - np.tile(actual_force, (2, 1))) / scale
        ),
        "batch_energy": float(np.max(np.abs(be.detach().numpy() - energy.item()))),
        "batch_force": float(
            np.linalg.norm(bf.detach().numpy() - np.tile(actual_force, (2, 1))) / scale
        ),
    }
    if retain_trace:
        result["trace"] = {
            "geometry": geometry_observation(geometry),
            "matrix": matrix.tolist(),
            "translation": translation.tolist(),
            "permutation": permutation.tolist(),
            "energy": float(energy.item()),
            "force": actual_force.tolist(),
            "transformed_energy": float(changed_energy.item()),
            "transformed_force": transformed_force.tolist(),
            "disconnected_energy": float(de.item()),
            "disconnected_force": df.detach().numpy().tolist(),
            "batch_energy": be.detach().numpy().tolist(),
            "batch_force": bf.detach().numpy().tolist(),
        }
    return result


def assert_symmetry(result, energy_tolerance=1e-8, relative_tolerance=1e-7):
    for key in ("energy_error", "additivity", "batch_energy"):
        assert result[key] < energy_tolerance, (key, result[key])
    for key in ("force_relative", "net_force", "torque", "disconnected_force", "batch_force"):
        assert result[key] < relative_tolerance, (key, result[key])
