import numpy as np
from scipy.stats import spearmanr

from asterion.contracts.errors import ContractError


def prediction_metrics(energy, force, target_energy, target_force, pointers):
    energy, force = np.asarray(energy), np.asarray(force)
    target_energy, target_force = np.asarray(target_energy), np.asarray(target_force)
    if energy.shape != target_energy.shape or force.shape != target_force.shape:
        raise ContractError("metric axes mismatch")
    if not all(np.isfinite(x).all() for x in (energy, force, target_energy, target_force)):
        raise ContractError("nonfinite prediction metrics")
    error = force - target_force
    per_geometry = np.array(
        [np.abs(error[a:b]).mean() for a, b in zip(pointers[:-1], pointers[1:])]
    )
    if not len(per_geometry) or len(per_geometry) != len(energy):
        raise ContractError("metric denominators require nonempty geometry units")
    return {
        "energy_mae_eV": float(np.abs(energy - target_energy).mean()),
        "energy_rmse_eV": float(np.sqrt(np.mean((energy - target_energy) ** 2))),
        "force_mae_eV_angstrom": float(per_geometry.mean()),
        "force_rmse_eV_angstrom": float(np.sqrt(np.mean(error**2))),
        "geometry_force_mae": per_geometry.tolist(),
        "geometries": len(energy),
        "atoms": len(force),
        "force_components": force.size,
    }


def paired_block_interval(first, second, blocks, seed=491, repetitions=2000):
    first, second, blocks = np.asarray(first), np.asarray(second), np.asarray(blocks)
    if first.shape != second.shape or first.shape != blocks.shape or not len(first):
        raise ContractError("paired block comparison requires matched nonempty units")
    if (
        first.ndim != 1
        or not np.isfinite(first).all()
        or not np.isfinite(second).all()
        or repetitions < 1
    ):
        raise ContractError(
            "finite one-dimensional paired errors and positive repetitions required"
        )
    if blocks.dtype.kind in "fiu" and not np.isfinite(blocks).all():
        raise ContractError("nonfinite temporal block identities")
    unique = np.unique(blocks)
    if len(unique) < 2:
        raise ContractError("uncertainty requires >=2 independent declared blocks")
    totals = np.array(
        [(first[blocks == block] - second[blocks == block]).sum() for block in unique]
    )
    counts = np.array([(blocks == block).sum() for block in unique])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(unique), size=(repetitions, len(unique)))
    samples = totals[indices].sum(1) / counts[indices].sum(1)
    return {
        "mean_improvement": float((first - second).mean()),
        "lower_95": float(np.quantile(samples, 0.025)),
        "upper_95": float(np.quantile(samples, 0.975)),
        "blocks": len(unique),
        "repetitions": repetitions,
        "seed": seed,
    }


def uncertainty_metrics(scores, errors, bound):
    scores, errors, bound = map(np.asarray, (scores, errors, bound))
    if not len(scores) or scores.shape != errors.shape or bound.shape != errors.shape:
        raise ContractError("uncertainty arrays require matched geometry axes")
    correlation = float(spearmanr(scores, errors).statistic)
    return {
        "spearman": correlation if np.isfinite(correlation) else None,
        "coverage": float(np.mean(errors <= bound)),
        "mean_bound": float(bound.mean()),
        "geometries": len(errors),
    }
