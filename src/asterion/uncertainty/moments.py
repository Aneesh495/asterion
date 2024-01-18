"""Scaled committee moments avoid overflow before validity policy can inspect a result."""

import numpy as np

from asterion.contracts.errors import ContractError


def scaled_moments(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 2 or values.shape[0] != 3 or not np.isfinite(values).all():
        raise ContractError("committee moments require three finite member observations")
    scale = np.max(np.abs(values), axis=0)
    denominator = np.where(scale > 0, scale, 1)
    normalized = values / denominator
    mean = normalized.sum(axis=0) / 3
    centered = normalized - mean
    deviation = np.sqrt(np.sum(centered * centered, axis=0) / 2)
    try:
        with np.errstate(over="raise", invalid="raise"):
            mean = mean * scale
            deviation = deviation * scale
    except FloatingPointError as error:
        raise ContractError("committee moment exceeds finite physical output range") from error
    if not np.isfinite(mean).all() or not np.isfinite(deviation).all():
        raise ContractError("nonfinite aggregated committee moment")
    return mean, deviation


def member_statistics(energies, forces, atom_counts):
    energies, forces = np.asarray(energies), np.asarray(forces)
    counts = np.asarray(atom_counts)
    if (
        counts.ndim != 1
        or not len(counts)
        or counts.dtype.kind not in "iu"
        or np.any(counts <= 0)
        or energies.shape != (3, len(counts))
        or forces.shape != (3, int(counts.sum()), 3)
    ):
        raise ContractError("committee member, geometry and Cartesian axes disagree")
    energy_mean, energy_std = scaled_moments(energies)
    force_mean, force_std = scaled_moments(forces)
    pointers = np.r_[0, counts.cumsum()]
    scores = []
    for begin, end in zip(pointers[:-1], pointers[1:]):
        block = force_std[begin:end]
        scale = float(np.max(block))
        scores.append(scale * float(np.sqrt(np.mean((block / scale) ** 2))) if scale else 0.0)
    score_array = np.asarray(scores)
    if not np.isfinite(score_array).all():
        raise ContractError("nonfinite aggregated force-disagreement score")
    return energy_mean, force_mean, energy_std, force_std, score_array


def conformal_parameters(scores, errors, alpha):
    ordered = np.sort(scores)
    center = float(ordered[(len(ordered) - 1) // 2] / 2 + ordered[len(ordered) // 2] / 2)
    floor = max(center * 0.01, 1e-8)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            ratios = errors / np.maximum(scores, floor)
    except FloatingPointError as error:
        raise ContractError("calibration ratio exceeds finite physical output range") from error
    rank = min(len(ratios), int(np.ceil((len(ratios) + 1) * (1 - alpha))))
    return float(np.sort(ratios)[rank - 1]), floor
