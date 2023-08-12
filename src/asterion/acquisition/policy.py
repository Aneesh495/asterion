from dataclasses import dataclass

import numpy as np

from asterion.contracts.errors import ContractError


@dataclass
class Choices:
    rows: list[int]
    scores: list[float]
    novelty: list[float]
    policy: str
    seed: int


def rank01(values):
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values))
    return ranks / max(len(values) - 1, 1)


def choose(pool_rows, descriptor, revealed, count, policy, seed, uncertainty=None, weight=0.5):
    pool_rows = np.asarray(pool_rows, dtype=np.int64)
    descriptor = np.asarray(descriptor, dtype=np.float64)
    if descriptor.shape[0] != len(pool_rows) or len(set(pool_rows.tolist())) != len(pool_rows):
        raise ContractError("acquisition descriptor rows mismatch")
    if not set(revealed) <= set(pool_rows) or count < 1 or count > len(pool_rows) - len(revealed):
        raise ContractError("acquisition budget exceeds remaining unique pool")
    if not np.isfinite(descriptor).all():
        raise ContractError("nonfinite acquisition descriptors")
    available = ~np.isin(pool_rows, revealed)
    indices = np.flatnonzero(available)
    if policy == "uniform":
        selected = np.random.default_rng(seed).choice(indices, count, replace=False)
        return Choices(pool_rows[selected].tolist(), [0.0] * count, [0.0] * count, policy, seed)
    if policy not in {"diversity", "uncertainty-diversity"}:
        raise ContractError("unknown acquisition policy")
    if policy == "uncertainty-diversity":
        if uncertainty is None or np.asarray(uncertainty).shape != (len(pool_rows),):
            raise ContractError("committee scores required for every allowed candidate")
        if not np.isfinite(uncertainty).all() or np.any(np.asarray(uncertainty) < 0):
            raise ContractError("invalid uncertainty acquisition scores")
        ranks = rank01(np.asarray(uncertainty))
    else:
        ranks = np.zeros(len(pool_rows))
    # Descriptor scaling uses coordinates from the candidate pool, without inspecting labels.
    scale = np.maximum(descriptor.std(0), 1e-6)
    normalized = (descriptor - descriptor.mean(0)) / scale
    selected_indices = np.flatnonzero(~available)
    distance = np.full(len(pool_rows), np.inf)
    if len(selected_indices):
        for index in selected_indices:
            distance = np.minimum(distance, np.linalg.norm(normalized - normalized[index], axis=-1))
    else:
        distance = np.linalg.norm(normalized - normalized.mean(0), axis=-1)
    choices, recorded_scores, recorded_novelty = [], [], []
    for _ in range(count):
        novelty_rank = rank01(distance)
        score = (
            novelty_rank if policy == "diversity" else weight * ranks + (1 - weight) * novelty_rank
        )
        score[~available] = -np.inf
        index = int(np.argmax(score))
        choices.append(int(pool_rows[index]))
        recorded_scores.append(float(score[index]))
        recorded_novelty.append(float(distance[index]))
        available[index] = False
        distance = np.minimum(distance, np.linalg.norm(normalized - normalized[index], axis=-1))
    return Choices(choices, recorded_scores, recorded_novelty, policy, seed)
