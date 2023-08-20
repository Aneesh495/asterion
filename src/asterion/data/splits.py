from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from asterion.artifacts.io import object_hash, read_json, write_json
from asterion.contracts.errors import ContractError, IntegrityError
from asterion.data.dataset import CoordinatePool


@dataclass(frozen=True)
class Split:
    dataset_hash: str
    protocol: str
    seed: int
    train_pool: tuple[int, ...]
    validation: tuple[int, ...]
    calibration: tuple[int, ...]
    evaluation: tuple[int, ...]
    temporal_guard: int = 0

    def validate(self, size: int):
        roles = [self.train_pool, self.validation, self.calibration, self.evaluation]
        for rows in roles:
            if not rows or len(rows) != len(set(rows)) or min(rows) < 0 or max(rows) >= size:
                raise ContractError("empty, duplicate or out-of-range split rows")
        for i, rows in enumerate(roles):
            for other in roles[i + 1 :]:
                if set(rows) & set(other):
                    raise ContractError("split roles overlap")
        return self

    @property
    def digest(self):
        return object_hash(asdict(self))

    def write(self, path: Path):
        value = asdict(self)
        if path.exists() and read_json(path) != {**value, "digest": self.digest}:
            if (
                object_hash({k: v for k, v in read_json(path).items() if k != "digest"})
                != self.digest
            ):
                raise IntegrityError("immutable split already exists with other contents")
        write_json(path, {**value, "digest": self.digest})


def author_split(
    pool: CoordinatePool,
    directory: Path,
    seed: int,
    evaluation_size: int = 5000,
    validation_size: int = 500,
    calibration_size: int = 500,
) -> Split:
    # Published rows are retained verbatim. Additional evaluation rows form an explicitly named
    # supplement because the authors' test CSV itself contains only 1000 entries.
    train = np.loadtxt(directory / f"index_train_{seed:02}.csv", dtype=int, delimiter=",").ravel()
    test = np.loadtxt(directory / f"index_test_{seed:02}.csv", dtype=int, delimiter=",").ravel()
    if len(train) != 1000 or len(test) != 1000:
        raise ContractError("unexpected author split size")
    remaining = np.array(sorted(set(range(len(pool))) - set(train.tolist()) - set(test.tolist())))
    np.random.default_rng(seed).shuffle(remaining)
    needed = max(0, evaluation_size - len(test)) + validation_size + calibration_size
    if needed > len(remaining):
        raise ContractError("insufficient rows for author supplement")
    validation = remaining[:validation_size]
    calibration = remaining[validation_size : validation_size + calibration_size]
    supplement = remaining[validation_size + calibration_size : needed]
    evaluation = np.r_[test, supplement] if evaluation_size >= len(test) else test[:evaluation_size]
    return Split(
        pool.dataset_hash,
        "author-plus-evaluation-supplement",
        seed,
        tuple(map(int, train)),
        tuple(map(int, validation)),
        tuple(map(int, calibration)),
        tuple(map(int, evaluation)),
    ).validate(len(pool))


def geometry_descriptors(pool: CoordinatePool, rows, chunk_size: int = 256) -> np.ndarray:
    """Species-pair sorted distances are invariant to rigid motion and same-species permutations."""
    rows = np.asarray(rows, dtype=np.int64)
    pairs = [(i, j) for i in range(len(pool.numbers)) for j in range(i + 1, len(pool.numbers))]
    categories = sorted(
        {tuple(sorted((int(pool.numbers[i]), int(pool.numbers[j])))) for i, j in pairs}
    )
    blocks = []
    for start in range(0, len(rows), chunk_size):
        xyz = pool.positions[rows[start : start + chunk_size]]
        values = []
        for species in categories:
            selected = [
                (i, j)
                for i, j in pairs
                if tuple(sorted((int(pool.numbers[i]), int(pool.numbers[j])))) == species
            ]
            first, second = np.array(selected).T
            values.append(np.sort(np.linalg.norm(xyz[:, first] - xyz[:, second], axis=-1), axis=-1))
        blocks.append(np.concatenate(values, axis=-1))
    if not blocks:
        raise ContractError("descriptor input cannot be empty")
    return np.concatenate(blocks)


def guarded_split(
    pool: CoordinatePool,
    seed: int,
    train_size=1000,
    evaluation_size=5000,
    validation_size=500,
    calibration_size=500,
    guard=100,
) -> Split:
    order = np.argsort(pool.original_indices)
    n = len(order)
    cuts = (int(n * 0.45), int(n * 0.60), int(n * 0.75))
    partitions = [
        order[: cuts[0]],
        order[cuts[0] : cuts[1]],
        order[cuts[1] : cuts[2]],
        order[cuts[2] :],
    ]
    rng = np.random.default_rng(seed)
    selected = []
    for part, size in zip(
        partitions, (train_size, validation_size, calibration_size, evaluation_size)
    ):
        low = pool.original_indices[part[0]] + guard
        high = pool.original_indices[part[-1]] - guard
        part = part[(pool.original_indices[part] > low) & (pool.original_indices[part] < high)]
        if len(part) < size:
            raise ContractError("temporal block too small after guards")
        selected.append(tuple(map(int, rng.choice(part, size=size, replace=False))))
    return Split(
        pool.dataset_hash,
        "temporal-block-guard",
        seed,
        selected[0],
        selected[1],
        selected[2],
        selected[3],
        temporal_guard=guard,
    ).validate(n)


def audit_leakage(pool: CoordinatePool, split: Split, near_threshold: float = 0.01):
    train = np.array(split.train_pool)
    test = np.array(split.evaluation)
    desc_train = geometry_descriptors(pool, train)
    desc_test = geometry_descriptors(pool, test)
    distance, neighbor = cKDTree(desc_train).query(desc_test)
    distance /= np.sqrt(desc_train.shape[1])
    temporal = np.sort(pool.original_indices[train])
    insertion = np.searchsorted(temporal, pool.original_indices[test])
    left = temporal[np.clip(insertion - 1, 0, len(temporal) - 1)]
    right = temporal[np.clip(insertion, 0, len(temporal) - 1)]
    gap = np.minimum(
        np.abs(left - pool.original_indices[test]), np.abs(right - pool.original_indices[test])
    )
    if split.temporal_guard and np.any(gap <= split.temporal_guard):
        raise IntegrityError("temporal guard violated")
    return {
        "split": split.digest,
        "distance_rms_angstrom": distance.tolist(),
        "nearest_train_rows": train[neighbor].tolist(),
        "temporal_gaps": gap.tolist(),
        "near_threshold": near_threshold,
        "near_count": int((distance < near_threshold).sum()),
        "evaluated": len(test),
        "minimum_gap": int(gap.min()),
    }
