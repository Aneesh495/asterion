import time
from pathlib import Path

import numpy as np

from asterion.artifacts.io import object_hash, read_json, write_json
from asterion.contracts.errors import IntegrityError, LabelAccessError
from asterion.data.splits import Split


class RevealLedger:
    """Hash-chained snapshots. One coordinator owns all reveals for a run."""

    def __init__(self, split: Split, directory: Path):
        self.split = split
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rounds = []
        previous = "genesis"
        seen: set[int] = set()
        for path in sorted(directory.glob("round-*.json")):
            value = read_json(path)
            digest = value.pop("digest")
            if object_hash(value) != digest or value["previous"] != previous:
                raise IntegrityError("reveal ledger hash chain broken")
            if value["split"] != split.digest or value["round"] != len(self.rounds):
                raise IntegrityError("reveal ledger lineage mismatch")
            rows = set(value["rows"])
            if not rows or rows & seen or not rows <= set(split.train_pool):
                raise LabelAccessError("invalid label reveal in persisted ledger")
            if value["total"] != len(seen | rows):
                raise IntegrityError("reveal count mismatch")
            seen |= rows
            value["digest"] = digest
            self.rounds.append(value)
            previous = digest

    @property
    def revealed(self) -> tuple[int, ...]:
        return tuple(row for record in self.rounds for row in record["rows"])

    @property
    def digest(self):
        return self.rounds[-1]["digest"] if self.rounds else "genesis"

    def reveal(self, rows, policy: str, model_hashes=(), score_hash: str = "initial"):
        rows = list(map(int, rows))
        if not rows or len(rows) != len(set(rows)) or set(rows) & set(self.revealed):
            raise LabelAccessError("each geometry must be revealed exactly once")
        if not set(rows) <= set(self.split.train_pool) or len(self.revealed) + len(rows) > 1000:
            raise LabelAccessError("reveal outside allowed pool or maximum budget")
        if not policy:
            raise LabelAccessError("acquisition policy identity required")
        value = {
            "round": len(self.rounds),
            "split": self.split.digest,
            "previous": self.digest,
            "rows": rows,
            "total": len(self.revealed) + len(rows),
            "policy": policy,
            "model_hashes": list(model_hashes),
            "scores": score_hash,
            "timestamp_ns": time.time_ns(),
        }
        value["digest"] = object_hash(value)
        path = self.directory / f"round-{len(self.rounds):04}.json"
        if path.exists():
            raise IntegrityError("concurrent reveal coordinator or immutable-round collision")
        write_json(path, value)
        self.rounds.append(value)
        return value


def fit_offsets(labels):
    if not labels:
        raise LabelAccessError("offset fitting requires revealed training labels")
    atom_count = {len(label.geometry.numbers) for label in labels}
    if len(atom_count) != 1:
        raise LabelAccessError("per-molecule offsets require fixed composition")
    energy = np.array([label.energy for label in labels])
    force = np.concatenate([label.forces for label in labels])
    return {
        "energy_per_atom": float(energy.mean() / next(iter(atom_count))),
        "energy_scale": max(float(energy.std()), 0.01),
        "force_scale": max(float(np.sqrt(np.mean(force**2))), 0.01),
        "fit_identities": [label.geometry.identity for label in labels],
    }
