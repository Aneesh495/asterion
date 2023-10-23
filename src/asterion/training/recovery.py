"""Rollback to an immutable training generation without discarding interrupted work."""

import os
import re
import uuid
from pathlib import Path

import numpy as np

from asterion.artifacts.io import file_hash, object_hash, read_json, safe_member, write_json
from asterion.contracts.errors import IntegrityError


def tree_digest(path: Path):
    if path.is_symlink():
        raise IntegrityError("linked training recovery input")
    if path.is_file():
        return file_hash(path)
    values = {}
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise IntegrityError("linked training recovery input")
        if item.is_file():
            values[str(item.relative_to(path))] = file_hash(item)
    return object_hash(values)


def finish_archive(directory: Path, plan_path: Path, plan):
    for item in plan["moves"]:
        original = safe_member(directory, item["original"])
        retained = safe_member(directory, item["retained"])
        if original.exists() and retained.exists():
            raise IntegrityError("training recovery archive has two possible originals")
        existing = original if original.exists() else retained
        if not existing.exists() or tree_digest(existing) != item["digest"]:
            raise IntegrityError("training recovery archive bytes changed")
        if original.exists():
            retained.parent.mkdir(parents=True, exist_ok=True)
            original.rename(retained)
            for parent in {original.parent, retained.parent}:
                descriptor = os.open(parent, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
    plan["status"] = "archived"
    write_json(plan_path, plan)


def checkpoint_history(directory: Path, record):
    completed = record["epoch"]
    if not isinstance(completed, int) or completed < 0:
        raise IntegrityError("invalid recovery epoch")
    history = record.get("training_history")
    if history is None:
        history = read_json(directory / "history.json")[:completed] if completed else []
    if len(history) != completed or any(
        row["epoch"] != index + 1
        or not np.isfinite(row["validation_force_mae"])
        or row["validation_force_mae"] < 0
        for index, row in enumerate(history)
    ):
        raise IntegrityError("recovery history does not end at the checkpoint epoch")
    if history and history[-1]["updates"] != record["updates"]:
        raise IntegrityError("recovery optimizer update count disagrees with history")
    if not history:
        return history, None
    best = min(history, key=lambda row: (row["validation_force_mae"], row["epoch"]))
    selection = {
        "checkpoint": f"epoch-{best['epoch']:05}",
        "epoch": best["epoch"],
        "score": best["validation_force_mae"],
        "rule": "minimum-validation-force-mae",
        "lineage": record["lineage"],
    }
    if "training_selection" in record and record["training_selection"] != selection:
        raise IntegrityError("checkpoint selection is inconsistent with validation history")
    best_path = directory / selection["checkpoint"]
    if not best_path.is_dir() or best_path.is_symlink():
        raise IntegrityError("selected recovery generation is missing")
    metadata = read_json(best_path / "checkpoint.json")
    if metadata["lineage"] != record["lineage"] or metadata["epoch"] != selection["epoch"]:
        raise IntegrityError("selected recovery checkpoint has another lineage")
    return history, selection


def reconcile_training(directory: Path, checkpoint: Path, record):
    checkpoint = checkpoint.resolve()
    if checkpoint.parent != directory.resolve() or not (
        checkpoint.name == "initial" or re.fullmatch(r"epoch-[0-9]{5}", checkpoint.name)
    ):
        raise IntegrityError("resume generation must belong to this training run")
    plan_path = directory / "RECOVERY.json"
    if plan_path.exists():
        plan = read_json(plan_path)
        if plan["status"] == "preparing":
            if plan["checkpoint_identity"] != record["identity"]:
                raise IntegrityError("unfinished recovery uses another checkpoint")
            finish_archive(directory, plan_path, plan)
    history, selection = checkpoint_history(directory, record)
    future = [
        path
        for path in directory.iterdir()
        if re.fullmatch(r"epoch-[0-9]{5}", path.name) and int(path.name[6:]) > record["epoch"]
    ]
    mutable = [
        directory / name
        for name in ("history.json", "selection.json", "latest.json", "completed.json")
        if (directory / name).exists()
    ]
    attempt = uuid.uuid4().hex
    archive = Path("recovery") / attempt
    plan = {
        "format": "asterion-training-recovery-v1",
        "checkpoint_identity": record["identity"],
        "lineage": record["lineage"],
        "epoch": record["epoch"],
        "status": "preparing",
        "moves": [
            {
                "original": path.name,
                "retained": str(archive / path.name),
                "digest": tree_digest(path),
            }
            for path in (*future, *mutable)
        ],
    }
    write_json(plan_path, plan)
    write_json(directory / archive / "RECOVERY.json", plan)
    finish_archive(directory, plan_path, plan)
    write_json(directory / "history.json", history)
    if selection is not None:
        write_json(directory / "selection.json", selection)
    write_json(directory / "latest.json", {"checkpoint": checkpoint.name, "epoch": record["epoch"]})
    plan["status"] = "restored"
    write_json(plan_path, plan)
    write_json(directory / archive / "RECOVERY.json", plan)
    return history, selection
