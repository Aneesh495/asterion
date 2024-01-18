"""Retain rejected initial observations without manufacturing a valid physical frame."""

from dataclasses import asdict

import numpy as np

from asterion.artifacts.io import atomic_npz, file_hash, object_hash, write_json


def reject_initial(
    directory,
    numbers,
    positions,
    velocities,
    config,
    detector,
    lineage,
    reason,
    result=None,
    error=None,
):
    arrays = {"numbers": numbers, "positions": positions, "velocities": velocities}
    kind = "geometry-rejection"
    if result is not None:
        kind = "force-result"
        arrays.update(
            energy=np.asarray(result.energy),
            forces=np.asarray(result.forces),
            uncertainty=np.asarray(result.uncertainty),
            uncertainty_available=np.asarray(result.uncertainty_available),
            uncertainty_quantity=np.asarray(result.uncertainty_quantity),
            neighbor_rebuilds=np.asarray(result.neighbor_rebuilds),
        )
    if error is not None:
        kind = "force-error"
    atomic_npz(directory / "failure.npz", **arrays)
    metadata = {
        "format": "asterion-rejected-initial-v1",
        "config": asdict(config),
        "lineage": lineage,
        "detector": detector.to_dict(),
        "status": "failed",
        "reason": reason,
        "completed_steps": 0,
        "requested_steps": config.steps,
        "physical_horizon_fs": 0.0,
        "valid_frames": 0,
        "observation_kind": kind,
        "evaluation_error": None
        if error is None
        else {"type": type(error).__name__, "message": str(error)},
        "uncertainty_quantity": "unavailable",
        "failure_sha256": file_hash(directory / "failure.npz"),
        "initial_identity": object_hash(
            {"numbers": numbers.tolist(), "positions": positions.tolist()}
        ),
    }
    write_json(directory / "trajectory.json", metadata)
    return metadata
