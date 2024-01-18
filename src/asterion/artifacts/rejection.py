"""Independently derive rejected-start reasons from unmodified raw observations."""

import numpy as np

from asterion.artifacts.io import file_hash, object_hash, read_json
from asterion.contracts.errors import AsterionError, IntegrityError


def require(condition, message):
    if not condition:
        raise IntegrityError(message)


def geometry_reason(numbers, positions, detector):
    policy = detector["policy"]
    if positions.shape != (len(numbers), 3) or not np.isfinite(positions).all():
        return "nonfinite-or-invalid-geometry"
    if not set(numbers.tolist()) <= set(policy["supported_elements"]):
        return "unsupported-chemistry"
    for atom in range(len(numbers)):
        if np.any(
            np.linalg.norm(positions[atom + 1 :] - positions[atom], axis=1)
            < policy["minimum_distance"]
        ):
            return "atomic-collision"
    for first, second, reference in detector["initial_bonds"]:
        if (
            np.linalg.norm(positions[first] - positions[second])
            > reference * policy["maximum_bond_ratio"]
        ):
            return "bond-domain-exceeded"
    return None


def force_reason(raw, numbers, policy):
    quantity = raw["uncertainty_quantity"].item()
    available = raw["uncertainty_available"].item()
    if quantity not in {
        "unavailable",
        "force-disagreement",
        "calibrated-mae-bound",
    } or available != (quantity != "unavailable"):
        return "invalid-uncertainty-contract"
    if not np.isfinite(raw["energy"]).all() or not np.isfinite(raw["forces"]).all():
        return "nonfinite-energy-or-force"
    if raw["forces"].shape != (len(numbers), 3):
        return "invalid-force-shape"
    if np.sqrt(np.sum(raw["forces"] ** 2, axis=1)).max() > policy["maximum_force"]:
        return "extreme-force"
    uncertainty = raw["uncertainty"].item()
    if not np.isfinite(uncertainty) or uncertainty < 0:
        return "invalid-uncertainty"
    if policy["maximum_uncertainty"] is not None and not available:
        return "uncertainty-unavailable"
    if policy["maximum_uncertainty"] is not None and uncertainty > policy["maximum_uncertainty"]:
        return "uncertainty-domain-exceeded"
    return None


def rejected_initial(directory):
    metadata = read_json(directory / "trajectory.json")
    require(
        metadata["format"] == "asterion-rejected-initial-v1"
        and metadata["status"] == "failed"
        and metadata["completed_steps"] == 0
        and metadata["valid_frames"] == 0
        and metadata["physical_horizon_fs"] == 0
        and metadata["requested_steps"] == metadata["config"]["steps"] > 0,
        "rejected initial state invents valid frames or completed physical work",
    )
    require(
        not (directory / "trajectory.npz").exists(),
        "rejected initial observation is also presented as a valid trajectory",
    )
    require(
        file_hash(directory / "failure.npz") == metadata["failure_sha256"],
        "rejected initial observation bytes changed",
    )
    with np.load(directory / "failure.npz", allow_pickle=False) as archive:
        raw = {key: archive[key] for key in archive.files}
    numbers, positions, velocities = (raw[key] for key in ("numbers", "positions", "velocities"))
    require(
        numbers.ndim == 1
        and numbers.dtype.kind in "iu"
        and 0 < len(numbers) <= 512
        and positions.shape == (len(numbers), 3)
        and positions.dtype.kind == "f"
        and velocities.shape == positions.shape
        and velocities.dtype.kind == "f"
        and np.isfinite(positions).all()
        and np.isfinite(velocities).all(),
        "rejected initial state has malformed retained input axes",
    )
    require(
        metadata["initial_identity"]
        == object_hash({"numbers": numbers.tolist(), "positions": positions.tolist()}),
        "rejected initial coordinate identity changed",
    )
    detector = metadata["detector"]
    policy = detector["policy"]
    bonds = []
    for first in range(len(numbers)):
        for second in range(first + 1, len(numbers)):
            distance = float(np.sqrt(np.sum((positions[first] - positions[second]) ** 2)))
            if distance < policy["bond_reference_radius"]:
                bonds.append([first, second, distance])
    require(
        len(bonds) == len(detector["initial_bonds"])
        and np.allclose(
            np.asarray(bonds).reshape(-1, 3),
            np.asarray(detector["initial_bonds"]).reshape(-1, 3),
            atol=1e-12,
            rtol=0,
        ),
        "rejected initial detector does not belong to retained geometry",
    )
    kind = metadata["observation_kind"]
    input_fields = {"numbers", "positions", "velocities"}
    reason = geometry_reason(numbers, positions, detector)
    if kind == "geometry-rejection":
        require(
            set(raw) == input_fields
            and reason == metadata["reason"]
            and reason is not None
            and metadata["evaluation_error"] is None,
            "rejected initial geometry reason lacks raw support",
        )
    else:
        require(reason is None, "initial force was evaluated outside the geometry policy")
        if kind == "force-error":
            error = metadata["evaluation_error"]
            require(
                set(raw) == input_fields
                and error["type"]
                and error["message"]
                and metadata["reason"] == f"force-evaluation:{error['message']}",
                "rejected initial evaluation error changed or invents outputs",
            )
        else:
            output_fields = {
                "energy",
                "forces",
                "uncertainty",
                "uncertainty_available",
                "uncertainty_quantity",
                "neighbor_rebuilds",
            }
            require(
                kind == "force-result"
                and set(raw) == input_fields | output_fields
                and raw["energy"].shape == ()
                and raw["uncertainty"].shape == ()
                and raw["uncertainty_available"].shape == ()
                and raw["uncertainty_available"].dtype.kind == "b"
                and raw["uncertainty_quantity"].shape == ()
                and raw["uncertainty_quantity"].dtype.kind in "US"
                and metadata["evaluation_error"] is None,
                "rejected initial force observation has malformed fields",
            )
            reason = force_reason(raw, numbers, policy)
            require(
                reason is not None and reason == metadata["reason"],
                "rejected initial force reason lacks raw support",
            )
    return {
        "completed_steps": 0,
        "physical_horizon_fs": 0.0,
        "duration_seconds_SI": 0.0,
        "ensemble": metadata["config"]["ensemble"],
        "position_error": 0.0,
        "velocity_error": 0.0,
        "heat_error": 0.0,
        "max_energy_drift_eV": None,
        "end_energy_drift_eV": None,
        "valid_frames": 0,
        "rejected_initial": True,
    }


def verify_rejected_model(directory, model):
    rejected_initial(directory)
    metadata = read_json(directory / "trajectory.json")
    if metadata["observation_kind"] == "geometry-rejection":
        return
    with np.load(directory / "failure.npz", allow_pickle=False) as archive:
        raw = {key: archive[key] for key in archive.files}
    from asterion.contracts.geometry import Chemistry
    from asterion.dynamics.learned import LearnedEvaluator

    try:
        observed = LearnedEvaluator(model, Chemistry(level=model.config.level))(
            raw["positions"], raw["numbers"]
        )
    except (AsterionError, FloatingPointError) as error:
        require(
            metadata["observation_kind"] == "force-error"
            and metadata["evaluation_error"]
            == {"type": type(error).__name__, "message": str(error)},
            "rejected initial error does not belong to saved weights",
        )
    else:
        require(
            metadata["observation_kind"] == "force-result"
            and raw["forces"].shape == observed.forces.shape
            and np.allclose(raw["forces"], observed.forces, equal_nan=True, atol=1e-9, rtol=1e-7)
            and np.allclose(raw["energy"], observed.energy, equal_nan=True, atol=1e-9, rtol=1e-8),
            "rejected initial observations do not belong to saved weights",
        )
