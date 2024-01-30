from pathlib import Path

import numpy as np

from asterion.artifacts.io import file_hash, read_json
from asterion.contracts.errors import IntegrityError
from asterion.data.archive import dataset_arrays


def replay_trajectory(directory: Path, tolerance=1e-9):
    """Independent SI accounting and integration equations from retained arrays."""
    metadata = read_json(directory / "trajectory.json")
    if metadata.get("format") == "asterion-rejected-initial-v1":
        from asterion.artifacts.rejection import rejected_initial

        return rejected_initial(directory)
    if file_hash(directory / "trajectory.npz") != metadata["trajectory_sha256"]:
        raise IntegrityError("trajectory bytes changed")
    required = {
        "positions",
        "velocities",
        "forces",
        "potential",
        "kinetic",
        "heat",
        "numbers",
        "step",
        "noise",
        "uncertainty",
        "neighbor_rebuilds",
        "temperature",
        "uncertainty_available",
        "uncertainty_quantity",
    }
    a = dataset_arrays(directory / "trajectory.npz", required)
    if not required <= a.keys() or not all(np.isfinite(value).all() for value in a.values()):
        raise IntegrityError("missing or nonfinite raw trajectory")
    count = len(a["positions"]) - 1
    if (
        a["uncertainty_available"].dtype.kind != "b"
        or a["uncertainty_quantity"].dtype.kind not in "iu"
        or not np.array_equal(a["uncertainty_available"], a["uncertainty_quantity"] != 0)
        or not set(a["uncertainty_quantity"].tolist()) <= {0, 1, 2}
    ):
        raise IntegrityError("trajectory uncertainty availability or physical quantity changed")
    quantities = {0: "unavailable", 1: "force-disagreement", 2: "calibrated-mae-bound"}
    if (
        len(set(a["uncertainty_quantity"].tolist())) != 1
        or metadata["uncertainty_quantity"] != quantities[int(a["uncertainty_quantity"][0])]
    ):
        raise IntegrityError("trajectory mixes or mislabels uncertainty quantities")
    if count != metadata["completed_steps"] or not np.array_equal(a["step"], np.arange(count + 1)):
        raise IntegrityError("trajectory completion count disagrees with raw steps")
    config = metadata["config"]
    if (
        config["ensemble"] not in {"NVE", "NVT"}
        or type(config["seed"]) is not int
        or config["seed"] < 0
        or not np.isfinite(config["temperature_K"])
        or config["temperature_K"] < 0
        or not np.isfinite(config["friction_per_fs"])
        or config["friction_per_fs"] <= 0
    ):
        raise IntegrityError(
            "trajectory stochastic configuration is outside its finite physical contract"
        )
    requested = metadata["requested_steps"]
    if (
        type(requested) is not int
        or requested < 1
        or requested != config["steps"]
        or type(metadata["completed_steps"]) is not int
        or not 0 <= count <= requested
        or not np.isfinite(config["timestep_fs"])
        or config["timestep_fs"] <= 0
    ):
        raise IntegrityError("trajectory requested workload or completion contract changed")
    status, reason = metadata["status"], metadata["reason"]
    if status == "passed":
        if count != requested or reason != "completed":
            raise IntegrityError("passed trajectory has not completed its declared workload")
    elif status == "failed":
        if count >= requested or not isinstance(reason, str) or not reason:
            raise IntegrityError("failed trajectory hides completed work or its failure reason")
        if file_hash(directory / "failure.npz") != metadata["failure_sha256"]:
            raise IntegrityError("failed trajectory candidate bytes changed")
    elif status == "interrupted":
        if not 0 < count <= requested or reason != "explicit-interruption":
            raise IntegrityError("interrupted trajectory has invalid progress or termination")
    else:
        raise IntegrityError("trajectory has an unknown completion status")
    dt = config["timestep_fs"]
    if count * dt != metadata["physical_horizon_fs"]:
        raise IntegrityError("physical horizon mismatch")
    frames, atoms = len(a["positions"]), len(a["numbers"])
    if (
        a["numbers"].ndim != 1
        or a["numbers"].dtype.kind not in "iu"
        or not 0 < atoms <= 512
        or a["positions"].shape != (frames, atoms, 3)
        or any(a[key].shape != (frames, atoms, 3) for key in ("velocities", "forces", "noise"))
        or any(
            a[key].shape != (frames,)
            for key in required - {"numbers", "positions", "velocities", "forces", "noise"}
        )
        or a["step"].dtype.kind not in "iu"
    ):
        raise IntegrityError("trajectory Cartesian, atom or observation axes changed")
    from asterion.artifacts.rejection import force_reason, geometry_reason

    detector = metadata["detector"]
    policy = detector["policy"]
    if not all(
        np.isfinite(policy[key]) and policy[key] > 0
        for key in (
            "minimum_distance",
            "maximum_force",
            "maximum_bond_ratio",
            "bond_reference_radius",
        )
    ) or (
        policy["maximum_uncertainty"] is not None
        and (not np.isfinite(policy["maximum_uncertainty"]) or policy["maximum_uncertainty"] <= 0)
    ):
        raise IntegrityError("trajectory validity policy has nonfinite or nonpositive limits")
    initial_force_reason = None
    for frame in range(frames):
        force_rejection = force_reason(
            {
                "energy": a["potential"][frame],
                "forces": a["forces"][frame],
                "uncertainty": a["uncertainty"][frame],
                "uncertainty_available": a["uncertainty_available"][frame],
                "uncertainty_quantity": np.asarray(
                    quantities[int(a["uncertainty_quantity"][frame])]
                ),
            },
            a["numbers"],
            policy,
        )
        if frame == 0:
            initial_force_reason = force_rejection
        if geometry_reason(a["numbers"], a["positions"][frame], detector) is not None:
            raise IntegrityError(
                "trajectory contains geometry outside its declared validity policy"
            )
        if force_rejection is not None and not (
            frame == 0 and count == 0 and status == "failed" and force_rejection == reason
        ):
            raise IntegrityError(
                "completed trajectory frame violates its force or uncertainty policy"
            )
    mass_table = {1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999}
    mass_kg = np.array([mass_table[int(z)] for z in a["numbers"]]) * 1.66053906892e-27
    si_velocity = a["velocities"] * 1e5
    kinetic = 0.5 * np.sum(mass_kg[None, :, None] * si_velocity**2, axis=(1, 2)) / 1.602176634e-19
    if not np.allclose(kinetic, a["kinetic"], rtol=tolerance, atol=tolerance):
        raise IntegrityError("independent SI kinetic energy mismatch")
    if len(mass_kg) > 1:
        center_velocity = np.sum(mass_kg[None, :, None] * si_velocity, axis=1) / np.sum(mass_kg)
        internal = si_velocity - center_velocity[:, None, :]
        expected_temperature = np.sum(mass_kg[None, :, None] * internal**2, axis=(1, 2)) / (
            (3 * len(mass_kg) - 3) * 1.380649e-23
        )
    else:
        expected_temperature = np.zeros(len(si_velocity))
    if not np.allclose(expected_temperature, a["temperature"], rtol=tolerance, atol=tolerance):
        raise IntegrityError("independent internal-temperature accounting mismatch")
    dt_seconds = dt * 1e-15
    acceleration = a["forces"] * (1.602176634e-19 / 1e-10) / mass_kg[None, :, None]
    acceleration *= 1e-20
    position_error = velocity_error = heat_error = 0.0
    rng = np.random.default_rng(config["seed"])
    if np.any(a["noise"][0]) or (config["ensemble"] == "NVE" and np.any(a["noise"])):
        raise IntegrityError("trajectory invents thermostat noise outside an NVT update")
    for i in range(count):
        if config["ensemble"] == "NVE":
            expected_position = (
                a["positions"][i] + dt * a["velocities"][i] + 0.5 * dt**2 * acceleration[i]
            )
            expected_velocity = a["velocities"][i] + 0.5 * dt * (
                acceleration[i] + acceleration[i + 1]
            )
            expected_heat = a["heat"][i]
        else:
            if not np.array_equal(a["noise"][i + 1], rng.normal(size=(atoms, 3))):
                raise IntegrityError("NVT noise does not match the declared seeded Gaussian stream")
            first_velocity = a["velocities"][i] + 0.5 * dt * acceleration[i]
            decay = np.exp(-config["friction_per_fs"] * dt)
            # Convert thermal SI velocities into angstrom/fs independently.
            noise_scale = (
                np.sqrt((1 - decay**2) * 1.380649e-23 * config["temperature_K"] / mass_kg) / 1e5
            )
            middle_velocity = decay * first_velocity + noise_scale[:, None] * a["noise"][i + 1]
            expected_position = a["positions"][i] + 0.5 * dt * (first_velocity + middle_velocity)
            expected_velocity = middle_velocity + 0.5 * dt * acceleration[i + 1]
            change = (
                0.5
                * np.sum(
                    mass_kg[:, None] * ((middle_velocity * 1e5) ** 2 - (first_velocity * 1e5) ** 2)
                )
                / 1.602176634e-19
            )
            expected_heat = a["heat"][i] + change
        position_error = max(
            position_error, float(np.max(np.abs(expected_position - a["positions"][i + 1])))
        )
        velocity_error = max(
            velocity_error, float(np.max(np.abs(expected_velocity - a["velocities"][i + 1])))
        )
        heat_error = max(heat_error, abs(expected_heat - a["heat"][i + 1]))
    if max(position_error, velocity_error, heat_error) > tolerance:
        raise IntegrityError("independent integrator replay mismatch")
    if status == "failed":
        attempted = dataset_arrays(directory / "failure.npz", {"positions"})["positions"]
        if count == 0 and initial_force_reason is not None:
            expected = a["positions"][0]
        elif config["ensemble"] == "NVE":
            expected = (
                a["positions"][-1] + dt * a["velocities"][-1] + 0.5 * dt**2 * acceleration[-1]
            )
        else:
            first = a["velocities"][-1] + 0.5 * dt * acceleration[-1]
            decay = np.exp(-config["friction_per_fs"] * dt)
            scale = np.sqrt((1 - decay**2) * 1.380649e-23 * config["temperature_K"] / mass_kg) / 1e5
            noisy = decay * first + scale[:, None] * rng.normal(size=(atoms, 3))
            expected = a["positions"][-1] + 0.5 * dt * (first + noisy)
        if attempted.shape != expected.shape or not np.allclose(
            attempted, expected, atol=tolerance, rtol=0, equal_nan=True
        ):
            raise IntegrityError("failed candidate was not produced by the declared integrator")
    total = a["potential"] + kinetic
    adjusted = total - a["heat"]
    drift = adjusted - adjusted[0]
    return {
        "completed_steps": count,
        "ensemble": config["ensemble"],
        "position_error": position_error,
        "velocity_error": velocity_error,
        "heat_error": heat_error,
        "max_energy_drift_eV": float(np.max(np.abs(drift))),
        "end_energy_drift_eV": float(drift[-1]),
        "physical_horizon_fs": count * dt,
        "duration_seconds_SI": count * dt_seconds,
    }
