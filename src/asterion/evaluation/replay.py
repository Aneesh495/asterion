from pathlib import Path

import numpy as np

from asterion.artifacts.io import file_hash, read_json
from asterion.contracts.errors import IntegrityError


def replay_trajectory(directory: Path, tolerance=1e-9):
    """Independent SI accounting and integration equations from retained arrays."""
    metadata = read_json(directory / "trajectory.json")
    if file_hash(directory / "trajectory.npz") != metadata["trajectory_sha256"]:
        raise IntegrityError("trajectory bytes changed")
    with np.load(directory / "trajectory.npz", allow_pickle=False) as data:
        a = {key: data[key] for key in data.files}
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
    dt = config["timestep_fs"]
    if count * dt != metadata["physical_horizon_fs"]:
        raise IntegrityError("physical horizon mismatch")
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
