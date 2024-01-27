"""Observe finite paths without claiming equilibrium or independent frames."""

import numpy as np

from asterion.artifacts.io import file_hash, read_json, write_json
from asterion.contracts.errors import ContractError
from asterion.evaluation.replay import replay_trajectory


def temperature_correlation(values, maximum_lag=200):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ContractError("temperature correlation requires a finite nonempty time series")
    centered = values - values.mean()
    variance = float(np.mean(centered**2))
    if variance == 0 or len(values) < 4:
        return {
            "available": False,
            "reason": "constant-or-short-series",
            "lags": [],
            "correlation": [],
            "integrated_time_steps": None,
        }
    lags = list(range(min(maximum_lag, len(values) // 2) + 1))
    correlation = [
        float(np.mean(centered[: len(values) - lag] * centered[lag:]) / variance) for lag in lags
    ]
    positive = []
    for value in correlation[1:]:
        if value <= 0:
            break
        positive.append(value)
    return {
        "available": True,
        "reason": None,
        "lags": lags,
        "correlation": correlation,
        "integrated_time_steps": max(1.0, 1 + 2 * sum(positive)),
    }


def bond_distribution(positions, numbers, bonds):
    positions, numbers = np.asarray(positions), np.asarray(numbers)
    if positions.ndim != 3 or positions.shape[1:] != (len(numbers), 3) or not len(positions):
        raise ContractError("bond distribution requires completed Cartesian frames")
    if not np.isfinite(positions).all():
        raise ContractError("nonfinite bond-distribution coordinates")
    bins = np.arange(121, dtype=float) / 20
    categories: dict[str, list[tuple[int, int]]] = {}
    for first, second, _ in bonds:
        if not 0 <= first < second < len(numbers):
            raise ContractError("bond reference has invalid atom axes")
        key = "-".join(map(str, sorted((int(numbers[first]), int(numbers[second])))))
        categories.setdefault(key, []).append((first, second))
    result = {}
    for key, pairs in sorted(categories.items()):
        indices = np.asarray(pairs, dtype=int)
        histogram = np.zeros(len(bins) - 1, dtype=np.int64)
        count, mean, m2, underflow, overflow = 0, 0.0, 0.0, 0, 0
        chunk = max(1, min(128, 262144 // len(pairs)))
        for begin in range(0, len(positions), chunk):
            xyz = positions[begin : begin + chunk]
            distance = np.linalg.norm(
                xyz[:, indices[:, 0]] - xyz[:, indices[:, 1]], axis=2
            ).reshape(-1)
            local_count, local_mean = len(distance), float(distance.mean())
            local_m2 = float(np.sum((distance - local_mean) ** 2))
            delta = local_mean - mean
            combined = count + local_count
            m2 += local_m2 + delta * delta * count * local_count / combined
            mean += delta * local_count / combined
            count = combined
            histogram += np.histogram(distance, bins)[0]
            underflow += int((distance < bins[0]).sum())
            overflow += int((distance > bins[-1]).sum())
        result[key] = {
            "pairs": len(pairs),
            "sample_count": count,
            "histogram": histogram.tolist(),
            "underflow": underflow,
            "overflow": overflow,
            "mean_angstrom": mean,
            "sample_std_angstrom": float(np.sqrt(m2 / (count - 1))) if count > 1 else None,
        }
    return {"bin_edges_angstrom": bins.tolist(), "species_pairs": result}


def analyze_trajectory(directory):
    replay = replay_trajectory(directory)
    metadata = read_json(directory / "trajectory.json")
    result = {
        "format": "asterion-finite-dynamics-statistics-v1",
        "status": metadata["status"],
        "reason": metadata["reason"],
        "completed_steps": replay["completed_steps"],
        "physical_horizon_fs": replay["physical_horizon_fs"],
        "ensemble": replay["ensemble"],
        "equilibrium_claim": False,
        "independent_frames_claim": False,
    }
    if replay["completed_steps"] == 0:
        result.update(available=False, reason_unavailable="no-completed-physical-frames")
        write_json(directory / "statistics.json", result)
        return result
    with np.load(directory / "trajectory.npz", allow_pickle=False) as raw:
        arrays = {key: raw[key] for key in raw.files}
    mass = np.asarray(
        [{1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999}[int(z)] for z in arrays["numbers"]]
    )
    center = np.sum(arrays["positions"] * mass[None, :, None], axis=1) / mass.sum()
    centered = arrays["positions"] - center[:, None, :]
    velocity_center = np.sum(arrays["velocities"] * mass[None, :, None], axis=1) / mass.sum()
    internal_velocity = arrays["velocities"] - velocity_center[:, None, :]
    angular_terms = (
        np.cross(centered * 1e-10, internal_velocity * 1e5)
        * mass[None, :, None]
        * 1.66053906892e-27
    )
    angular = angular_terms.sum(axis=1)
    angular_change = float(np.linalg.norm(angular - angular[0], axis=1).max())
    angular_roundoff = float(
        64
        * len(mass)
        * np.finfo(np.float64).eps
        * np.linalg.norm(angular_terms, axis=2).sum(axis=1).max()
    )
    temperature = arrays["temperature"][1:]
    result.update(
        available=True,
        source_trajectory_sha256=file_hash(directory / "trajectory.npz"),
        sample_frames=len(temperature),
        initial_frame_excluded=True,
        temperature_mean_K=float(temperature.mean()),
        temperature_sample_std_K=float(temperature.std(ddof=1)) if len(temperature) > 1 else None,
        temperature_correlation=temperature_correlation(temperature),
        maximum_center_of_mass_displacement_angstrom=float(
            np.linalg.norm(center - center[0], axis=1).max()
        ),
        maximum_internal_angular_momentum_change_SI=angular_change,
        angular_momentum_roundoff_bound_SI=angular_roundoff,
        angular_change_above_roundoff=angular_change > angular_roundoff,
        angular_momentum_unit="kg m^2/s",
        maximum_heat_adjusted_energy_drift_eV=replay["max_energy_drift_eV"],
        bond_reference_radius_angstrom=metadata["detector"]["policy"]["bond_reference_radius"],
        bond_distances=bond_distribution(
            arrays["positions"][1:], arrays["numbers"], metadata["detector"]["initial_bonds"]
        ),
    )
    write_json(directory / "statistics.json", result)
    return result


def matched_timestep_observations(directories):
    if len(directories) != 3:
        raise ContractError("timestep study requires three declared refinements")
    records = [read_json(path / "trajectory.json") for path in directories]
    for path in directories:
        replay_trajectory(path)
    dt = [record["config"]["timestep_fs"] for record in records]
    if not np.allclose(np.asarray(dt[:-1]) / np.asarray(dt[1:]), 2, rtol=0, atol=1e-12) or any(
        record["config"]["ensemble"] != "NVE" for record in records
    ):
        raise ContractError("timestep study requires successively halved NVE timesteps")
    if any(record["status"] != "passed" for record in records):
        return {
            "available": False,
            "reason": "retained-failed-refinement",
            "statuses": [record["status"] for record in records],
            "timesteps_fs": dt,
        }
    horizons = [record["physical_horizon_fs"] for record in records]
    if not np.allclose(horizons, horizons[0], atol=1e-12, rtol=0):
        raise ContractError("timestep refinements changed completed physical horizon")
    arrays = []
    for path in directories:
        with np.load(path / "trajectory.npz", allow_pickle=False) as archive:
            arrays.append({key: archive[key] for key in ("positions", "velocities", "numbers")})
    if len({record["lineage"] for record in records}) != 1 or any(
        not np.array_equal(
            arrays[0][key][0] if key != "numbers" else arrays[0][key],
            a[key][0] if key != "numbers" else a[key],
        )
        for a in arrays[1:]
        for key in ("positions", "velocities", "numbers")
    ):
        raise ContractError("timestep refinements changed model or initial physical state")
    finest = arrays[-1]["positions"]
    coarse = float(np.sqrt(np.mean((arrays[0]["positions"] - finest[::4]) ** 2)))
    middle = float(np.sqrt(np.mean((arrays[1]["positions"] - finest[::2]) ** 2)))
    return {
        "available": True,
        "timesteps_fs": dt,
        "physical_horizon_fs": horizons[0],
        "coarse_to_finest_coordinate_rms_angstrom": coarse,
        "middle_to_finest_coordinate_rms_angstrom": middle,
        "observed_error_ratio": coarse / middle if middle > 0 else None,
        "convergence_order_claim": False,
    }
