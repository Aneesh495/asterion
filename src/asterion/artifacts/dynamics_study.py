"""Derive finite-path structural reports and bind every learned study to retained inputs."""

import numpy as np

from asterion.artifacts.io import file_hash, read_json, safe_member
from asterion.artifacts.studies import model_trajectory
from asterion.contracts.errors import IntegrityError
from asterion.training.identity import state_fingerprint


def require(condition, message):
    if not condition:
        raise IntegrityError(message)


def agree(actual, expected, message):
    require(np.allclose(actual, expected, atol=1e-12, rtol=1e-9), message)


def verify_statistics(directory, saved, replay):
    metadata = read_json(directory / "trajectory.json")
    require(
        saved["format"] == "asterion-finite-dynamics-statistics-v1"
        and saved["status"] == metadata["status"]
        and saved["reason"] == metadata["reason"]
        and saved["ensemble"] == replay["ensemble"]
        and saved["completed_steps"] == replay["completed_steps"]
        and saved["physical_horizon_fs"] == replay["physical_horizon_fs"]
        and saved["equilibrium_claim"] is False
        and saved["independent_frames_claim"] is False,
        "structural report changes physical scope or termination",
    )
    if not replay["completed_steps"]:
        require(
            saved["available"] is False
            and saved["reason_unavailable"] == "no-completed-physical-frames",
            "empty physical path claims structural statistics",
        )
        return
    require(
        saved["available"] is True
        and saved["initial_frame_excluded"] is True
        and saved["source_trajectory_sha256"] == file_hash(directory / "trajectory.npz"),
        "structural report source or sampling convention changed",
    )
    with np.load(directory / "trajectory.npz", allow_pickle=False) as archive:
        raw = {key: archive[key] for key in archive.files}
    t = raw["temperature"][1:]
    require(
        saved["sample_frames"] == len(t), "structural statistics include missing or repeated frames"
    )
    agree(saved["temperature_mean_K"], np.sum(t) / len(t), "false time-averaged temperature")
    if len(t) > 1:
        agree(
            saved["temperature_sample_std_K"],
            np.sqrt(np.sum((t - np.sum(t) / len(t)) ** 2) / (len(t) - 1)),
            "false temperature variation",
        )
    else:
        require(
            saved["temperature_sample_std_K"] is None, "single frame invents temperature variation"
        )
    correlation = saved["temperature_correlation"]
    centered = t - np.sum(t) / len(t)
    variance = np.dot(centered, centered) / len(t)
    if variance == 0 or len(t) < 4:
        require(
            correlation["available"] is False and correlation["integrated_time_steps"] is None,
            "constant or short path invents temperature correlation",
        )
    else:
        lags = list(range(min(200, len(t) // 2) + 1))
        values = [
            float(np.dot(centered[: len(t) - lag], centered[lag:]) / ((len(t) - lag) * variance))
            for lag in lags
        ]
        require(
            correlation["available"] is True and correlation["lags"] == lags,
            "temperature correlation lag population changed",
        )
        agree(correlation["correlation"], values, "temperature time-correlation trace changed")
        positive = 0.0
        for value in values[1:]:
            if value <= 0:
                break
            positive += value
        agree(
            correlation["integrated_time_steps"],
            max(1.0, 1 + 2 * positive),
            "false integrated correlation time",
        )
    mass = np.array([{1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999}[int(z)] for z in raw["numbers"]])
    center = np.einsum("tac,a->tc", raw["positions"], mass) / mass.sum()
    center_velocity = np.einsum("tac,a->tc", raw["velocities"], mass) / mass.sum()
    momentum = (
        (raw["velocities"] - center_velocity[:, None, :]) * mass[None, :, None] * 1.66053906892e-22
    )
    angular = np.cross((raw["positions"] - center[:, None, :]) * 1e-10, momentum).sum(1)
    agree(
        saved["maximum_center_of_mass_displacement_angstrom"],
        np.sqrt(((center - center[0]) ** 2).sum(1)).max(),
        "false center-of-mass displacement",
    )
    require(saved["angular_momentum_unit"] == "kg m^2/s", "angular momentum unit changed")
    require(
        np.isclose(
            saved["maximum_internal_angular_momentum_change_SI"],
            np.sqrt(((angular - angular[0]) ** 2).sum(1)).max(),
            atol=1e-50,
            rtol=1e-8,
        ),
        "false angular momentum change",
    )
    agree(
        saved["maximum_heat_adjusted_energy_drift_eV"],
        replay["max_energy_drift_eV"],
        "thermostat heat was omitted from drift",
    )
    radius = metadata["detector"]["policy"]["bond_reference_radius"]
    require(saved["bond_reference_radius_angstrom"] == radius, "bond reference radius changed")
    groups: dict[str, list[tuple[int, int]]] = {}
    initial = raw["positions"][0]
    for first in range(len(mass)):
        for second in range(first + 1, len(mass)):
            if np.linalg.norm(initial[first] - initial[second]) < radius:
                key = "-".join(
                    map(str, sorted((int(raw["numbers"][first]), int(raw["numbers"][second]))))
                )
                groups.setdefault(key, []).append((first, second))
    bonds = saved["bond_distances"]
    edges = np.arange(121, dtype=float) / 20
    agree(bonds["bin_edges_angstrom"], edges, "bond distribution bins changed")
    require(set(bonds["species_pairs"]) == set(groups), "bond species population changed")
    for key, pairs in groups.items():
        distances = np.concatenate(
            [
                np.linalg.norm(raw["positions"][1:, a] - raw["positions"][1:, b], axis=1)
                for a, b in pairs
            ]
        )
        row = bonds["species_pairs"][key]
        require(
            row["pairs"] == len(pairs)
            and row["sample_count"] == len(distances)
            and np.array_equal(row["histogram"], np.histogram(distances, edges)[0])
            and row["underflow"] == int((distances < 0).sum())
            and row["overflow"] == int((distances > 6).sum()),
            "bond-distance distribution omits physical samples",
        )
        agree(row["mean_angstrom"], distances.mean(), "false bond-distance mean")
        if len(distances) > 1:
            agree(
                row["sample_std_angstrom"], distances.std(ddof=1), "false bond-distance variation"
            )
        else:
            require(row["sample_std_angstrom"] is None, "single bond sample invents variation")


def verify_dynamics_study(root, path, config, source, models, owners, datasets):
    report_path = safe_member(root, path)
    base = report_path.parent
    report = read_json(report_path)
    protocol = config["dynamics_study"]
    require(
        report["format"] == "asterion-learned-dynamics-study-v1"
        and report["source"]["digest"] == source["digest"]
        and report["protocol"] == protocol
        and report["force_dtype"] == "float64"
        and report["force_device"] == "cpu"
        and report["execution_status"] == "completed",
        "learned dynamics study source, execution or frozen protocol changed",
    )
    expected = {
        (m, s, ensemble, dt)
        for m in protocol["molecules"]
        for s in config["seeds"]
        for ensemble, dt in [
            ("NVT", protocol["timesteps_fs"][0]),
            *(("NVE", dt) for dt in protocol["timesteps_fs"]),
        ]
    }
    require(
        len(report["cases"]) == len(expected)
        and {(r["molecule"], r["seed"], r["ensemble"], r["timestep_fs"]) for r in report["cases"]}
        == expected,
        "separate NVT or refinement population is missing or repeated",
    )
    refined: dict = {}
    total = 0
    for row in report["cases"]:
        directory = safe_member(base, row["path"])
        require(
            row["checkpoint"] in models
            and row["execution_state"] == state_fingerprint(models[row["checkpoint"]]),
            "learned dynamics study substituted saved weights",
        )
        model = models[row["checkpoint"]]
        owner, unit = owners[row["checkpoint"]]
        require(
            unit["molecule"] == row["molecule"]
            and unit["seed"] == row["seed"]
            and unit["budget"] == max(config["budgets"])
            and unit["model"] == "equivariant",
            "dynamics study used another selected model population",
        )
        split = read_json(owner / "split.json")
        identity = f"{row['molecule']}:{split['dataset_hash']}:{split['evaluation'][0]}"
        require(row["initial_geometry"] == identity, "dynamics study initial label role changed")
        metadata, replay = model_trajectory(directory, model, row["checkpoint"], source)
        dt_index = protocol["timesteps_fs"].index(row["timestep_fs"])
        steps = (
            protocol["steps"]
            if row["ensemble"] == "NVT"
            else protocol["refinement_steps"] * 2**dt_index
        )
        require(
            metadata == row["record"]
            and metadata["config"]["steps"] == steps
            and metadata["config"]["timestep_fs"] == row["timestep_fs"]
            and metadata["config"]["temperature_K"] == protocol["temperature_K"]
            and metadata["config"]["ensemble"] == row["ensemble"]
            and metadata["config"]["seed"] == row["seed"],
            "learned dynamics study changes workload or hides failure",
        )
        if not replay.get("rejected_initial"):
            with np.load(directory / "trajectory.npz", allow_pickle=False) as raw:
                initial_position, initial_velocity, numbers = (
                    raw["positions"][0],
                    raw["velocities"][0],
                    raw["numbers"],
                )
        else:
            with np.load(directory / "failure.npz", allow_pickle=False) as raw:
                initial_position, initial_velocity, numbers = (
                    raw["positions"],
                    raw["velocities"],
                    raw["numbers"],
                )
        with np.load(datasets[f"rmd17_{row['molecule']}.npz"], allow_pickle=False) as author:
            require(
                np.array_equal(initial_position, author["coords"][split["evaluation"][0]])
                and np.array_equal(numbers, author["nuclear_charges"]),
                "study initial author geometry changed",
            )
        mass = np.array([{1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999}[int(z)] for z in numbers])
        velocity = np.random.default_rng(row["seed"] * 1000).normal(size=initial_velocity.shape)
        velocity *= (
            np.sqrt(1.380649e-23 * protocol["temperature_K"] / (mass * 1.66053906892e-27))[:, None]
            / 1e5
        )
        velocity -= np.sum(velocity * mass[:, None], axis=0) / mass.sum()
        observed_temperature = np.sum((velocity * 1e5) ** 2 * mass[:, None] * 1.66053906892e-27) / (
            (3 * len(mass) - 3) * 1.380649e-23
        )
        velocity *= np.sqrt(protocol["temperature_K"] / observed_temperature)
        agree(initial_velocity, velocity, "study initial velocity or temperature changed")
        require(
            row["started_ns"] < row["ended_ns"]
            and np.isclose(
                row["seconds"], (row["ended_ns"] - row["started_ns"]) * 1e-9, atol=0, rtol=1e-12
            ),
            "dynamics study timing changed",
        )
        require(
            row["statistics"] == read_json(directory / "statistics.json"),
            "study statistics binding changed",
        )
        verify_statistics(directory, row["statistics"], replay)
        total += replay["completed_steps"]
        if row["ensemble"] == "NVE":
            refined.setdefault((row["molecule"], row["seed"]), {})[row["timestep_fs"]] = (
                directory,
                metadata,
            )
    require(
        len(report["refinements"]) == len(refined)
        and {(r["molecule"], r["seed"]) for r in report["refinements"]} == set(refined),
        "matched-horizon refinement summaries are missing or repeated",
    )
    for row in report["refinements"]:
        records = [refined[(row["molecule"], row["seed"])][dt] for dt in protocol["timesteps_fs"]]
        saved = row["observation"]
        statuses = [metadata["status"] for _, metadata in records]
        require(
            saved["timesteps_fs"] == protocol["timesteps_fs"], "refinement timestep order changed"
        )
        if any(status != "passed" for status in statuses):
            require(
                saved["available"] is False and saved["statuses"] == statuses,
                "failed refinement was hidden",
            )
            continue
        require(saved["available"] is True, "completed refinement was omitted")
        horizons = [metadata["physical_horizon_fs"] for _, metadata in records]
        agree(horizons, [horizons[0]] * 3, "refinement changed completed physical horizon")
        agree(saved["physical_horizon_fs"], horizons[0], "refinement summary changed horizon")
        positions = []
        for directory, _ in records:
            with np.load(directory / "trajectory.npz", allow_pickle=False) as raw:
                positions.append(raw["positions"].copy())
        coarse = np.sqrt(np.mean((positions[0] - positions[2][::4]) ** 2))
        middle = np.sqrt(np.mean((positions[1] - positions[2][::2]) ** 2))
        agree(
            saved["coarse_to_finest_coordinate_rms_angstrom"],
            coarse,
            "false coarse refinement difference",
        )
        agree(
            saved["middle_to_finest_coordinate_rms_angstrom"],
            middle,
            "false middle refinement difference",
        )
        if middle > 0:
            agree(
                saved["observed_error_ratio"], coarse / middle, "false observed convergence ratio"
            )
        else:
            require(
                saved["observed_error_ratio"] is None,
                "zero refinement difference invents convergence",
            )
        require(
            saved["convergence_order_claim"] is False, "finite study invents asymptotic convergence"
        )
    return {"cases": len(report["cases"]), "completed_steps": total}
