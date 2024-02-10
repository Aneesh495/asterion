import ast
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from asterion.artifacts.io import file_hash, read_json, safe_member
from asterion.artifacts.source import project_root
from asterion.contracts.errors import IntegrityError
from asterion.evaluation.replay import replay_trajectory


def require(condition, message):
    if not condition:
        raise IntegrityError(message)


def verify_full_gates(root: Path, report, config, observations, candidate=False):
    """Derive full gate populations from retained raw records, never from summary counts."""
    path = root / "GATE_OBSERVATIONS.json"
    require(path.is_file(), "full gate observation index is missing")
    index = read_json(path)
    from asterion.artifacts.receipts import verify_command_receipts, verify_stage_receipts

    names = (
        "tests",
        "data",
        "pilot",
        "train",
        "evaluate",
        "guarded-shift",
        "simulate",
        "acquire",
        "ablate",
        "reference",
        "secondary",
        "properties",
        "derivatives",
        "analytic",
        "stress",
        "bench",
        "demo",
    )
    finals = verify_stage_receipts(root, names, scientific=True)
    commands = tuple(name for name in names if name not in {"tests", "data"})
    verify_command_receipts(root, commands, finals)
    required = {
        "junit",
        "resource_pilot",
        "properties",
        "derivatives",
        "analytic",
        "simulation_recovery",
        "training_recovery",
        "faults",
        "memory",
        "benchmark",
        "benchmark_dynamics",
        "dynamics_study",
        "ablations",
        "acquisitions",
        "reference",
        "secondary",
        "controls",
        "guarded_shift",
    }
    require(required <= index.keys(), "full gate observation families are missing")
    require(
        config["dynamics_study"]["steps"] >= 1000
        and config["dynamics_study"]["refinement_steps"] >= 1000
        and len(config["dynamics_study"]["molecules"]) >= 2
        and config["dynamics_study"]["timesteps_fs"] == [0.1, 0.05, 0.025]
        and index["dynamics_study"] == read_json(root / "UNITS.json")["dynamics_study"],
        "separate thermostat or matched-horizon refinement study shortened",
    )
    counts = {}
    test_functions: list[tuple[str, str]] = []
    for source in (project_root() / "tests").rglob("*.py"):
        tree = ast.parse(source.read_text())
        test_functions.extend(
            (str(source), node.name)
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        )
    require(len(test_functions) >= 400, "authored semantic test population is below 400")
    tests = ET.parse(safe_member(root, index["junit"])).getroot()
    cases = tests.findall(".//testcase")
    require(
        len(cases) >= 400
        and not tests.findall(".//failure")
        and not tests.findall(".//error")
        and not tests.findall(".//skipped"),
        "full test invocation has failed or skipped cases",
    )
    counts["semantic_tests"] = len(test_functions)
    properties = read_json(safe_member(root, index["properties"]))
    require(
        properties["count"] == len(properties["cases"]) >= 10000,
        "generated graph case count mismatch",
    )
    require(
        len({case["seed"] for case in properties["cases"]}) == len(properties["cases"]),
        "generated cases repeat identities",
    )
    require(
        all(case["status"] == "passed" for case in properties["cases"]),
        "generated property failure hidden",
    )
    # Recompute a bounded independent subset with NumPy all-pairs arithmetic.
    for case in properties["cases"][:: max(1, len(properties["cases"]) // 100)]:
        rng = np.random.default_rng(case["seed"])
        atoms = int(rng.integers(1, 21))
        rng.choice([1, 6, 7, 8], atoms)
        xyz = rng.normal(size=(atoms, 3))
        cutoff = float(rng.uniform(0.1, 5.0))
        total = sum(
            i != j and np.linalg.norm(xyz[i] - xyz[j]) < cutoff
            for i in range(atoms)
            for j in range(atoms)
        )
        require(
            case["atoms"] == atoms and case["cutoff"] == cutoff and case["edges"] == total,
            "independent generated graph observation mismatch",
        )
    counts["property_cases"] = len(properties["cases"])
    derivatives = read_json(safe_member(root, index["derivatives"]))
    from asterion.artifacts.derivatives import verify_derivative_campaign
    from asterion.training.checkpoint import load_checkpoint

    models = {}
    checkpoint_paths = {}
    evaluation_boundaries = {}
    for entry in read_json(root / "UNITS.json")["training"]:
        directory = safe_member(root, entry["path"])
        selected = read_json(directory / "selection.json")
        model, checkpoint = load_checkpoint(safe_member(directory, selected["checkpoint"]))
        models[checkpoint["identity"]] = model
        checkpoint_paths[checkpoint["identity"]] = safe_member(directory, selected["checkpoint"])
        prefix = read_json(directory / "normalization.json")["fit_identities"][0].rsplit(":", 1)[0]
        evaluation_boundaries[checkpoint["identity"]] = [
            f"{prefix}:{row}" for row in read_json(directory / "split.json")["evaluation"][:8]
        ]
    derivative_data = {
        record["name"]: Path(record["path"])
        for record in read_json(root / "DATASETS.json")["files"]
    }
    from asterion.artifacts.resource_pilot import verify_resource_pilot

    counts["resource_pilot"] = verify_resource_pilot(
        root,
        index["resource_pilot"],
        config,
        read_json(root / "SOURCE_INPUTS.json"),
        derivative_data,
    )
    verify_derivative_campaign(
        root, index["derivatives"], read_json(root / "SOURCE_INPUTS.json"), models, derivative_data
    )
    require(len(derivatives["symmetry"]) >= 1200, "symmetry trial workload shortened")
    require(
        len(derivatives["force"]) >= 1000 and len(derivatives["hessian"]) >= 100,
        "force or Hessian difference workload shortened",
    )
    for trial in derivatives["symmetry"]:
        require(
            max(trial[key] for key in ("energy_error", "additivity", "batch_energy")) < 1e-7
            and max(
                trial[key]
                for key in (
                    "force_relative",
                    "net_force",
                    "torque",
                    "disconnected_force",
                    "batch_force",
                )
            )
            < 1e-7,
            "retained symmetry or physical-identity failure",
        )
    require(
        all(
            trial["relative"] < 1e-4
            for family in ("force", "hessian")
            for trial in derivatives[family]
        ),
        "retained derivative failure",
    )
    counts.update(
        symmetry_trials=len(derivatives["symmetry"]),
        force_checks=len(derivatives["force"]),
        hessian_checks=len(derivatives["hessian"]),
    )
    analytic = read_json(safe_member(root, index["analytic"]))
    require(
        analytic["fixtures"] == len(analytic["outcomes"]) >= 100,
        "analytic fixture population shortened",
    )
    for fixture in analytic["outcomes"]:
        require(
            fixture["status"] == "passed" and len(fixture["runs"]) == 3,
            "analytic fixture failed or lacks three timesteps",
        )
        horizons = []
        for run in fixture["runs"]:
            replay = replay_trajectory(safe_member(root, run["path"]))
            require(replay["completed_steps"] > 0, "empty analytic integration")
            horizons.append(replay["physical_horizon_fs"])
        require(
            max(horizons) - min(horizons) < 1e-12, "timestep refinement changed physical horizon"
        )
    counts["analytic_dynamics"] = len(analytic["outcomes"])
    recovery_count = 0
    for family in ("simulation_recovery", "training_recovery"):
        raw = read_json(safe_member(root, index[family]))
        require(
            raw["cases"] == len(raw["outcomes"]) > 0
            and all(x["status"] == "passed" for x in raw["outcomes"]),
            "missing or failed interruption population",
        )
        recovery_count += len(raw["outcomes"])
    require(recovery_count >= 50, "interruption workload shortened")
    counts["recovery_cases"] = recovery_count
    faults = read_json(safe_member(root, index["faults"]))
    require(
        faults["cases"] == len(faults["outcomes"]) >= 100
        and all(x["status"] == "rejected" and x["error"] for x in faults["outcomes"]),
        "failure/resource cases missing or accepted",
    )
    counts["fault_cases"] = len(faults["outcomes"])
    memory = read_json(safe_member(root, index["memory"]))
    require(
        memory["elapsed_seconds"] >= 1800
        and memory["completed_calls"] > 0
        and len(memory["samples"]) >= 300,
        "30-minute memory stress shortened or empty",
    )
    from asterion.artifacts.io import read_json as read_environment

    env = read_environment(root / "ENVIRONMENT.json")
    factor = 1024 if env["rss_native_units"] == "KiB" else 1
    rss = [sample["peak_rss_native"] * factor for sample in memory["samples"]]
    require(
        max(rss[60:]) - min(rss[60:]) <= max(min(rss[60:]) * 0.25, 128 * 1024**2),
        "inference memory is not bounded under declared tolerance",
    )
    counts["stress_seconds"] = memory["elapsed_seconds"]
    benchmark = read_json(safe_member(root, index["benchmark"]))
    from asterion.artifacts.runtime import verify_runtime

    require(
        benchmark["checkpoint"] in models
        and benchmark["geometry_identities"] == evaluation_boundaries[benchmark["checkpoint"]],
        "benchmark does not use a selected trained model and its retained evaluation geometries",
    )
    require(
        safe_member(root, index["benchmark_dynamics"])
        == safe_member(root, index["benchmark"]).parent / "benchmark-dynamics",
        "benchmark dynamics observation index changed",
    )
    import torch

    require(
        benchmark["dtype"] == f"torch.{config['training']['dtype']}"
        and benchmark["device"]
        == ("cuda:0" if config["training"]["device"] == "cuda" else config["training"]["device"]),
        "benchmark does not measure the declared research backend and precision",
    )
    timed_model, _ = load_checkpoint(
        checkpoint_paths[benchmark["checkpoint"]], dtype=getattr(torch, config["training"]["dtype"])
    )
    counts.update(
        runtime=verify_runtime(
            root,
            index["benchmark"],
            timed_model,
            read_json(root / "SOURCE_INPUTS.json"),
            derivative_data,
            full=True,
        )
    )
    if candidate:
        require(not (root / index["controls"]).exists(), "candidate already contains controls")
        counts["negative_controls"] = 0
    else:
        controls = read_json(safe_member(root, index["controls"]))
        require(
            controls["profile"] == "full"
            and controls.get("phase") == "candidate"
            and len(controls["controls"]) >= 12
            and controls["positive_before"]["exit"] == controls["positive_after"]["exit"] == 0,
            "full verifier controls absent or positive verification failed",
        )
        require(
            all(control["exit"] != 0 and control["stderr"] for control in controls["controls"])
            and sum(control["semantic_rehashed"] for control in controls["controls"]) >= 4,
            "negative controls lack genuine semantic rejections",
        )
        require(controls.get("source_bundle_unchanged"), "candidate changed during controls")
        require(
            controls.get("candidate_manifest_sha256") == report.get("candidate_manifest_sha256"),
            "controls tested another candidate",
        )
        counts["negative_controls"] = len(controls["controls"])
    acquisitions = read_json(safe_member(root, index["acquisitions"]))
    from asterion.artifacts.studies import verify_acquisition

    dataset_paths = {
        record["name"]: Path(record["path"])
        for record in read_json(root / "DATASETS.json")["files"]
    }
    verify_acquisition(
        root, index["acquisitions"], config, read_json(root / "SOURCE_INPUTS.json"), dataset_paths
    )
    expected = {
        (m, s, p, b)
        for m in config["acquisition_molecules"]
        for s in config["seeds"]
        for p in ("uniform", "diversity", "uncertainty-diversity")
        for b in (100, 300, 1000)
    }
    require(
        len(config["acquisition_molecules"]) >= 2
        and {(x["molecule"], x["trial"], x["policy"], x["budget"]) for x in acquisitions}
        == expected
        and all(x["completed_labels"] == x["budget"] for x in acquisitions),
        "acquisition policies, seeds, molecules or budgets are missing",
    )
    from asterion.artifacts.ablations import verify_ablations
    from asterion.artifacts.shift import verify_guarded_study

    require(
        len(config["molecules"][:2]) == 2, "architectural ablations require a two-molecule core"
    )
    verify_ablations(
        root, index["ablations"], config, read_json(root / "SOURCE_INPUTS.json"), dataset_paths
    )
    verify_guarded_study(
        root, index["guarded_shift"], config, read_json(root / "SOURCE_INPUTS.json"), dataset_paths
    )
    reference = read_json(safe_member(root, index["reference"]))
    require(
        len(reference) >= 12
        and all(
            x["exit"] == 0
            and x["origin"] == "independent-reference-retraining"
            and x["version"] == "0.3.14"
            for x in reference
        ),
        "matched reference retraining incomplete",
    )
    expected_reference = {(m, seed) for m in config["molecules"] for seed in config["seeds"]}
    require(
        {(row["molecule"], row["seed"]) for row in reference} == expected_reference
        and len(reference) == len(expected_reference),
        "reference hides a seed or repeats a training identity",
    )
    reference_base = safe_member(root, index["reference"]).parent
    dataset_records = read_json(root / "DATASETS.json")["files"]
    datasets = {record["name"]: record for record in dataset_records}
    conversion = 4184 / (6.02214076e23 * 1.602176634e-19)
    for row in reference:
        directory = safe_member(root, str((reference_base / row["path"]).relative_to(root)))
        from asterion.artifacts.reference import reference_history
        from asterion.training.reference_env import locked_versions

        history_path = safe_member(directory, row["history_path"])
        require(
            file_hash(history_path) == row["history_sha256"], "reference training journal changed"
        )
        require(
            reference_history(history_path, 400, 1000, 8) == row["history"],
            "reference optimizer work counter changed",
        )
        comparison = read_json(directory / "ENVIRONMENT.json")
        lock = safe_member(project_root(), comparison["lock"])
        require(
            comparison["cuda_available"]
            and comparison["python"][:2] == [3, 12]
            and file_hash(lock) == comparison["lock_sha256"]
            and locked_versions(lock) == comparison["versions"],
            "reference environment differs from the frozen comparison stack",
        )
        require(
            row.get("budget") == 1000 and row.get("evaluation_exit") == 0,
            "reference labels or actual prediction execution missing",
        )
        require(
            row.get("selection") == "standard-final-export", "reference selection protocol changed"
        )
        model_path = safe_member(directory, row["model_path"])
        require(file_hash(model_path) == row["model_sha256"], "reference model bytes changed")
        prediction = directory / "evaluation-predictions.npz"
        require(
            file_hash(prediction) == row["predictions_sha256"], "reference prediction bytes changed"
        )
        for role in ("train", "validation", "evaluation"):
            require(
                file_hash(directory / f"{role}.xyz") == row["inputs"][role],
                "reference exported input changed",
            )
        split = read_json(directory / "split.json")
        require(split["digest"] == row["split"], "reference split lineage changed")
        require(
            len(row["training_rows"]) == len(set(row["training_rows"])) == 1000
            and set(row["training_rows"]) == set(split["train_pool"]),
            "reference trained on unmatched or repeated labels",
        )
        with np.load(prediction, allow_pickle=False) as raw:
            force = raw["force"]
            target = raw["target_force"]
            energy = raw["energy"]
            target_energy = raw["target_energy"]
            pointers = raw["pointers"]
            positions = raw["positions"]
            numbers = raw["numbers"]
            identities = raw["identities"]
        require(
            len(energy) >= 5000
            and pointers.shape == (len(energy) + 1,)
            and pointers[0] == 0
            and pointers[-1] == len(force)
            and np.all(np.diff(pointers) > 0),
            "reference denominators or held-out count changed",
        )
        require(
            energy.shape == target_energy.shape == (len(energy),)
            and identities.shape == (len(energy),)
            and force.ndim == 2
            and force.shape[1] == 3
            and numbers.shape == (len(force),)
            and force.shape == target.shape == positions.shape
            and all(
                np.isfinite(x).all() for x in (force, target, energy, target_energy, positions)
            ),
            "reference output is empty, malformed or nonfinite",
        )
        rows = [int(identity.rsplit(":", 1)[1]) for identity in identities]
        require(
            rows == split["evaluation"] and not set(rows) & set(row["training_rows"]),
            "reference evaluation identities changed or leak labels",
        )
        record = datasets[f"rmd17_{row['molecule']}.npz"]
        require(record["sha256"] == row["dataset"], "reference dataset lineage changed")
        with np.load(record["path"], allow_pickle=False) as author:
            from asterion.data.extxyz import reference_arrays

            for role, selected in (
                ("train", row["training_rows"]),
                ("validation", split["validation"]),
                ("evaluation", split["evaluation"]),
            ):
                raw_input = reference_arrays(directory / f"{role}.xyz", predictions=False)
                require(
                    raw_input["identities"].tolist()
                    == [f"{row['molecule']}:{row['dataset']}:{index}" for index in selected],
                    "reference exported role identities changed",
                )
                require(
                    np.allclose(
                        raw_input["target_force"],
                        author["forces"][selected].reshape(-1, 3) * conversion,
                        atol=1e-12,
                        rtol=0,
                    )
                    and np.allclose(
                        raw_input["target_energy"],
                        author["energies"][selected] * conversion,
                        atol=1e-10,
                        rtol=0,
                    )
                    and np.array_equal(
                        raw_input["positions"], author["coords"][selected].reshape(-1, 3)
                    )
                    and np.array_equal(
                        raw_input["numbers"], np.tile(author["nuclear_charges"], len(selected))
                    ),
                    "reference was trained or selected using different physical input labels",
                )
            require(
                np.allclose(
                    target, author["forces"][rows].reshape(-1, 3) * conversion, atol=1e-8, rtol=0
                ),
                "reference target force units or rows changed",
            )
            require(
                np.allclose(
                    target_energy, author["energies"][rows] * conversion, atol=1e-8, rtol=0
                ),
                "reference target energy units or rows changed",
            )
            require(
                np.allclose(positions, author["coords"][rows].reshape(-1, 3), atol=1e-8, rtol=0)
                and np.array_equal(numbers, np.tile(author["nuclear_charges"], len(rows))),
                "reference geometry or atom ordering changed",
            )
        errors = [
            np.abs(force[a:b] - target[a:b]).sum() / ((b - a) * 3)
            for a, b in zip(pointers[:-1], pointers[1:])
        ]
        require(
            np.isclose(
                np.mean(errors), row["metrics"]["force_mae_eV_angstrom"], atol=1e-12, rtol=1e-10
            )
            and row["metrics"]["geometries"] == len(energy),
            "false reference force metric",
        )
        require(
            np.isclose(
                np.abs(energy - target_energy).mean(),
                row["metrics"]["energy_mae_eV"],
                atol=1e-12,
                rtol=1e-10,
            ),
            "false reference energy metric",
        )
    secondary = read_json(safe_member(root, index["secondary"]))
    from asterion.artifacts.studies import verify_secondary

    verify_secondary(root, index["secondary"], config, read_json(root / "SOURCE_INPUTS.json"))
    require(
        secondary["training"]["status"] == "passed"
        and len(secondary["trajectories"]) >= 10
        and secondary["metrics"]["trained"]["geometries"] >= 1000
        and all(x["record"]["requested_steps"] >= 5000 for x in secondary["trajectories"]),
        "extended-system held-out evaluation or actual trajectory population missing",
    )
    learning = {}
    for molecule in config["molecules"]:
        molecular_observations = [
            x
            for x in observations
            if x["molecule"] == molecule and x["model"] == "equivariant" and x["budget"] == 1000
        ]
        learning[molecule] = bool(molecular_observations) and all(
            x["force_mae"] <= 0.7 * x["zero_force_mae"]
            and x["force_mae"] < x["untrained_force_mae"]
            and x["positive_learning_interval"]
            for x in molecular_observations
        )
    require(sum(learning.values()) >= 3, "demonstrable molecular force-learning gate failed")
    for key, value in counts.items():
        require(
            report["counts"].get(key) == value,
            f"summary counter disagrees with retained observations: {key}",
        )
    return counts
