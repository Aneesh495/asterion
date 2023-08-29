import ast
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from asterion.artifacts.io import read_json, safe_member
from asterion.artifacts.source import project_root
from asterion.contracts.errors import IntegrityError
from asterion.evaluation.replay import replay_trajectory


def require(condition, message):
    if not condition:
        raise IntegrityError(message)


def verify_full_gates(root: Path, report, config, observations):
    """Derive full gate populations from retained raw records, never from summary counts."""
    path = root / "GATE_OBSERVATIONS.json"
    require(path.is_file(), "full gate observation index is missing")
    index = read_json(path)
    required = {
        "junit",
        "properties",
        "derivatives",
        "analytic",
        "simulation_recovery",
        "training_recovery",
        "faults",
        "memory",
        "benchmark",
        "ablations",
        "acquisitions",
        "reference",
        "secondary",
        "controls",
    }
    require(required <= index.keys(), "full gate observation families are missing")
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
    rss = [sample["peak_rss_native"] for sample in memory["samples"]]
    require(
        max(rss[60:]) - min(rss[60:]) <= max(min(rss[60:]) * 0.25, 128 * 1024**2),
        "inference memory is not bounded under declared tolerance",
    )
    counts["stress_seconds"] = memory["elapsed_seconds"]
    benchmark = read_json(safe_member(root, index["benchmark"]))
    require(
        benchmark["repetitions"] >= 30 and benchmark["long_repetitions"] >= 10,
        "benchmark repetitions shortened",
    )
    require(
        all(sample["seconds"] > 0 for sample in benchmark["samples"]), "invalid raw latency sample"
    )
    controls = read_json(safe_member(root, index["controls"]))
    require(
        controls["profile"] == "full"
        and len(controls["controls"]) >= 12
        and controls["positive_before"]["exit"] == controls["positive_after"]["exit"] == 0,
        "full verifier controls absent or positive verification failed",
    )
    require(
        all(control["exit"] != 0 and control["stderr"] for control in controls["controls"])
        and sum(control["semantic_rehashed"] for control in controls["controls"]) >= 4,
        "negative controls lack genuine semantic rejections",
    )
    counts["negative_controls"] = len(controls["controls"])
    acquisitions = read_json(safe_member(root, index["acquisitions"]))
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
    ablations = read_json(safe_member(root, index["ablations"]))
    require(
        {x["variant"] for x in ablations}
        >= {"lmax0", "lmax1", "no-higher-order", "hard-cutoff", "energy-only"}
        and len({x["molecule"] for x in ablations}) >= 2
        and all(
            len({x["seed"] for x in ablations if x["variant"] == variant}) >= 3
            for variant in {x["variant"] for x in ablations}
        ),
        "required ablation population incomplete",
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
    secondary = read_json(safe_member(root, index["secondary"]))
    require(
        secondary["training"]["status"] == "passed"
        and len(secondary["trajectories"]) >= 10
        and secondary["metrics"]["trained"]["geometries"] >= 1000
        and all(x["record"]["requested_steps"] >= 5000 for x in secondary["trajectories"]),
        "extended-system held-out evaluation or actual trajectory population missing",
    )
    learning = {}
    for molecule in config["molecules"]:
        rows = [
            x
            for x in observations
            if x["molecule"] == molecule and x["model"] == "equivariant" and x["budget"] == 1000
        ]
        learning[molecule] = bool(rows) and all(
            x["force_mae"] <= 0.7 * x["zero_force_mae"]
            and x["force_mae"] < x["untrained_force_mae"]
            for x in rows
        )
    require(sum(learning.values()) >= 3, "demonstrable molecular force-learning gate failed")
    for key, value in counts.items():
        require(
            report["counts"].get(key) == value,
            f"summary counter disagrees with retained observations: {key}",
        )
    return counts
