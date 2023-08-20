import math
import subprocess
from pathlib import Path

import numpy as np

from asterion.artifacts.io import file_hash, object_hash, read_json, safe_member
from asterion.artifacts.source import project_root, source_inputs
from asterion.contracts.errors import IntegrityError
from asterion.contracts.geometry import Geometry, collate
from asterion.evaluation.replay import replay_trajectory
from asterion.training.checkpoint import load_checkpoint

FULL_REQUIREMENTS = {
    "semantic_tests": 400,
    "property_cases": 10000,
    "symmetry_trials": 1200,
    "force_checks": 1000,
    "hessian_checks": 100,
    "analytic_dynamics": 100,
    "learned_steps": 1000000,
    "recovery_cases": 50,
    "fault_cases": 100,
    "stress_seconds": 1800,
    "negative_controls": 12,
}


def require(condition, message):
    if not condition:
        raise IntegrityError(message)


def check_manifest(root: Path):
    manifest = read_json(root / "MANIFEST.json")
    require(manifest.get("format") == "asterion-evidence-v1", "unknown evidence format")
    require(bool(manifest["files"]), "empty artifact manifest")
    actual_files = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "MANIFEST.json"
    }
    require(actual_files == set(manifest["files"]), "manifest omissions or extra artifact files")
    for relative, expected in manifest["files"].items():
        path = safe_member(root, relative)
        require(
            path.is_file() and file_hash(path) == expected,
            f"artifact checksum mismatch: {relative}",
        )
    return manifest


def check_source(root: Path):
    source = read_json(root / "SOURCE_INPUTS.json")
    require(object_hash(source["hashes"]) == source["digest"], "source digest mismatch")
    require(not source["dirty"], "evidence generated from uncommitted source")
    current = source_inputs()
    require(
        current["hashes"] == source["hashes"], "executable inputs changed since evidence generation"
    )
    repository = project_root()
    for path, expected in source["hashes"].items():
        result = subprocess.run(
            ["git", "show", f"{source['commit']}:{path}"], cwd=repository, capture_output=True
        )
        import hashlib

        require(
            result.returncode == 0 and hashlib.sha256(result.stdout).hexdigest() == expected,
            f"source commit does not own frozen input {path}",
        )
    return source


def check_dataset(records):
    files = {}
    for record in records["files"]:
        path = Path(record["path"])
        require(
            path.is_file() and file_hash(path) == record["sha256"],
            "external dataset missing or changed",
        )
        files[record["name"]] = path
    return files


def check_reveals(directory, split):
    previous = "genesis"
    seen: set[int] = set()
    ordered = []
    for index, path in enumerate(sorted((directory / "reveal").glob("round-*.json"))):
        record = read_json(path)
        digest = record.pop("digest")
        require(object_hash(record) == digest, "reveal ledger hash mismatch")
        require(
            record["round"] == index and record["previous"] == previous,
            "reveal round chain mismatch",
        )
        require(record["split"] == split["digest"], "reveal split identity mismatch")
        rows = record["rows"]
        require(len(rows) == len(set(rows)) and not set(rows) & seen, "duplicated label reveals")
        require(set(rows) <= set(split["train_pool"]), "prohibited label reveal")
        seen.update(rows)
        ordered.extend(rows)
        require(record["total"] == len(seen) <= 1000, "reveal completed count mismatch")
        previous = digest
    require(bool(seen), "empty label reveal ledger")
    return ordered, previous


def independent_metrics(arrays):
    e, f = arrays["energy"], arrays["force"]
    te, tf = arrays["target_energy"], arrays["target_force"]
    p = arrays["pointers"]
    require(e.shape == te.shape and f.shape == tf.shape, "prediction axes disagree")
    require(
        p.shape == (len(e) + 1,) and p[0] == 0 and p[-1] == len(f) and np.all(np.diff(p) > 0),
        "invalid geometry denominators",
    )
    require(
        len(e) > 0 and all(np.isfinite(a).all() for a in (e, f, te, tf, arrays["positions"])),
        "empty or nonfinite predictions",
    )
    geometry_error = np.array(
        [
            sum(abs(float(x)) for x in (f[a:b] - tf[a:b]).flat) / ((b - a) * 3)
            for a, b in zip(p[:-1], p[1:])
        ]
    )
    return {
        "energy_mae_eV": float(sum(abs(float(x)) for x in e - te) / len(e)),
        "force_mae_eV_angstrom": float(geometry_error.mean()),
        "geometry_force_mae": geometry_error,
        "geometries": len(e),
        "atoms": len(f),
    }


def read_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def verify_training(root: Path, entry, source, datasets, profile):
    directory = safe_member(root, entry["path"])
    config = read_json(directory / "config.json")
    split = read_json(directory / "split.json")
    digest = split.pop("digest")
    require(object_hash(split) == digest, "split fingerprint mismatch")
    split["digest"] = digest
    require(
        config["split"] == digest and config["source"] == source["digest"],
        "training source/split lineage mismatch",
    )
    roles = [split[key] for key in ("train_pool", "validation", "calibration", "evaluation")]
    require(
        all(len(rows) == len(set(rows)) and len(rows) > 0 for rows in roles), "invalid split units"
    )
    for i, rows in enumerate(roles):
        require(all(not set(rows) & set(other) for other in roles[i + 1 :]), "label roles overlap")
    revealed, reveal_hash = check_reveals(directory, split)
    train_config = config["training"]
    require(
        len(revealed) == train_config["budget"] == entry["budget"], "training label budget mismatch"
    )
    require(config["reveal"] == reveal_hash, "checkpoint reveal lineage mismatch")
    normalization = read_json(directory / "normalization.json")
    fit_rows = [int(identity.rsplit(":", 1)[1]) for identity in normalization["fit_identities"]]
    require(fit_rows == revealed, "normalizer fitted outside revealed training labels")
    history = read_json(directory / "history.json")
    require(len(history) == train_config["epochs"] > 0, "shortened training workload")
    updates_per_epoch = math.ceil(
        len(revealed) / (train_config["batch_size"] * train_config["accumulation"])
    )
    for i, item in enumerate(history):
        require(
            item["epoch"] == i + 1 and item["updates"] == (i + 1) * updates_per_epoch,
            "optimizer update count mismatch",
        )
        require(
            item["examples_processed"] == (i + 1) * len(revealed)
            and item["unique_training_geometries"] == len(revealed),
            "padding or inflated example count",
        )
    if train_config["force_weight"] > 0:
        require(history[0]["force_gradient_squared"] > 1e-20, "detached force-loss gradient")
    selection = read_json(directory / "selection.json")
    minimum = min(item["validation_force_mae"] for item in history)
    require(
        abs(selection["score"] - minimum) < 1e-12, "checkpoint selection is not validation-optimal"
    )
    require(
        abs(history[selection["epoch"] - 1]["validation_force_mae"] - minimum) < 1e-12,
        "selection epoch does not match raw validation history",
    )
    model, checkpoint = load_checkpoint(directory / selection["checkpoint"])
    initial, initial_record = load_checkpoint(directory / "initial")
    require(
        checkpoint["origin"] == "original-force-loss-training" and checkpoint["updates"] > 0,
        "imported or untrained checkpoint presented as original training",
    )
    require(
        initial_record["origin"] == "original-random-initialization"
        and initial_record["updates"] == 0,
        "untrained comparison provenance invalid",
    )
    require(
        checkpoint["epoch"] == selection["epoch"] and checkpoint["lineage"] == object_hash(config),
        "selected tensor checkpoint lineage mismatch",
    )
    change = sum(
        (p.detach() - q.detach()).square().sum().item()
        for p, q in zip(model.parameters(), initial.parameters())
    )
    require(change > 1e-18, "saved trained weights equal random initialization")
    arrays = read_arrays(directory / "evaluation-predictions.npz")
    untrained = read_arrays(directory / "evaluation-untrained.npz")
    for key in ("target_energy", "target_force", "pointers", "positions", "numbers", "identities"):
        require(
            np.array_equal(untrained[key], arrays[key]),
            "untrained comparison uses different held-out units",
        )
    metrics = read_json(directory / "evaluation-metrics.json")
    require(
        metrics["checkpoint"] == checkpoint["identity"],
        "reported predictions use another checkpoint",
    )
    rows = [int(identity.rsplit(":", 1)[1]) for identity in arrays["identities"]]
    require(
        rows == split["evaluation"] and len(rows) == len(set(rows)),
        "evaluation rows changed or repeated",
    )
    if profile == "full":
        require(len(rows) >= 5000, "shortened held-out geometry workload")
    molecule = entry["molecule"]
    path = datasets[f"rmd17_{molecule}.npz"]
    require(
        file_hash(path) == config["dataset"] == split["dataset_hash"],
        "training dataset lineage mismatch",
    )
    with np.load(path, allow_pickle=False) as reference:
        energies = reference["energies"][rows]
        forces = reference["forces"][rows].reshape(-1, 3)
        coordinates = reference["coords"][rows].reshape(-1, 3)
        numbers = np.tile(reference["nuclear_charges"], len(rows))
        train_energies = reference["energies"][revealed]
    conversion = 4184 / (6.02214076e23 * 1.602176634e-19)
    require(
        np.allclose(arrays["target_energy"], energies * conversion, atol=1e-10, rtol=0),
        "swapped physical energy units",
    )
    require(
        np.allclose(arrays["target_force"], forces * conversion, atol=1e-12, rtol=0),
        "swapped physical force units",
    )
    require(
        np.array_equal(arrays["positions"], coordinates)
        and np.array_equal(arrays["numbers"], numbers),
        "prediction geometries differ from released held-out coordinates",
    )
    require(
        abs(
            normalization["energy_per_atom"]
            - float(train_energies.mean() * conversion / len(numbers) * len(rows))
        )
        < 1e-8,
        "offset does not match training-only labels",
    )
    computed = independent_metrics(arrays)
    initial_computed = independent_metrics(untrained)
    for key in ("energy_mae_eV", "force_mae_eV_angstrom", "geometries", "atoms"):
        require(
            np.isclose(computed[key], metrics["trained"][key], rtol=1e-10, atol=1e-12),
            "false summary metric",
        )
        require(
            np.isclose(initial_computed[key], metrics["untrained"][key], rtol=1e-10, atol=1e-12),
            "false untrained metric",
        )
    zero_geometry = np.array(
        [
            np.abs(arrays["target_force"][a:b]).mean()
            for a, b in zip(arrays["pointers"][:-1], arrays["pointers"][1:])
        ]
    )
    require(
        np.isclose(zero_geometry.mean(), metrics["zero_force"]["force_mae_eV_angstrom"]),
        "missing or false zero-force baseline",
    )
    # Probe selection depends on frozen source and geometry identity rather than a handpicked prefix.
    selected = sorted(
        range(len(rows)),
        key=lambda i: object_hash([source["digest"], str(arrays["identities"][i])]),
    )[:8]
    require(len(selected) == 8, "at least eight replay probes required")
    for i in selected:
        begin, end = arrays["pointers"][i : i + 2]
        geometry = Geometry(
            arrays["numbers"][begin:end],
            arrays["positions"][begin:end],
            str(arrays["identities"][i]),
        )
        energy, force = model(collate([geometry]))
        require(
            np.allclose(energy.detach().numpy(), arrays["energy"][i : i + 1], rtol=1e-8, atol=1e-9)
            and np.allclose(
                force.detach().numpy(), arrays["force"][begin:end], rtol=1e-7, atol=1e-9
            ),
            "saved-checkpoint prediction replay mismatch",
        )
        initial_energy, initial_force = initial(collate([geometry]))
        require(
            np.allclose(
                initial_energy.detach().numpy(),
                untrained["energy"][i : i + 1],
                rtol=1e-8,
                atol=1e-9,
            )
            and np.allclose(
                initial_force.detach().numpy(), untrained["force"][begin:end], rtol=1e-7, atol=1e-9
            ),
            "saved random-initialization replay mismatch",
        )
    return {
        "molecule": molecule,
        "model": entry["model"],
        "seed": entry["seed"],
        "budget": entry["budget"],
        "geometries": len(rows),
        "updates": history[-1]["updates"],
        "force_mae": computed["force_mae_eV_angstrom"],
        "zero_force_mae": float(zero_geometry.mean()),
        "probes": len(selected),
    }


def verify(root: Path, expected_profile="full"):
    root = root.resolve()
    check_manifest(root)
    source = check_source(root)
    report = read_json(root / "ACCEPTANCE.json")
    require(
        report["profile"] == expected_profile,
        "profile mismatch; diagnostic evidence cannot certify full research",
    )
    require(report["status"] == "passed", "bundle declares an incomplete campaign")
    config = read_json(root / "CAMPAIGN.json")
    require(config["profile"] == expected_profile, "campaign profile lineage mismatch")
    datasets = check_dataset(read_json(root / "DATASETS.json"))
    units = read_json(root / "UNITS.json")
    expected = {
        (m, k, s, b)
        for m in config["molecules"]
        for k in config["models"]
        for s in config["seeds"]
        for b in config["budgets"]
    }
    observed = [(e["molecule"], e["model"], e["seed"], e["budget"]) for e in units["training"]]
    require(
        len(observed) == len(set(observed)) and set(observed) == expected,
        "hidden seed or missing matched baseline",
    )
    if expected_profile == "full":
        require(
            set(config["molecules"]) == {"aspirin", "ethanol", "malonaldehyde", "salicylic"}
            and set(config["models"]) == {"invariant", "equivariant"}
            and config["budgets"] == [100, 300, 1000]
            and len(config["seeds"]) == 3,
            "full profile scientific population shortened",
        )
    observations = [
        verify_training(root, entry, source, datasets, expected_profile)
        for entry in units["training"]
    ]
    require(
        report["completed_training_runs"] == len(observations), "inflated completed training count"
    )
    trajectories = [
        replay_trajectory(safe_member(root, entry["path"])) for entry in units["trajectories"]
    ]
    learned_steps = sum(item["completed_steps"] for item in trajectories)
    require(
        learned_steps > 0 and report["completed_learned_steps"] == learned_steps,
        "zero or inflated learned integration count",
    )
    for entry in units["trajectories"]:
        metadata = read_json(safe_member(root, entry["path"]) / "trajectory.json")
        require(
            metadata["requested_steps"] == config["simulation"]["steps"],
            "shortened simulation workload",
        )
    if expected_profile == "full":
        for key, threshold in FULL_REQUIREMENTS.items():
            require(report["counts"].get(key, 0) >= threshold, f"full requirement unmet: {key}")
        require(
            learned_steps >= FULL_REQUIREMENTS["learned_steps"], "million-step dynamics gate unmet"
        )
        require(
            report["reference"]["status"] == "passed" and report["secondary"]["status"] == "passed",
            "reference retraining or extended-system study absent",
        )
        require(
            all(item["status"] == "passed" for item in report["gates"].values()),
            "full profile has unmet gates",
        )
    import json

    commands = [json.loads(line) for line in (root / "COMMANDS.jsonl").read_text().splitlines()]
    require(
        commands and all(item["exit"] == 0 for item in commands),
        "failed or absent evidence command exits",
    )
    require(report["source"] == source["digest"], "report source identity mismatch")
    return {
        "status": "passed",
        "profile": expected_profile,
        "source": source["digest"],
        "training_runs": len(observations),
        "learned_steps": learned_steps,
        "replayed_probes": sum(item["probes"] for item in observations),
        "results": observations,
    }
