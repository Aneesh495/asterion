import math
import subprocess
from pathlib import Path

import numpy as np

from asterion.artifacts.io import file_hash, object_hash, read_json, safe_member
from asterion.artifacts.source import project_root, source_inputs
from asterion.contracts.errors import IntegrityError
from asterion.contracts.geometry import Chemistry, Geometry, collate
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


def check_dataset(records, storage: Path | None = None):
    release = read_json(project_root() / "configs/rmd17-release-v4.json")
    require(
        records["version"] == release["version"] == 4 and records["doi"] == release["doi"],
        "dataset release identity changed",
    )
    require(
        records["family"] == "rMD17"
        and records["level"] == "PBE/def2-SVP"
        and records["license"] == release["license"],
        "dataset family, electronic level or license changed",
    )
    expected = (
        {f"rmd17_{m}.npz" for m in ("aspirin", "ethanol", "malonaldehyde", "salicylic")}
        | {f"index_{role}_{seed:02}.csv" for role in ("train", "test") for seed in (1, 2, 3)}
        | {"readme.txt"}
    )
    names = [record["name"] for record in records["files"]]
    require(
        set(names) == expected and len(names) == len(expected),
        "pinned dataset population is missing or repeated",
    )
    pinned = {row["name"]: row for row in release["files"]}
    files = {}
    for record in records["files"]:
        path = storage.resolve() / "rmd17" / record["name"] if storage else Path(record["path"])
        require(
            path.is_file() and file_hash(path) == record["sha256"],
            "external dataset missing or changed",
        )
        reference = pinned[record["name"]]
        require(
            all(
                record[key] == reference[key]
                for key in ("id", "size", "download_url", "supplied_md5")
            ),
            "dataset file differs from the pinned author release",
        )
        require(
            path.stat().st_size == reference["size"]
            and file_hash(path, "md5") == reference["supplied_md5"],
            "dataset bytes do not match the pinned author release",
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
    require(
        e.ndim == 1 and f.ndim == 2 and f.shape[1] == 3 and arrays["positions"].shape == f.shape,
        "physical prediction axes must be scalar energies and Cartesian vectors",
    )
    require(p.dtype.kind in "iu", "geometry pointers must be integer indices")
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
        "energy_rmse_eV": float(np.sqrt(sum(float(x) ** 2 for x in e - te) / len(e))),
        "force_rmse_eV_angstrom": float(
            np.sqrt(sum(float(x) ** 2 for x in (f - tf).flat) / f.size)
        ),
        "force_components": f.size,
    }


def read_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def verify_training(
    root: Path,
    entry,
    source,
    datasets,
    profile,
    expected_architecture=None,
    expected_loss=None,
    intervals_required=True,
):
    family = entry.get("dataset_family", "rmd17")
    require(family in {"rmd17", "md22"}, "unsupported evidence dataset family")
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
        config["architecture"]["kind"] == entry["model"] and train_config["seed"] == entry["seed"],
        "model kind or seed differs from declared unit",
    )
    if profile == "full":
        campaign = read_json(root / "CAMPAIGN.json")
        require(
            train_config["epochs"] == campaign["epochs"]
            and train_config["batch_size"] == campaign["batch_size"],
            "training workload differs from frozen full configuration",
        )
        architecture = (
            expected_architecture if expected_architecture is not None else campaign["architecture"]
        )
        loss = expected_loss if expected_loss is not None else campaign["training"]
        require(
            all(config["architecture"][key] == value for key, value in architecture.items()),
            "trained architecture differs from declared experiment",
        )
        require(
            all(train_config[key] == value for key, value in loss.items()),
            "training precision or objective differs from declared experiment",
        )
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
    require(
        1 <= selection["epoch"] <= len(history), "selection epoch is outside completed training"
    )
    minimum = min(item["validation_force_mae"] for item in history)
    require(
        abs(selection["score"] - minimum) < 1e-12, "checkpoint selection is not validation-optimal"
    )
    require(
        abs(history[selection["epoch"] - 1]["validation_force_mae"] - minimum) < 1e-12,
        "selection epoch does not match raw validation history",
    )
    model, checkpoint = load_checkpoint(safe_member(directory, selection["checkpoint"]))
    require(
        object_hash(checkpoint["architecture"]) == object_hash(config["architecture"]),
        "selected checkpoint architecture differs from training",
    )
    require(
        checkpoint.get("training_history") == history[: selection["epoch"]],
        "selected checkpoint lost its immutable training history",
    )
    require(
        model.config.level == ("PBE/def2-SVP" if family == "rmd17" else "PBE+MBD"),
        "trained model electronic level differs from its dataset family",
    )
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
    require(
        arrays["identities"].tolist()
        == [f"{entry['molecule']}:{split['dataset_hash']}:{row}" for row in rows],
        "prediction namespace differs from its molecule or dataset",
    )
    if profile == "full":
        require(
            len(rows) >= (5000 if family == "rmd17" else 1000),
            "shortened held-out geometry workload",
        )
    molecule = entry["molecule"]
    path = datasets[f"rmd17_{molecule}.npz"] if family == "rmd17" else datasets["md22"]
    require(
        file_hash(path) == config["dataset"] == split["dataset_hash"],
        "training dataset lineage mismatch",
    )
    with np.load(path, allow_pickle=False) as reference:
        energy_key, force_key, coordinate_key, numbers_key = (
            ("energies", "forces", "coords", "nuclear_charges")
            if family == "rmd17"
            else ("E", "F", "R", "z")
        )
        energies = reference[energy_key][rows].reshape(-1)
        forces = reference[force_key][rows].reshape(-1, 3)
        coordinates = reference[coordinate_key][rows].reshape(-1, 3)
        numbers = np.tile(reference[numbers_key], len(rows))
        train_energies = reference[energy_key][revealed]
        train_forces = reference[force_key][revealed]
        expected_composition = tuple(
            int(np.count_nonzero(reference[numbers_key] == z)) for z in model.config.elements
        )
        original_indices = reference["old_indices"][rows] if family == "rmd17" else np.asarray(rows)
    conversion = 4184 / (6.02214076e23 * 1.602176634e-19)
    require(
        np.isclose(
            normalization["energy_scale"],
            max(float(np.std(train_energies * conversion)), 0.01),
            atol=1e-10,
            rtol=1e-9,
        )
        and np.isclose(
            normalization["force_scale"],
            max(float(np.sqrt(np.mean((train_forces * conversion) ** 2))), 0.01),
            atol=1e-12,
            rtol=1e-10,
        )
        and np.isclose(
            model.config.energy_scale, normalization["energy_scale"], atol=1e-12, rtol=1e-10
        )
        and np.isclose(
            model.config.energy_offset, normalization["energy_per_atom"], atol=1e-10, rtol=1e-10
        ),
        "model normalization differs from permitted training labels",
    )
    require(
        model.config.composition is not None
        and tuple(model.config.composition) == expected_composition,
        "trained composition metadata differs from source atoms",
    )
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
    mean = read_arrays(directory / "evaluation-mean-force.npz")
    expected_force = np.tile(train_forces.mean(0) * conversion, (len(rows), 1))
    expected_energy = np.repeat(train_energies.mean() * conversion, len(rows))
    require(
        np.allclose(mean["force"], expected_force, atol=1e-12, rtol=1e-10)
        and np.allclose(mean["energy"], expected_energy, atol=1e-10, rtol=0),
        "mean-force baseline is not fitted on training-only labels",
    )
    mean_errors = np.asarray(
        [
            np.abs(expected_force[a:b] - arrays["target_force"][a:b]).mean()
            for a, b in zip(arrays["pointers"][:-1], arrays["pointers"][1:])
        ]
    )
    require(
        np.isclose(
            mean_errors.mean(),
            metrics["mean_force"]["force_mae_eV_angstrom"],
            atol=1e-12,
            rtol=1e-10,
        ),
        "false mean-force baseline metric",
    )
    if profile == "full" and family == "rmd17" and intervals_required:
        paired = read_json(directory / "learning-paired.json")
        require(
            paired["temporal_block_span_original_indices"] == 1000
            and paired["geometry_count"] == len(rows),
            "learning interval blocks or denominators changed",
        )
        labels = original_indices // 1000
        unique = np.unique(labels)
        require(len(unique) >= 2, "learning uncertainty has insufficient temporal blocks")
        for name, baseline in (
            ("zero_force", zero_geometry),
            ("untrained", initial_computed["geometry_force_mae"]),
        ):
            differences = baseline - computed["geometry_force_mae"]
            sums = np.asarray(
                [sum(float(x) for x in differences[labels == block]) for block in unique]
            )
            sizes = np.asarray([np.count_nonzero(labels == block) for block in unique])
            draws = np.random.default_rng(491).integers(0, len(unique), size=(2000, len(unique)))
            samples = np.asarray([sums[index].sum() / sizes[index].sum() for index in draws])
            expected = {
                "mean_improvement": float(differences.mean()),
                "lower_95": float(np.quantile(samples, 0.025)),
                "upper_95": float(np.quantile(samples, 0.975)),
                "blocks": len(unique),
                "repetitions": 2000,
                "seed": 491,
            }
            require(
                all(
                    np.isclose(paired["comparisons"][name][key], value, atol=1e-12, rtol=1e-10)
                    for key, value in expected.items()
                ),
                "paired learning interval disagrees with independent block resampling",
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
            Chemistry(level=model.config.level),
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
        "untrained_force_mae": initial_computed["force_mae_eV_angstrom"],
        "probes": len(selected),
        "checkpoint": checkpoint["identity"],
        "positive_learning_interval": profile != "full"
        or family != "rmd17"
        or not intervals_required
        or all(value["lower_95"] > 0 for value in paired["comparisons"].values()),
    }


def verify(root: Path, expected_profile="full", candidate=False, storage: Path | None = None):
    root = root.resolve()
    check_manifest(root)
    source = check_source(root)
    report = read_json(root / "ACCEPTANCE.json")
    require(
        report["profile"] == expected_profile,
        "profile mismatch; diagnostic evidence cannot certify full research",
    )
    require(
        not candidate or expected_profile == "full", "candidate verification requires full profile"
    )
    require(
        report["status"] == ("candidate" if candidate else "passed"),
        "bundle declares an incomplete campaign",
    )
    config = read_json(root / "CAMPAIGN.json")
    if expected_profile == "full":
        from asterion.training.campaign import validate_campaign

        validate_campaign(config)
    require(config["profile"] == expected_profile, "campaign profile lineage mismatch")
    datasets = check_dataset(read_json(root / "DATASETS.json"), storage)
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
    from asterion.artifacts.shift import verify_temporal_slice

    for entry, observation in zip(units["training"], observations, strict=True):
        observation["temporal_slice"] = verify_temporal_slice(
            safe_member(root, entry["path"]),
            config,
            source,
            datasets[f"rmd17_{entry['molecule']}.npz"],
            entry["molecule"],
        )
    require(
        report["completed_training_runs"] == len(observations), "inflated completed training count"
    )
    trajectories = [
        replay_trajectory(safe_member(root, entry["path"])) for entry in units["trajectories"]
    ]
    checkpoints = {}
    for entry in units["training"]:
        directory = safe_member(root, entry["path"])
        selection = read_json(directory / "selection.json")
        model, metadata = load_checkpoint(directory / selection["checkpoint"])
        checkpoints[metadata["identity"]] = model
    for entry in units["trajectories"]:
        directory = safe_member(root, entry["path"])
        metadata = read_json(directory / "trajectory.json")
        require(
            metadata["lineage"] in checkpoints, "trajectory has no retained trained-model lineage"
        )
        model = checkpoints[metadata["lineage"]]
        if metadata.get("format") == "asterion-rejected-initial-v1":
            from asterion.artifacts.rejection import verify_rejected_model

            verify_rejected_model(directory, model)
            continue
        arrays = read_arrays(directory / "trajectory.npz")
        indices = sorted(
            range(len(arrays["positions"])),
            key=lambda i: object_hash([source["digest"], entry["path"], i]),
        )[:8]
        for i in indices:
            geometry = Geometry(arrays["numbers"], arrays["positions"][i], f"trajectory-probe-{i}")
            energy, force = model(collate([geometry]))
            require(
                np.allclose(energy.detach().numpy(), arrays["potential"][i], rtol=1e-8, atol=1e-9)
                and np.allclose(force.detach().numpy(), arrays["forces"][i], rtol=1e-7, atol=1e-9),
                "trajectory force or energy does not belong to saved weights",
            )
    if "dynamics_study" in config:
        from asterion.artifacts.dynamics_study import verify_dynamics_study

        owners = {}
        for entry in units["training"]:
            owner = safe_member(root, entry["path"])
            selection = read_json(owner / "selection.json")
            identity = read_json(safe_member(owner, selection["checkpoint"]) / "checkpoint.json")[
                "identity"
            ]
            owners[identity] = (owner, entry)
        require("dynamics_study" in units, "separate dynamics study observations missing")
        verify_dynamics_study(
            root, units["dynamics_study"], config, source, checkpoints, owners, datasets
        )
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
        from asterion.artifacts.full_gates import verify_full_gates

        verify_full_gates(
            root,
            report,
            config,
            observations,
            candidate=candidate,
            dataset_paths=datasets,
            storage=storage,
        )
        required_starts = {
            (m, seed, kelvin, start)
            for m in config["molecules"]
            for seed in config["seeds"]
            for kelvin in (100, 300, 500)
            for start in range(config["simulation"]["starts_per_temperature"])
        }
        actual_starts = [
            (entry["molecule"], entry["seed"], entry["temperature_K"], entry["start"])
            for entry in units["trajectories"]
        ]
        require(
            len(actual_starts) == len(set(actual_starts)) and set(actual_starts) == required_starts,
            "learned trajectory population hides failed or missing starts",
        )
        trajectory_metadata = [
            read_json(safe_member(root, entry["path"]) / "trajectory.json")
            for entry in units["trajectories"]
        ]
        require(
            all(item["requested_steps"] == 10000 for item in trajectory_metadata),
            "full starts must request 10000 steps",
        )
        require(
            sum(item["status"] == "passed" for item in trajectory_metadata)
            / len(trajectory_metadata)
            >= 0.9,
            "core learned trajectory completion is below 90 percent",
        )
        for key, threshold in FULL_REQUIREMENTS.items():
            if candidate and key == "negative_controls":
                require(
                    report["counts"].get(key) == 0,
                    "candidate cannot claim controls before execution",
                )
                continue
            require(report["counts"].get(key, 0) >= threshold, f"full requirement unmet: {key}")
        require(
            learned_steps >= FULL_REQUIREMENTS["learned_steps"], "million-step dynamics gate unmet"
        )
        require(
            report["reference"]["status"] == ("candidate" if candidate else "passed")
            and report["secondary"]["status"] == ("candidate" if candidate else "passed"),
            "reference retraining or extended-system study absent",
        )
        require(
            all(
                item["status"] == ("candidate" if candidate else "passed")
                for item in report["gates"].values()
            ),
            "full profile has unmet gates",
        )
    import json

    commands = [json.loads(line) for line in (root / "COMMANDS.jsonl").read_text().splitlines()]
    if expected_profile != "full":
        require(
            commands and all(item["exit"] == 0 for item in commands),
            "failed or absent evidence command exits",
        )
    require(report["source"] == source["digest"], "report source identity mismatch")
    return {
        "status": "candidate-verified" if candidate else "passed",
        "phase": "candidate" if candidate else "final",
        "manifest_sha256": file_hash(root / "MANIFEST.json"),
        "profile": expected_profile,
        "source": source["digest"],
        "training_runs": len(observations),
        "learned_steps": learned_steps,
        "replayed_probes": sum(item["probes"] for item in observations),
        "results": observations,
    }
