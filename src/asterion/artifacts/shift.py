"""Independent temporal-role, physical-label and saved-weight checks."""

from pathlib import Path

import numpy as np

from asterion.artifacts.io import file_hash, object_hash, read_json, safe_member
from asterion.contracts.errors import IntegrityError
from asterion.contracts.geometry import Geometry, collate
from asterion.training.checkpoint import load_checkpoint


def require(condition, message):
    if not condition:
        raise IntegrityError(message)


def check_rmd_observations(arrays, author_path, rows, molecule, dataset_hash):
    from asterion.artifacts.verify import independent_metrics

    metrics = independent_metrics(arrays)
    require(
        arrays["identities"].tolist() == [f"{molecule}:{dataset_hash}:{row}" for row in rows],
        "physical observation identities changed",
    )
    conversion = 4184 / (6.02214076e23 * 1.602176634e-19)
    with np.load(author_path, allow_pickle=False) as author:
        require(
            np.allclose(
                arrays["target_energy"],
                author["energies"][rows].reshape(-1) * conversion,
                atol=1e-10,
                rtol=0,
            )
            and np.allclose(
                arrays["target_force"],
                author["forces"][rows].reshape(-1, 3) * conversion,
                atol=1e-12,
                rtol=0,
            ),
            "shift physical labels, units or rows changed",
        )
        require(
            np.array_equal(arrays["positions"], author["coords"][rows].reshape(-1, 3))
            and np.array_equal(arrays["numbers"], np.tile(author["nuclear_charges"], len(rows))),
            "shift coordinates or atom ordering changed",
        )
        atoms = len(author["nuclear_charges"])
    require(
        np.array_equal(arrays["pointers"], np.arange(len(rows) + 1) * atoms),
        "shift geometry pointers changed",
    )
    return metrics


def prediction_probes(model, arrays, source, role):
    probes = sorted(
        range(len(arrays["energy"])),
        key=lambda i: object_hash([source["digest"], str(arrays["identities"][i]), role]),
    )[:8]
    require(len(probes) == 8, "shift saved-weight probe population shortened")
    for index in probes:
        begin, end = arrays["pointers"][index : index + 2]
        geometry = Geometry(
            arrays["numbers"][begin:end],
            arrays["positions"][begin:end],
            str(arrays["identities"][index]),
        )
        energy, force = model(collate([geometry]))
        require(
            np.allclose(
                energy.detach().numpy(), arrays["energy"][index : index + 1], atol=1e-9, rtol=1e-8
            )
            and np.allclose(
                force.detach().numpy(), arrays["force"][begin:end], atol=1e-9, rtol=1e-7
            ),
            "shift predictions do not belong to the saved checkpoint",
        )
    return len(probes)


def verify_temporal_slice(directory: Path, config, source, author_path, molecule):
    from asterion.artifacts.verify import read_arrays

    original = read_json(directory / "split.json")
    split = read_json(directory / "shift-split.json")
    require(
        object_hash({k: v for k, v in split.items() if k != "digest"}) == split["digest"],
        "shift split fingerprint changed",
    )
    require(
        split["protocol"] == "temporal-slice-excluding-primary-roles"
        and split["temporal_guard"] == 0,
        "temporal slice falsely declares a guarded training task",
    )
    require(
        all(
            split[key] == original[key]
            for key in ("dataset_hash", "seed", "train_pool", "validation", "calibration")
        ),
        "temporal slice changed its trained model roles",
    )
    rows = split["evaluation"]
    forbidden = {
        row
        for role in ("train_pool", "validation", "calibration", "evaluation")
        for row in original[role]
    }
    require(
        len(rows) == len(set(rows)) == config["evaluation_size"] and not set(rows) & forbidden,
        "temporal slice shortened, repeats or leaks primary roles",
    )
    require(file_hash(author_path) == split["dataset_hash"], "temporal slice dataset changed")
    with np.load(author_path, allow_pickle=False) as author:
        indices = author["old_indices"]
        order = np.argsort(indices)
        block = order[int(len(order) * 0.75) :]
        allowed = set(
            block[
                (indices[block] > indices[block[0]] + 100)
                & (indices[block] < indices[block[-1]] - 100)
            ].tolist()
        )
    require(set(rows) <= allowed, "temporal slice entered another original-index block")
    arrays = read_arrays(directory / "shift-predictions.npz")
    observed = check_rmd_observations(arrays, author_path, rows, molecule, split["dataset_hash"])
    recorded = read_json(directory / "shift-metrics.json")
    require(recorded["split"] == split["digest"], "temporal slice metric uses another split")
    for key, value in observed.items():
        require(
            np.allclose(recorded["metrics"][key], value, atol=1e-12, rtol=1e-10),
            "false temporal slice metric",
        )
    selection = read_json(directory / "selection.json")
    model, _ = load_checkpoint(safe_member(directory, selection["checkpoint"]))
    return {
        "geometries": len(rows),
        "probes": prediction_probes(model, arrays, source, "temporal-slice"),
        "force_mae": observed["force_mae_eV_angstrom"],
    }


def check_guarded_roles(split, original_indices, config):
    definition = config.get(
        "guarded_shift", {"budget": max(config["budgets"]), "temporal_guard": 100}
    )
    guard = definition["temporal_guard"]
    require(
        split["protocol"] == "temporal-block-guard" and split["temporal_guard"] == guard > 0,
        "guarded comparison lost its protocol",
    )
    order = np.argsort(original_indices)
    cuts = [0, int(len(order) * 0.45), int(len(order) * 0.60), int(len(order) * 0.75), len(order)]
    counts = [
        1000,
        config["validation_size"],
        config["calibration_size"],
        config["evaluation_size"],
    ]
    rng = np.random.default_rng(split["seed"])
    for role, begin, end, count in zip(
        ("train_pool", "validation", "calibration", "evaluation"),
        cuts[:-1],
        cuts[1:],
        counts,
        strict=True,
    ):
        block = order[begin:end]
        indices = original_indices[block]
        allowed = block[(indices > indices[0] + guard) & (indices < indices[-1] - guard)]
        expected = rng.choice(allowed, size=count, replace=False).tolist()
        require(
            split[role] == expected, "guarded role differs from frozen original-index selection"
        )


def verify_guarded_study(root, report_path, config, source, datasets):
    from asterion.artifacts.verify import verify_training

    path = safe_member(root, report_path)
    units = read_json(path)
    expected = {
        (m, k, s) for m in config["molecules"] for k in config["models"] for s in config["seeds"]
    }
    observed = [(u["molecule"], u["model"], u["seed"]) for u in units]
    require(
        set(observed) == expected and len(observed) == len(expected),
        "guarded study hides or repeats a model seed",
    )
    results = []
    for unit in units:
        require(unit["budget"] == max(config["budgets"]), "guarded model label budget changed")
        directory = safe_member(path.parent, unit["path"])
        split = read_json(directory / "split.json")
        author_path = datasets[f"rmd17_{unit['molecule']}.npz"]
        with np.load(author_path, allow_pickle=False) as author:
            check_guarded_roles(split, author["old_indices"], config)
        entry = {**unit, "path": str(directory.relative_to(root))}
        measured = verify_training(root, entry, source, datasets, config["profile"])
        require(
            np.isclose(
                measured["force_mae"],
                unit["metrics"]["trained"]["force_mae_eV_angstrom"],
                atol=1e-12,
                rtol=1e-10,
            ),
            "guarded study summary metric changed",
        )
        results.append(measured)
    return results
