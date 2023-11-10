import numpy as np

from asterion.artifacts.io import read_json, safe_member
from asterion.contracts.errors import IntegrityError


def verify_ablations(root, report_path, config, source, datasets):
    from asterion.artifacts.verify import verify_training

    changes: dict[str, dict[str, int | bool]] = {
        "lmax0": {"lmax": 0},
        "lmax1": {"lmax": 1},
        "no-higher-order": {"higher_order": False},
        "hard-cutoff": {"smooth": False},
        "energy-only": {},
    }
    path = safe_member(root, report_path)
    units = read_json(path)
    expected = {
        (variant, molecule, seed)
        for variant in changes
        for molecule in config["molecules"][:2]
        for seed in config["seeds"]
    }
    observed = [(u["variant"], u["molecule"], u["seed"]) for u in units]
    if len(observed) != len(expected) or set(observed) != expected:
        raise IntegrityError("ablation model/seed population is missing or repeated")
    label_sets: dict[tuple[str, int], list[str]] = {}
    results = []
    for unit in units:
        variant = unit["variant"]
        directory = safe_member(path.parent, unit["path"])
        entry = {**unit, "path": str(directory.relative_to(root))}
        architecture = {**config["architecture"], **changes[variant]}
        objective = {**config["training"]}
        if variant == "energy-only":
            objective["force_weight"] = 0.0
        measured = verify_training(
            root,
            entry,
            source,
            datasets,
            config["profile"],
            expected_architecture=architecture,
            expected_loss=objective,
            intervals_required=False,
        )
        actual = read_json(directory / "config.json")
        if not all(
            actual["architecture"][key] == value for key, value in architecture.items()
        ) or not all(actual["training"][key] == value for key, value in objective.items()):
            raise IntegrityError("ablation does not implement its declared intervention")
        labels = read_json(directory / "normalization.json")["fit_identities"]
        key = (unit["molecule"], unit["seed"])
        if key in label_sets and label_sets[key] != labels:
            raise IntegrityError("architectural ablations trained on different labels")
        label_sets[key] = labels
        if unit["budget"] != max(config["budgets"]) or unit["model"] != "equivariant":
            raise IntegrityError("ablation altered the label budget or conservative model family")
        if not np.isclose(
            measured["force_mae"],
            unit["metrics"]["trained"]["force_mae_eV_angstrom"],
            atol=1e-12,
            rtol=1e-10,
        ):
            raise IntegrityError("ablation summary metric changed")
        results.append(measured)
    return results
