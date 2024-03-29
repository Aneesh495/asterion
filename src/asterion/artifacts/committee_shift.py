"""Check disjoint temporal committee observations without recalibrating or training."""

from pathlib import Path

import numpy as np

from asterion.artifacts.acquisition_analysis import verify_uncertainty_summary
from asterion.artifacts.io import object_hash, read_json, safe_member
from asterion.contracts.errors import IntegrityError
from asterion.contracts.geometry import Chemistry, Geometry, collate


def require(condition, message):
    if not condition:
        raise IntegrityError(message)


def verify_committee_shift(
    directory: Path, row, config, source, primary, fitted, models, author_path
):
    from asterion.artifacts.studies import committee_arrays
    from asterion.artifacts.verify import read_arrays

    shifted = row.get("shift")
    require(isinstance(shifted, dict), "committee temporal slice is missing")
    split = read_json(directory / "shift-split.json")
    with np.load(author_path, allow_pickle=False) as author:
        total = len(author["coords"])
        original = author["old_indices"]
        order = sorted(range(total), key=lambda i: (int(original[i]), i))
        forbidden = {
            int(i)
            for role in ("train_pool", "validation", "calibration", "evaluation")
            for i in primary[role]
        }
        eligible = [i for i in order[total * 3 // 4 :] if i not in forbidden]
        count = config["evaluation_size"]
        require(len(eligible) >= count, "committee temporal suffix has too few unused rows")
        rows = (
            np.random.default_rng(primary["seed"] + 9100)
            .choice(eligible, count, replace=False)
            .tolist()
        )
        expected = {
            k: primary[k]
            for k in ("dataset_hash", "seed", "train_pool", "validation", "calibration")
        }
        expected.update(
            protocol="committee-temporal-suffix-excluding-primary-roles",
            evaluation=rows,
            temporal_guard=0,
        )
        require(
            split == {**expected, "digest": object_hash(expected)},
            "committee slice changed identities or crossed primary roles",
        )
        require(
            shifted["protocol"] == expected["protocol"]
            and shifted["split"] == split["digest"]
            and shifted["geometries"] == count
            and shifted["temporal_guard"] == 0
            and shifted["calibration"] == "unchanged-primary-calibration"
            and shifted["interpretation"]
            == "Disjoint late-time slice; primary training is not temporally guarded and distribution shift is not guaranteed.",
            "committee shift population or calibration claim changed",
        )
        require(
            read_json(directory / f"shift-{row['round']}.json") == shifted,
            "committee shift report and acquisition summary disagree",
        )
        expected_files = [f"shift-{row['round']}-member-{i}.npz" for i in (1, 2, 3)]
        require(
            shifted["members"] == expected_files
            and shifted["observations"] == f"uncertainty-shift-{row['round']}.npz",
            "committee shift observations point to other members or rounds",
        )
        members = [read_arrays(safe_member(directory, name)) for name in expected_files]
        identities = [f"{row['molecule']}:{primary['dataset_hash']}:{i}" for i in rows]
        conversion = 4184 / (6.02214076e23 * 1.602176634e-19)
        target_force = author["forces"][rows].reshape(-1, 3) * conversion
        positions = author["coords"][rows].reshape(-1, 3)
        numbers = np.tile(author["nuclear_charges"], len(rows))
        atoms = len(author["nuclear_charges"])
        pointers = np.arange(count + 1, dtype=np.int64) * atoms
        for model, data in zip(models, members):
            require(
                data["identities"].tolist() == identities
                and np.array_equal(data["pointers"], pointers)
                and np.array_equal(data["positions"], positions)
                and np.array_equal(data["numbers"], numbers)
                and np.allclose(data["target_force"], target_force, atol=1e-12, rtol=0)
                and np.allclose(
                    data["target_energy"], author["energies"][rows] * conversion, atol=1e-10, rtol=0
                ),
                "committee shift coordinates, units or targets changed",
            )
            probes = sorted(
                range(count),
                key=lambda i: object_hash([source["digest"], "committee-shift", identities[i]]),
            )[:8]
            for i in probes:
                a, b = pointers[i : i + 2]
                geometry = Geometry(
                    numbers[a:b], positions[a:b], identities[i], Chemistry(level=model.config.level)
                )
                energy, force = model(collate([geometry]))
                require(
                    np.allclose(energy.detach().numpy(), data["energy"][i], atol=1e-9, rtol=1e-8)
                    and np.allclose(
                        force.detach().numpy(), data["force"][a:b], atol=1e-9, rtol=1e-7
                    ),
                    "committee shift predictions belong to another saved checkpoint",
                )
    mean, std, scores, errors = committee_arrays(members)
    saved = read_arrays(safe_member(directory, shifted["observations"]))
    bounds = fitted["multiplier"] * np.maximum(scores, fitted["floor"])
    require(
        np.array_equal(saved["identities"], members[0]["identities"])
        and np.array_equal(saved["pointers"], pointers)
        and np.array_equal(saved["target_force"], members[0]["target_force"])
        and np.allclose(saved["mean_force"], mean, atol=1e-12, rtol=1e-10)
        and np.allclose(saved["component_std"], std, atol=1e-12, rtol=1e-10)
        and np.allclose(saved["scores"], scores, atol=1e-12, rtol=1e-10)
        and np.allclose(saved["errors"], errors, atol=1e-12, rtol=1e-10)
        and np.allclose(saved["bounds"], bounds, atol=1e-12, rtol=1e-10),
        "committee shift means, disagreement, errors or reused bounds changed",
    )
    verify_uncertainty_summary(shifted["uncertainty"], scores, errors, bounds)
    return {
        "geometries": count,
        "members": 3,
        "replayed_probes": min(8, count) * 3,
        "protocol": expected["protocol"],
    }
