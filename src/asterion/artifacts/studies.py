from pathlib import Path

import numpy as np

from asterion.artifacts.io import file_hash, object_hash, read_json, safe_member
from asterion.contracts.errors import IntegrityError
from asterion.contracts.geometry import Chemistry, Geometry, collate
from asterion.evaluation.replay import replay_trajectory
from asterion.training.checkpoint import load_checkpoint


def require(condition, message):
    if not condition:
        raise IntegrityError(message)


def model_trajectory(directory, model, identity, source):
    metadata = read_json(directory / "trajectory.json")
    require(metadata["lineage"] == identity, "study trajectory uses another model")
    replay = replay_trajectory(directory)
    with np.load(directory / "trajectory.npz", allow_pickle=False) as raw:
        arrays = {key: raw[key] for key in raw.files}
    probes = sorted(
        range(len(arrays["positions"])),
        key=lambda index: object_hash([source["digest"], str(directory.name), index]),
    )[:8]
    for index in probes:
        geometry = Geometry(
            arrays["numbers"],
            arrays["positions"][index],
            f"study-probe-{index}",
            Chemistry(level=model.config.level),
        )
        energy, force = model(collate([geometry]))
        require(
            np.allclose(energy.detach().numpy(), arrays["potential"][index], atol=1e-9, rtol=1e-8)
            and np.allclose(force.detach().numpy(), arrays["forces"][index], atol=1e-9, rtol=1e-7),
            "study trajectory does not belong to saved model weights",
        )
    return metadata, replay


def verify_secondary(root: Path, report_path: str, config, source):
    from asterion.artifacts.verify import verify_training

    report_file = safe_member(root, report_path)
    directory = report_file.parent
    report = read_json(report_file)
    dataset = report["dataset"]
    author_path = Path(dataset["path"])
    require(
        author_path.is_file() and file_hash(author_path) == dataset["sha256"],
        "secondary author data missing or changed",
    )
    require(dataset["level"] == "PBE+MBD", "secondary electronic level changed")
    entry = {
        "molecule": dataset["name"],
        "model": "equivariant",
        "seed": 1,
        "budget": config["secondary"]["train"],
        "path": str((directory / "model").relative_to(root)),
        "dataset_family": "md22",
    }
    observed = verify_training(root, entry, source, {"md22": author_path}, config["profile"])
    model_dir = directory / "model"
    selection = read_json(model_dir / "selection.json")
    model, checkpoint = load_checkpoint(model_dir / selection["checkpoint"])
    require(
        model.config.level == dataset["level"], "secondary model lost its independent label task"
    )
    starts = report["trajectories"]
    require(
        {row["start"] for row in starts} == set(range(config["secondary"]["starts"]))
        and len(starts) == config["secondary"]["starts"],
        "secondary starts are missing or repeated",
    )
    steps = failed = 0
    for row in starts:
        metadata, replay = model_trajectory(
            directory / f"trajectory-{row['start']:03}", model, checkpoint["identity"], source
        )
        require(
            metadata["requested_steps"] == config["secondary"]["steps"],
            "secondary physical workload shortened",
        )
        require(metadata == row["record"], "secondary summary hides trajectory termination")
        steps += replay["completed_steps"]
        failed += metadata["status"] != "passed"
    require(
        steps > 0 and steps == report["completed_steps"] and failed == report["failed_starts"],
        "secondary useful progress or failure denominator changed",
    )
    require(
        observed["force_mae"] == report["metrics"]["trained"]["force_mae_eV_angstrom"],
        "secondary summary force metric changed",
    )
    return {"model": observed, "completed_steps": steps, "failed_starts": failed}


def committee_arrays(member_arrays):
    require(len(member_arrays) == 3, "study committee must contain three raw members")
    first = member_arrays[0]
    for other in member_arrays[1:]:
        require(
            all(
                np.array_equal(first[key], other[key])
                for key in (
                    "identities",
                    "pointers",
                    "target_force",
                    "target_energy",
                    "positions",
                    "numbers",
                )
            ),
            "committee members evaluated different geometry identities",
        )
    force = np.stack([value["force"] for value in member_arrays])
    require(np.isfinite(force).all(), "nonfinite study member forces")
    mean = np.add.reduce(force) / 3
    variance = np.add.reduce((force - mean) ** 2) / 2
    std = np.sqrt(variance)
    pointers = first["pointers"]
    scores = np.array(
        [
            np.sqrt(np.sum(std[a:b] ** 2) / ((b - a) * 3))
            for a, b in zip(pointers[:-1], pointers[1:])
        ]
    )
    errors = np.array(
        [
            np.sum(np.abs(mean[a:b] - first["target_force"][a:b])) / ((b - a) * 3)
            for a, b in zip(pointers[:-1], pointers[1:])
        ]
    )
    return mean, std, scores, errors


def independent_choices(rows, coordinates, numbers, revealed, count, policy, seed, scores):
    """Express the declared greedy policy directly from permitted coordinate observations."""
    available = [index for index, row in enumerate(rows) if row not in set(revealed)]
    if policy == "uniform":
        chosen = np.random.default_rng(seed).choice(np.asarray(available), count, replace=False)
        return [rows[int(index)] for index in chosen], [0.0] * count, [0.0] * count
    require(policy in {"diversity", "uncertainty-diversity"}, "unknown retained acquisition policy")
    pairs: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for first in range(len(numbers)):
        for second in range(first + 1, len(numbers)):
            category = tuple(sorted((int(numbers[first]), int(numbers[second]))))
            pairs.setdefault(category, []).append((first, second))
    columns = []
    for category in sorted(pairs):
        values = [
            np.sqrt(np.sum((coordinates[:, a] - coordinates[:, b]) ** 2, axis=1))
            for a, b in pairs[category]
        ]
        columns.append(np.sort(np.stack(values, axis=1), axis=1))
    descriptor = np.concatenate(columns, axis=1)
    descriptor = (descriptor - descriptor.mean(axis=0)) / np.maximum(descriptor.std(axis=0), 1e-6)
    selected = [index for index, row in enumerate(rows) if row in set(revealed)]
    distance = np.min(
        np.stack(
            [np.sqrt(np.sum((descriptor - descriptor[index]) ** 2, axis=1)) for index in selected]
        ),
        axis=0,
    )

    def ranks(values):
        order = sorted(range(len(values)), key=lambda index: (float(values[index]), index))
        rank = np.empty(len(values), dtype=float)
        for position, index in enumerate(order):
            rank[index] = position / max(len(order) - 1, 1)
        return rank

    uncertainty_rank = ranks(scores)
    choices = []
    chosen_scores = []
    novelty = []
    for _ in range(count):
        values = ranks(distance)
        if policy == "uncertainty-diversity":
            values = (values + uncertainty_rank) / 2
        index = max(available, key=lambda index: (float(values[index]), -index))
        choices.append(rows[index])
        chosen_scores.append(float(values[index]))
        novelty.append(float(distance[index]))
        available.remove(index)
        distance = np.minimum(
            distance, np.sqrt(np.sum((descriptor - descriptor[index]) ** 2, axis=1))
        )
    return choices, chosen_scores, novelty


def verify_acquisition(root: Path, report_path: str, config, source, datasets):
    from asterion.artifacts.verify import read_arrays, verify_training

    report_file = safe_member(root, report_path)
    base = report_file.parent
    outcomes = read_json(report_file)
    initial: dict[tuple[str, int], list[int]] = {}
    for row in outcomes:
        directory = safe_member(root, str((base / row["path"]).relative_to(root)))
        require(
            len(row["members"]) == 3 and len(set(row["members"])) == 3,
            "acquisition member paths missing or repeated",
        )
        evaluation = []
        calibration = []
        model_ids = []
        member_rows = []
        for member, path in enumerate(row["members"], 1):
            unit_dir = safe_member(directory, path)
            seed = row["trial"] * 100 + member
            unit = {
                "molecule": row["molecule"],
                "seed": seed,
                "budget": row["budget"],
                "model": "equivariant",
                "path": str(unit_dir.relative_to(root)),
            }
            observed = verify_training(
                root, unit, source, datasets, config["profile"], intervals_required=False
            )
            model_ids.append(observed["checkpoint"])
            normalizer = read_json(unit_dir / "normalization.json")
            require(
                read_json(unit_dir / "config.json")["reveal"] == row["reveal"],
                "acquisition outcome points to another reveal round",
            )
            member_rows.append(normalizer["fit_identities"])
            evaluation.append(read_arrays(unit_dir / "evaluation-predictions.npz"))
            calibration.append(
                read_arrays(directory / f"calibration-{row['round']}-member-{member}.npz")
            )
            split = read_json(unit_dir / "split.json")
            data = calibration[-1]
            expected_ids = [
                f"{row['molecule']}:{split['dataset_hash']}:{index}"
                for index in split["calibration"]
            ]
            require(
                data["identities"].tolist() == expected_ids,
                "calibration labels entered from another role",
            )
            conversion = 4184 / (6.02214076e23 * 1.602176634e-19)
            with np.load(datasets[f"rmd17_{row['molecule']}.npz"], allow_pickle=False) as author:
                indices = split["calibration"]
                require(
                    np.allclose(
                        data["target_energy"],
                        author["energies"][indices] * conversion,
                        atol=1e-10,
                        rtol=0,
                    )
                    and np.allclose(
                        data["target_force"],
                        author["forces"][indices].reshape(-1, 3) * conversion,
                        atol=1e-12,
                        rtol=0,
                    ),
                    "calibration physical labels changed",
                )
                require(
                    np.array_equal(data["positions"], author["coords"][indices].reshape(-1, 3))
                    and np.array_equal(
                        data["numbers"], np.tile(author["nuclear_charges"], len(indices))
                    ),
                    "calibration geometry changed",
                )
            selection = read_json(unit_dir / "selection.json")
            model, _ = load_checkpoint(unit_dir / selection["checkpoint"])
            probes = sorted(
                range(len(expected_ids)),
                key=lambda index: object_hash([source["digest"], expected_ids[index]]),
            )[:8]
            for index in probes:
                begin, end = data["pointers"][index : index + 2]
                geometry = Geometry(
                    data["numbers"][begin:end], data["positions"][begin:end], expected_ids[index]
                )
                energy, force = model(collate([geometry]))
                require(
                    np.allclose(
                        energy.detach().numpy(), data["energy"][index], atol=1e-9, rtol=1e-8
                    )
                    and np.allclose(
                        force.detach().numpy(), data["force"][begin:end], atol=1e-9, rtol=1e-7
                    ),
                    "calibration predictions belong to another checkpoint",
                )
        require(
            member_rows[0] == member_rows[1] == member_rows[2],
            "committee did not train on identical revealed rows",
        )
        mean, std, scores, errors = committee_arrays(evaluation)
        saved = read_arrays(directory / f"uncertainty-evaluation-{row['round']}.npz")
        require(
            np.array_equal(saved["identities"], evaluation[0]["identities"]),
            "committee evaluation identities changed",
        )
        require(
            np.allclose(mean, saved["mean_force"], atol=1e-12, rtol=1e-10)
            and np.allclose(std, saved["component_std"], atol=1e-12, rtol=1e-10),
            "committee means or disagreement were fabricated",
        )
        require(
            np.allclose(scores, saved["scores"], atol=1e-12, rtol=1e-10)
            and np.allclose(errors, saved["errors"], atol=1e-12, rtol=1e-10),
            "committee scores or geometry errors changed",
        )
        _, _, cal_scores, cal_errors = committee_arrays(calibration)
        fitted = read_json(directory / f"calibration-{row['round']}.json")
        require(
            tuple(fitted["committee"]) == tuple(model_ids), "calibration uses another committee"
        )
        require(
            fitted["alpha"] == 0.1 and fitted["sample_count"] == len(cal_scores) >= 10,
            "calibration workload or target changed",
        )
        floor = max(float(np.median(cal_scores) * 0.01), 1e-8)
        ratios = np.sort(cal_errors / np.maximum(cal_scores, floor))
        rank = min(len(ratios), int(np.ceil((len(ratios) + 1) * 0.9)))
        multiplier = float(ratios[rank - 1])
        require(
            np.isclose(fitted["floor"], floor, atol=1e-12, rtol=1e-10)
            and np.isclose(fitted["multiplier"], multiplier, atol=1e-12, rtol=1e-10),
            "calibration quantile disagrees with raw held-out errors",
        )
        bounds = multiplier * np.maximum(scores, floor)
        require(
            np.allclose(bounds, saved["bounds"], atol=1e-12, rtol=1e-10),
            "committee coverage bounds changed",
        )
        require(
            np.isclose(np.mean(errors <= bounds), row["uncertainty"]["coverage"]),
            "false committee coverage",
        )
        require(
            len(errors) == row["uncertainty"]["geometries"], "false committee geometry denominator"
        )
        first_round = read_json(
            next((directory / row["members"][0] / "reveal").glob("round-0000.json"))
        )
        key = (row["molecule"], row["trial"])
        if key in initial:
            require(
                initial[key] == first_round["rows"],
                "acquisition policies did not share the same initial pool",
            )
        else:
            initial[key] = first_round["rows"]
        if row["round"]:
            candidate = read_json(directory / f"candidates-{row['round']}.json")
            round_record = read_json(
                directory / row["members"][0] / "reveal" / f"round-{row['round']:04}.json"
            )
            require(
                object_hash(candidate) == round_record["scores"]
                and candidate["previous_reveal"] == round_record["previous"],
                "candidate scores and reveal lineage disagree",
            )
            require(
                candidate["choices"]["rows"] == round_record["rows"],
                "revealed labels differ from selected candidates",
            )
            pool = read_arrays(directory / f"candidate-pool-{row['round'] - 1}.npz")
            pool_rows = candidate["pool"]
            atoms = len(pool["numbers"]) // len(pool_rows)
            xyz = pool["positions"].reshape(len(pool_rows), atoms, 3)
            matrix = pool["member_forces"]
            require(
                matrix.shape == (3, len(pool_rows) * atoms, 3),
                "candidate member force axes changed",
            )
            standard = np.sqrt(np.sum((matrix - matrix.mean(0)) ** 2, axis=0) / 2)
            pool_scores = np.sqrt(
                np.mean(standard.reshape(len(pool_rows), atoms, 3) ** 2, axis=(1, 2))
            )
            require(
                np.allclose(pool_scores, candidate["scores"], atol=1e-12, rtol=1e-10),
                "candidate disagreement scores were fabricated",
            )
            split = read_json(directory / row["members"][0] / "split.json")
            require(pool_rows == split["train_pool"], "candidate pool crossed a label role")
            expected_ids = [
                f"{row['molecule']}:{split['dataset_hash']}:{index}" for index in pool_rows
            ]
            require(
                pool["identities"].tolist() == expected_ids, "candidate score identities changed"
            )
            with np.load(datasets[f"rmd17_{row['molecule']}.npz"], allow_pickle=False) as author:
                require(
                    np.array_equal(xyz, author["coords"][pool_rows])
                    and np.array_equal(
                        pool["numbers"], np.tile(author["nuclear_charges"], len(pool_rows))
                    ),
                    "candidate coordinates or atom order changed",
                )
            probes = sorted(
                range(len(pool_rows)),
                key=lambda index: object_hash(
                    [source["digest"], expected_ids[index], "candidate-pool"]
                ),
            )[:8]
            for member in (1, 2, 3):
                previous = directory / f"round-{row['round'] - 1}-member-{member}"
                selection = read_json(previous / "selection.json")
                model, record = load_checkpoint(previous / selection["checkpoint"])
                require(
                    record["identity"] == candidate["committee"][member - 1],
                    "candidate scores use another committee",
                )
                for index in probes:
                    begin, end = pool["pointers"][index : index + 2]
                    geometry = Geometry(
                        pool["numbers"][begin:end],
                        pool["positions"][begin:end],
                        expected_ids[index],
                    )
                    energy, force = model(collate([geometry]))
                    require(
                        np.allclose(
                            energy.detach().numpy(),
                            pool["member_energies"][member - 1, index],
                            atol=1e-9,
                            rtol=1e-8,
                        )
                        and np.allclose(
                            force.detach().numpy(),
                            matrix[member - 1, begin:end],
                            atol=1e-9,
                            rtol=1e-7,
                        ),
                        "candidate predictions do not belong to saved member weights",
                    )
            revealed = [
                int(identity.rsplit(":", 1)[1])
                for identity in member_rows[0]
                if int(identity.rsplit(":", 1)[1]) not in round_record["rows"]
            ]
            seed = row["trial"] * 100 + row["round"]
            require(
                candidate["choices"]["seed"] == seed
                and candidate["choices"]["policy"] == row["policy"],
                "acquisition seed or policy changed",
            )
            choice, choice_scores, novelty = independent_choices(
                pool_rows,
                xyz,
                pool["numbers"][:atoms],
                revealed,
                len(round_record["rows"]),
                row["policy"],
                seed,
                pool_scores,
            )
            require(
                choice == round_record["rows"]
                and np.allclose(
                    choice_scores, candidate["choices"]["scores"], atol=1e-12, rtol=1e-10
                )
                and np.allclose(novelty, candidate["choices"]["novelty"], atol=1e-10, rtol=1e-9),
                "independent acquisition decision reconstruction failed",
            )
    return {"outcomes": len(outcomes), "shared_initial_pools": len(initial)}
