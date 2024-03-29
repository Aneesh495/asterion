"""Export independently checked reveal rounds and their original coordinates."""

from dataclasses import dataclass
from pathlib import Path

from asterion.artifacts.io import file_hash, read_json, safe_member
from asterion.artifacts.studies import verify_acquisition
from asterion.artifacts.verify import read_arrays
from asterion.contracts.errors import IntegrityError
from asterion.data.archive import dataset_arrays


@dataclass(frozen=True)
class AcquisitionArtifacts:
    root: Path
    report: str
    config: dict
    source: dict
    datasets: dict[str, Path]


def acquisition_payload(inputs: AcquisitionArtifacts | None):
    if inputs is None:
        return {
            "status": "unavailable",
            "reason": "No acquisition observations attached",
            "rounds": [],
        }
    root = inputs.root.resolve()
    report = safe_member(root, inputs.report)
    report_hash = file_hash(report)
    outcomes = read_json(report)
    expected = {
        (molecule, trial, policy, index, budget)
        for molecule in inputs.config["acquisition_molecules"]
        for trial in inputs.config["seeds"]
        for policy in ("uniform", "diversity", "uncertainty-diversity")
        for index, budget in enumerate(inputs.config["acquisition_budgets"])
    }
    observed = {
        (row["molecule"], row["trial"], row["policy"], row["round"], row["budget"])
        for row in outcomes
    }
    if observed != expected or len(outcomes) != len(expected):
        raise IntegrityError("viewer acquisition population is missing or repeats a declared round")
    verify_acquisition(root, inputs.report, inputs.config, inputs.source, inputs.datasets)
    authors = {}
    rounds = []
    for row in outcomes:
        molecule = row["molecule"]
        directory = safe_member(root, str((report.parent / row["path"]).relative_to(root)))
        members = [safe_member(directory, path) for path in row["members"]]
        split = read_json(members[0] / "split.json")
        dataset_path = Path(inputs.datasets[f"rmd17_{molecule}.npz"])
        if molecule not in authors:
            if file_hash(dataset_path) != split["dataset_hash"]:
                raise IntegrityError("viewer author coordinates differ from the acquisition split")
            authors[molecule] = dataset_arrays(dataset_path, {"coords", "nuclear_charges"})
        author = authors[molecule]
        fit = read_json(members[0] / "normalization.json")["fit_identities"]
        reveal_file = members[0] / "reveal" / f"round-{row['round']:04}.json"
        reveal = read_json(reveal_file)
        if reveal["digest"] != row["reveal"] or reveal["total"] != row["budget"]:
            raise IntegrityError("viewer acquired labels differ from the retained reveal")
        geometries = []
        for identity in fit:
            index = int(identity.rsplit(":", 1)[1])
            geometries.append(
                {
                    "identity": identity,
                    "positions": author["coords"][index].tolist(),
                    "numbers": author["nuclear_charges"].tolist(),
                    "force": None,
                    "reference": None,
                    "energy": None,
                    "selected_this_round": index in reveal["rows"],
                }
            )
        if len(geometries) != row["budget"]:
            raise IntegrityError("viewer acquired-coordinate denominator changed")
        checkpoints = []
        for member in members:
            selection = read_json(member / "selection.json")
            checkpoint = safe_member(member, selection["checkpoint"])
            metadata = read_json(checkpoint / "checkpoint.json")
            checkpoints.append(
                {
                    "identity": metadata["identity"],
                    "metadata_sha256": file_hash(checkpoint / "checkpoint.json"),
                    "normalization_sha256": file_hash(member / "normalization.json"),
                }
            )
        rounds.append(
            {
                **{key: row[key] for key in ("molecule", "trial", "policy", "round", "budget")},
                "geometries": geometries,
                "new_labels": len(reveal["rows"]),
                "uncertainty": row["uncertainty"],
                "shift": row["shift"],
                "force_mae": float(
                    read_arrays(directory / f"uncertainty-evaluation-{row['round']}.npz")[
                        "errors"
                    ].mean()
                ),
                "cost": row["cost"],
                "diversity": row["selected_diversity"],
                "lineage": {
                    "source": inputs.source["digest"],
                    "dataset": split["dataset_hash"],
                    "split": split["digest"],
                    "reveal": reveal["digest"],
                    "reveal_sha256": file_hash(reveal_file),
                    "uncertainty_sha256": file_hash(
                        directory / f"uncertainty-evaluation-{row['round']}.npz"
                    ),
                    "shift_split_sha256": file_hash(directory / "shift-split.json"),
                    "shift_observations_sha256": file_hash(
                        directory / row["shift"]["observations"]
                    ),
                    "shift_member_sha256": {
                        name: file_hash(directory / name) for name in row["shift"]["members"]
                    },
                    "cost_sha256": file_hash(directory / f"cost-{row['round']}.json"),
                    "checkpoints": checkpoints,
                },
            }
        )
    for molecule, author in authors.items():
        path = Path(inputs.datasets[f"rmd17_{molecule}.npz"])
        expected_hash = next(
            row["lineage"]["dataset"] for row in rounds if row["molecule"] == molecule
        )
        if file_hash(path) != expected_hash:
            raise IntegrityError("viewer author data changed during acquisition export")
    if file_hash(report) != report_hash:
        raise IntegrityError("viewer acquisition report changed during export")
    paired = report.parent / "paired-policy-outcomes.json"
    return {
        "status": "verified observations",
        "source": inputs.source["digest"],
        "report_sha256": report_hash,
        "paired_sha256": file_hash(paired),
        "paired": read_json(paired),
        "rounds": rounds,
    }
