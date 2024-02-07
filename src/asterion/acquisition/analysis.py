"""Measured acquisition work, coordinate diversity and paired held-out policy outcomes."""

import numpy as np

from asterion.artifacts.io import file_hash, read_json, safe_member, write_json
from asterion.contracts.errors import ContractError
from asterion.evaluation.metrics import paired_block_interval


def interval(start, end):
    if type(start) is not int or type(end) is not int or not 0 < start < end:
        raise ContractError("acquisition timing requires observed increasing clock readings")
    return {"started_ns": start, "ended_ns": end, "seconds": (end - start) * 1e-9}


def selected_diversity(descriptors, pool_rows, selected):
    descriptors = np.asarray(descriptors, dtype=float)
    rows = list(pool_rows)
    selected = list(selected)
    if (
        descriptors.ndim != 2
        or descriptors.shape[0] != len(rows)
        or not descriptors.shape[1]
        or not np.isfinite(descriptors).all()
        or len(set(rows)) != len(rows)
        or len(selected) < 2
        or len(set(selected)) != len(selected)
        or not set(selected) <= set(rows)
    ):
        raise ContractError("selected coordinate diversity requires unique allowed descriptor rows")
    normalized = (descriptors - descriptors.mean(0)) / np.maximum(descriptors.std(0), 1e-6)
    lookup = {row: index for index, row in enumerate(rows)}
    values = normalized[[lookup[row] for row in selected]]
    nearest = []
    total = 0.0
    for begin in range(0, len(values), 32):
        distance = np.linalg.norm(values[begin : begin + 32, None, :] - values[None, :, :], axis=2)
        total += float(distance.sum())
        local = np.arange(len(distance))
        distance[local, begin + local] = np.inf
        nearest.extend(distance.min(1).tolist())
    return {
        "selected_geometries": len(values),
        "descriptor_dimensions": descriptors.shape[1],
        "normalization_scope": "all-permitted-candidate-coordinates",
        "mean_nearest_descriptor_distance": float(np.mean(nearest)),
        "mean_pairwise_descriptor_distance": total / (len(values) * (len(values) - 1)),
    }


def round_cost(
    run, round_index, clock, training_calls, decision, calibration_evaluation, candidate_inference
):
    members = []
    for member, call in enumerate(training_calls, 1):
        directory = run / f"round-{round_index}-member-{member}"
        history = read_json(directory / "history.json")
        if not history:
            raise ContractError("acquisition cost cannot count empty training")
        members.append(
            {
                "member": member,
                "call": call,
                "history_sha256": file_hash(directory / "history.json"),
                "epochs": len(history),
                "updates": history[-1]["updates"],
                "examples_processed": history[-1]["examples_processed"],
                "epoch_train_and_validation_seconds": sum(
                    row["elapsed_seconds"] for row in history
                ),
            }
        )
    record = {
        "format": "asterion-acquisition-cost-v1",
        "clock": clock,
        "decision_and_reveal": decision,
        "members": members,
        "calibration_and_evaluation": calibration_evaluation,
        "candidate_inference": candidate_inference,
        "completed_optimizer_updates": sum(row["updates"] for row in members),
        "processed_training_examples": sum(row["examples_processed"] for row in members),
        "epoch_seconds": sum(row["epoch_train_and_validation_seconds"] for row in members),
        "includes_checkpoint_and_observation_io": True,
        "excludes_shared_preprocessing_and_initial_reveal": True,
        "excludes_cost_summary_and_final_pairing_publication": True,
        "oracle_cost_claim": "released-label access; no electronic-structure calculation measured",
    }
    write_json(run / f"cost-{round_index}.json", record)
    return record


def paired_policy_outcomes(directory, outcomes, pools):
    expected = {(row["molecule"], row["trial"], row["budget"]) for row in outcomes}
    comparisons = []
    for molecule, trial, budget in sorted(expected):
        rows = [
            row
            for row in outcomes
            if (row["molecule"], row["trial"], row["budget"]) == (molecule, trial, budget)
        ]
        if {row["policy"] for row in rows} != {
            "uniform",
            "diversity",
            "uncertainty-diversity",
        } or len(rows) != 3:
            raise ContractError("paired policy outcomes require all three controls")
        arrays = {}
        for row in rows:
            path = (
                safe_member(directory, row["path"]) / f"uncertainty-evaluation-{row['round']}.npz"
            )
            with np.load(path, allow_pickle=False) as raw:
                arrays[row["policy"]] = {
                    "identities": raw["identities"].copy(),
                    "errors": raw["errors"].copy(),
                }
        uniform = arrays["uniform"]
        indices = np.array([int(identity.rsplit(":", 1)[1]) for identity in uniform["identities"]])
        blocks = pools[molecule].original_indices[indices] // 1000
        for policy in ("diversity", "uncertainty-diversity"):
            current = arrays[policy]
            if not np.array_equal(current["identities"], uniform["identities"]):
                raise ContractError("policy comparison changed held-out geometry identities")
            row = {
                "molecule": molecule,
                "trial": trial,
                "budget": budget,
                "policy": policy,
                "control": "uniform",
                "geometries": len(indices),
                "uniform_geometry_force_mae": float(uniform["errors"].mean()),
                "policy_geometry_force_mae": float(current["errors"].mean()),
                "temporal_block_span": 1000,
                "independent_geometry_claim": False,
            }
            if len(np.unique(blocks)) < 2:
                row["paired_interval"] = {
                    "available": False,
                    "reason": "fewer-than-two-temporal-blocks",
                }
            else:
                row["paired_interval"] = {
                    "available": True,
                    **paired_block_interval(uniform["errors"], current["errors"], blocks),
                }
            comparisons.append(row)
    write_json(directory / "paired-policy-outcomes.json", comparisons)
    return comparisons
