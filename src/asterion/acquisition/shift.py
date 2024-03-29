"""Measure committee calibration on a disjoint temporal suffix without refitting it."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from asterion.artifacts.io import atomic_npz, write_json
from asterion.contracts.errors import ContractError
from asterion.evaluation.metrics import uncertainty_metrics
from asterion.training.engine import predict


def temporal_slice(pool, primary, count):
    if type(count) is not int or count <= 0:
        raise ContractError("committee slice needs a positive geometry count")
    forbidden = (
        set(primary.train_pool)
        | set(primary.validation)
        | set(primary.calibration)
        | set(primary.evaluation)
    )
    order = np.argsort(pool.original_indices, kind="stable")
    eligible = np.array(
        [int(row) for row in order[len(order) * 3 // 4 :] if int(row) not in forbidden],
        dtype=np.int64,
    )
    if len(eligible) < count:
        raise ContractError("committee temporal slice cannot meet its frozen distinct denominator")
    selected = np.random.default_rng(primary.seed + 9100).choice(eligible, count, replace=False)
    return replace(
        primary,
        protocol="committee-temporal-suffix-excluding-primary-roles",
        evaluation=tuple(map(int, selected)),
        temporal_guard=0,
    ).validate(len(pool))


def shifted_observation(
    oracle, split, models, committee, calibration, batch_size, directory: Path, round_index
):
    labels = oracle.labels(split.evaluation, split.evaluation, "evaluation")
    member_files = []
    for member, model in enumerate(models, 1):
        arrays, _ = predict(model, labels, batch_size)
        filename = f"shift-{round_index}-member-{member}.npz"
        atomic_npz(directory / filename, **arrays)
        member_files.append(filename)
    output = committee.infer([label.geometry for label in labels])
    pointers = np.r_[0, np.cumsum([len(label.geometry.numbers) for label in labels])]
    target = np.concatenate([label.forces for label in labels])
    errors = np.array(
        [
            np.abs(output.forces_mean[a:b] - target[a:b]).mean()
            for a, b in zip(pointers[:-1], pointers[1:])
        ]
    )
    bounds = calibration.bounds(output.force_score)
    filename = f"uncertainty-shift-{round_index}.npz"
    atomic_npz(
        directory / filename,
        identities=output.identities,
        scores=output.force_score,
        errors=errors,
        bounds=bounds,
        mean_force=output.forces_mean,
        target_force=target,
        component_std=output.force_component_std,
        pointers=pointers,
    )
    report = {
        "protocol": split.protocol,
        "split": split.digest,
        "geometries": len(labels),
        "calibration": "unchanged-primary-calibration",
        "temporal_guard": 0,
        "interpretation": "Disjoint late-time slice; primary training is not temporally guarded and distribution shift is not guaranteed.",
        "members": member_files,
        "observations": filename,
        "uncertainty": uncertainty_metrics(output.force_score, errors, bounds),
    }
    write_json(directory / f"shift-{round_index}.json", report)
    return report
