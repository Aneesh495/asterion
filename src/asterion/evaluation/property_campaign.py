import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from asterion.artifacts.io import write_json
from asterion.contracts.geometry import Geometry, collate
from asterion.evaluation.checks import (
    assert_symmetry,
    force_difference,
    hessian_difference,
    symmetry_trial,
)
from asterion.graphs.radius import RadiusGraph


def graph_cases(output: Path, count=10000):
    observations = []
    for seed in range(count):
        rng = np.random.default_rng(seed)
        atoms = int(rng.integers(1, 21))
        geometry = Geometry(
            rng.choice([1, 6, 7, 8], atoms), rng.normal(size=(atoms, 3)), f"property-{seed}"
        )
        batch = collate([geometry])
        cutoff = float(rng.uniform(0.1, 5.0))
        graph = RadiusGraph(cutoff, block_size=5)(batch)
        pairs = [
            (j, i)
            for j in range(atoms)
            for i in range(atoms)
            if i != j and np.linalg.norm(geometry.positions[i] - geometry.positions[j]) < cutoff
        ]
        actual = list(zip(graph.target.tolist(), graph.source.tolist()))
        if pairs != actual:
            raise AssertionError(f"independent graph mismatch for seed {seed}")
        if graph.size and not torch.equal(
            torch.sort(graph.source).values, torch.sort(graph.target).values
        ):
            raise AssertionError("directed symmetry violated")
        observations.append(
            {
                "seed": seed,
                "atoms": atoms,
                "cutoff": cutoff,
                "edges": graph.size,
                "status": "passed",
            }
        )
        if seed and seed % 1000 == 0:
            print(f"graph property cases completed={seed}", flush=True)
    write_json(output, {"count": count, "cases": observations})
    return observations


def derivative_campaign(
    model,
    geometries,
    directory: Path,
    symmetry=1200,
    forces=1000,
    hessians=100,
    checkpoint_identity=None,
    source=None,
):
    from asterion.training.identity import state_fingerprint

    if not geometries or any(
        type(count) is not int or count < 1 for count in (symmetry, forces, hessians)
    ):
        raise ValueError("derivative campaign requires nonempty geometry and trial populations")
    if next(model.parameters()).dtype != torch.float64 or next(model.parameters()).is_cuda:
        raise ValueError("derivative reference campaign requires CPU float64")
    directory.mkdir(parents=True, exist_ok=True)
    symmetry_results, force_results, hessian_results = [], [], []
    started = time.perf_counter()
    for seed in range(symmetry):
        result = symmetry_trial(model, geometries[seed % len(geometries)], seed, retain_trace=True)
        assert_symmetry(result, energy_tolerance=1e-7)
        symmetry_results.append(result)
    for seed in range(forces):
        geometry = geometries[seed % len(geometries)]
        displacement = (
            np.random.default_rng(seed + 1900).normal(size=geometry.positions.shape) * 0.001
        )
        geometry = replace(geometry, positions=geometry.positions + displacement)
        result = force_difference(model, geometry, retain_trace=True)
        if result["relative"] >= 1e-4:
            raise AssertionError(f"force finite difference failed: {result['relative']}")
        force_results.append(
            {"seed": seed, **{k: v for k, v in result.items() if k not in {"finite", "autograd"}}}
        )
    for seed in range(hessians):
        geometry = geometries[seed % len(geometries)]
        direction = np.random.default_rng(seed + 3900).normal(size=geometry.positions.shape)
        direction /= np.linalg.norm(direction)
        result = hessian_difference(model, geometry, direction, retain_trace=True)
        if result["relative"] >= 1e-4:
            raise AssertionError("independent Hessian-vector difference failed")
        hessian_results.append(
            {"seed": seed, **{k: v for k, v in result.items() if k not in {"product", "finite"}}}
        )
    record = {
        "symmetry": symmetry_results,
        "force": force_results,
        "hessian": hessian_results,
        "elapsed_seconds": time.perf_counter() - started,
        "dtype": "float64",
        "checkpoint": checkpoint_identity,
        "source": source,
        "executed_state": state_fingerprint(model),
        "base_geometries": [g.identity for g in geometries],
        "format": "asterion-raw-derivatives-v1",
    }
    write_json(directory / "derivatives.json", record)
    return record
