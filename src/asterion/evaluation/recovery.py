from pathlib import Path

import numpy as np
import torch

from asterion.artifacts.io import read_json, write_json
from asterion.contracts.errors import AsterionError
from asterion.contracts.geometry import Chemistry, Geometry
from asterion.data.dataset import CoordinatePool, LabelOracle
from asterion.data.splits import Split
from asterion.dynamics.analytic import HarmonicBond
from asterion.dynamics.integrator import SimulationConfig, simulate
from asterion.models.potential import ModelConfig
from asterion.training.checkpoint import load_checkpoint
from asterion.training.engine import TrainConfig, train


def simulation_recovery(directory: Path, cases=50):
    outcomes = []
    numbers = np.array([6, 1])
    positions = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
    for seed in range(cases):
        velocity = np.random.default_rng(seed).normal(size=(2, 3)) * 0.001
        config = SimulationConfig(
            steps=40, ensemble="NVT" if seed % 2 else "NVE", seed=seed, checkpoint_every=5
        )
        whole = directory / f"whole-{seed:03}"
        resumed = directory / f"resumed-{seed:03}"
        simulate(HarmonicBond(), numbers, positions, velocity, whole, config)
        interrupt_at = (seed % 7 + 1) * 5
        simulate(
            HarmonicBond(), numbers, positions, velocity, resumed, config, stop_after=interrupt_at
        )
        simulate(HarmonicBond(), numbers, positions, velocity, resumed, config, resume=True)
        with (
            np.load(whole / "trajectory.npz", allow_pickle=False) as a,
            np.load(resumed / "trajectory.npz", allow_pickle=False) as b,
        ):
            for key in a.files:
                if not np.array_equal(a[key], b[key]):
                    raise AssertionError(f"simulation recovery differs: {seed}/{key}")
        outcomes.append(
            {
                "case": seed,
                "interruption_step": interrupt_at,
                "ensemble": config.ensemble,
                "status": "passed",
            }
        )
    write_json(directory / "recovery.json", {"cases": len(outcomes), "outcomes": outcomes})
    return outcomes


def training_recovery(directory: Path, cases=5):
    rng = np.random.default_rng(811)
    positions = np.array([[[0.0, 0.0, 0.0], [r, 0.0, 0.0]] for r in rng.uniform(0.8, 1.2, size=40)])
    pool = CoordinatePool(
        np.array([6, 1]),
        positions,
        np.arange(40),
        "analytic-bond",
        "bond",
        Chemistry(level="analytic"),
    )
    evaluator = HarmonicBond()
    outputs = [evaluator(x, pool.numbers) for x in positions]
    oracle = LabelOracle(
        pool, np.array([r.energy for r in outputs]), np.array([r.forces for r in outputs])
    )
    split = Split(
        pool.dataset_hash,
        "recovery",
        1,
        tuple(range(20)),
        tuple(range(20, 25)),
        tuple(range(25, 30)),
        tuple(range(30, 40)),
    )
    outcomes = []
    for seed in range(cases):
        config = TrainConfig(seed=seed, budget=16, epochs=3, batch_size=4, checkpoint_every=1)
        architecture = ModelConfig(kind="invariant", width=4, depth=1, radial=4)
        whole = directory / f"whole-{seed}"
        resumed = directory / f"resumed-{seed}"
        train(oracle, split, whole, architecture, config)
        interrupted = train(oracle, split, resumed, architecture, config, stop_after=1)
        train(oracle, split, resumed, architecture, config, resume=Path(interrupted["checkpoint"]))
        a, _ = load_checkpoint(whole / "epoch-00003")
        b, _ = load_checkpoint(resumed / "epoch-00003")
        for key, value in a.state_dict().items():
            if not torch.equal(value, b.state_dict()[key]):
                raise AssertionError(f"training recovery weights differ: {key}")
        first = read_json(whole / "history.json")
        second = read_json(resumed / "history.json")
        for r, s in zip(first, second):
            for key in r.keys() - {"elapsed_seconds"}:
                if r[key] != s[key]:
                    raise AssertionError(f"recovery training history differs: {key}")
        outcomes.append({"case": seed, "seed": seed, "interruption_epoch": 1, "status": "passed"})
    write_json(directory / "recovery.json", {"cases": len(outcomes), "outcomes": outcomes})
    return outcomes


def fault_cases(directory: Path, count=100):
    directory.mkdir(parents=True, exist_ok=True)
    outcomes = []
    for seed in range(count):
        xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        numbers = np.array([6, 1])
        chemistry = Chemistry()
        mode = seed % 5
        if mode == 0:
            xyz[1, seed % 3] = float("nan")
        elif mode == 1:
            xyz[1] = xyz[0]
        elif mode == 2:
            numbers[seed % 2] = 2 + seed
        elif mode == 3:
            chemistry = Chemistry(charge=seed + 1)
        else:
            chemistry = Chemistry(periodic=True)
        try:
            Geometry(numbers, xyz, f"invalid-{seed}", chemistry).validate()
        except AsterionError as error:
            outcomes.append({"case": seed, "mode": mode, "error": str(error), "status": "rejected"})
        else:
            raise AssertionError(f"invalid geometry accepted: {seed}")
    write_json(directory / "faults.json", {"cases": len(outcomes), "outcomes": outcomes})
    return outcomes
