from pathlib import Path

from asterion.artifacts.io import write_json
from asterion.contracts.units import thermal_velocities
from asterion.data.dataset import load_md22
from asterion.data.secondary import acquire_md22
from asterion.data.splits import guarded_split
from asterion.dynamics.detectors import ValidityPolicy
from asterion.dynamics.integrator import SimulationConfig, simulate
from asterion.dynamics.learned import LearnedEvaluator
from asterion.evaluation.replay import replay_trajectory
from asterion.models.potential import ModelConfig
from asterion.training.checkpoint import load_checkpoint
from asterion.training.engine import TrainConfig, evaluate_run, train


def secondary_campaign(config, storage: Path, directory: Path):
    path, manifest = acquire_md22(Path("configs/md22-release.json"), storage)
    pool, oracle = load_md22(path, manifest["name"], manifest["sha256"], manifest["level"])
    definition = config["secondary"]
    split = guarded_split(
        pool,
        1,
        train_size=definition["train"],
        evaluation_size=definition["evaluation"],
        validation_size=500,
        calibration_size=500,
    )
    run = directory / "model"
    result = train(
        oracle,
        split,
        run,
        ModelConfig(**config["architecture"]),
        TrainConfig(
            seed=1,
            budget=definition["train"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            **config["training"],
        ),
    )
    metrics = evaluate_run(oracle, split, run, config["batch_size"])
    model, checkpoint = load_checkpoint(Path(result["checkpoint"]))
    trajectories = []
    for start in range(definition["starts"]):
        geometry = pool.geometry(split.evaluation[start])
        run = directory / f"trajectory-{start:03}"
        record = simulate(
            LearnedEvaluator(model, pool.chemistry),
            geometry.numbers,
            geometry.positions,
            thermal_velocities(geometry.numbers, 300, start),
            run,
            SimulationConfig(steps=definition["steps"], seed=start),
            ValidityPolicy(),
            lineage=checkpoint["identity"],
        )
        trajectories.append({"start": start, "record": record, "replay": replay_trajectory(run)})
    result = {
        "dataset": {**manifest, "path": str(path.resolve())},
        "training": result,
        "metrics": metrics,
        "trajectories": trajectories,
        "completed_steps": sum(item["record"]["completed_steps"] for item in trajectories),
        "failed_starts": sum(item["record"]["status"] != "passed" for item in trajectories),
    }
    write_json(directory / "secondary.json", result)
    return result
