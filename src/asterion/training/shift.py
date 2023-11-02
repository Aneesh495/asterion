"""Train a distinct temporal-block task rather than relabel an author-split model."""

from pathlib import Path

from asterion.artifacts.io import write_json
from asterion.contracts.errors import ContractError
from asterion.data.dataset import load_rmd17
from asterion.data.splits import audit_leakage, guarded_split
from asterion.models.potential import ModelConfig
from asterion.training.campaign import write_learning_intervals
from asterion.training.engine import TrainConfig, evaluate_run, train


def guarded_campaign(config, storage: Path, directory: Path):
    definition = config.get(
        "guarded_shift", {"budget": max(config["budgets"]), "temporal_guard": 100}
    )
    budget, guard = definition["budget"], definition["temporal_guard"]
    if not isinstance(guard, int) or guard < 1 or budget != max(config["budgets"]):
        raise ContractError(
            "guarded comparison must retain the maximum label budget and positive guard"
        )
    outcomes = []
    for molecule in config["molecules"]:
        pool, oracle = load_rmd17(storage / "rmd17" / f"rmd17_{molecule}.npz", molecule)
        for seed in config["seeds"]:
            split = guarded_split(
                pool,
                seed,
                train_size=1000,
                evaluation_size=config["evaluation_size"],
                validation_size=config["validation_size"],
                calibration_size=config["calibration_size"],
                guard=guard,
            )
            audit = audit_leakage(pool, split)
            for kind in config["models"]:
                name = f"{kind}-{molecule}-{budget}-{seed}"
                run = directory / name
                result = train(
                    oracle,
                    split,
                    run,
                    ModelConfig(kind=kind, **config["architecture"]),
                    TrainConfig(
                        seed=seed,
                        budget=budget,
                        epochs=config["epochs"],
                        batch_size=config["batch_size"],
                        **config["training"],
                    ),
                )
                metrics = evaluate_run(oracle, split, run, config["batch_size"])
                write_learning_intervals(pool, split, run, metrics)
                write_json(run / "leakage.json", audit)
                outcomes.append(
                    {
                        "molecule": molecule,
                        "model": kind,
                        "seed": seed,
                        "budget": budget,
                        "path": name,
                        "result": result,
                        "metrics": metrics,
                        "protocol": split.protocol,
                        "temporal_guard": guard,
                    }
                )
                write_json(directory / "guarded-units.json", outcomes)
    return outcomes
