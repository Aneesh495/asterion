from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from asterion.artifacts.io import read_json, write_json
from asterion.contracts.errors import ContractError
from asterion.data.dataset import load_rmd17
from asterion.data.splits import Split, audit_leakage, author_split, guarded_split
from asterion.models.potential import ModelConfig
from asterion.training.engine import TrainConfig, evaluate_run, train


def validate_campaign(config):
    required = {
        "profile",
        "molecules",
        "models",
        "seeds",
        "budgets",
        "epochs",
        "batch_size",
        "architecture",
        "training",
        "simulation",
    }
    if not required <= config.keys() or config["profile"] not in {"diagnostic", "full"}:
        raise ContractError("invalid campaign schema or profile")
    if not config["molecules"] or not set(config["molecules"]) <= {
        "aspirin",
        "ethanol",
        "malonaldehyde",
        "salicylic",
    }:
        raise ContractError("invalid primary molecule population")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ContractError("seeds must be independent unique identities")
    if not config["budgets"] or any(not 1 <= budget <= 1000 for budget in config["budgets"]):
        raise ContractError("primary labels cannot exceed 1000 structures")
    if config["profile"] == "full":
        if (
            set(config["molecules"]) != {"aspirin", "ethanol", "malonaldehyde", "salicylic"}
            or config["budgets"] != [100, 300, 1000]
            or len(config["seeds"]) != 3
        ):
            raise ContractError("full campaign population is fixed")
        from asterion.artifacts.source import project_root

        frozen = read_json(project_root() / "configs/full.json")
        observed = deepcopy(config)
        # Interpreter location is host-specific. Scientific and resource budgets are not.
        observed.get("reference", {}).pop("python", None)
        frozen.get("reference", {}).pop("python", None)
        if observed != frozen:
            raise ContractError("full workload differs from the frozen research configuration")
    return config


def load_split(path):
    value = read_json(path)
    digest = value.pop("digest")
    for key in ("train_pool", "validation", "calibration", "evaluation"):
        value[key] = tuple(value[key])
    split = Split(**value)
    if split.digest != digest:
        raise ContractError("persisted split digest invalid")
    return split


def unit_name(molecule, kind, budget, seed):
    return f"{kind}-{molecule}-{budget}-{seed}"


def train_campaign(config, storage: Path, directory: Path, resume=False):
    validate_campaign(config)
    outcomes = []
    for molecule in config["molecules"]:
        pool, oracle = load_rmd17(storage / "rmd17" / f"rmd17_{molecule}.npz", molecule)
        for seed in config["seeds"]:
            split = author_split(
                pool,
                storage / "rmd17",
                seed,
                config["evaluation_size"],
                config["validation_size"],
                config["calibration_size"],
            )
            for kind in config["models"]:
                for budget in config["budgets"]:
                    name = unit_name(molecule, kind, budget, seed)
                    run = directory / name
                    if resume and (run / "completed.json").exists():
                        result = read_json(run / "completed.json")
                        # Completion is not permission to ignore changed inputs.
                        frozen = read_json(run / "config.json")
                        if (
                            frozen["dataset"] != pool.dataset_hash
                            or frozen["training"]["epochs"] != config["epochs"]
                        ):
                            raise ContractError(
                                "completed run cannot be reused with changed data/workload"
                            )
                    else:
                        checkpoint = None
                        if resume and (run / "latest.json").exists():
                            checkpoint = run / read_json(run / "latest.json")["checkpoint"]
                        architecture = ModelConfig(kind=kind, **config["architecture"])
                        training = TrainConfig(
                            seed=seed,
                            budget=budget,
                            epochs=config["epochs"],
                            batch_size=config["batch_size"],
                            **config["training"],
                        )
                        print(f"training {name}", flush=True)
                        result = train(
                            oracle, split, run, architecture, training, resume=checkpoint
                        )
                    outcomes.append(
                        {
                            "molecule": molecule,
                            "model": kind,
                            "seed": seed,
                            "budget": budget,
                            "path": name,
                            "result": result,
                        }
                    )
                    write_json(directory / "training-units.json", outcomes)
    return outcomes


def evaluate_campaign(config, storage: Path, directory: Path):
    outcomes = []
    for entry in read_json(directory / "training-units.json"):
        molecule = entry["molecule"]
        pool, oracle = load_rmd17(storage / "rmd17" / f"rmd17_{molecule}.npz", molecule)
        run = directory / entry["path"]
        split = load_split(run / "split.json")
        result = evaluate_run(oracle, split, run, config["batch_size"])
        audit = audit_leakage(pool, split)
        write_json(run / "leakage.json", audit)
        from asterion.evaluation.metrics import paired_block_interval

        blocks = pool.original_indices[list(split.evaluation)] // 1000
        comparisons = {}
        for baseline in ("zero_force", "untrained"):
            comparisons[baseline] = paired_block_interval(
                result[baseline]["geometry_force_mae"],
                result["trained"]["geometry_force_mae"],
                blocks,
            )
        write_json(
            run / "learning-paired.json",
            {
                "temporal_block_span_original_indices": 1000,
                "geometry_count": len(split.evaluation),
                "comparisons": comparisons,
                "assumption": "temporal blocks are a declared correlation heuristic, not a proof of independence",
            },
        )
        # Shift evaluation excludes every role used for fitting, selection and calibration.
        forbidden = (
            set(split.train_pool)
            | set(split.validation)
            | set(split.calibration)
            | set(split.evaluation)
        )
        frozen = guarded_split(
            pool,
            entry["seed"],
            evaluation_size=config["evaluation_size"] + len(forbidden),
            validation_size=config["validation_size"],
            calibration_size=config["calibration_size"],
        )
        shifted = tuple(row for row in frozen.evaluation if row not in forbidden)[
            : config["evaluation_size"]
        ]
        if len(shifted) != config["evaluation_size"]:
            raise ContractError("guarded shift cannot meet the frozen distinct evaluation count")
        stress = replace(split, protocol="temporal-held-out-shift", evaluation=shifted)
        stress.write(run / "shift-split.json")
        # Preserve the primary artifact while a separately named evaluation uses the same trained weights.
        from asterion.artifacts.io import atomic_npz
        from asterion.training.checkpoint import load_checkpoint
        from asterion.training.engine import predict

        model, _ = load_checkpoint(run / read_json(run / "selection.json")["checkpoint"])
        labels = oracle.labels(shifted, shifted, "evaluation")
        arrays, metrics = predict(model, labels, config["batch_size"])
        atomic_npz(run / "shift-predictions.npz", **arrays)
        write_json(run / "shift-metrics.json", {"split": stress.digest, "metrics": metrics})
        outcomes.append({**entry, "evaluation": result, "shift": metrics})
        write_json(directory / "evaluation-units.json", outcomes)
    return outcomes


def ablation_campaign(config, storage: Path, directory: Path):
    variants: dict[str, dict] = {
        "lmax0": {"lmax": 0},
        "lmax1": {"lmax": 1},
        "no-higher-order": {"higher_order": False},
        "hard-cutoff": {"smooth": False},
        "energy-only": {},
    }
    outcomes = []
    core = config["molecules"][:2]
    if config["profile"] == "full" and len(core) != 2:
        raise ContractError("full ablations require two molecules")
    for molecule in core:
        pool, oracle = load_rmd17(storage / "rmd17" / f"rmd17_{molecule}.npz", molecule)
        for seed in config["seeds"]:
            split = author_split(
                pool,
                storage / "rmd17",
                seed,
                config["evaluation_size"],
                config["validation_size"],
                config["calibration_size"],
            )
            for name, changes in variants.items():
                architecture = ModelConfig(**{**config["architecture"], **changes})
                train_options = {**config["training"]}
                if name == "energy-only":
                    train_options["force_weight"] = 0.0
                training = TrainConfig(
                    seed=seed,
                    budget=config["budgets"][-1],
                    epochs=config["epochs"],
                    batch_size=config["batch_size"],
                    **train_options,
                )
                run = directory / f"{name}-{molecule}-{seed}"
                result = train(oracle, split, run, architecture, training)
                metrics = evaluate_run(oracle, split, run, config["batch_size"])
                outcomes.append(
                    {
                        "variant": name,
                        "molecule": molecule,
                        "seed": seed,
                        "result": result,
                        "metrics": metrics,
                    }
                )
                write_json(directory / "ablation-units.json", outcomes)
    return outcomes
