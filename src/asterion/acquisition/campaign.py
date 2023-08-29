from dataclasses import asdict
from pathlib import Path

import numpy as np

from asterion.acquisition.policy import choose
from asterion.artifacts.io import atomic_npz, object_hash, write_json
from asterion.data.dataset import load_rmd17
from asterion.data.ledger import RevealLedger
from asterion.data.splits import author_split, geometry_descriptors
from asterion.evaluation.metrics import uncertainty_metrics
from asterion.models.potential import ModelConfig
from asterion.training.checkpoint import load_checkpoint
from asterion.training.engine import TrainConfig, evaluate_run, train
from asterion.uncertainty.committee import Committee


def acquisition_campaign(config, storage: Path, directory: Path):
    outcomes = []
    for molecule in config["acquisition_molecules"]:
        pool, oracle = load_rmd17(storage / "rmd17" / f"rmd17_{molecule}.npz", molecule)
        for trial in config["seeds"]:
            split = author_split(
                pool,
                storage / "rmd17",
                trial,
                config["evaluation_size"],
                config["validation_size"],
                config["calibration_size"],
            )
            descriptors = geometry_descriptors(pool, split.train_pool)
            budgets = config["acquisition_budgets"]
            initial = np.random.default_rng(trial).permutation(split.train_pool)[: budgets[0]]
            for policy in ("uniform", "diversity", "uncertainty-diversity"):
                run = directory / f"{molecule}-{trial}-{policy}"
                ledger = RevealLedger(split, run / "reveal")
                ledger.reveal(initial, "shared-initial-pool")
                committee = None
                disagreement = None
                for round_index, budget in enumerate(budgets):
                    if round_index:
                        if committee is None or disagreement is None:
                            raise RuntimeError("acquisition round lacks prior trained committee")
                        choices = choose(
                            split.train_pool,
                            descriptors,
                            ledger.revealed,
                            budget - len(ledger.revealed),
                            policy,
                            trial * 100 + round_index,
                            uncertainty=disagreement.force_score,
                        )
                        candidate = {
                            "pool": list(split.train_pool),
                            "scores": disagreement.force_score.tolist(),
                            "choices": asdict(choices),
                            "committee": committee.identities,
                            "split": split.digest,
                            "previous_reveal": ledger.digest,
                        }
                        write_json(run / f"candidates-{round_index}.json", candidate)
                        ledger.reveal(
                            choices.rows, policy, committee.identities, object_hash(candidate)
                        )
                    models = []
                    identities = []
                    metrics = []
                    for member in (1, 2, 3):
                        path = run / f"round-{round_index}-member-{member}"
                        training = TrainConfig(
                            seed=trial * 100 + member,
                            budget=budget,
                            epochs=config["epochs"],
                            batch_size=config["batch_size"],
                            **config["training"],
                        )
                        result = train(
                            oracle,
                            split,
                            path,
                            ModelConfig(**config["architecture"]),
                            training,
                            ledger=ledger,
                        )
                        model, checkpoint = load_checkpoint(Path(result["checkpoint"]))
                        models.append(model)
                        identities.append(checkpoint["identity"])
                        metrics.append(evaluate_run(oracle, split, path, config["batch_size"]))
                    committee = Committee(
                        models,
                        identities,
                        [trial * 100 + i for i in (1, 2, 3)],
                        forbidden_rows=set(ledger.revealed) | set(split.validation),
                    )
                    calibration_labels = oracle.labels(
                        split.calibration, split.calibration, "calibration"
                    )
                    calibration, calibration_prediction, calibration_errors = committee.calibrate(
                        calibration_labels
                    )
                    write_json(run / f"calibration-{round_index}.json", asdict(calibration))
                    atomic_npz(
                        run / f"calibration-{round_index}.npz",
                        identities=calibration_prediction.identities,
                        scores=calibration_prediction.force_score,
                        errors=calibration_errors,
                        bounds=calibration.bounds(calibration_prediction.force_score),
                    )
                    evaluation = oracle.labels(split.evaluation, split.evaluation, "evaluation")
                    output = committee.infer([label.geometry for label in evaluation])
                    pointers = np.r_[
                        0, np.cumsum([len(label.geometry.numbers) for label in evaluation])
                    ]
                    target = np.concatenate([label.forces for label in evaluation])
                    error = np.array(
                        [
                            np.abs(output.forces_mean[a:b] - target[a:b]).mean()
                            for a, b in zip(pointers[:-1], pointers[1:])
                        ]
                    )
                    uncertainty = uncertainty_metrics(
                        output.force_score, error, calibration.bounds(output.force_score)
                    )
                    atomic_npz(
                        run / f"uncertainty-evaluation-{round_index}.npz",
                        identities=output.identities,
                        scores=output.force_score,
                        errors=error,
                        bounds=calibration.bounds(output.force_score),
                        mean_force=output.forces_mean,
                        target_force=target,
                        component_std=output.force_component_std,
                        pointers=pointers,
                    )
                    disagreement = committee.infer([pool.geometry(row) for row in split.train_pool])
                    outcomes.append(
                        {
                            "molecule": molecule,
                            "trial": trial,
                            "policy": policy,
                            "round": round_index,
                            "budget": budget,
                            "reveal": ledger.digest,
                            "metrics": metrics,
                            "uncertainty": uncertainty,
                            "completed_labels": len(ledger.revealed),
                        }
                    )
                    write_json(directory / "acquisition-units.json", outcomes)
    return outcomes
