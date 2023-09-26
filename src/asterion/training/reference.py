import os
import subprocess
from pathlib import Path

import numpy as np

from asterion.artifacts.io import atomic_npz, file_hash, write_json
from asterion.contracts.errors import ResourceError
from asterion.contracts.units import SYMBOLS
from asterion.data.dataset import load_rmd17
from asterion.data.extxyz import reference_arrays
from asterion.data.splits import author_split
from asterion.evaluation.metrics import prediction_metrics


def write_xyz(path, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for label in labels:
            stream.write(f"{len(label.geometry.numbers)}\n")
            stream.write(
                f"Properties=species:S:1:pos:R:3:REF_forces:R:3 REF_energy={label.energy:.17g} "
                f'asterion_id="{label.geometry.identity}" pbc="F F F"\n'
            )
            for z, position, force in zip(
                label.geometry.numbers, label.geometry.positions, label.forces
            ):
                values = " ".join(f"{value:.17g}" for value in np.r_[position, force])
                stream.write(f"{SYMBOLS[int(z)]} {values}\n")


def reference_campaign(config, storage: Path, directory: Path):
    definition = config["reference"]
    executable = Path(definition.get("python", ".reference-venv/bin/python")).resolve()
    if not executable.is_file():
        raise ResourceError(
            "reference requires a separate pinned MACE environment; see docs/REFERENCE.md"
        )
    probe = subprocess.run(
        [
            str(executable),
            "-c",
            "import importlib.metadata; print(importlib.metadata.version('mace-torch'))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    if probe.stdout.strip() != definition["version"]:
        raise ResourceError("MACE reference version differs from frozen comparison definition")
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
            run = directory / f"{molecule}-{seed}"
            if run.exists():
                raise ResourceError("reference comparison run already exists")
            run.mkdir(parents=True)
            split.write(run / "split.json")
            rows = np.random.default_rng(seed).permutation(split.train_pool)[:1000]
            for role, selected in (
                ("train", rows),
                ("validation", split.validation),
                ("evaluation", split.evaluation),
            ):
                write_xyz(run / f"{role}.xyz", oracle.labels(selected, selected, role))
            args = [
                str(executable),
                "-m",
                "mace.cli.run_train",
                f"--name={molecule}-{seed}",
                "--model=MACE",
                f"--train_file={run / 'train.xyz'}",
                f"--valid_file={run / 'validation.xyz'}",
                f"--test_file={run / 'evaluation.xyz'}",
                f"--seed={seed}",
                "--energy_key=REF_energy",
                "--forces_key=REF_forces",
                "--E0s=average",
                f"--r_max={config['architecture']['cutoff']}",
                "--num_channels=24",
                "--max_L=2",
                "--correlation=3",
                "--num_interactions=2",
                "--batch_size=8",
                "--valid_batch_size=8",
                "--max_num_epochs=400",
                "--device=cuda",
                "--default_dtype=float32",
                f"--model_dir={run / 'models'}",
                f"--checkpoints_dir={run / 'checkpoints'}",
                f"--log_dir={run / 'logs'}",
            ]
            with (run / "stdout.txt").open("w") as stdout, (run / "stderr.txt").open("w") as stderr:
                completed = subprocess.run(args, stdout=stdout, stderr=stderr, timeout=72 * 3600)
            outcome = {
                "molecule": molecule,
                "seed": seed,
                "split": split.digest,
                "dataset": pool.dataset_hash,
                "origin": "independent-reference-retraining",
                "implementation": "mace-torch",
                "version": definition["version"],
                "command": args,
                "exit": completed.returncode,
                "inputs": {
                    role: file_hash(run / f"{role}.xyz")
                    for role in ("train", "validation", "evaluation")
                },
            }
            outcomes.append(outcome)
            write_json(directory / "reference-units.json", outcomes)
            if completed.returncode:
                raise ResourceError(f"MACE reference training failed: {molecule}/{seed}")
            exported_model = run / "models" / f"{molecule}-{seed}.model"
            if not exported_model.is_file() or exported_model.is_symlink():
                raise ResourceError("MACE standard final export is absent")
            evaluation_args = [
                str(executable),
                "-m",
                "mace.cli.eval_configs",
                f"--model={exported_model}",
                f"--configs={run / 'evaluation.xyz'}",
                f"--output={run / 'predictions.xyz'}",
                "--info_prefix=PRED_",
                "--device=cuda",
                "--default_dtype=float32",
                "--batch_size=8",
            ]
            # MACE's own trusted export is a full Torch module. Only this isolated
            # reference process loads it; Asterion checkpoint loading remains pickle-free.
            with (
                (run / "evaluation.stdout").open("w") as stdout,
                (run / "evaluation.stderr").open("w") as stderr,
            ):
                evaluation_process = subprocess.run(
                    evaluation_args,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=12 * 3600,
                    env={**os.environ, "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"},
                )
            if evaluation_process.returncode:
                raise ResourceError("MACE retained-output evaluation failed")
            arrays = reference_arrays(run / "predictions.xyz")
            if arrays["identities"].tolist() != [
                pool.geometry(row).identity for row in split.evaluation
            ]:
                raise ResourceError("MACE evaluation changed geometry identities")
            atomic_npz(run / "evaluation-predictions.npz", **arrays)
            metrics = prediction_metrics(
                arrays["energy"],
                arrays["force"],
                arrays["target_energy"],
                arrays["target_force"],
                arrays["pointers"],
            )
            write_json(run / "evaluation-metrics.json", metrics)
            outcome.update(
                path=str(run.relative_to(directory)),
                budget=1000,
                training_rows=rows.tolist(),
                model_path=str(exported_model.relative_to(run)),
                model_sha256=file_hash(exported_model),
                evaluation_command=evaluation_args,
                evaluation_exit=evaluation_process.returncode,
                predictions_sha256=file_hash(run / "evaluation-predictions.npz"),
                selection="standard-final-export",
                metrics=metrics,
            )
            write_json(directory / "reference-units.json", outcomes)
    return outcomes
