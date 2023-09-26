import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

from asterion.artifacts.io import atomic_npz, object_hash, read_json, write_json
from asterion.artifacts.source import source_inputs
from asterion.contracts.errors import ContractError, IntegrityError, ResourceError
from asterion.contracts.geometry import collate
from asterion.data.ledger import RevealLedger, fit_offsets
from asterion.evaluation.metrics import prediction_metrics
from asterion.models.potential import ModelConfig, make_model
from asterion.training.checkpoint import load_checkpoint, restore_optimizer, save_checkpoint


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 1
    budget: int = 100
    epochs: int = 80
    batch_size: int = 8
    learning_rate: float = 0.003
    energy_weight: float = 0.1
    force_weight: float = 1.0
    accumulation: int = 1
    clip_norm: float = 100.0
    checkpoint_every: int = 10
    threads: int = 2
    dtype: str = "float64"
    device: str = "cpu"

    def validate(self):
        if (
            not 1 <= self.budget <= 1000
            or min(
                self.epochs, self.batch_size, self.accumulation, self.checkpoint_every, self.threads
            )
            < 1
        ):
            raise ContractError("invalid training size")
        if (
            min(self.energy_weight, self.force_weight) < 0
            or self.energy_weight + self.force_weight == 0
        ):
            raise ContractError("positive training loss weight required")
        if self.dtype not in {"float64", "float32"} or self.device not in {"cpu", "cuda"}:
            raise ContractError("training supports audited CPU/CUDA precision contracts")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ResourceError("declared CUDA training device is unavailable")
        return self


def predict(model, labels, batch_size=8):
    if not labels:
        raise ContractError("empty prediction set")
    parameter = next(model.parameters())
    energies, forces, targets, target_forces, pointers = [], [], [], [], [0]
    model.eval()
    for start in range(0, len(labels), batch_size):
        selected = labels[start : start + batch_size]
        batch = collate(selected, dtype=parameter.dtype, device=parameter.device)
        assert batch.energy is not None and batch.forces is not None
        energy, force = model(batch)
        energies.append(energy.detach().cpu().numpy())
        forces.append(force.detach().cpu().numpy())
        targets.append(batch.energy.cpu().numpy())
        target_forces.append(batch.forces.cpu().numpy())
        for label in selected:
            pointers.append(pointers[-1] + len(label.geometry.numbers))
    arrays = {
        "energy": np.concatenate(energies),
        "force": np.concatenate(forces),
        "target_energy": np.concatenate(targets),
        "target_force": np.concatenate(target_forces),
        "pointers": np.array(pointers),
        "identities": np.array([label.geometry.identity for label in labels]),
        "numbers": np.concatenate([label.geometry.numbers for label in labels]),
        "positions": np.concatenate([label.geometry.positions for label in labels]),
    }
    return arrays, prediction_metrics(
        arrays["energy"],
        arrays["force"],
        arrays["target_energy"],
        arrays["target_force"],
        arrays["pointers"],
    )


def train(
    oracle,
    split,
    directory: Path,
    architecture: ModelConfig,
    config: TrainConfig,
    ledger: RevealLedger | None = None,
    resume: Path | None = None,
    stop_after: int | None = None,
):
    config.validate()
    split.validate(len(oracle.pool))
    if split.dataset_hash != oracle.pool.dataset_hash:
        raise IntegrityError("training split belongs to another dataset")
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(config.device == "cpu")
    dtype = torch.float64 if config.dtype == "float64" else torch.float32
    if directory.exists() and resume is None:
        raise IntegrityError(
            "training run already exists; choose a new identity or explicit resume"
        )
    directory.mkdir(parents=True, exist_ok=True)
    ledger = ledger or RevealLedger(split, directory / "reveal")
    if not ledger.revealed:
        rows = np.random.default_rng(config.seed).permutation(split.train_pool)[: config.budget]
        ledger.reveal(rows, "fixed-label-budget")
    if len(ledger.revealed) != config.budget:
        raise ContractError("training budget disagrees with revealed labels")
    labels = oracle.labels(ledger.revealed, ledger.revealed, "train")
    validation = oracle.labels(split.validation, split.validation, "validation")
    normalization = fit_offsets(labels)
    architecture = replace(
        architecture,
        energy_offset=normalization["energy_per_atom"],
        energy_scale=normalization["energy_scale"],
        level=oracle.pool.chemistry.level,
        composition=tuple(
            int(np.count_nonzero(oracle.pool.numbers == z)) for z in architecture.elements
        ),
    )
    source = source_inputs()
    lineage = {
        "split": split.digest,
        "dataset": oracle.pool.dataset_hash,
        "reveal": ledger.digest,
        "training": asdict(config),
        "architecture": architecture.to_dict(),
        "source": source["digest"],
    }
    lineage_hash = object_hash(lineage)
    history = []
    completed = 0
    updates = 0
    best = float("inf")
    selected_path = None
    selected_epoch = -1
    torch.manual_seed(config.seed)
    model = make_model(architecture).to(device=config.device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    if resume is not None:
        model, record = load_checkpoint(resume, device=config.device, dtype=dtype)
        if record["lineage"] != lineage_hash:
            raise IntegrityError("resume source/data/config/reveal lineage mismatch")
        restore_optimizer(
            resume,
            optimizer := torch.optim.Adam(model.parameters(), lr=config.learning_rate),
            record,
        )
        completed, updates = record["epoch"], record["updates"]
        history = read_json(directory / "history.json")[:completed]
        selection = read_json(directory / "selection.json")
        best, selected_epoch = selection["score"], selection["epoch"]
        selected_path = directory / selection["checkpoint"]
    else:
        split.write(directory / "split.json")
        local_reveal = directory / "reveal"
        local_reveal.mkdir(parents=True, exist_ok=True)
        for round_record in ledger.rounds:
            write_json(local_reveal / f"round-{round_record['round']:04}.json", round_record)
        write_json(directory / "config.json", lineage)
        write_json(directory / "SOURCE_INPUTS.json", source)
        write_json(directory / "normalization.json", normalization)
        save_checkpoint(
            directory / "initial",
            model,
            optimizer,
            {
                "origin": "original-random-initialization",
                "lineage": lineage_hash,
                "epoch": 0,
                "updates": 0,
                "examples_processed": 0,
            },
        )
        initial_arrays, initial_metrics = predict(model, validation, config.batch_size)
        atomic_npz(directory / "initial-validation.npz", **initial_arrays)
        write_json(directory / "initial-validation.json", initial_metrics)
    initial_weights = {k: v.detach().clone() for k, v in model.named_parameters()}
    for epoch in range(completed, config.epochs):
        model.train()
        factor = 0.1 + 0.9 * (1 + np.cos(np.pi * epoch / config.epochs)) / 2
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = config.learning_rate * factor
        started = time.perf_counter()
        order = np.random.default_rng(config.seed * 1000003 + epoch).permutation(len(labels))
        totals = {"energy_loss": 0.0, "force_loss": 0.0, "force_gradient_squared": 0.0}
        for group in range(0, len(order), config.batch_size * config.accumulation):
            optimizer.zero_grad(set_to_none=True)
            group_order = order[group : group + config.batch_size * config.accumulation]
            for start in range(0, len(group_order), config.batch_size):
                indices = group_order[start : start + config.batch_size]
                batch = collate([labels[i] for i in indices], dtype=dtype, device=config.device)
                energy, force = model(batch, create_graph=True)
                energy_loss = (
                    ((energy - batch.energy) / normalization["energy_scale"]).square().mean()
                )
                force_loss = ((force - batch.forces) / normalization["force_scale"]).square().mean()
                loss = config.energy_weight * energy_loss + config.force_weight * force_loss
                if not torch.isfinite(loss):
                    raise ContractError("nonfinite training loss")
                if epoch == 0 and group == 0 and start == 0 and config.force_weight > 0:
                    mixed = torch.autograd.grad(
                        force_loss, tuple(model.parameters()), allow_unused=True, retain_graph=True
                    )
                    totals["force_gradient_squared"] = sum(
                        g.square().sum().item() for g in mixed if g is not None
                    )
                    if totals["force_gradient_squared"] <= 1e-20:
                        raise ContractError("force loss does not reach energy model parameters")
                (loss * (len(indices) / len(group_order))).backward()
                totals["energy_loss"] += energy_loss.item() * len(indices) / len(labels)
                totals["force_loss"] += force_loss.item() * len(indices) / len(labels)
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm)
            if not torch.isfinite(gradient):
                raise ContractError("nonfinite parameter gradient")
            optimizer.step()
            updates += 1
        _, metrics = predict(model, validation, config.batch_size)
        score = metrics["force_mae_eV_angstrom"]
        improved = score < best
        if improved:
            best, selected_epoch = score, epoch + 1
        record = {
            "epoch": epoch + 1,
            "updates": updates,
            "examples_processed": (epoch + 1) * len(labels),
            "unique_training_geometries": len(labels),
            "validation_force_mae": score,
            "validation_energy_mae": metrics["energy_mae_eV"],
            "selected": improved,
            "elapsed_seconds": time.perf_counter() - started,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **totals,
        }
        history.append(record)
        write_json(directory / "history.json", history)
        need_checkpoint = (
            improved or (epoch + 1) % config.checkpoint_every == 0 or epoch + 1 == config.epochs
        )
        if stop_after is not None and epoch + 1 >= stop_after:
            need_checkpoint = True
        if need_checkpoint:
            checkpoint = directory / f"epoch-{epoch + 1:05}"
            metadata = {
                "origin": "original-force-loss-training",
                "lineage": lineage_hash,
                "epoch": epoch + 1,
                "updates": updates,
                "examples_processed": (epoch + 1) * len(labels),
                "unique_training_geometries": len(labels),
                "seed": config.seed,
                "parameter_change_norm": float(
                    sum(
                        (p - initial_weights[k]).square().sum().item()
                        for k, p in model.named_parameters()
                    )
                    ** 0.5
                ),
            }
            save_checkpoint(checkpoint, model, optimizer, metadata)
            write_json(
                directory / "latest.json", {"checkpoint": checkpoint.name, "epoch": epoch + 1}
            )
            if improved:
                selected_path = checkpoint
                write_json(
                    directory / "selection.json",
                    {
                        "checkpoint": checkpoint.name,
                        "epoch": epoch + 1,
                        "score": score,
                        "rule": "minimum-validation-force-mae",
                        "lineage": lineage_hash,
                    },
                )
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"epoch={epoch + 1} updates={updates} validation_force_mae={score:.6f}", flush=True
            )
        if stop_after is not None and epoch + 1 >= stop_after:
            return {"status": "interrupted", "checkpoint": str(checkpoint), "epochs": epoch + 1}
    if selected_path is None:
        raise IntegrityError("no selected checkpoint")
    write_json(
        directory / "completed.json",
        {
            "status": "passed",
            "epochs": config.epochs,
            "updates": updates,
            "examples_processed": config.epochs * len(labels),
            "lineage": lineage_hash,
            "selected_epoch": selected_epoch,
            "selected_checkpoint": selected_path.name,
        },
    )
    return {
        "status": "passed",
        "checkpoint": str(selected_path),
        "epochs": config.epochs,
        "updates": updates,
        "lineage": lineage_hash,
    }


def evaluate_run(oracle, split, directory: Path, batch_size=8, role="evaluation"):
    rows = getattr(split, role)
    labels = oracle.labels(rows, rows, role)
    selection = read_json(directory / "selection.json")
    model, checkpoint = load_checkpoint(directory / selection["checkpoint"])
    initial, _ = load_checkpoint(directory / "initial")
    arrays, metrics = predict(model, labels, batch_size)
    initial_arrays, initial_metrics = predict(initial, labels, batch_size)
    normalization = read_json(directory / "normalization.json")
    training_rows = [
        int(identity.rsplit(":", 1)[1]) for identity in normalization["fit_identities"]
    ]
    training_labels = oracle.labels(training_rows, training_rows, "train")
    baseline_energy = np.mean([label.energy for label in training_labels])
    mean_force = np.mean(np.stack([label.forces for label in training_labels]), axis=0)
    mean_prediction = np.tile(mean_force, (len(labels), 1))
    zero = prediction_metrics(
        np.repeat(baseline_energy, len(labels)),
        np.zeros_like(arrays["target_force"]),
        arrays["target_energy"],
        arrays["target_force"],
        arrays["pointers"],
    )
    atomic_npz(directory / f"{role}-predictions.npz", **arrays)
    atomic_npz(directory / f"{role}-untrained.npz", **initial_arrays)
    mean_metrics = prediction_metrics(
        np.repeat(baseline_energy, len(labels)),
        mean_prediction,
        arrays["target_energy"],
        arrays["target_force"],
        arrays["pointers"],
    )
    atomic_npz(
        directory / f"{role}-mean-force.npz",
        force=mean_prediction,
        energy=np.repeat(baseline_energy, len(labels)),
        identities=arrays["identities"],
    )
    result = {
        "trained": metrics,
        "untrained": initial_metrics,
        "zero_force": zero,
        "mean_force": mean_metrics,
        "checkpoint": checkpoint["identity"],
        "split": split.digest,
        "relative_improvement_zero": 1
        - metrics["force_mae_eV_angstrom"] / zero["force_mae_eV_angstrom"],
    }
    write_json(directory / f"{role}-metrics.json", result)
    return result
