"""Measure original gradient mechanisms on real coordinates before research training."""

import resource
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from asterion.artifacts.io import read_json, write_json
from asterion.artifacts.source import environment, source_inputs
from asterion.contracts.errors import ContractError, IntegrityError
from asterion.contracts.geometry import collate
from asterion.data.dataset import load_md22, load_rmd17
from asterion.models.potential import ModelConfig, make_model
from asterion.training.checkpoint import save_checkpoint
from asterion.training.identity import state_fingerprint


def resource_pilot(config, storage: Path, directory: Path, repetitions=3):
    if type(repetitions) is not int or repetitions < 3:
        raise ContractError("resource pilot requires at least three genuine repetitions")
    if directory.exists():
        raise IntegrityError("resource pilot observation identity already exists")
    directory.mkdir(parents=True)
    execution = config["training"]
    device = execution["device"]
    dtype = getattr(torch, execution["dtype"])
    if device == "cuda" and not torch.cuda.is_available():
        raise ContractError("configured CUDA pilot backend is unavailable")
    systems = []
    system_paths = {}
    for molecule in config["molecules"]:
        pool, _ = load_rmd17(storage / "rmd17" / f"rmd17_{molecule}.npz", molecule)
        systems.append((molecule, [pool.geometry(row) for row in range(8)]))
        system_paths[molecule] = storage / "rmd17" / f"rmd17_{molecule}.npz"
    if "secondary" in config:
        manifest = read_json(Path("configs/md22-release.json"))
        pool, _ = load_md22(
            storage / "md22" / manifest["filename"],
            manifest["name"],
            manifest["sha256"],
            manifest["level"],
        )
        systems.append((manifest["name"], [pool.geometry(row) for row in range(8)]))
        system_paths[manifest["name"]] = storage / "md22" / manifest["filename"]
    report = {
        "format": "asterion-resource-pilot-v1",
        "source": source_inputs(),
        "environment": environment(),
        "repetitions": repetitions,
        "systems": [],
        "samples": [],
        "status": "running",
        "objective": "quadratic current energy/force predictions; gradient mechanism only",
        "supervised_labels_consumed": 0,
        "training_or_accuracy_claim": False,
        "full_campaign_hours": None,
        "cost_forecast_status": "unavailable; reference, validation and I/O not projected",
    }
    write_json(directory / "pilot.json", report)
    try:
        for name, geometries in systems:
            for kind in config["models"]:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(491)
                    architecture = replace(
                        ModelConfig(kind=kind, **config["architecture"]),
                        level=geometries[0].chemistry.level,
                    )
                    model = make_model(architecture).to(device=device, dtype=dtype)
                path = f"{name}-{kind}"
                checkpoint = save_checkpoint(
                    directory / path,
                    model,
                    None,
                    {"origin": "original-resource-pilot-initialization", "seed": 491},
                )
                fingerprint = state_fingerprint(model)
                report["systems"].append(
                    {
                        "name": name,
                        "kind": kind,
                        "checkpoint": checkpoint["identity"],
                        "dataset_path": str(system_paths[name].resolve()),
                        "path": path,
                        "architecture": asdict(architecture),
                        "executed_state": fingerprint,
                        "geometries": [
                            {
                                "identity": g.identity,
                                "numbers": g.numbers.tolist(),
                                "positions": g.positions.tolist(),
                                "chemistry": asdict(g.chemistry),
                            }
                            for g in geometries
                        ],
                    }
                )
                for size in (1, 8):
                    for mode in ("energy-force", "force-loss-backward"):
                        for repetition in range(repetitions):
                            batch = collate(geometries[:size], dtype=dtype, device=device)
                            model.zero_grad(set_to_none=True)
                            if next(model.parameters()).is_cuda:
                                torch.cuda.synchronize()
                                torch.cuda.reset_peak_memory_stats()
                            started = time.perf_counter_ns()
                            energy, force = model(batch, create_graph=mode == "force-loss-backward")
                            if mode == "force-loss-backward":
                                (force.square().mean() + energy.square().mean() * 1e-6).backward()
                            if next(model.parameters()).is_cuda:
                                torch.cuda.synchronize()
                            ended = time.perf_counter_ns()
                            if not torch.isfinite(energy).all() or not torch.isfinite(force).all():
                                raise ContractError("nonfinite original resource-pilot prediction")
                            gradients = [
                                parameter.grad
                                for parameter in model.parameters()
                                if parameter.grad is not None
                            ]
                            if any(not torch.isfinite(value).all() for value in gradients):
                                raise ContractError("nonfinite original force-loss pilot gradient")
                            report["samples"].append(
                                {
                                    "name": name,
                                    "kind": kind,
                                    "mode": mode,
                                    "batch_geometries": size,
                                    "atoms": len(batch.numbers),
                                    "repetition": repetition,
                                    "started_ns": started,
                                    "ended_ns": ended,
                                    "seconds": (ended - started) * 1e-9,
                                    "dtype": str(dtype),
                                    "device": str(next(model.parameters()).device),
                                    "native_peak_rss": resource.getrusage(
                                        resource.RUSAGE_SELF
                                    ).ru_maxrss,
                                    "peak_device_bytes": torch.cuda.max_memory_allocated()
                                    if next(model.parameters()).is_cuda
                                    else 0,
                                    "energy": energy.detach().cpu().tolist()
                                    if repetition == 0
                                    else None,
                                    "forces": force.detach().cpu().tolist()
                                    if repetition == 0
                                    else None,
                                    "gradient_squared_norm": sum(
                                        float(value.detach().double().square().sum().cpu())
                                        for value in gradients
                                    )
                                    if mode == "force-loss-backward"
                                    else None,
                                    "collation_timed": False,
                                    "optimizer_update": False,
                                }
                            )
                            write_json(directory / "pilot.json", report)
                if state_fingerprint(model) != fingerprint:
                    raise IntegrityError("pilot changed original parameter state")
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        write_json(directory / "pilot.json", report)
        raise
    write_json(directory / "pilot.json", report)
    return {
        "status": report["status"],
        "samples": len(report["samples"]),
        "artifact": str(directory / "pilot.json"),
    }
