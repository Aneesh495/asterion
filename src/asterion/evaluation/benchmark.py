import resource
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from asterion.artifacts.io import write_json
from asterion.contracts.errors import ContractError, IntegrityError
from asterion.contracts.geometry import collate
from asterion.graphs.radius import NeighborCache


def synchronize(model):
    if next(model.parameters()).is_cuda:
        torch.cuda.synchronize()


def benchmark(
    model,
    geometries,
    output: Path,
    repetitions=30,
    long_repetitions=10,
    long_steps=100,
    checkpoint_identity=None,
    source=None,
):
    if not geometries or any(
        type(value) is not int or value < 1 for value in (repetitions, long_repetitions, long_steps)
    ):
        raise ContractError(
            "benchmark requires nonempty geometries and positive repetition/work counts"
        )
    for geometry in geometries:
        geometry.validate()
        if len(geometry.numbers) < 2:
            raise ContractError("dynamics benchmarking requires at least two atoms")
    if len({geometry.identity for geometry in geometries}) != len(geometries):
        raise ContractError("benchmark geometry identities must be unique")
    if output.exists() or (output.parent / "benchmark-dynamics").exists():
        raise IntegrityError("benchmark observation identity already exists")
    from asterion.training.identity import state_fingerprint

    fingerprint = state_fingerprint(model)
    parameter = next(model.parameters())
    torch.set_num_threads(2)
    samples = []
    cache = NeighborCache(model.config.cutoff)
    for batch_size in (1, 2, 4, 8):
        selected = [geometries[i % len(geometries)] for i in range(batch_size)]
        for mode in (
            "graph",
            "cached_graph",
            "energy_force",
            "force_loss_backward",
            "hessian_vector",
            "host_device_transfer",
        ):
            for repetition in range(repetitions):
                batch = collate(selected, dtype=parameter.dtype, device=parameter.device)
                batch.positions.requires_grad_(True)
                model.zero_grad(set_to_none=True)
                previous_rebuilds = cache.rebuilds
                if parameter.is_cuda:
                    torch.cuda.reset_peak_memory_stats(parameter.device)
                synchronize(model)
                start = time.perf_counter_ns()
                observation = None
                if mode == "graph":
                    edges = model.graph_builder(batch)
                elif mode == "cached_graph":
                    edges = cache(batch)
                elif mode == "host_device_transfer":
                    # CPU clone and an explicit return copy expose transfer work on either backend.
                    copied = batch.positions.detach().cpu().clone().to(parameter.device)
                    returned = copied.cpu()
                    edges = None
                else:
                    energy, force = model(
                        batch, create_graph=mode in {"force_loss_backward", "hessian_vector"}
                    )
                    if mode == "force_loss_backward":
                        (force.square().mean() + energy.square().mean() * 1e-6).backward()
                    if mode == "hessian_vector":
                        direction = torch.ones_like(force) / np.sqrt(force.numel())
                        product = -torch.autograd.grad((force * direction).sum(), batch.positions)[
                            0
                        ]
                    edges = None
                synchronize(model)
                ended = time.perf_counter_ns()
                elapsed = (ended - start) * 1e-9
                if mode == "host_device_transfer":
                    if not torch.equal(returned, batch.positions.detach().cpu()):
                        raise IntegrityError("host/device transfer changed coordinates")
                elif edges is None:
                    if not torch.isfinite(energy).all() or not torch.isfinite(force).all():
                        raise IntegrityError("nonfinite timed model output")
                    if mode == "hessian_vector" and not torch.isfinite(product).all():
                        raise IntegrityError("nonfinite timed Hessian product")
                    if repetition == 0:
                        observation = {
                            "energy": energy.detach().cpu().tolist(),
                            "forces": force.detach().cpu().tolist(),
                        }
                        if mode == "hessian_vector":
                            observation["hessian_product"] = product.detach().cpu().tolist()
                samples.append(
                    {
                        "mode": mode,
                        "batch_geometries": batch_size,
                        "atoms": len(batch.numbers),
                        "edges": edges.size if edges is not None else None,
                        "repetition": repetition,
                        "seconds": elapsed,
                        "started_ns": start,
                        "ended_ns": ended,
                        "dtype": str(parameter.dtype),
                        "device": str(parameter.device),
                        "peak_rss_native": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        "peak_device_allocated_bytes": torch.cuda.max_memory_allocated(
                            parameter.device
                        )
                        if parameter.is_cuda
                        else 0,
                        "cache_rebuilt": cache.rebuilds != previous_rebuilds
                        if mode == "cached_graph"
                        else None,
                        "includes_collation": False,
                        "observation": observation,
                    }
                )
    for repetition in range(long_repetitions):
        synchronize(model)
        start = time.perf_counter_ns()
        for _ in range(long_steps):
            energy, force = model(
                collate([geometries[0]], dtype=parameter.dtype, device=parameter.device)
            )
        synchronize(model)
        ended = time.perf_counter_ns()
        samples.append(
            {
                "mode": "long_energy_force",
                "repetition": repetition,
                "completed_calls": long_steps,
                "seconds": (ended - start) * 1e-9,
                "started_ns": start,
                "ended_ns": ended,
                "includes_collation": True,
                "dtype": str(parameter.dtype),
                "device": str(parameter.device),
                "observation": {
                    "energy": energy.detach().cpu().tolist(),
                    "forces": force.detach().cpu().tolist(),
                },
            }
        )
    from asterion.contracts.units import thermal_velocities
    from asterion.dynamics.detectors import ValidityPolicy
    from asterion.dynamics.integrator import SimulationConfig, simulate
    from asterion.dynamics.learned import LearnedEvaluator
    from asterion.evaluation.replay import replay_trajectory

    for repetition in range(long_repetitions):
        geometry = geometries[repetition % len(geometries)]
        directory = output.parent / "benchmark-dynamics" / f"start-{repetition:03}"
        synchronize(model)
        started = time.perf_counter_ns()
        observed = simulate(
            LearnedEvaluator(model, geometry.chemistry),
            geometry.numbers,
            geometry.positions,
            thermal_velocities(geometry.numbers, 100, repetition),
            directory,
            SimulationConfig(steps=long_steps, timestep_fs=0.1, seed=repetition),
            ValidityPolicy(),
            lineage=fingerprint,
        )
        synchronize(model)
        ended = time.perf_counter_ns()
        elapsed = (ended - started) * 1e-9
        replay = replay_trajectory(directory)
        samples.append(
            {
                "mode": "original_dynamics",
                "repetition": repetition,
                "seconds": elapsed,
                "started_ns": started,
                "ended_ns": ended,
                "requested_steps": long_steps,
                "completed_steps": replay["completed_steps"],
                "physical_horizon_fs": replay["physical_horizon_fs"],
                "completed_steps_per_second": replay["completed_steps"] / elapsed,
                "status": observed["status"],
                "reason": observed["reason"],
                "path": str(directory.relative_to(output.parent)),
                "includes_collation": True,
                "includes_neighbor_updates": True,
                "includes_serialization": True,
                "dtype": str(parameter.dtype),
                "device": str(parameter.device),
            }
        )
    if state_fingerprint(model) != fingerprint:
        raise IntegrityError("benchmark changed model parameters")
    result = {
        "format": "asterion-runtime-v2",
        "samples": samples,
        "repetitions": repetitions,
        "long_repetitions": long_repetitions,
        "long_steps": long_steps,
        "includes_graph_and_coordinate_gradient": True,
        "executed_state": fingerprint,
        "checkpoint": checkpoint_identity,
        "source": source,
        "architecture": asdict(model.config),
        "threads": torch.get_num_threads(),
        "dtype": str(parameter.dtype),
        "device": str(parameter.device),
        "timing_clock": "perf_counter_ns",
        "timing_obligation": "energy-plus-coordinate-gradient",
        "geometry_identities": [geometry.identity for geometry in geometries],
        "geometries": [
            {
                "identity": geometry.identity,
                "numbers": geometry.numbers.tolist(),
                "positions": geometry.positions.tolist(),
                "chemistry": asdict(geometry.chemistry),
            }
            for geometry in geometries
        ],
    }
    write_json(output, result)
    return result


def memory_stress(model, geometry, output: Path, seconds=1800):
    samples = []
    started = time.monotonic()
    completed = 0
    last_sample = started
    parameter = next(model.parameters())
    while time.monotonic() - started < seconds:
        batch = collate([geometry], dtype=parameter.dtype, device=parameter.device)
        energy, force = model(batch)
        if not torch.isfinite(energy).all() or not torch.isfinite(force).all():
            raise RuntimeError("nonfinite stress inference")
        completed += 1
        now = time.monotonic()
        if now - last_sample >= 5:
            samples.append(
                {
                    "elapsed_seconds": now - started,
                    "completed_calls": completed,
                    "peak_rss_native": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                }
            )
            last_sample = now
    result = {
        "requested_seconds": seconds,
        "elapsed_seconds": time.monotonic() - started,
        "completed_calls": completed,
        "samples": samples,
    }
    write_json(output, result)
    return result
