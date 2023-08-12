import resource
import time
from pathlib import Path

import torch

from asterion.artifacts.io import write_json
from asterion.contracts.geometry import collate
from asterion.graphs.radius import NeighborCache


def synchronize(model):
    if next(model.parameters()).is_cuda:
        torch.cuda.synchronize()


def benchmark(model, geometries, output: Path, repetitions=30, long_repetitions=10, long_steps=100):
    parameter = next(model.parameters())
    torch.set_num_threads(2)
    samples = []
    cache = NeighborCache(model.config.cutoff)
    for batch_size in (1, 2, 4, 8):
        selected = [geometries[i % len(geometries)] for i in range(batch_size)]
        for mode in ("graph", "cached_graph", "energy_force", "force_loss_backward"):
            for repetition in range(repetitions):
                batch = collate(selected, dtype=parameter.dtype, device=parameter.device)
                batch.positions.requires_grad_(True)
                model.zero_grad(set_to_none=True)
                synchronize(model)
                start = time.perf_counter_ns()
                if mode == "graph":
                    edges = model.graph_builder(batch)
                elif mode == "cached_graph":
                    edges = cache(batch)
                else:
                    energy, force = model(batch, create_graph=mode == "force_loss_backward")
                    if mode == "force_loss_backward":
                        (force.square().mean() + energy.square().mean() * 1e-6).backward()
                    edges = None
                synchronize(model)
                elapsed = (time.perf_counter_ns() - start) * 1e-9
                samples.append(
                    {
                        "mode": mode,
                        "batch_geometries": batch_size,
                        "atoms": len(batch.numbers),
                        "edges": edges.size if edges is not None else None,
                        "repetition": repetition,
                        "seconds": elapsed,
                        "dtype": str(parameter.dtype),
                        "device": str(parameter.device),
                        "peak_rss_native": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    }
                )
    for repetition in range(long_repetitions):
        synchronize(model)
        start = time.perf_counter_ns()
        for _ in range(long_steps):
            model(collate([geometries[0]], dtype=parameter.dtype, device=parameter.device))
        synchronize(model)
        samples.append(
            {
                "mode": "long_energy_force",
                "repetition": repetition,
                "completed_calls": long_steps,
                "seconds": (time.perf_counter_ns() - start) * 1e-9,
            }
        )
    result = {
        "samples": samples,
        "repetitions": repetitions,
        "long_repetitions": long_repetitions,
        "long_steps": long_steps,
        "includes_graph_and_coordinate_gradient": True,
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
