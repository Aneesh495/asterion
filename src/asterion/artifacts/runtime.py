"""Verify timing populations, physical work and saved-weight observations independently."""

from dataclasses import replace

import numpy as np
import torch

from asterion.artifacts.derivatives import geometry_from_trace
from asterion.artifacts.io import file_hash, object_hash, read_json, safe_member
from asterion.contracts.errors import IntegrityError
from asterion.contracts.geometry import collate
from asterion.evaluation.replay import replay_trajectory
from asterion.training.identity import state_fingerprint


def require(condition, message):
    if not condition:
        raise IntegrityError(message)


def timing(sample):
    start, end = sample["started_ns"], sample["ended_ns"]
    require(
        type(start) is int and type(end) is int and 0 < start < end,
        "runtime clock observation is missing or reversed",
    )
    elapsed = sample["seconds"]
    require(
        np.isfinite(elapsed)
        and elapsed > 0
        and np.isclose(elapsed, (end - start) * 1e-9, rtol=1e-12, atol=0),
        "runtime duration disagrees with monotonic clock observations",
    )
    return start, end


def model_observation(model, geometries, observation, hessian=False):
    parameter = next(model.parameters())
    batch = collate(geometries, dtype=parameter.dtype, device=parameter.device)
    energy, force = model(batch, create_graph=hessian)
    tolerance = 1e-6 if parameter.dtype == torch.float32 else 1e-9
    for key, actual in (("energy", energy), ("forces", force)):
        retained = np.asarray(observation[key], dtype=float)
        expected = actual.detach().cpu().numpy()
        require(
            retained.shape == expected.shape
            and np.isfinite(retained).all()
            and np.allclose(retained, expected, atol=tolerance, rtol=tolerance * 10),
            "runtime observation differs from executed saved weights",
        )
    if hessian:
        direction = torch.ones_like(force) / np.sqrt(force.numel())
        product = -torch.autograd.grad((force * direction).sum(), batch.positions)[0]
        retained = np.asarray(observation["hessian_product"], dtype=float)
        require(
            retained.shape == product.shape
            and np.isfinite(retained).all()
            and np.allclose(
                retained, product.detach().cpu().numpy(), atol=tolerance, rtol=tolerance * 10
            ),
            "runtime curvature differs from executed saved weights",
        )


def dynamics_observation(directory, sample, geometry, model, fingerprint, steps):
    replay = replay_trajectory(directory)
    metadata = read_json(directory / "trajectory.json")
    config = metadata["config"]
    require(
        config["steps"] == steps
        and config["timestep_fs"] == 0.1
        and config["ensemble"] == "NVE"
        and config["seed"] == sample["repetition"]
        and metadata["lineage"] == fingerprint,
        "runtime dynamics workload or executed-state lineage changed",
    )
    require(
        sample["requested_steps"] == steps
        and sample["completed_steps"] == replay["completed_steps"] <= steps
        and sample["physical_horizon_fs"] == replay["physical_horizon_fs"]
        and sample["status"] == metadata["status"]
        and sample["reason"] == metadata["reason"],
        "runtime completed work or failure differs from raw trajectory",
    )
    require(
        (
            sample["status"] == "passed"
            and replay["completed_steps"] == steps
            and sample["reason"] == "completed"
        )
        or (
            sample["status"] == "failed"
            and isinstance(sample["reason"], str)
            and bool(sample["reason"])
            and replay["completed_steps"] < steps
        ),
        "runtime dynamics status hides incomplete work",
    )
    require(
        np.isfinite(sample["completed_steps_per_second"])
        and np.isclose(
            sample["completed_steps_per_second"],
            replay["completed_steps"] / sample["seconds"],
            atol=1e-12,
            rtol=1e-12,
        ),
        "runtime dynamics throughput uses requested rather than completed work",
    )
    require(
        all(
            sample[key] is True
            for key in ("includes_collation", "includes_neighbor_updates", "includes_serialization")
        ),
        "runtime dynamics omits end-to-end work",
    )
    with np.load(directory / "trajectory.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    require(
        np.array_equal(arrays["numbers"], geometry.numbers)
        and np.array_equal(arrays["positions"][0], geometry.positions)
        and not arrays["uncertainty_available"].any(),
        "runtime dynamics starts from another geometry or changes force provider",
    )
    mass = np.array([{1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999}[int(z)] for z in geometry.numbers])
    velocity = np.random.default_rng(sample["repetition"]).normal(size=geometry.positions.shape)
    velocity *= np.sqrt(1.380649e-23 * 100 / (1.66053906892e-27 * mass))[:, None] / 1e5
    velocity -= np.sum(velocity * mass[:, None], axis=0) / mass.sum()
    observed_temperature = np.sum((1.66053906892e-27 * mass[:, None]) * (velocity * 1e5) ** 2) / (
        (3 * len(mass) - 3) * 1.380649e-23
    )
    velocity *= np.sqrt(100 / observed_temperature)
    require(
        np.allclose(velocity, arrays["velocities"][0], atol=1e-12, rtol=1e-10),
        "runtime dynamics initial temperature or RNG population changed",
    )
    rebuilds, reference = 0, None
    for position, actual in zip(arrays["positions"], arrays["neighbor_rebuilds"], strict=True):
        if reference is None or np.linalg.norm(position - reference, axis=1).max() >= 0.25:
            rebuilds += 1
            reference = position
        require(actual == rebuilds, "runtime neighbor-cache rebuild trace changed")
    for frame in sorted(set(np.linspace(0, replay["completed_steps"], 4, dtype=int))):
        changed = replace(geometry, positions=arrays["positions"][frame])
        model_observation(
            model,
            [changed],
            {"energy": [arrays["potential"][frame]], "forces": arrays["forces"][frame]},
        )
    return replay


def verify_runtime(root, report_path, model, source=None, datasets=None, full=False):
    path = safe_member(root, report_path)
    report = read_json(path)
    require(
        report["format"] == "asterion-runtime-v2"
        and report["timing_clock"] == "perf_counter_ns"
        and report["timing_obligation"] == "energy-plus-coordinate-gradient"
        and report["includes_graph_and_coordinate_gradient"] is True,
        "runtime format or force obligation changed",
    )
    fingerprint = state_fingerprint(model)
    parameter = next(model.parameters())
    require(
        report["executed_state"] == fingerprint
        and report["dtype"] == str(parameter.dtype)
        and object_hash(report["architecture"]) == object_hash(model.config.to_dict())
        and report["device"] in {"cpu", "cuda:0"}
        and type(report["threads"]) is int
        and report["threads"] == 2,
        "runtime precision, architecture or parameter identity changed",
    )
    if source is not None:
        require(report["source"]["digest"] == source["digest"], "runtime source identity changed")
    geometries = [geometry_from_trace(row) for row in report["geometries"]]
    require(
        report["geometry_identities"] == [geometry.identity for geometry in geometries]
        and len(set(report["geometry_identities"])) == len(geometries) > 0,
        "runtime geometry identities are repeated or incomplete",
    )
    if datasets is not None:
        for geometry in geometries:
            molecule, digest, row = geometry.identity.split(":")
            author_path = datasets[f"rmd17_{molecule}.npz"]
            require(file_hash(author_path) == digest, "runtime geometry dataset bytes changed")
            with np.load(author_path, allow_pickle=False) as author:
                require(
                    np.array_equal(author["nuclear_charges"], geometry.numbers)
                    and np.array_equal(author["coords"][int(row)], geometry.positions),
                    "runtime geometry differs from author coordinates",
                )
    repetitions, long_repetitions, steps = (
        report[key] for key in ("repetitions", "long_repetitions", "long_steps")
    )
    require(
        all(type(value) is int and value > 0 for value in (repetitions, long_repetitions, steps)),
        "runtime repetition or work budget is invalid",
    )
    if full:
        require(
            repetitions >= 30 and long_repetitions >= 10 and steps >= 100 and len(geometries) == 8,
            "runtime required workload shortened",
        )
        require(report["checkpoint"] is not None, "runtime trained checkpoint identity is absent")
    modes = {
        "graph",
        "cached_graph",
        "energy_force",
        "force_loss_backward",
        "hessian_vector",
        "host_device_transfer",
    }
    expected = {
        (mode, size, repetition)
        for mode in modes
        for size in (1, 2, 4, 8)
        for repetition in range(repetitions)
    }
    expected |= {
        (mode, 1, repetition)
        for mode in ("long_energy_force", "original_dynamics")
        for repetition in range(long_repetitions)
    }
    observed = set()
    previous_end = 0
    total_steps = 0
    for sample in report["samples"]:
        mode = sample["mode"]
        size = sample["batch_geometries"] if mode in modes else 1
        identity = (mode, size, sample["repetition"])
        require(
            identity in expected and identity not in observed,
            "runtime matrix has missing, extra or repeated work identities",
        )
        observed.add(identity)
        start, end = timing(sample)
        require(start >= previous_end, "runtime samples overlap or change execution order")
        previous_end = end
        require(
            sample["dtype"] == report["dtype"] and sample["device"] == report["device"],
            "runtime sample mixes execution precision or device",
        )
        selected = [geometries[i % len(geometries)] for i in range(size)]
        if mode in modes:
            require(
                sample["atoms"] == sum(len(g.numbers) for g in selected)
                and sample["includes_collation"] is False,
                "runtime batch scaling or timed collation contract changed",
            )
            require(
                type(sample["peak_rss_native"]) is int
                and sample["peak_rss_native"] > 0
                and type(sample["peak_device_allocated_bytes"]) is int
                and sample["peak_device_allocated_bytes"] >= 0
                and (report["device"] != "cpu" or sample["peak_device_allocated_bytes"] == 0),
                "runtime memory observations are invalid",
            )
            if mode in {"graph", "cached_graph"}:
                edges = sum(
                    sum(
                        i != j
                        and np.linalg.norm(g.positions[i] - g.positions[j]) < model.config.cutoff
                        for i in range(len(g.numbers))
                        for j in range(len(g.numbers))
                    )
                    for g in selected
                )
                require(sample["edges"] == edges, "runtime graph edge work differs from geometry")
            else:
                require(sample["edges"] is None, "runtime non-graph mode invents edge timing")
            require(
                sample["cache_rebuilt"] is (sample["repetition"] == 0)
                if mode == "cached_graph"
                else sample["cache_rebuilt"] is None,
                "runtime neighbor reuse or rebuild work changed",
            )
            if mode in {"energy_force", "force_loss_backward", "hessian_vector"}:
                if sample["repetition"] == 0:
                    model_observation(
                        model, selected, sample["observation"], hessian=mode == "hessian_vector"
                    )
                else:
                    require(sample["observation"] is None, "runtime observation retention changed")
            else:
                require(sample["observation"] is None, "runtime transfer or topology claims forces")
        elif mode == "long_energy_force":
            require(
                sample["completed_calls"] == steps and sample["includes_collation"] is True,
                "runtime repeated inference work shortened",
            )
            model_observation(model, [geometries[0]], sample["observation"])
        else:
            require(
                sample["path"] == f"benchmark-dynamics/start-{sample['repetition']:03}",
                "runtime dynamics observation path changed",
            )
            directory = safe_member(path.parent, sample["path"])
            replay = dynamics_observation(
                directory,
                sample,
                geometries[sample["repetition"] % len(geometries)],
                model,
                fingerprint,
                steps,
            )
            total_steps += replay["completed_steps"]
    require(observed == expected, "runtime matrix omits required measured work")
    require(state_fingerprint(model) == fingerprint, "runtime verifier mutated saved weights")
    return {
        "short_samples": len(modes) * 4 * repetitions,
        "long_samples": 2 * long_repetitions,
        "completed_dynamics_steps": total_steps,
    }
