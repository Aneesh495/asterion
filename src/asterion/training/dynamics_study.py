"""Execute separate NVT and matched-horizon studies with selected original weights."""

import time

import numpy as np

from asterion.artifacts.io import read_json, safe_member, write_json
from asterion.artifacts.source import source_inputs
from asterion.contracts.errors import ContractError
from asterion.contracts.units import thermal_velocities
from asterion.data.dataset import load_rmd17
from asterion.dynamics.integrator import SimulationConfig, simulate
from asterion.dynamics.learned import LearnedEvaluator
from asterion.evaluation.dynamics_statistics import (
    analyze_trajectory,
    matched_timestep_observations,
)
from asterion.training.campaign import load_split
from asterion.training.checkpoint import load_checkpoint
from asterion.training.identity import state_fingerprint


def study_protocol(config):
    specification = config["dynamics_study"]
    steps = specification["steps"]
    refinement = specification["refinement_steps"]
    timesteps = specification["timesteps_fs"]
    temperature = specification["temperature_K"]
    molecules = specification["molecules"]
    if (
        any(type(value) is not int or value < 1 for value in (steps, refinement))
        or len(timesteps) != 3
        or not all(np.isfinite(dt) and dt > 0 for dt in timesteps)
        or not np.allclose(
            np.asarray(timesteps[:-1]) / np.asarray(timesteps[1:]), 2, atol=1e-12, rtol=0
        )
        or not np.isfinite(temperature)
        or temperature <= 0
        or not molecules
        or len(set(molecules)) != len(molecules)
        or not set(molecules) <= set(config["molecules"])
    ):
        raise ContractError("invalid separate dynamics-study protocol")
    return specification


def dynamics_study(config, storage, runs):
    protocol = study_protocol(config)
    root = runs / "simulation/studies"
    root.mkdir(parents=True)
    report = {
        "format": "asterion-learned-dynamics-study-v1",
        "source": source_inputs(),
        "protocol": protocol,
        "force_dtype": "float64",
        "force_device": "cpu",
        "execution_status": "running",
        "cases": [],
        "refinements": [],
    }
    entries = [
        entry
        for entry in read_json(runs / "training-units.json")
        if entry["model"] == "equivariant"
        and entry["budget"] == max(config["budgets"])
        and entry["molecule"] in protocol["molecules"]
    ]
    expected = {(molecule, seed) for molecule in protocol["molecules"] for seed in config["seeds"]}
    if {(entry["molecule"], entry["seed"]) for entry in entries} != expected or len(entries) != len(
        expected
    ):
        raise ContractError("dynamics study selected-model population is incomplete")
    for entry in entries:
        training = safe_member(runs, entry["path"])
        model, checkpoint = load_checkpoint(
            safe_member(training, read_json(training / "selection.json")["checkpoint"])
        )
        pool, _ = load_rmd17(
            storage / "rmd17" / f"rmd17_{entry['molecule']}.npz", entry["molecule"]
        )
        geometry = pool.geometry(load_split(training / "split.json").evaluation[0])
        velocities = thermal_velocities(
            geometry.numbers, protocol["temperature_K"], entry["seed"] * 1000
        )
        refinements = []
        trials = [("NVT", protocol["timesteps_fs"][0], protocol["steps"])]
        trials.extend(
            ("NVE", dt, protocol["refinement_steps"] * 2**index)
            for index, dt in enumerate(protocol["timesteps_fs"])
        )
        for ensemble, timestep, steps in trials:
            relative = f"{entry['molecule']}-{entry['seed']}/{ensemble.lower()}-{timestep:.3f}"
            directory = root / relative
            started = time.perf_counter_ns()
            observed = simulate(
                LearnedEvaluator(model, pool.chemistry),
                geometry.numbers,
                geometry.positions,
                velocities,
                directory,
                SimulationConfig(
                    steps=steps,
                    timestep_fs=timestep,
                    ensemble=ensemble,
                    temperature_K=protocol["temperature_K"],
                    seed=entry["seed"],
                ),
                lineage=checkpoint["identity"],
            )
            ended = time.perf_counter_ns()
            statistics = analyze_trajectory(directory)
            report["cases"].append(
                {
                    "molecule": entry["molecule"],
                    "seed": entry["seed"],
                    "ensemble": ensemble,
                    "timestep_fs": timestep,
                    "path": relative,
                    "checkpoint": checkpoint["identity"],
                    "execution_state": state_fingerprint(model),
                    "initial_geometry": geometry.identity,
                    "started_ns": started,
                    "ended_ns": ended,
                    "seconds": (ended - started) * 1e-9,
                    "record": observed,
                    "statistics": statistics,
                }
            )
            if ensemble == "NVE":
                refinements.append(directory)
            write_json(root / "study.json", report)
        report["refinements"].append(
            {
                "molecule": entry["molecule"],
                "seed": entry["seed"],
                "observation": matched_timestep_observations(refinements),
            }
        )
        write_json(root / "study.json", report)
    report["execution_status"] = "completed"
    write_json(root / "study.json", report)
    return report
