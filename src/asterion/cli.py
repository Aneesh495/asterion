import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from asterion import __version__
from asterion.artifacts.bundle import CommandJournal, build_diagnostic
from asterion.artifacts.controls import controls
from asterion.artifacts.io import read_json, write_json
from asterion.artifacts.source import environment, project_root, source_inputs
from asterion.contracts.errors import AsterionError, ContractError, IntegrityError
from asterion.contracts.units import thermal_velocities
from asterion.data.dataset import load_rmd17
from asterion.data.release import acquire_rmd17
from asterion.dynamics.detectors import ValidityPolicy
from asterion.dynamics.integrator import SimulationConfig, simulate
from asterion.dynamics.learned import LearnedEvaluator
from asterion.evaluation.replay import replay_trajectory
from asterion.models.potential import ModelConfig, make_model
from asterion.training.campaign import (
    evaluate_campaign,
    load_split,
    train_campaign,
    validate_campaign,
)
from asterion.training.checkpoint import load_checkpoint

COMMANDS = (
    "doctor",
    "build",
    "data",
    "pilot",
    "train",
    "evaluate",
    "sample-efficiency",
    "acquire",
    "ablate",
    "simulate",
    "bench",
    "stress",
    "docs",
    "demo",
    "diagnostic",
    "acceptance",
    "verify",
    "controls",
    "release",
    "properties",
    "derivatives",
    "analytic",
    "secondary",
    "reference",
    "guarded-shift",
)


def selected_model(runs, device="cpu", dtype=torch.float64):
    units = read_json(runs / "training-units.json")
    entry = next(entry for entry in units if entry["model"] == "equivariant")
    run = runs / entry["path"]
    selection = read_json(run / "selection.json")
    return entry, run, *load_checkpoint(run / selection["checkpoint"], device=device, dtype=dtype)


def simulate_campaign(config, storage: Path, runs: Path):
    outcomes = []
    for entry in read_json(runs / "training-units.json"):
        if entry["model"] != "equivariant" or entry["budget"] != max(config["budgets"]):
            continue
        run = runs / entry["path"]
        model, checkpoint = load_checkpoint(run / read_json(run / "selection.json")["checkpoint"])
        pool, _ = load_rmd17(
            storage / "rmd17" / f"rmd17_{entry['molecule']}.npz", entry["molecule"]
        )
        split = load_split(run / "split.json")
        definition = config["simulation"]
        for kelvin in definition["temperatures_K"]:
            for start in range(definition["starts_per_temperature"]):
                geometry = pool.geometry(split.evaluation[start])
                name = f"{entry['molecule']}-{entry['seed']}-{kelvin}-{start}"
                path = runs / "simulation" / name
                record = simulate(
                    LearnedEvaluator(model),
                    geometry.numbers,
                    geometry.positions,
                    thermal_velocities(geometry.numbers, kelvin, entry["seed"] * 1000 + start),
                    path,
                    SimulationConfig(
                        steps=definition["steps"],
                        timestep_fs=definition["timestep_fs"],
                        temperature_K=kelvin,
                        seed=entry["seed"],
                    ),
                    ValidityPolicy(),
                    lineage=checkpoint["identity"],
                )
                outcomes.append(
                    {
                        "path": str(path.relative_to(runs)),
                        "molecule": entry["molecule"],
                        "seed": entry["seed"],
                        "temperature_K": kelvin,
                        "start": start,
                        "record": record,
                        "replay": replay_trajectory(path),
                    }
                )
                write_json(runs / "simulation-units.json", outcomes)
    if "dynamics_study" in config:
        from asterion.training.dynamics_study import dynamics_study

        dynamics_study(config, storage, runs)
    return outcomes


def demo(config, storage, runs):
    from asterion.viewer.export import export_viewer

    entry, run, model, checkpoint = selected_model(runs)
    simulations = read_json(runs / "simulation-units.json")
    trajectory = runs / simulations[0]["path"]
    output = runs / "viewer.html"
    export_viewer(
        run / "evaluation-predictions.npz",
        trajectory,
        output,
        {
            "source": read_json(run / "SOURCE_INPUTS.json"),
            "checkpoint": checkpoint["identity"],
            "split": read_json(run / "split.json")["digest"],
            "run": runs.name,
            "molecule": entry["molecule"],
        },
    )
    # A real fail-closed model evaluation uses an extreme-force validity policy.
    pool, _ = load_rmd17(storage / "rmd17" / f"rmd17_{entry['molecule']}.npz", entry["molecule"])
    split = load_split(run / "split.json")
    geometry = pool.geometry(split.evaluation[0])
    failure = runs / "demo-failure"
    result = simulate(
        LearnedEvaluator(model),
        geometry.numbers,
        geometry.positions,
        np.zeros_like(geometry.positions),
        failure,
        SimulationConfig(steps=50),
        ValidityPolicy(maximum_force=0.00001),
        lineage=checkpoint["identity"],
    )
    if result["status"] != "failed" or result["reason"] != "extreme-force":
        raise IntegrityError("demo failure intervention did not activate the force detector")
    export_viewer(
        run / "evaluation-predictions.npz",
        failure,
        runs / "failure-viewer.html",
        {
            "checkpoint": checkpoint["identity"],
            "intervention": "strict-force-limit",
            "run": runs.name,
        },
    )
    return {
        "viewer": str(output),
        "failure_viewer": str(runs / "failure-viewer.html"),
        "failure": result,
    }


def doctor(config, storage):
    from asterion.contracts.geometry import Geometry, collate

    torch.set_num_threads(2)
    model = make_model(ModelConfig(width=4, depth=1, radial=4)).double()
    geometry = Geometry(np.array([6, 1]), np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), "doctor")
    e, f = model(collate([geometry]), create_graph=True)
    f.square().sum().backward()
    gradients = sum(p.grad.square().sum().item() for p in model.parameters() if p.grad is not None)
    result = {
        **environment(),
        "version": __version__,
        "cpu_float64_double_backward": gradients > 0,
        "energy_finite": bool(torch.isfinite(e).all()),
        "full_device_required": config["training"]["device"],
        "rmd17_files_present": [p.name for p in (storage / "rmd17").glob("*.npz")],
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        result["cuda_memory_bytes"] = properties.total_memory
        result["cuda_name"] = properties.name
    return result


def full_acceptance(config, storage, runs, evidence, config_path, resume=False):
    from asterion.artifacts.campaign import execute_research

    if config["profile"] != "full":
        raise ContractError(
            "make acceptance requires configs/full.json; use diagnostic for bounded evidence"
        )
    return execute_research(config, config_path, storage, runs, evidence, resume)


def dispatch(args):
    root = project_root()
    config = validate_campaign(read_json(args.config))
    storage = args.storage.resolve()
    runs = root / "runs" / args.run_id
    if args.command == "verify":
        from asterion.artifacts.verify import verify

        return verify(args.evidence, args.profile or config["profile"], candidate=args.candidate)
    if args.command == "controls":
        return controls(
            args.evidence,
            runs / "controls.json",
            args.profile or config["profile"],
            candidate=args.candidate,
        )
    if args.command == "doctor":
        return doctor(config, storage)
    if args.command == "data":
        primary = acquire_rmd17(root / "configs/rmd17-release-v4.json", storage)
        if "secondary" in config:
            from asterion.data.secondary import acquire_md22

            acquire_md22(root / "configs/md22-release.json", storage)
        return primary
    if args.command == "build":
        subprocess.run(["uv", "build", "--offline"], cwd=root, check=True)
        return {"status": "passed", "directory": str(root / "dist")}
    if args.command == "docs":
        from asterion.evaluation.documentation import check_docs

        return check_docs(root)
    if args.command == "acceptance":
        return full_acceptance(config, storage, runs, args.evidence, args.config, args.resume)
    if args.command == "release":
        from asterion.artifacts.verify import verify

        verify(args.evidence, "full")
        subprocess.run(["uv", "build", "--offline"], cwd=root, check=True)
        return {"status": "passed", "source": source_inputs()["digest"]}
    journal = CommandJournal(runs / "COMMANDS.jsonl")
    if args.command == "pilot":
        from asterion.evaluation.resource_pilot import resource_pilot

        return journal.execute("pilot", lambda: resource_pilot(config, storage, runs / "pilot"))
    if args.command in {"train", "sample-efficiency"}:
        return journal.execute("train", lambda: train_campaign(config, storage, runs, args.resume))
    if args.command == "evaluate":
        return journal.execute("evaluate", lambda: evaluate_campaign(config, storage, runs))
    if args.command == "simulate":
        return journal.execute("simulate", lambda: simulate_campaign(config, storage, runs))
    if args.command == "demo":
        return journal.execute("demo", lambda: demo(config, storage, runs))
    if args.command == "acquire":
        from asterion.acquisition.campaign import acquisition_campaign

        return journal.execute(
            "acquire", lambda: acquisition_campaign(config, storage, runs / "acquisition")
        )
    if args.command == "ablate":
        from asterion.training.campaign import ablation_campaign

        return journal.execute(
            "ablate", lambda: ablation_campaign(config, storage, runs / "ablations")
        )
    if args.command == "guarded-shift":
        from asterion.training.shift import guarded_campaign

        return journal.execute(
            "guarded-shift", lambda: guarded_campaign(config, storage, runs / "guarded-shift")
        )
    if args.command == "secondary":
        from asterion.training.secondary import secondary_campaign

        return journal.execute(
            "secondary", lambda: secondary_campaign(config, storage, runs / "secondary")
        )
    if args.command == "reference":
        from asterion.training.reference import reference_campaign

        return journal.execute(
            "reference", lambda: reference_campaign(config, storage, runs / "reference")
        )
    if args.command == "properties":
        from asterion.evaluation.property_campaign import graph_cases

        return journal.execute(
            "properties", lambda: {"completed": len(graph_cases(runs / "properties.json"))}
        )
    if args.command == "analytic":
        from asterion.evaluation.analytic_campaign import analytic_campaign

        return journal.execute(
            "analytic", lambda: {"completed": len(analytic_campaign(runs / "analytic"))}
        )
    execution = (
        config["training"] if args.command == "bench" else {"device": "cpu", "dtype": "float64"}
    )
    entry, run, model, checkpoint = selected_model(
        runs, device=execution["device"], dtype=getattr(torch, execution["dtype"])
    )
    pool, _ = load_rmd17(storage / "rmd17" / f"rmd17_{entry['molecule']}.npz", entry["molecule"])
    split = load_split(run / "split.json")
    geometries = [pool.geometry(row) for row in split.evaluation[:8]]
    if args.command == "derivatives":
        from asterion.evaluation.property_campaign import derivative_campaign

        return journal.execute(
            "derivatives",
            lambda: derivative_campaign(
                model,
                geometries,
                runs / "derivatives",
                checkpoint_identity=checkpoint["identity"],
                source=source_inputs(project_root()),
            ),
        )
    if args.command == "bench":
        from asterion.evaluation.benchmark import benchmark

        result = journal.execute(
            "bench",
            lambda: benchmark(
                model,
                geometries,
                runs / "benchmark.json",
                checkpoint_identity=checkpoint["identity"],
                source=source_inputs(project_root()),
            ),
        )
        return {"raw_samples": len(result["samples"]), "artifact": str(runs / "benchmark.json")}
    if args.command == "stress":
        from asterion.evaluation.benchmark import memory_stress
        from asterion.evaluation.recovery import fault_cases, simulation_recovery, training_recovery

        journal.execute(
            "simulation-recovery", lambda: simulation_recovery(runs / "simulation-recovery")
        )
        journal.execute("training-recovery", lambda: training_recovery(runs / "training-recovery"))
        journal.execute("faults", lambda: fault_cases(runs / "faults"))
        return journal.execute(
            "memory-stress",
            lambda: memory_stress(model, geometries[0], runs / "memory-stress.json"),
        )
    if args.command == "diagnostic":
        return journal.execute(
            "bundle", lambda: build_diagnostic(config, storage, runs, args.evidence)
        )
    raise ContractError(f"unsupported command {args.command}")


def main():
    parser = argparse.ArgumentParser(description="Asterion molecular research commands")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--config", type=Path, default=Path("configs/diagnostic.json"))
    parser.add_argument("--storage", type=Path, default=Path("storage"))
    parser.add_argument("--run-id", default="diagnostic")
    parser.add_argument("--evidence", type=Path, default=Path("evidence/diagnostic"))
    parser.add_argument("--profile", choices=("full", "diagnostic"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--candidate",
        action="store_true",
        help="check a full candidate without granting release status",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}", args.run_id):
        parser.error("run-id must be a bounded simple identifier")
    try:
        result = dispatch(args)
        import json

        print(json.dumps(result, sort_keys=True, allow_nan=False, default=str))
    except (
        AsterionError,
        FileNotFoundError,
        KeyError,
        ValueError,
        AssertionError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"asterion: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
