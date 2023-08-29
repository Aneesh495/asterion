from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from asterion.artifacts.io import atomic_npz, file_hash, object_hash, read_json, write_json
from asterion.contracts.errors import AsterionError, ContractError, IntegrityError
from asterion.contracts.units import (
    ACCELERATION,
    BOLTZMANN_EV_K,
    kinetic_energy,
    masses,
    temperature,
)
from asterion.dynamics.analytic import ForceResult
from asterion.dynamics.detectors import Detector, ValidityPolicy


@dataclass(frozen=True)
class SimulationConfig:
    steps: int = 250
    timestep_fs: float = 0.1
    ensemble: str = "NVE"
    temperature_K: float = 300.0
    friction_per_fs: float = 0.01
    seed: int = 1
    checkpoint_every: int = 50

    def validate(self):
        if self.steps < 1 or not np.isfinite(self.timestep_fs) or self.timestep_fs <= 0:
            raise ContractError("positive steps and timestep required")
        if (
            self.ensemble not in {"NVE", "NVT"}
            or self.temperature_K < 0
            or self.friction_per_fs <= 0
        ):
            raise ContractError("invalid ensemble, temperature or friction")
        return self


@dataclass
class State:
    positions: np.ndarray
    velocities: np.ndarray
    forces: np.ndarray
    energy: float
    step: int = 0
    heat_eV: float = 0.0


def verlet(state: State, result: ForceResult, mass, dt) -> State:
    acceleration = ACCELERATION * state.forces / mass[:, None]
    positions = state.positions + dt * state.velocities + 0.5 * dt**2 * acceleration
    velocities = (
        state.velocities + 0.5 * dt * ACCELERATION * (state.forces + result.forces) / mass[:, None]
    )
    return State(
        positions, velocities, result.forces.copy(), result.energy, state.step + 1, state.heat_eV
    )


def snapshot(
    state: State, numbers, uncertainty=0.0, rebuilds=0, noise=None, uncertainty_available=False
):
    mass = masses(numbers)
    return {
        "positions": state.positions.copy(),
        "velocities": state.velocities.copy(),
        "forces": state.forces.copy(),
        "potential": state.energy,
        "kinetic": kinetic_energy(state.velocities, mass),
        "step": state.step,
        "heat": state.heat_eV,
        "uncertainty": uncertainty,
        "uncertainty_available": uncertainty_available,
        "neighbor_rebuilds": rebuilds,
        "temperature": temperature(state.velocities, mass) if len(numbers) > 1 else 0.0,
        "noise": np.zeros_like(state.positions) if noise is None else noise.copy(),
    }


def write_trajectory(
    directory: Path, frames, numbers, config, detector, lineage, status, reason, attempted=None
):
    arrays = {key: np.asarray([frame[key] for frame in frames]) for key in frames[0]}
    arrays["numbers"] = np.asarray(numbers)
    atomic_npz(directory / "trajectory.npz", **arrays)
    if attempted is not None:
        atomic_npz(directory / "failure.npz", positions=attempted)
    completed = len(frames) - 1
    metadata = {
        "config": asdict(config),
        "lineage": lineage,
        "detector": detector.to_dict(),
        "completed_steps": completed,
        "requested_steps": config.steps,
        "physical_horizon_fs": completed * config.timestep_fs,
        "status": status,
        "reason": reason,
        "trajectory_sha256": file_hash(directory / "trajectory.npz"),
        "initial_identity": object_hash(
            {"numbers": list(map(int, numbers)), "positions": frames[0]["positions"].tolist()}
        ),
    }
    if attempted is not None:
        metadata["failure_sha256"] = file_hash(directory / "failure.npz")
    write_json(directory / "trajectory.json", metadata)
    return metadata


def simulate(
    evaluator,
    numbers,
    positions,
    velocities,
    directory: Path,
    config: SimulationConfig,
    policy=ValidityPolicy(),
    lineage="analytic",
    resume=False,
    stop_after: int | None = None,
):
    config.validate()
    numbers = np.asarray(numbers, dtype=np.int64)
    mass = masses(numbers)
    detector = Detector(numbers, positions, policy)
    if directory.exists() and not resume:
        raise IntegrityError("simulation run collision")
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)
    frames = []
    state = None
    if resume:
        checkpoint = read_json(directory / "resume.json")
        if checkpoint["config"] != asdict(config) or checkpoint["lineage"] != lineage:
            raise IntegrityError("simulation resume lineage changed")
        path = directory / checkpoint["arrays"]
        if file_hash(path) != checkpoint["sha256"]:
            raise IntegrityError("simulation checkpoint arrays changed")
        with np.load(path, allow_pickle=False) as data:
            for i in range(len(data["step"])):
                frames.append({key: data[key][i] for key in data.files})
        last = frames[-1]
        state = State(
            last["positions"],
            last["velocities"],
            last["forces"],
            float(last["potential"]),
            int(last["step"]),
            float(last["heat"]),
        )
        rng.bit_generator.state = checkpoint["rng"]
    else:
        if velocities.shape != positions.shape or not np.isfinite(velocities).all():
            raise ContractError("finite velocity axes required")
        reason = detector.geometry(positions)
        if reason:
            raise ContractError(f"invalid initial state: {reason}")
        initial = evaluator(positions, numbers)
        state = State(positions.copy(), velocities.copy(), initial.forces.copy(), initial.energy)
        frames.append(
            snapshot(
                state,
                numbers,
                initial.uncertainty,
                initial.neighbor_rebuilds,
                uncertainty_available=initial.uncertainty_available,
            )
        )
        reason = detector.forces(initial)
        if reason:
            return write_trajectory(
                directory, frames, numbers, config, detector, lineage, "failed", reason, positions
            )
    dt = config.timestep_fs
    while state.step < config.steps:
        noise = None
        heat = state.heat_eV
        if config.ensemble == "NVE":
            candidate = (
                state.positions
                + dt * state.velocities
                + 0.5 * dt**2 * ACCELERATION * state.forces / mass[:, None]
            )
        else:
            # BAOAB: kick, half drift, exact Ornstein-Uhlenbeck velocity update, half drift, kick.
            half_velocity = (
                state.velocities + 0.5 * dt * ACCELERATION * state.forces / mass[:, None]
            )
            middle_position = state.positions + 0.5 * dt * half_velocity
            damping = np.exp(-config.friction_per_fs * dt)
            sigma = np.sqrt(
                (1 - damping**2) * BOLTZMANN_EV_K * config.temperature_K * ACCELERATION / mass
            )
            noise = rng.normal(size=half_velocity.shape)
            noisy_velocity = damping * half_velocity + sigma[:, None] * noise
            heat += kinetic_energy(noisy_velocity, mass) - kinetic_energy(half_velocity, mass)
            candidate = middle_position + 0.5 * dt * noisy_velocity
        reason = detector.geometry(candidate)
        if reason:
            return write_trajectory(
                directory, frames, numbers, config, detector, lineage, "failed", reason, candidate
            )
        try:
            result = evaluator(candidate, numbers)
        except (AsterionError, FloatingPointError) as error:
            return write_trajectory(
                directory,
                frames,
                numbers,
                config,
                detector,
                lineage,
                "failed",
                f"force-evaluation:{error}",
                candidate,
            )
        reason = detector.forces(result)
        if reason:
            return write_trajectory(
                directory, frames, numbers, config, detector, lineage, "failed", reason, candidate
            )
        if config.ensemble == "NVE":
            state = verlet(state, result, mass, dt)
        else:
            final_velocity = (
                noisy_velocity + 0.5 * dt * ACCELERATION * result.forces / mass[:, None]
            )
            state = State(
                candidate, final_velocity, result.forces.copy(), result.energy, state.step + 1, heat
            )
        if not np.isfinite(state.velocities).all():
            raise ContractError("nonfinite integrated velocities")
        frames.append(
            snapshot(
                state,
                numbers,
                result.uncertainty,
                result.neighbor_rebuilds,
                noise,
                uncertainty_available=result.uncertainty_available,
            )
        )
        if state.step % config.checkpoint_every == 0 or state.step == stop_after:
            name = f"state-{state.step:07}.npz"
            atomic_npz(
                directory / name,
                **{key: np.array([frame[key] for frame in frames]) for key in frames[0]},
            )
            write_json(
                directory / "resume.json",
                {
                    "config": asdict(config),
                    "lineage": lineage,
                    "arrays": name,
                    "sha256": file_hash(directory / name),
                    "rng": rng.bit_generator.state,
                },
            )
        if stop_after is not None and state.step >= stop_after:
            return write_trajectory(
                directory,
                frames,
                numbers,
                config,
                detector,
                lineage,
                "interrupted",
                "explicit-interruption",
            )
    return write_trajectory(
        directory, frames, numbers, config, detector, lineage, "passed", "completed"
    )
