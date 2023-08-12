from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from asterion.artifacts.io import write_json
from asterion.dynamics.analytic import Harmonic, LennardJones
from asterion.dynamics.detectors import ValidityPolicy
from asterion.dynamics.integrator import SimulationConfig, simulate
from asterion.evaluation.replay import replay_trajectory


def analytic_campaign(directory: Path, fixtures=100):
    directory.mkdir(parents=True, exist_ok=True)
    outcomes = []
    for seed in range(fixtures):
        rng = np.random.default_rng(seed + 712)
        numbers = np.array([1, 1]) if seed % 2 else np.array([6, 6])
        mass = 1.008 if seed % 2 else 12.011
        equilibrium = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        displacement = rng.normal(size=(2, 3)) * 0.1
        positions = equilibrium + displacement
        velocity = rng.normal(size=(2, 3)) * 0.001
        stiffness = float(rng.uniform(0.5, 3.0))
        # Reference frequency derives from SI, without calling the production acceleration helper.
        omega = np.sqrt(stiffness * 1.602176634e-19 / (mass * 1.66053906892e-27 * 1e-20)) * 1e-15
        horizon = 20.0
        exact = (
            equilibrium
            + displacement * np.cos(omega * horizon)
            + velocity / omega * np.sin(omega * horizon)
        )
        errors = []
        runs = []
        for dt in (0.4, 0.2, 0.1):
            run = directory / f"harmonic-{seed:04}-{dt}"
            simulate(
                Harmonic(equilibrium, stiffness),
                numbers,
                positions,
                velocity,
                run,
                SimulationConfig(steps=round(horizon / dt), timestep_fs=dt),
                ValidityPolicy(maximum_bond_ratio=10),
            )
            with np.load(run / "trajectory.npz", allow_pickle=False) as data:
                error = float(np.linalg.norm(data["positions"][-1] - exact))
            replay = replay_trajectory(run)
            errors.append(error)
            runs.append({"path": str(run), "dt": dt, "error": error, "replay": replay})
        ratios = [errors[0] / max(errors[1], 1e-16), errors[1] / max(errors[2], 1e-16)]
        if not all(3.6 < ratio < 4.4 for ratio in ratios):
            raise AssertionError(f"analytic convergence order failed for fixture {seed}: {ratios}")
        outcomes.append(
            {
                "fixture": seed,
                "mass_amu": mass,
                "stiffness_eV_angstrom2": stiffness,
                "horizon_fs": horizon,
                "ratios": ratios,
                "runs": runs,
                "status": "passed",
            }
        )
    # An independently expressed Lennard-Jones ODE checks the force sign and SI acceleration.
    for seed in range(10):
        radius = 1.1 + seed * 0.03
        numbers = np.array([6, 6])
        positions = np.array([[0.0, 0.0, 0.0], [radius, 0.0, 0.0]])
        initial = np.r_[positions.ravel(), np.zeros(6)]

        def ode(time, state):
            xyz = state[:6].reshape(2, 3)
            delta = xyz[1] - xyz[0]
            r = np.linalg.norm(delta)
            factor = 24 * 0.01 * ((1 / r) ** 6 - 2 * (1 / r) ** 12) / r**2
            f = np.array([factor * delta, -factor * delta])
            acceleration = f * 1.602176634e-19 / (12.011 * 1.66053906892e-27) * 1e-10
            return np.r_[state[6:], acceleration.ravel()]

        reference = solve_ivp(ode, (0, 10), initial, rtol=1e-11, atol=1e-13)
        runs = []
        for dt in (0.2, 0.1, 0.05):
            run = directory / f"lj-{seed:04}-{dt}"
            simulate(
                LennardJones(),
                numbers,
                positions,
                np.zeros_like(positions),
                run,
                SimulationConfig(steps=round(10 / dt), timestep_fs=dt),
                ValidityPolicy(maximum_bond_ratio=10),
            )
            with np.load(run / "trajectory.npz", allow_pickle=False) as data:
                error = float(
                    np.linalg.norm(data["positions"][-1] - reference.y[:6, -1].reshape(2, 3))
                )
            if error > 1e-4:
                raise AssertionError("independent LJ ODE mismatch")
            runs.append(
                {"path": str(run), "dt": dt, "error": error, "replay": replay_trajectory(run)}
            )
        outcomes.append({"fixture": f"lj-{seed}", "runs": runs, "status": "passed"})
    write_json(directory / "analytic.json", {"fixtures": len(outcomes), "outcomes": outcomes})
    return outcomes
