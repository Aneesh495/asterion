import numpy as np
import pytest

from asterion.contracts.units import ACCELERATION, masses, temperature, thermal_velocities
from asterion.dynamics.analytic import Harmonic, LennardJones
from asterion.dynamics.detectors import ValidityPolicy
from asterion.dynamics.integrator import SimulationConfig, simulate
from asterion.evaluation.replay import replay_trajectory


@pytest.mark.parametrize("seed", range(8))
def test_thermal_temperature(seed):
    numbers = np.array([6, 1, 1, 8])
    velocity = thermal_velocities(numbers, 300, seed)
    np.testing.assert_allclose(np.sum(masses(numbers)[:, None] * velocity, axis=0), 0, atol=1e-15)
    assert abs(temperature(velocity, masses(numbers)) - 300) < 1e-10


@pytest.mark.parametrize("dt", [0.4, 0.2, 0.1])
def test_analytic_harmonic_horizon(tmp_path, dt):
    numbers = np.array([6, 6])
    center = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    positions = center + np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    run = tmp_path / "harmonic"
    simulate(
        Harmonic(center, 2.0),
        numbers,
        positions,
        np.zeros_like(positions),
        run,
        SimulationConfig(steps=round(10 / dt), timestep_fs=dt),
        ValidityPolicy(maximum_bond_ratio=5.0),
    )
    with np.load(run / "trajectory.npz") as data:
        exact = center + (positions - center) * np.cos(np.sqrt(2 * ACCELERATION / 12.011) * 10)
        assert np.linalg.norm(data["positions"][-1] - exact) < 1e-5
    assert replay_trajectory(run)["completed_steps"] == round(10 / dt)


def test_second_order_refinement(tmp_path):
    numbers = np.array([1, 1])
    center = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    positions = center + 0.1
    errors = []
    for dt in (0.4, 0.2, 0.1):
        directory = tmp_path / str(dt)
        simulate(
            Harmonic(center),
            numbers,
            positions,
            np.zeros_like(positions),
            directory,
            SimulationConfig(steps=round(20 / dt), timestep_fs=dt),
        )
        with np.load(directory / "trajectory.npz") as data:
            exact = center + 0.1 * np.cos(np.sqrt(ACCELERATION / 1.008) * 20)
            errors.append(np.linalg.norm(data["positions"][-1] - exact))
    assert 3.8 < errors[0] / errors[1] < 4.2
    assert 3.8 < errors[1] / errors[2] < 4.2


@pytest.mark.parametrize("ensemble", ["NVE", "NVT"])
def test_simulation_atomic_resume(tmp_path, ensemble):
    numbers = np.array([6, 6])
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    velocity = thermal_velocities(numbers, 100, 13)
    config = SimulationConfig(steps=30, ensemble=ensemble, checkpoint_every=10)
    evaluator = LennardJones()
    simulate(evaluator, numbers, positions, velocity, tmp_path / "whole", config)
    simulate(evaluator, numbers, positions, velocity, tmp_path / "resume", config, stop_after=10)
    simulate(evaluator, numbers, positions, velocity, tmp_path / "resume", config, resume=True)
    with (
        np.load(tmp_path / "whole" / "trajectory.npz") as first,
        np.load(tmp_path / "resume" / "trajectory.npz") as second,
    ):
        for key in first.files:
            np.testing.assert_array_equal(first[key], second[key])
    replay_trajectory(tmp_path / "resume")


def test_force_detector_preserves_failed_start(tmp_path):
    numbers = np.array([1, 1])
    positions = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]])
    record = simulate(
        LennardJones(),
        numbers,
        positions,
        np.zeros_like(positions),
        tmp_path / "failure",
        SimulationConfig(steps=30),
        ValidityPolicy(maximum_force=1),
    )
    assert record["status"] == "failed" and record["completed_steps"] == 0
    assert record["reason"] == "extreme-force"
    assert (tmp_path / "failure" / "failure.npz").exists()
