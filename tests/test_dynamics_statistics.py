from copy import deepcopy

import numpy as np
import pytest

from asterion.artifacts.dynamics_study import verify_statistics
from asterion.contracts.errors import ContractError, IntegrityError
from asterion.dynamics.analytic import HarmonicBond
from asterion.dynamics.integrator import SimulationConfig, simulate
from asterion.evaluation.dynamics_statistics import (
    analyze_trajectory,
    bond_distribution,
    matched_timestep_observations,
    temperature_correlation,
)
from asterion.evaluation.replay import replay_trajectory
from asterion.training.dynamics_study import study_protocol


@pytest.fixture
def path_statistics(tmp_path):
    numbers = np.array([6, 6])
    positions = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
    velocities = np.array([[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]])
    directory = tmp_path / "nvt"
    simulate(
        HarmonicBond(stiffness=200),
        numbers,
        positions,
        velocities,
        directory,
        SimulationConfig(steps=30, timestep_fs=0.05, ensemble="NVT", temperature_K=300, seed=491),
    )
    return directory, analyze_trajectory(directory)


def test_finite_nvt_statistics_replay_heat_and_exclude_initial_frame(path_statistics):
    directory, statistics = path_statistics
    assert statistics["sample_frames"] == 30 and statistics["equilibrium_claim"] is False
    verify_statistics(directory, statistics, replay_trajectory(directory))
    with np.load(directory / "trajectory.npz", allow_pickle=False) as raw:
        np.testing.assert_allclose(statistics["temperature_mean_K"], raw["temperature"][1:].mean())


def test_structural_report_cannot_claim_correlated_frames_are_independent(path_statistics):
    directory, statistics = path_statistics
    statistics["independent_frames_claim"] = True
    with pytest.raises(IntegrityError, match="physical scope"):
        verify_statistics(directory, statistics, replay_trajectory(directory))


def test_bond_histogram_denominator_is_completed_frames_times_reference_pairs(path_statistics):
    _, statistics = path_statistics
    row = statistics["bond_distances"]["species_pairs"]["6-6"]
    assert row["pairs"] == 1 and row["sample_count"] == sum(row["histogram"]) == 30


def test_bond_report_cannot_drop_a_physical_sample(path_statistics):
    directory, statistics = path_statistics
    row = statistics["bond_distances"]["species_pairs"]["6-6"]
    index = next(i for i, n in enumerate(row["histogram"]) if n)
    row["histogram"][index] -= 1
    with pytest.raises(IntegrityError, match="omits physical samples"):
        verify_statistics(directory, statistics, replay_trajectory(directory))


def test_thermostat_statistics_cannot_omit_recorded_heat(path_statistics):
    directory, statistics = path_statistics
    statistics["maximum_heat_adjusted_energy_drift_eV"] += 1
    with pytest.raises(IntegrityError, match="thermostat heat"):
        verify_statistics(directory, statistics, replay_trajectory(directory))


def test_temperature_correlation_preserves_time_order():
    ordered = temperature_correlation(np.sin(np.arange(100) / 10))
    permuted = temperature_correlation(
        np.random.default_rng(491).permutation(np.sin(np.arange(100) / 10))
    )
    assert ordered["correlation"][1] > 0.9
    assert ordered["integrated_time_steps"] > permuted["integrated_time_steps"]


def test_angular_momentum_check_uses_si_scale_tolerance(path_statistics):
    directory, statistics = path_statistics
    statistics["maximum_internal_angular_momentum_change_SI"] += 1e-40
    with pytest.raises(IntegrityError, match="angular momentum"):
        verify_statistics(directory, statistics, replay_trajectory(directory))


def test_constant_temperature_path_does_not_invent_correlation_time():
    result = temperature_correlation(np.full(100, 300.0))
    assert result["available"] is False and result["integrated_time_steps"] is None


def test_bond_histogram_preserves_samples_above_declared_display_range():
    positions = np.array([[[0.0, 0.0, 0.0], [7.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]]])
    report = bond_distribution(positions, np.array([6, 6]), [(0, 1, 1.0)])
    row = report["species_pairs"]["6-6"]
    assert row["sample_count"] == 2 and row["overflow"] == 2 and sum(row["histogram"]) == 0
    assert row["mean_angstrom"] == 7.5


def refinements(tmp_path, last_steps=80, change_initial=False):
    paths = []
    for i, (dt, steps) in enumerate(((0.1, 20), (0.05, 40), (0.025, last_steps))):
        directory = tmp_path / f"refinement-{i}"
        velocity = np.array([[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]])
        if change_initial and i == 2:
            velocity *= 2
        simulate(
            HarmonicBond(stiffness=200),
            np.array([6, 6]),
            np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]]),
            velocity,
            directory,
            SimulationConfig(steps=steps, timestep_fs=dt),
            lineage="analytic-bond",
        )
        paths.append(directory)
    return paths


def test_timestep_refinement_compares_identical_completed_physical_horizon(tmp_path):
    result = matched_timestep_observations(refinements(tmp_path))
    assert result["physical_horizon_fs"] == 2
    assert (
        result["coarse_to_finest_coordinate_rms_angstrom"]
        > result["middle_to_finest_coordinate_rms_angstrom"]
        > 0
    )
    assert result["convergence_order_claim"] is False


def test_timestep_refinement_rejects_shorter_fine_horizon(tmp_path):
    with pytest.raises(ContractError, match="physical horizon"):
        matched_timestep_observations(refinements(tmp_path, last_steps=79))


def test_timestep_refinement_rejects_changed_initial_velocities(tmp_path):
    with pytest.raises(ContractError, match="initial physical state"):
        matched_timestep_observations(refinements(tmp_path, change_initial=True))


def test_dynamics_study_protocol_rejects_unmatched_timestep_halving():
    config = {
        "molecules": ["ethanol"],
        "dynamics_study": {
            "molecules": ["ethanol"],
            "steps": 10,
            "refinement_steps": 10,
            "temperature_K": 300,
            "timesteps_fs": [0.1, 0.06, 0.025],
        },
    }
    with pytest.raises(ContractError, match="protocol"):
        study_protocol(deepcopy(config))
