from copy import deepcopy

import numpy as np
import pytest

from asterion.artifacts.dynamics_study import verify_statistics
from asterion.artifacts.io import atomic_npz, file_hash, read_json, write_json
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


def test_learned_momentum_cancellation_is_checked_at_retained_scale(model, geometry, tmp_path):
    from asterion.contracts.units import thermal_velocities
    from asterion.dynamics.learned import LearnedEvaluator

    simulate(
        LearnedEvaluator(model),
        geometry.numbers,
        geometry.positions,
        thermal_velocities(geometry.numbers, 100, 491),
        tmp_path / "learned",
        SimulationConfig(steps=8),
        lineage="original-untrained-model-fixture",
    )
    statistics = analyze_trajectory(tmp_path / "learned")
    assert statistics["angular_momentum_roundoff_bound_SI"] > 0
    assert statistics["angular_change_above_roundoff"] is False
    verify_statistics(tmp_path / "learned", statistics, replay_trajectory(tmp_path / "learned"))


def test_angular_momentum_roundoff_scale_cannot_be_removed(path_statistics):
    directory, statistics = path_statistics
    statistics["angular_momentum_roundoff_bound_SI"] = 0
    with pytest.raises(IntegrityError, match="conditioning"):
        verify_statistics(directory, statistics, replay_trajectory(directory))


def test_angular_resolution_claim_requires_a_boolean(path_statistics):
    directory, statistics = path_statistics
    statistics["angular_change_above_roundoff"] = int(statistics["angular_change_above_roundoff"])
    with pytest.raises(IntegrityError, match="conditioning"):
        verify_statistics(directory, statistics, replay_trajectory(directory))


def test_valid_integration_prefix_cannot_claim_completed_nvt_workload(path_statistics):
    directory, _ = path_statistics
    with np.load(directory / "trajectory.npz", allow_pickle=False) as raw:
        arrays = {
            key: raw[key].copy() if key == "numbers" else raw[key][:-1].copy() for key in raw.files
        }
    atomic_npz(directory / "trajectory.npz", **arrays)
    metadata = read_json(directory / "trajectory.json")
    metadata["trajectory_sha256"] = file_hash(directory / "trajectory.npz")
    metadata["completed_steps"] -= 1
    metadata["physical_horizon_fs"] = (
        metadata["completed_steps"] * metadata["config"]["timestep_fs"]
    )
    write_json(directory / "trajectory.json", metadata)
    with pytest.raises(IntegrityError, match="has not completed"):
        replay_trajectory(directory)


def test_nvt_replay_binds_noise_to_the_declared_seed(path_statistics):
    directory, _ = path_statistics
    record = read_json(directory / "trajectory.json")
    record["config"]["seed"] += 1
    write_json(directory / "trajectory.json", record)
    with pytest.raises(IntegrityError, match="seeded Gaussian stream"):
        replay_trajectory(directory)


def test_valid_frames_cannot_relax_their_force_domain_after_execution(path_statistics):
    directory, _ = path_statistics
    record = read_json(directory / "trajectory.json")
    record["detector"]["policy"]["maximum_force"] = 1e-12
    write_json(directory / "trajectory.json", record)
    with pytest.raises(IntegrityError, match="force or uncertainty policy"):
        replay_trajectory(directory)


def test_json_exponent_overflow_cannot_disable_a_validity_limit(path_statistics):
    directory, _ = path_statistics
    path = directory / "trajectory.json"
    path.write_text(path.read_text().replace('"maximum_force":100.0', '"maximum_force":1e309'))
    with pytest.raises(IntegrityError, match="nonfinite or nonpositive"):
        replay_trajectory(directory)


def test_json_exponent_overflow_cannot_hide_nonfinite_thermostat_arithmetic(path_statistics):
    directory, _ = path_statistics
    path = directory / "trajectory.json"
    record = read_json(path)
    record["config"]["temperature_K"] = 1
    write_json(path, record)
    path.write_text(path.read_text().replace('"temperature_K":1,', '"temperature_K":1e309,'))
    with pytest.raises(IntegrityError, match="stochastic configuration"):
        replay_trajectory(directory)


def test_failed_initial_candidate_is_bound_to_actual_retained_positions(tmp_path):
    from asterion.dynamics.analytic import ForceResult

    numbers = np.array([6, 6])
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    directory = tmp_path / "failed"
    simulate(
        lambda p, n: ForceResult(0.0, np.full_like(p, 1000)),
        numbers,
        positions,
        np.zeros_like(positions),
        directory,
        SimulationConfig(steps=8),
    )
    atomic_npz(directory / "failure.npz", positions=positions + 1)
    metadata = read_json(directory / "trajectory.json")
    metadata["failure_sha256"] = file_hash(directory / "failure.npz")
    write_json(directory / "trajectory.json", metadata)
    with pytest.raises(IntegrityError, match="not produced by"):
        replay_trajectory(directory)


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
