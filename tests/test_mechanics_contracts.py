import numpy as np
import pytest

from asterion.artifacts.io import atomic_npz, file_hash, read_json, write_json
from asterion.contracts.errors import ContractError, IntegrityError
from asterion.contracts.units import kinetic_energy, masses, temperature, thermal_velocities
from asterion.dynamics.analytic import Harmonic
from asterion.dynamics.detectors import ValidityPolicy
from asterion.dynamics.integrator import SimulationConfig, simulate
from asterion.evaluation.replay import replay_trajectory


def test_internal_temperature_excludes_collective_translation():
    velocity = np.tile([0.04, -0.03, 0.02], (3, 1))
    assert temperature(velocity, np.array([1.0, 6.0, 8.0])) < 1e-25


def test_internal_temperature_is_galilean_invariant():
    velocity = np.array([[0.03, 0.02, 0.0], [-0.02, 0.0, 0.01], [0.0, -0.01, -0.02]])
    mass = np.array([1.0, 6.0, 8.0])
    np.testing.assert_allclose(
        temperature(velocity, mass), temperature(velocity + [3.0, 2.0, -1.0], mass), rtol=1e-13
    )


def test_total_temperature_uses_all_three_cartesian_degrees():
    velocity = np.array([[0.01, 0.02, 0.03]])
    mass = np.array([12.011])
    si_twice_kinetic = 12.011 * 1.66053906892e-27 * np.sum((velocity * 1e5) ** 2)
    expected = si_twice_kinetic / (3 * 1.380649e-23)
    np.testing.assert_allclose(temperature(velocity, mass, removed_dof=0), expected, rtol=1e-14)


def test_internal_temperature_matches_independent_mass_weighted_si_accounting():
    velocity = np.array([[0.02, 0.01, 0.0], [-0.01, 0.03, 0.01]])
    mass = np.array([1.008, 12.011])
    kilograms = mass * 1.66053906892e-27
    meters_per_second = velocity * 1e5
    center = np.sum(kilograms[:, None] * meters_per_second, axis=0) / sum(kilograms)
    expected = np.sum(kilograms[:, None] * (meters_per_second - center) ** 2) / (3 * 1.380649e-23)
    np.testing.assert_allclose(temperature(velocity, mass), expected, rtol=1e-14)


def test_internal_temperature_does_not_mutate_retained_velocities():
    velocity = np.array([[0.02, 0.0, 0.0], [-0.01, 0.0, 0.0]])
    before = velocity.copy()
    velocity.setflags(write=False)
    assert temperature(velocity, np.array([1.0, 12.0])) > 0
    np.testing.assert_array_equal(velocity, before)


def test_internal_single_atom_temperature_is_undefined():
    with pytest.raises(ContractError, match="degrees"):
        temperature(np.array([[0.02, 0.0, 0.0]]), np.array([1.0]))


def test_temperature_rejects_unimplemented_rotational_constraint_count():
    with pytest.raises(ContractError, match="center-of-mass"):
        temperature(np.zeros((3, 3)), np.ones(3), removed_dof=6)


def test_mass_lookup_rejects_fractional_atomic_number():
    with pytest.raises(ContractError, match="integer"):
        masses(np.array([1.5, 6.0]))


def test_mass_lookup_rejects_empty_element_population():
    with pytest.raises(ContractError, match="nonempty"):
        masses(np.array([], dtype=int))


def test_kinetic_energy_rejects_negative_mass():
    with pytest.raises(ContractError, match="positive"):
        kinetic_energy(np.zeros((2, 3)), np.array([1.0, -1.0]))


def test_kinetic_energy_rejects_nonfinite_mass():
    with pytest.raises(ContractError, match="finite"):
        kinetic_energy(np.zeros((2, 3)), np.array([1.0, float("nan")]))


def test_kinetic_energy_rejects_matrix_mass_axis():
    with pytest.raises(ContractError, match="vector"):
        kinetic_energy(np.zeros((2, 3)), np.ones((2, 1)))


def test_kinetic_energy_rejects_nonfinite_velocity():
    with pytest.raises(ContractError, match="velocities"):
        kinetic_energy(np.array([[float("inf"), 0.0, 0.0]]), np.ones(1))


def test_thermal_initialization_has_exact_zero_velocity_at_zero_temperature():
    np.testing.assert_array_equal(thermal_velocities(np.array([1, 6, 8]), 0, 4), np.zeros((3, 3)))


def test_thermal_initialization_scales_with_square_root_temperature():
    numbers = np.array([1, 6, 8])
    np.testing.assert_allclose(
        thermal_velocities(numbers, 1200, 4),
        2 * thermal_velocities(numbers, 300, 4),
        atol=1e-16,
        rtol=1e-14,
    )


def test_simulation_rejects_noninteger_step_workload():
    with pytest.raises(ContractError, match="steps"):
        SimulationConfig(steps=10.5).validate()


def test_simulation_rejects_zero_checkpoint_period():
    with pytest.raises(ContractError):
        SimulationConfig(checkpoint_every=0).validate()


def test_simulation_rejects_nonfinite_bath_temperature():
    with pytest.raises(ContractError, match="temperature"):
        SimulationConfig(temperature_K=float("nan")).validate()


def test_simulation_rejects_nonfinite_friction():
    with pytest.raises(ContractError, match="friction"):
        SimulationConfig(friction_per_fs=float("inf")).validate()


def test_simulation_rejects_invalid_rng_seed():
    with pytest.raises(ContractError, match="seed"):
        SimulationConfig(seed=-1).validate()


def test_detector_rejects_nan_force_threshold():
    with pytest.raises(ContractError, match="finite"):
        ValidityPolicy(maximum_force=float("nan")).validate()


def test_detector_rejects_infinite_uncertainty_threshold():
    with pytest.raises(ContractError, match="positive"):
        ValidityPolicy(maximum_uncertainty=float("inf")).validate()


def test_detector_rejects_unsupported_chemistry_policy():
    with pytest.raises(ContractError, match="element"):
        ValidityPolicy(supported_elements=(1, 26)).validate()


@pytest.fixture
def interrupted(tmp_path):
    numbers = np.array([6, 6])
    positions = np.array([[0.1, 0.0, 0.0], [3.1, 0.0, 0.0]])
    equilibrium = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    velocity = np.zeros_like(positions)
    config = SimulationConfig(steps=4, checkpoint_every=2)
    evaluator = Harmonic(equilibrium)
    run = tmp_path / "run"
    simulate(evaluator, numbers, positions, velocity, run, config, stop_after=2)
    return evaluator, numbers, positions, velocity, run, config


def test_simulation_resume_rejects_changed_detector_policy(interrupted):
    evaluator, numbers, positions, velocity, run, config = interrupted
    with pytest.raises(IntegrityError, match="lineage"):
        simulate(
            evaluator,
            numbers,
            positions,
            velocity,
            run,
            config,
            ValidityPolicy(maximum_force=50),
            resume=True,
        )


def test_simulation_resume_rejects_changed_initial_velocity(interrupted):
    evaluator, numbers, positions, velocity, run, config = interrupted
    with pytest.raises(IntegrityError, match="lineage"):
        simulate(evaluator, numbers, positions, velocity + 0.01, run, config, resume=True)


def test_simulation_resume_rejects_changed_initial_position(interrupted):
    evaluator, numbers, positions, velocity, run, config = interrupted
    with pytest.raises(IntegrityError, match="lineage"):
        simulate(evaluator, numbers, positions + 0.01, velocity, run, config, resume=True)


def test_simulation_resume_rejects_changed_atom_mass_identity(interrupted):
    evaluator, _, positions, velocity, run, config = interrupted
    with pytest.raises(IntegrityError, match="lineage"):
        simulate(evaluator, np.array([1, 6]), positions, velocity, run, config, resume=True)


def test_replay_rejects_rehashed_false_internal_temperature(interrupted):
    evaluator, numbers, positions, velocity, run, config = interrupted
    simulate(evaluator, numbers, positions, velocity, run, config, resume=True)
    with np.load(run / "trajectory.npz") as raw:
        arrays = {key: raw[key] for key in raw.files}
    arrays["temperature"][-1] = 300
    atomic_npz(run / "trajectory.npz", **arrays)
    record = read_json(run / "trajectory.json")
    record["trajectory_sha256"] = file_hash(run / "trajectory.npz")
    write_json(run / "trajectory.json", record)
    with pytest.raises(IntegrityError, match="temperature"):
        replay_trajectory(run)


def test_simulation_rejects_nonfinite_initial_position_before_run_creation(tmp_path):
    numbers = np.array([6, 6])
    positions = np.array([[float("nan"), 0.0, 0.0], [3.0, 0.0, 0.0]])
    with pytest.raises(ContractError, match="initial positions"):
        simulate(
            Harmonic(np.zeros((2, 3))),
            numbers,
            positions,
            np.zeros_like(positions),
            tmp_path / "run",
            SimulationConfig(),
        )
    assert not (tmp_path / "run").exists()


def test_simulation_rejects_fractional_element_before_mass_coercion(tmp_path):
    positions = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    with pytest.raises(ContractError, match="integer"):
        simulate(
            Harmonic(positions),
            np.array([1.5, 6.0]),
            positions,
            np.zeros_like(positions),
            tmp_path / "run",
            SimulationConfig(),
        )
    assert not (tmp_path / "run").exists()
