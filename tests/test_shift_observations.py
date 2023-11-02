from copy import deepcopy

import numpy as np
import pytest

from asterion.artifacts.shift import check_guarded_roles, check_rmd_observations
from asterion.artifacts.verify import independent_metrics
from asterion.contracts.errors import IntegrityError
from asterion.contracts.geometry import Chemistry
from asterion.data.dataset import CoordinatePool
from asterion.data.splits import guarded_split


@pytest.fixture
def observations(tmp_path):
    rows = [2, 5, 9]
    coordinates = np.arange(60, dtype=float).reshape(10, 2, 3) * 0.1
    force = np.arange(60, dtype=float).reshape(10, 2, 3) * 0.2
    energy = np.arange(10, dtype=float)
    path = tmp_path / "author.npz"
    np.savez(path, coords=coordinates, forces=force, energies=energy, nuclear_charges=[6, 1])
    conversion = 4184 / (6.02214076e23 * 1.602176634e-19)
    targets = force[rows].reshape(-1, 3) * conversion
    arrays = {
        "energy": energy[rows] * conversion + 1,
        "target_energy": energy[rows] * conversion,
        "force": targets + 2,
        "target_force": targets,
        "numbers": np.tile([6, 1], 3),
        "positions": coordinates[rows].reshape(-1, 3),
        "pointers": np.arange(4) * 2,
        "identities": np.array([f"ethanol:hash:{row}" for row in rows]),
    }
    return arrays, path, rows


def check(case):
    arrays, path, rows = case
    return check_rmd_observations(arrays, path, rows, "ethanol", "hash")


def test_raw_physical_observation_errors_use_declared_geometry_units(observations):
    metric = check(observations)
    assert metric["geometries"] == 3 and metric["atoms"] == 6
    assert np.isclose(metric["force_mae_eV_angstrom"], 2)
    assert np.isclose(metric["energy_mae_eV"], 1)
    assert np.isclose(metric["force_rmse_eV_angstrom"], 2)


def test_raw_physical_observation_rejects_invented_identity(observations):
    observations[0]["identities"][0] = "ethanol:hash:3"
    with pytest.raises(IntegrityError, match="identities"):
        check(observations)


def test_raw_physical_observation_rejects_other_molecule_namespace(observations):
    arrays, path, rows = observations
    arrays["identities"] = np.array([f"aspirin:hash:{row}" for row in rows])
    with pytest.raises(IntegrityError, match="identities"):
        check((arrays, path, rows))


def test_raw_physical_observation_rejects_swapped_energy_scale(observations):
    observations[0]["target_energy"] *= 23.06
    with pytest.raises(IntegrityError, match="labels"):
        check(observations)


def test_raw_physical_observation_rejects_swapped_force_scale(observations):
    observations[0]["target_force"] *= 23.06
    with pytest.raises(IntegrityError, match="labels"):
        check(observations)


def test_raw_physical_observation_rejects_atom_reordering(observations):
    observations[0]["numbers"] = np.tile([1, 6], 3)
    with pytest.raises(IntegrityError, match="ordering"):
        check(observations)


def test_raw_physical_observation_rejects_altered_coordinate(observations):
    observations[0]["positions"][0, 0] += 0.01
    with pytest.raises(IntegrityError, match="coordinates"):
        check(observations)


def test_raw_physical_observation_rejects_false_atom_denominators(observations):
    observations[0]["pointers"] = np.array([0, 1, 4, 6])
    with pytest.raises(IntegrityError, match="pointers"):
        check(observations)


def test_independent_metric_rejects_fractional_geometry_offsets(observations):
    observations[0]["pointers"] = observations[0]["pointers"].astype(float)
    with pytest.raises(IntegrityError, match="integer"):
        independent_metrics(observations[0])


def test_independent_metric_rejects_extra_energy_axis(observations):
    observations[0]["energy"] = observations[0]["energy"][:, None]
    observations[0]["target_energy"] = observations[0]["target_energy"][:, None]
    with pytest.raises(IntegrityError, match="Cartesian"):
        independent_metrics(observations[0])


def test_independent_metric_rejects_missing_force_component(observations):
    observations[0]["force"] = observations[0]["force"][:, :2]
    observations[0]["target_force"] = observations[0]["target_force"][:, :2]
    with pytest.raises(IntegrityError, match="Cartesian"):
        independent_metrics(observations[0])


@pytest.fixture
def guarded():
    original = np.random.default_rng(51).permutation(10000)
    positions = np.zeros((10000, 2, 3))
    positions[:, 1, 0] = 1
    pool = CoordinatePool(np.array([6, 1]), positions, original, "hash", "bond", Chemistry())
    split = guarded_split(pool, 3, evaluation_size=100, validation_size=50, calibration_size=50)
    from dataclasses import asdict

    config = {
        "budgets": [1000],
        "evaluation_size": 100,
        "validation_size": 50,
        "calibration_size": 50,
    }
    value = {
        key: list(item) if isinstance(item, tuple) else item for key, item in asdict(split).items()
    }
    return value, original, config


def test_independent_guard_selection_matches_original_temporal_order(guarded):
    split, original, config = guarded
    split = {
        key: list(value) if isinstance(value, tuple) else value for key, value in split.items()
    }
    check_guarded_roles(split, original, config)
    assert max(original[split["train_pool"]]) < 4400
    assert min(original[split["evaluation"]]) > 7600


def test_guarded_study_rejects_training_slice_mislabeled_as_guarded(guarded):
    split, original, config = deepcopy(guarded)
    split["protocol"] = "temporal-slice-excluding-primary-roles"
    with pytest.raises(IntegrityError, match="protocol"):
        check_guarded_roles(split, original, config)


def test_guarded_study_rejects_zero_guard(guarded):
    split, original, config = deepcopy(guarded)
    split["temporal_guard"] = 0
    with pytest.raises(IntegrityError, match="protocol"):
        check_guarded_roles(split, original, config)


def test_guarded_study_rejects_author_random_training_rows(guarded):
    split, original, config = deepcopy(guarded)
    split["train_pool"] = list(range(1000))
    with pytest.raises(IntegrityError, match="selection"):
        check_guarded_roles(split, original, config)


def test_guarded_study_rejects_selected_validation_row_from_another_block(guarded):
    split, original, config = deepcopy(guarded)
    split = {
        key: list(value) if isinstance(value, tuple) else value for key, value in split.items()
    }
    split["validation"][0] = split["train_pool"][0]
    with pytest.raises(IntegrityError, match="selection"):
        check_guarded_roles(split, original, config)


def test_guarded_study_rejects_repeated_evaluation_seed_draw(guarded):
    split, original, config = deepcopy(guarded)
    split = {
        key: list(value) if isinstance(value, tuple) else value for key, value in split.items()
    }
    split["evaluation"][1] = split["evaluation"][0]
    with pytest.raises(IntegrityError, match="selection"):
        check_guarded_roles(split, original, config)
