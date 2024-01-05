import json
from copy import deepcopy

import numpy as np
import pytest

from asterion.artifacts.derivatives import (
    force_observation,
    hessian_observation,
    replay_probe,
    symmetry_observation,
)
from asterion.contracts.errors import IntegrityError
from asterion.evaluation.checks import force_difference, hessian_difference, symmetry_trial


@pytest.fixture
def force_row(model, geometry):
    result = force_difference(model, geometry, retain_trace=True)
    return {key: value for key, value in result.items() if key not in {"finite", "autograd"}}


@pytest.fixture
def hessian_row(model, geometry):
    direction = np.arange(1, geometry.positions.size + 1, dtype=float).reshape(
        geometry.positions.shape
    )
    direction /= np.linalg.norm(direction)
    result = hessian_difference(model, geometry, direction, retain_trace=True)
    return {key: value for key, value in result.items() if key not in {"finite", "product"}}


@pytest.fixture
def symmetry_row(model, geometry):
    return symmetry_trial(model, geometry, 3, retain_trace=True)


def test_raw_force_difference_recomputes_negative_energy_gradient(force_row):
    _, finite = force_observation(force_row)
    trace = force_row["trace"]
    expected = -(np.array(trace["plus_energy"]) - np.array(trace["minus_energy"])) / (
        2 * force_row["step"]
    )
    np.testing.assert_array_equal(finite, expected)


def test_raw_force_difference_rejects_plausible_false_summary(force_row):
    force_row["relative"] += 0.01
    with pytest.raises(IntegrityError, match="summary"):
        force_observation(force_row)


def test_raw_force_difference_rejects_changed_energy_observation(force_row):
    force_row["trace"]["plus_energy"][0][0] += 0.01
    with pytest.raises(IntegrityError, match="summary"):
        force_observation(force_row)


def test_raw_force_difference_rejects_swapped_force_sign(force_row):
    force_row["trace"]["force"] = (-np.array(force_row["trace"]["force"])).tolist()
    with pytest.raises(IntegrityError):
        force_observation(force_row)


def test_raw_force_difference_rejects_zero_step(force_row):
    force_row["step"] = 0
    with pytest.raises(IntegrityError, match="step"):
        force_observation(force_row)


def test_raw_force_difference_rejects_missing_cartesian_axis(force_row):
    force_row["trace"]["force"] = np.array(force_row["trace"]["force"])[:, :2].tolist()
    with pytest.raises(IntegrityError, match="vector"):
        force_observation(force_row)


def test_raw_force_difference_rejects_nonfinite_observation(force_row):
    force_row["trace"]["minus_energy"][0][0] = float("nan")
    with pytest.raises(IntegrityError, match="vector"):
        force_observation(force_row)


def test_saved_weight_force_probe_replays_retained_observations(model, force_row):
    replay_probe(model, "force", force_row)


def test_saved_weight_force_probe_rejects_fake_force_even_with_consistent_summary(model, force_row):
    force_row["trace"]["force"] = np.zeros_like(force_row["trace"]["force"]).tolist()
    with pytest.raises(IntegrityError, match="saved weights"):
        replay_probe(model, "force", force_row)


def test_raw_hessian_difference_recomputes_directional_force_derivative(hessian_row):
    _, direction = hessian_observation(hessian_row)
    np.testing.assert_allclose(np.sum(direction**2), 1)


def test_raw_hessian_difference_rejects_unscaled_direction(hessian_row):
    hessian_row["trace"]["direction"] = (2 * np.array(hessian_row["trace"]["direction"])).tolist()
    with pytest.raises(IntegrityError, match="normalized"):
        hessian_observation(hessian_row)


def test_raw_hessian_difference_rejects_changed_force_trace(hessian_row):
    hessian_row["trace"]["plus_force"][0][0] += 0.1
    with pytest.raises(IntegrityError, match="summary"):
        hessian_observation(hessian_row)


def test_raw_hessian_difference_rejects_changed_curvature_product(hessian_row):
    hessian_row["trace"]["product"][0][0] += 0.1
    with pytest.raises(IntegrityError, match="summary"):
        hessian_observation(hessian_row)


def test_saved_weight_hessian_probe_replays_actual_curvature(model, hessian_row):
    replay_probe(model, "hessian", hessian_row)


def test_saved_weight_hessian_probe_rejects_substituted_curvature(model, hessian_row):
    hessian_row["trace"]["product"][0][0] += 0.1
    with pytest.raises(IntegrityError, match="weights"):
        replay_probe(model, "hessian", hessian_row)


def test_raw_symmetry_recomputes_rotation_reflection_force_and_torque(symmetry_row):
    _, matrix, permutation = symmetry_observation(symmetry_row)
    np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-14)
    assert np.linalg.det(matrix) < 0
    assert len(permutation) == len(symmetry_row["trace"]["force"])


def test_raw_symmetry_rejects_nonorthogonal_matrix(symmetry_row):
    symmetry_row["trace"]["matrix"][0][0] += 0.1
    with pytest.raises(IntegrityError, match="orthogonal"):
        symmetry_observation(symmetry_row)


def test_raw_symmetry_rejects_false_reflection_label(symmetry_row):
    symmetry_row["reflection"] = False
    with pytest.raises(IntegrityError, match="reflection"):
        symmetry_observation(symmetry_row)


def test_raw_symmetry_rejects_repeated_atom_permutation(symmetry_row):
    symmetry_row["trace"]["permutation"][1] = symmetry_row["trace"]["permutation"][0]
    with pytest.raises(IntegrityError, match="permutation"):
        symmetry_observation(symmetry_row)


def test_raw_symmetry_rejects_force_rotation_without_original_vector(symmetry_row):
    symmetry_row["trace"]["transformed_force"][0][0] += 0.1
    with pytest.raises(IntegrityError, match="summary"):
        symmetry_observation(symmetry_row)


def test_raw_symmetry_rejects_fabricated_additivity(symmetry_row):
    symmetry_row["trace"]["disconnected_energy"] += 0.01
    with pytest.raises(IntegrityError, match="summary"):
        symmetry_observation(symmetry_row)


def test_raw_symmetry_rejects_changed_batch_order_force(symmetry_row):
    symmetry_row["trace"]["batch_force"][0][0] += 0.1
    with pytest.raises(IntegrityError, match="summary"):
        symmetry_observation(symmetry_row)


def test_saved_weight_symmetry_probe_replays_all_retained_systems(model, symmetry_row):
    replay_probe(model, "symmetry", symmetry_row)


def test_saved_weight_symmetry_probe_rejects_constant_energy_forgery(model, symmetry_row):
    altered = deepcopy(symmetry_row)
    altered["trace"]["energy"] += 1
    with pytest.raises(IntegrityError, match="saved weights"):
        replay_probe(model, "symmetry", altered)


def test_raw_derivative_probe_replays_after_json_roundtrip(model, symmetry_row):
    stored = json.loads(json.dumps(symmetry_row))
    symmetry_observation(stored)
    replay_probe(model, "symmetry", stored)
