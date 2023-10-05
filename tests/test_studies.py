import numpy as np
import pytest

from asterion.acquisition.policy import choose
from asterion.artifacts.studies import committee_arrays, independent_choices
from asterion.contracts.errors import IntegrityError
from asterion.contracts.geometry import Chemistry
from asterion.data.dataset import CoordinatePool
from asterion.data.splits import geometry_descriptors


def study():
    random = np.random.default_rng(4)
    numbers = np.array([6, 8, 1, 1])
    coordinates = random.normal(size=(12, 4, 3))
    pool = CoordinatePool(numbers, coordinates, np.arange(12), "hash", "fixture", Chemistry())
    return pool, numbers, coordinates


def test_independent_uniform_selection_matches_declared_rng():
    pool, numbers, xyz = study()
    expected = choose(
        list(range(12)), geometry_descriptors(pool, range(12)), [0, 1], 4, "uniform", 8
    )
    rows, scores, novelty = independent_choices(
        list(range(12)), xyz, numbers, [0, 1], 4, "uniform", 8, np.arange(12)
    )
    assert rows == expected.rows and scores == expected.scores and novelty == expected.novelty


def test_independent_diversity_selection_reconstructs_greedy_path():
    pool, numbers, xyz = study()
    expected = choose(
        list(range(12)), geometry_descriptors(pool, range(12)), [0, 1], 4, "diversity", 8
    )
    rows, scores, novelty = independent_choices(
        list(range(12)), xyz, numbers, [0, 1], 4, "diversity", 8, np.arange(12)
    )
    assert rows == expected.rows
    assert np.allclose(scores, expected.scores) and np.allclose(novelty, expected.novelty)


def test_independent_uncertainty_selection_reconstructs_score_tradeoff():
    pool, numbers, xyz = study()
    signal = np.random.default_rng(3).random(12)
    expected = choose(
        list(range(12)),
        geometry_descriptors(pool, range(12)),
        [0, 1],
        4,
        "uncertainty-diversity",
        8,
        signal,
    )
    rows, scores, novelty = independent_choices(
        list(range(12)), xyz, numbers, [0, 1], 4, "uncertainty-diversity", 8, signal
    )
    assert rows == expected.rows
    assert np.allclose(scores, expected.scores) and np.allclose(novelty, expected.novelty)


def test_independent_policy_rejects_unknown_strategy():
    _, numbers, xyz = study()
    with pytest.raises(IntegrityError, match="unknown"):
        independent_choices(list(range(12)), xyz, numbers, [0, 1], 4, "unknown", 8, np.arange(12))


def members():
    reference = {
        "identities": np.array(["a", "b"]),
        "pointers": np.array([0, 1, 2]),
        "target_force": np.zeros((2, 3)),
        "target_energy": np.zeros(2),
        "positions": np.zeros((2, 3)),
        "numbers": np.array([6, 6]),
    }
    return [{**reference, "force": np.full((2, 3), value)} for value in (1.0, 2.0, 3.0)]


def test_committee_mean_is_independently_recomputed():
    mean, std, scores, errors = committee_arrays(members())
    assert np.array_equal(mean, np.full((2, 3), 2.0))
    assert np.array_equal(std, np.ones((2, 3)))
    assert np.array_equal(scores, np.ones(2)) and np.array_equal(errors, np.full(2, 2.0))


def test_committee_raw_check_rejects_missing_member():
    with pytest.raises(IntegrityError, match="three"):
        committee_arrays(members()[:2])


def test_committee_raw_check_rejects_swapped_target_identity():
    values = members()
    values[1]["identities"] = np.array(["b", "a"])
    with pytest.raises(IntegrityError, match="different geometry"):
        committee_arrays(values)


def test_committee_raw_check_rejects_changed_reference_force():
    values = members()
    values[1]["target_force"] = np.ones((2, 3))
    with pytest.raises(IntegrityError, match="different geometry"):
        committee_arrays(values)


def test_committee_raw_check_rejects_nonfinite_member():
    values = members()
    values[1]["force"][0, 0] = float("nan")
    with pytest.raises(IntegrityError, match="nonfinite"):
        committee_arrays(values)
