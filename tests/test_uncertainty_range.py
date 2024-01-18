import numpy as np
import pytest

from asterion.contracts.errors import ContractError
from asterion.uncertainty.committee import Calibration
from asterion.uncertainty.moments import member_statistics, scaled_moments


def calibration(multiplier=1, floor=1e-8):
    return Calibration(
        multiplier,
        floor,
        0.1,
        10,
        ("member1", "member2", "member3"),
        tuple(f"geometry:{i}" for i in range(10)),
    )


def test_committee_mean_of_large_equal_members_does_not_overflow():
    mean, deviation = scaled_moments(np.full((3, 1), 1.7e308))
    assert mean[0] == 1.7e308 and deviation[0] == 0


def test_large_finite_member_disagreement_avoids_squared_overflow():
    mean, deviation = scaled_moments(np.array([[1e200], [0.0], [-1e200]]))
    assert mean[0] == 0 and deviation[0] == 1e200


def test_force_score_uses_scaled_rms_over_cartesian_components():
    forces = np.zeros((3, 1, 3))
    forces[:, 0, 0] = [1e200, 0, -1e200]
    *_, score = member_statistics(np.zeros((3, 1)), forces, [1])
    np.testing.assert_allclose(score, [1e200 / np.sqrt(3)], rtol=1e-15)


def test_float32_outputs_are_aggregated_without_float32_variance_overflow():
    values = np.array([[1e30], [0.0], [-1e30]], dtype=np.float32)
    mean, deviation = scaled_moments(values)
    assert mean[0] == 0 and np.isfinite(deviation).all()
    assert deviation[0] == float(values[0, 0])


def test_committee_rejects_unrepresentable_sample_deviation():
    with pytest.raises(ContractError, match="finite physical output range"):
        scaled_moments(np.array([[1.7e308], [-1.7e308], [1.7e308]]))


def test_member_statistics_rejects_atom_partitions_that_hide_force_components():
    with pytest.raises(ContractError, match="axes"):
        member_statistics(np.zeros((3, 2)), np.zeros((3, 4, 3)), [1, 2])


def test_member_statistics_matches_independent_ordinary_numpy_moments():
    rng = np.random.default_rng(491)
    energy, force = rng.normal(size=(3, 2)), rng.normal(size=(3, 5, 3))
    energy_mean, force_mean, energy_std, force_std, scores = member_statistics(
        energy, force, [2, 3]
    )
    np.testing.assert_allclose(energy_mean, energy.mean(0), atol=1e-15)
    np.testing.assert_allclose(force_mean, force.mean(0), atol=1e-15)
    np.testing.assert_allclose(energy_std, energy.std(0, ddof=1), atol=1e-15)
    np.testing.assert_allclose(force_std, force.std(0, ddof=1), atol=1e-15)
    np.testing.assert_allclose(
        scores,
        [np.sqrt(np.mean(force.std(0, ddof=1)[a:b] ** 2)) for a, b in ((0, 2), (2, 5))],
        atol=1e-15,
    )


def test_calibration_rejects_overflow_in_final_bound():
    with pytest.raises(ContractError, match="bound exceeds"):
        calibration(multiplier=2).bounds(np.array([1e308]))


def test_calibration_rejects_unrepresentable_error_to_disagreement_ratio():
    with pytest.raises(ContractError, match="ratio exceeds"):
        Calibration.fit(
            np.zeros(10),
            np.full(10, 1e308),
            0.1,
            ("member1", "member2", "member3"),
            [f"g:{i}" for i in range(10)],
        )


def test_calibration_median_of_large_scores_remains_finite():
    result = Calibration.fit(
        np.full(10, 1.7e308),
        np.full(10, 1e100),
        0.1,
        ("member1", "member2", "member3"),
        [f"g:{i}" for i in range(10)],
    )
    assert np.isfinite(result.floor) and result.floor == 1.7e306
    np.testing.assert_allclose(result.bounds(np.full(10, 1.7e308)), np.full(10, 1e100), rtol=1e-15)


def test_zero_error_bound_stays_available_at_large_finite_disagreement():
    np.testing.assert_array_equal(calibration(multiplier=0).bounds(np.array([1.7e308])), [0])
