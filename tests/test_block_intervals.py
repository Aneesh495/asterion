import numpy as np
import pytest

from asterion.contracts.errors import ContractError
from asterion.evaluation.metrics import paired_block_interval


def test_block_point_estimate_preserves_geometry_weighting():
    result = paired_block_interval([10.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0, 1, 1])
    assert result["mean_improvement"] == 4.0


def test_constant_paired_improvement_has_constant_interval():
    result = paired_block_interval([3.0, 4.0, 5.0], [1.0, 2.0, 3.0], [0, 1, 2])
    assert result["lower_95"] == result["upper_95"] == 2.0


def test_negative_learning_is_retained_as_negative_interval():
    result = paired_block_interval([1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [0, 1, 2])
    assert result["mean_improvement"] == result["upper_95"] == -1.0


def test_zero_learning_is_not_positive():
    result = paired_block_interval([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [0, 1, 2])
    assert result["mean_improvement"] == result["lower_95"] == 0.0


def test_block_resampling_is_reproducible():
    first = paired_block_interval([1.0, 3.0, 5.0], [0.0, 1.0, 2.0], [0, 1, 2], seed=4)
    second = paired_block_interval([1.0, 3.0, 5.0], [0.0, 1.0, 2.0], [0, 1, 2], seed=4)
    assert first == second


def test_block_resampling_is_row_permutation_invariant():
    first = paired_block_interval([1.0, 3.0, 5.0, 7.0], [0.0, 1.0, 2.0, 3.0], [0, 0, 1, 2])
    second = paired_block_interval([7.0, 3.0, 1.0, 5.0], [3.0, 1.0, 0.0, 2.0], [2, 0, 0, 1])
    assert first == second


def test_block_interval_rejects_single_correlation_unit():
    with pytest.raises(ContractError, match="independent"):
        paired_block_interval([1.0, 2.0], [0.0, 0.0], [0, 0])


def test_block_interval_rejects_empty_observations():
    with pytest.raises(ContractError, match="nonempty"):
        paired_block_interval([], [], [])


def test_block_interval_rejects_unpaired_shapes():
    with pytest.raises(ContractError, match="matched"):
        paired_block_interval([1.0, 2.0], [0.0], [0, 1])


def test_block_interval_rejects_nonfinite_error():
    with pytest.raises(ContractError, match="finite"):
        paired_block_interval([1.0, float("nan")], [0.0, 0.0], [0, 1])


def test_block_interval_rejects_nonfinite_block():
    with pytest.raises(ContractError, match="block identities"):
        paired_block_interval([1.0, 2.0], [0.0, 0.0], [0.0, float("nan")])


def test_block_interval_rejects_zero_repetitions():
    with pytest.raises(ContractError, match="repetitions"):
        paired_block_interval([1.0, 2.0], [0.0, 0.0], [0, 1], repetitions=0)


def test_block_interval_rejects_atom_matrix_as_geometry_axis():
    with pytest.raises(ContractError, match="one-dimensional"):
        paired_block_interval(np.ones((2, 3)), np.zeros((2, 3)), np.ones((2, 3)))
