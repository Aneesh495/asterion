from dataclasses import asdict

import numpy as np
import pytest

from asterion.acquisition.policy import choose
from asterion.artifacts.io import read_json, write_json
from asterion.contracts.errors import ContractError, IntegrityError, LabelAccessError
from asterion.contracts.geometry import Chemistry
from asterion.data.dataset import CoordinatePool, LabelOracle
from asterion.data.ledger import RevealLedger
from asterion.data.splits import Split, geometry_descriptors
from asterion.uncertainty.committee import Calibration


@pytest.fixture
def split():
    return Split(
        "data-hash",
        "test",
        1,
        tuple(range(1000)),
        tuple(range(1000, 1010)),
        tuple(range(1010, 1020)),
        tuple(range(1020, 1030)),
    ).validate(1030)


def test_immutable_reveal_chain(tmp_path, split):
    ledger = RevealLedger(split, tmp_path)
    ledger.reveal(range(100), "initial")
    ledger.reveal(range(100, 300), "uniform")
    assert RevealLedger(split, tmp_path).revealed == tuple(range(300))
    with pytest.raises(LabelAccessError):
        ledger.reveal([99], "duplicate")
    with pytest.raises(LabelAccessError):
        ledger.reveal(split.evaluation, "evaluation-leak")
    path = tmp_path / "round-0000.json"
    value = read_json(path)
    value["rows"][0] = 900
    write_json(path, value)
    with pytest.raises(IntegrityError):
        RevealLedger(split, tmp_path)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("policy", ["uniform", "diversity", "uncertainty-diversity"])
def test_exact_budget_and_no_repeat(seed, policy):
    rng = np.random.default_rng(seed)
    rows = np.arange(50)
    descriptor = rng.normal(size=(50, 5))
    scores = rng.random(50)
    chosen = choose(rows, descriptor, range(10), 20, policy, seed, scores)
    assert len(chosen.rows) == len(set(chosen.rows)) == 20
    assert not set(chosen.rows) & set(range(10))
    assert chosen == choose(rows, descriptor, range(10), 20, policy, seed, scores)


def test_label_oracle_access_boundary():
    pool = CoordinatePool(
        np.array([1, 1]),
        np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]] * 3),
        np.arange(3),
        "hash",
        "test",
        Chemistry(),
    )
    oracle = LabelOracle(pool, np.zeros(3), np.zeros((3, 2, 3)))
    assert len(oracle.labels([0], [0], "train")) == 1
    with pytest.raises(LabelAccessError):
        oracle.labels([1], [0], "train")
    with pytest.raises(LabelAccessError):
        oracle.labels([0], [0], "implicit")


def test_descriptor_permutation_and_rotation():
    rng = np.random.default_rng(62)
    xyz = rng.normal(size=(4, 3))
    permutation = np.array([0, 2, 1, 3])
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    pool = CoordinatePool(
        np.array([6, 1, 1, 8]),
        np.array([xyz, xyz[permutation] @ rotation]),
        np.arange(2),
        "hash",
        "test",
        Chemistry(),
    )
    descriptor = geometry_descriptors(pool, [0, 1])
    np.testing.assert_allclose(descriptor[0], descriptor[1], atol=1e-14)


def test_finite_sample_calibration():
    score = np.linspace(0.01, 0.1, 100)
    error = 2 * score
    calibration = Calibration.fit(score, error, 0.1, ("a", "b", "c"), tuple(map(str, range(100))))
    assert abs(calibration.multiplier - 2) < 1e-12
    np.testing.assert_allclose(calibration.bounds(score), error)


@pytest.mark.parametrize("field", ["train_pool", "validation", "calibration", "evaluation"])
def test_split_rejects_empty_roles(split, field):
    value = asdict(split)
    value[field] = ()
    with pytest.raises(ContractError):
        Split(**value).validate(1030)
