import numpy as np
import pytest

from asterion.artifacts.io import object_hash, read_json, write_json
from asterion.contracts.errors import ContractError, LabelAccessError
from asterion.contracts.geometry import Chemistry
from asterion.data.dataset import CoordinatePool, LabelOracle
from asterion.data.ledger import RevealLedger
from asterion.data.splits import Split


@pytest.fixture
def pool():
    return CoordinatePool(
        np.array([6, 1]),
        np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]] * 4),
        np.arange(4),
        "row-fixture",
        "fixture",
        Chemistry(),
    )


def oracle(pool):
    return LabelOracle(pool, np.arange(4, dtype=float), np.zeros((4, 2, 3)))


def test_coordinate_lookup_rejects_boolean_as_integer_alias(pool):
    with pytest.raises(ContractError, match="integer identity"):
        pool.geometry(True)


def test_label_access_rejects_fractional_row_before_integer_conversion(pool):
    with pytest.raises(LabelAccessError, match="fractions"):
        oracle(pool).labels([1.5], [1], "train")


def test_label_access_rejects_string_row_before_integer_conversion(pool):
    with pytest.raises(LabelAccessError, match="strings"):
        oracle(pool).labels(["1"], [1], "train")


def test_label_access_rejects_boolean_allow_list(pool):
    with pytest.raises(LabelAccessError, match="masks"):
        oracle(pool).labels([1], [True], "train")


def test_label_access_preserves_native_integer_identities(pool):
    labels = oracle(pool).labels(
        np.array([1, 2], dtype=np.int64), np.array([1, 2], dtype=np.uint64), "train"
    )
    assert [label.geometry.identity for label in labels] == [
        "fixture:row-fixture:1",
        "fixture:row-fixture:2",
    ]


def split():
    return Split("row-fixture", "fixture", 1, (0, 1, 2), (3,), (4,), (5,))


def test_reveal_rejects_fractional_identity_without_publishing_round(tmp_path):
    ledger = RevealLedger(split(), tmp_path)
    with pytest.raises(LabelAccessError, match="fractions"):
        ledger.reveal([1.5], "uniform")
    assert not list(tmp_path.glob("round-*.json")) and ledger.revealed == ()


def test_rehashed_persisted_reveal_cannot_alias_float_to_integer(tmp_path):
    ledger = RevealLedger(split(), tmp_path)
    ledger.reveal([1], "initial")
    path = tmp_path / "round-0000.json"
    record = read_json(path)
    record["rows"] = [1.0]
    record["digest"] = object_hash({key: value for key, value in record.items() if key != "digest"})
    write_json(path, record)
    with pytest.raises(LabelAccessError, match="fractions"):
        RevealLedger(split(), tmp_path)


def test_rehashed_persisted_reveal_cannot_hide_repeated_identity(tmp_path):
    ledger = RevealLedger(split(), tmp_path)
    ledger.reveal([1], "initial")
    path = tmp_path / "round-0000.json"
    record = read_json(path)
    record["rows"] = [1, 1]
    record["digest"] = object_hash({key: value for key, value in record.items() if key != "digest"})
    write_json(path, record)
    with pytest.raises(LabelAccessError, match="repeats"):
        RevealLedger(split(), tmp_path)
