import json
from copy import deepcopy

import pytest

from asterion.artifacts.io import read_json_lines
from asterion.artifacts.reference import reference_history
from asterion.contracts.errors import IntegrityError


@pytest.fixture
def journal(tmp_path):
    rows = [
        {"mode": "eval", "epoch": None, "loss": 4.0, "time": 0.1, "head": "Default", "mae_f": 1.0}
    ]
    for epoch in range(2):
        rows.extend([{"mode": "opt", "epoch": epoch, "loss": 3.0, "time": 0.2} for _ in range(2)])
        rows.append(
            {
                "mode": "eval",
                "epoch": epoch,
                "head": "Default",
                "loss": 2.0,
                "time": 0.1,
                "mae_f": 0.5,
            }
        )
    path = tmp_path / "training.txt"
    return path, rows


def save(case):
    path, rows = case
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_reference_history_counts_actual_nonpadding_optimizer_batches(journal):
    result = reference_history(save(journal), 2, 8, 4)
    assert result == {"epochs": 2, "updates": 4, "examples_processed": 16, "initial_validation": 1}


def test_reference_history_rejects_declared_epoch_without_optimizer_work(journal):
    path, rows = deepcopy(journal)
    rows.pop(1)
    with pytest.raises(IntegrityError, match="incomplete"):
        reference_history(save((path, rows)), 2, 8, 4)


def test_reference_history_rejects_duplicated_optimizer_batch(journal):
    path, rows = deepcopy(journal)
    rows.append(rows[1])
    with pytest.raises(IntegrityError, match="incomplete"):
        reference_history(save((path, rows)), 2, 8, 4)


def test_reference_history_rejects_hidden_validation_epoch(journal):
    path, rows = deepcopy(journal)
    rows.pop()
    with pytest.raises(IntegrityError, match="incomplete"):
        reference_history(save((path, rows)), 2, 8, 4)


def test_reference_history_rejects_padding_changed_label_denominator(journal):
    with pytest.raises(IntegrityError, match="drop-last"):
        reference_history(save(journal), 2, 9, 4)


def test_reference_history_rejects_nonfinite_optimizer_loss(journal):
    path, rows = deepcopy(journal)
    rows[1]["loss"] = float("nan")
    with pytest.raises(IntegrityError, match="nonfinite"):
        reference_history(save((path, rows)), 2, 8, 4)


def test_reference_history_rejects_epoch_from_another_training_workload(journal):
    path, rows = deepcopy(journal)
    rows[1]["epoch"] = 2
    with pytest.raises(IntegrityError, match="frozen"):
        reference_history(save((path, rows)), 2, 8, 4)


def test_reference_history_rejects_other_validation_head(journal):
    path, rows = deepcopy(journal)
    rows[-1]["head"] = "foundation"
    with pytest.raises(IntegrityError, match="identity"):
        reference_history(save((path, rows)), 2, 8, 4)


def test_line_journal_rejects_nonobject_records(tmp_path):
    path = tmp_path / "journal"
    path.write_text("[]\n")
    with pytest.raises(IntegrityError, match="object"):
        read_json_lines(path)


def test_line_journal_rejects_empty_observations(tmp_path):
    path = tmp_path / "journal"
    path.touch()
    with pytest.raises(IntegrityError, match="empty"):
        read_json_lines(path)


def test_line_journal_bounds_single_row_allocation(tmp_path):
    path = tmp_path / "journal"
    path.write_text(json.dumps({"padding": "x" * 16384}) + "\n")
    with pytest.raises(IntegrityError, match="row budget"):
        read_json_lines(path)


def test_line_journal_bounds_actual_completed_population(journal):
    with pytest.raises(IntegrityError, match="row budget"):
        read_json_lines(save(journal), maximum_rows=3)
