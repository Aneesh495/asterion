import numpy as np
import pytest
import torch

from asterion.artifacts.controls import refresh_manifest
from asterion.artifacts.io import read_json, write_json
from asterion.artifacts.verify import check_manifest, independent_metrics
from asterion.contracts.errors import IntegrityError
from asterion.evaluation.analytic_campaign import analytic_campaign
from asterion.evaluation.recovery import training_recovery
from asterion.training.checkpoint import load_checkpoint, save_checkpoint


def test_safe_checkpoint_roundtrip(model, tmp_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    metadata = {"origin": "unit-test", "epoch": 0, "updates": 0}
    record = save_checkpoint(tmp_path / "saved", model, optimizer, metadata)
    loaded, replay = load_checkpoint(tmp_path / "saved")
    assert record["identity"] == replay["identity"]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[name], rtol=0, atol=0)


def test_checkpoint_metadata_tamper(model, tmp_path):
    save_checkpoint(tmp_path / "saved", model, None, {"origin": "unit-test"})
    path = tmp_path / "saved" / "checkpoint.json"
    data = read_json(path)
    data["origin"] = "original-force-loss-training"
    write_json(path, data)
    with pytest.raises(IntegrityError):
        load_checkpoint(tmp_path / "saved")


@pytest.mark.parametrize("intervention", ["missing", "changed", "extra"])
def test_manifest_negative_paths(tmp_path, intervention):
    write_json(tmp_path / "raw.json", {"value": 17})
    refresh_manifest(tmp_path)
    check_manifest(tmp_path)
    if intervention == "missing":
        (tmp_path / "raw.json").unlink()
    elif intervention == "changed":
        write_json(tmp_path / "raw.json", {"value": 18})
    else:
        write_json(tmp_path / "extra.json", {"value": 19})
    with pytest.raises(IntegrityError):
        check_manifest(tmp_path)


def test_independent_geometry_denominator():
    arrays = {
        "energy": np.array([0.0, 0.0]),
        "target_energy": np.array([1.0, 3.0]),
        "force": np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0]]),
        "target_force": np.zeros((3, 3)),
        "pointers": np.array([0, 1, 3]),
        "positions": np.zeros((3, 3)),
    }
    metric = independent_metrics(arrays)
    assert metric["force_mae_eV_angstrom"] == 2.0
    assert metric["energy_mae_eV"] == 2.0


def test_real_force_training_resume(tmp_path):
    result = training_recovery(tmp_path, cases=1)
    assert result[0]["status"] == "passed"
    history = read_json(tmp_path / "whole-0" / "history.json")
    assert history[0]["force_gradient_squared"] > 0
    assert history[-1]["updates"] == 12


def test_independent_analytic_campaign(tmp_path):
    outcomes = analytic_campaign(tmp_path, fixtures=2)
    assert len(outcomes) == 12
