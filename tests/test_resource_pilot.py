import shutil

import numpy as np
import pytest

from asterion.artifacts.io import read_json, write_json
from asterion.artifacts.resource_pilot import verify_resource_pilot
from asterion.contracts.errors import IntegrityError
from asterion.evaluation.resource_pilot import resource_pilot


@pytest.fixture(scope="module")
def pilot(tmp_path_factory):
    root = tmp_path_factory.mktemp("resource-pilot")
    storage = root / "storage"
    (storage / "rmd17").mkdir(parents=True)
    positions = np.array([[[0.0, 0.0, 0.0], [1.1, 0.2, 0.0], [0.0, -0.4, 1.3]]] * 8)
    data = storage / "rmd17/rmd17_fixture.npz"
    np.savez_compressed(
        data,
        nuclear_charges=np.array([6, 1, 8]),
        coords=positions,
        energies=np.zeros(8),
        forces=np.zeros_like(positions),
        old_indices=np.arange(8),
    )
    config = {
        "molecules": ["fixture"],
        "models": ["invariant", "equivariant"],
        "architecture": {"width": 2, "depth": 1, "radial": 3, "lmax": 2, "cutoff": 5.0},
        "training": {"dtype": "float64", "device": "cpu"},
    }
    resource_pilot(config, storage, root / "pilot")
    return root, config, {"rmd17_fixture.npz": data}


@pytest.fixture
def observation(pilot, tmp_path):
    original, config, datasets = pilot
    shutil.copytree(original / "pilot", tmp_path / "pilot")
    return tmp_path, config, datasets, read_json(tmp_path / "pilot/pilot.json")


def verify(observation):
    root, config, datasets, report = observation
    write_json(root / "pilot/pilot.json", report)
    return verify_resource_pilot(root, "pilot/pilot.json", config, report["source"], datasets)


def test_resource_pilot_replays_untrained_parameters_and_mixed_gradients(observation):
    assert verify(observation) == {"systems": 2, "samples": 24}


def test_resource_pilot_rejects_missing_backward_repetition(observation):
    row = next(row for row in observation[3]["samples"] if row["mode"] == "force-loss-backward")
    observation[3]["samples"].remove(row)
    with pytest.raises(IntegrityError, match="omitted required"):
        verify(observation)


def test_resource_pilot_rejects_architecture_relabelled_without_parameter_change(observation):
    observation[3]["systems"][0]["architecture"]["width"] += 1
    with pytest.raises(IntegrityError, match="architecture differs"):
        verify(observation)


def test_resource_pilot_rejects_false_training_or_accuracy_claim(observation):
    observation[3]["training_or_accuracy_claim"] = True
    with pytest.raises(IntegrityError, match="untrained mechanism"):
        verify(observation)


def test_resource_pilot_rejects_unsupported_campaign_duration_projection(observation):
    observation[3]["full_campaign_hours"] = 72
    with pytest.raises(IntegrityError, match="untrained mechanism"):
        verify(observation)


def test_resource_pilot_rejects_substituted_gradient_norm(observation):
    row = next(
        row
        for row in observation[3]["samples"]
        if row["mode"] == "force-loss-backward" and row["repetition"] == 0
    )
    row["gradient_squared_norm"] += 1
    with pytest.raises(IntegrityError, match="mixed-gradient observation"):
        verify(observation)


def test_resource_pilot_rejects_invented_cpu_device_memory(observation):
    observation[3]["samples"][0]["peak_device_bytes"] = 1
    with pytest.raises(IntegrityError, match="backend, memory"):
        verify(observation)


def test_resource_pilot_rejects_duplicate_case_identity(observation):
    observation[3]["samples"].append(observation[3]["samples"][0])
    with pytest.raises(IntegrityError, match="work matrix"):
        verify(observation)


def test_resource_pilot_refuses_to_overwrite_prior_measurements(pilot):
    root, config, _ = pilot
    with pytest.raises(IntegrityError, match="already exists"):
        resource_pilot(config, root / "storage", root / "pilot")
