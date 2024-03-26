import numpy as np
import pytest
import torch

from asterion.artifacts.io import read_json, write_json
from asterion.contracts.errors import ContractError, IntegrityError
from asterion.contracts.geometry import Geometry
from asterion.evaluation.benchmark import memory_stress


class BoundedInference(torch.nn.Module):
    def __init__(self, fail_after=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
        self.calls = 0
        self.fail_after = fail_after

    def forward(self, batch):
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError("injected retained inference failure")
        self.calls += 1
        return self.weight.sum().reshape(1), batch.positions * self.weight


@pytest.fixture
def geometry():
    return Geometry(
        np.array([6, 1]), np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), "stress-contract-fixture"
    )


def test_stress_completion_matches_retained_progress(tmp_path, geometry):
    # A short duration is only a unit contract test, never 1800-second acceptance evidence.
    model = BoundedInference()
    path = tmp_path / "stress.json"
    result = memory_stress(model, geometry, path, seconds=0.02)
    assert result["completed_calls"] == model.calls > 0
    assert result["elapsed_seconds"] >= 0.02
    assert read_json(path) == result
    assert read_json(path.with_suffix(".progress.json")) == {
        **result,
        "status": "passed",
        "error": None,
    }


def test_stress_failure_retains_observed_calls_without_completed_output(tmp_path, geometry):
    model = BoundedInference(fail_after=2)
    path = tmp_path / "stress.json"
    with pytest.raises(RuntimeError, match="injected"):
        memory_stress(model, geometry, path, seconds=30)
    progress = read_json(path.with_suffix(".progress.json"))
    assert progress["status"] == "interrupted" and progress["completed_calls"] == model.calls == 2
    assert progress["error"].endswith("injected retained inference failure")
    assert not path.exists()


@pytest.mark.parametrize("duration", [0.0, -1.0, float("inf"), float("nan")])
def test_stress_rejects_invalid_duration_without_fake_completion(tmp_path, geometry, duration):
    with pytest.raises(ContractError, match="finite positive"):
        memory_stress(BoundedInference(), geometry, tmp_path / "stress.json", seconds=duration)
    assert not list(tmp_path.iterdir())


def test_stress_refuses_to_replace_retained_partial_identity(tmp_path, geometry):
    path = tmp_path / "stress.json"
    write_json(path.with_suffix(".progress.json"), {"retained": "interrupted attempt"})
    with pytest.raises(IntegrityError, match="already exists"):
        memory_stress(BoundedInference(), geometry, path, seconds=0.02)
    assert read_json(path.with_suffix(".progress.json")) == {"retained": "interrupted attempt"}
