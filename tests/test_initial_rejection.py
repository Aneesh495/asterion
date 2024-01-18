import json
import re
from dataclasses import replace

import numpy as np
import pytest

from asterion.artifacts.io import atomic_npz, file_hash, read_json, write_json
from asterion.artifacts.rejection import verify_rejected_model
from asterion.contracts.errors import ContractError, IntegrityError
from asterion.contracts.geometry import collate
from asterion.dynamics.analytic import ForceResult
from asterion.dynamics.detectors import Detector, ValidityPolicy
from asterion.dynamics.integrator import SimulationConfig, simulate
from asterion.dynamics.learned import LearnedEvaluator
from asterion.dynamics.rejection import reject_initial
from asterion.evaluation.replay import replay_trajectory
from asterion.viewer.export import export_viewer


@pytest.fixture
def initial(geometry, tmp_path):
    return geometry, tmp_path / "rejection", SimulationConfig(steps=5)


def run(initial, result):
    geometry, path, config = initial
    return simulate(
        lambda positions, numbers: result,
        geometry.numbers,
        geometry.positions,
        np.zeros_like(geometry.positions),
        path,
        config,
    )


def test_initial_nonfinite_energy_is_retained_without_a_valid_frame(initial):
    geometry, path, _ = initial
    record = run(initial, ForceResult(float("nan"), np.zeros_like(geometry.positions)))
    assert record["valid_frames"] == record["completed_steps"] == 0
    assert not (path / "trajectory.npz").exists()
    with np.load(path / "failure.npz", allow_pickle=False) as raw:
        assert np.isnan(raw["energy"])
    assert replay_trajectory(path)["max_energy_drift_eV"] is None


def test_initial_nonfinite_force_preserves_the_invalid_value(initial):
    geometry, path, _ = initial
    forces = np.zeros_like(geometry.positions)
    forces[0, 2] = float("inf")
    run(initial, ForceResult(0.0, forces))
    with np.load(path / "failure.npz", allow_pickle=False) as raw:
        assert np.isposinf(raw["forces"][0, 2])
    assert replay_trajectory(path)["rejected_initial"] is True


def test_initial_wrong_force_shape_has_no_invented_cartesian_axis(initial):
    geometry, path, _ = initial
    record = run(initial, ForceResult(0.0, np.zeros((len(geometry.numbers), 2))))
    assert record["reason"] == "invalid-force-shape"
    with np.load(path / "failure.npz", allow_pickle=False) as raw:
        assert raw["forces"].shape == (len(geometry.numbers), 2)
    replay_trajectory(path)


def test_initial_evaluation_error_preserves_inputs_and_unknown_outputs(initial):
    geometry, path, config = initial

    def fail(positions, numbers):
        raise ContractError("retained initial evaluation failure")

    record = simulate(
        fail, geometry.numbers, geometry.positions, np.zeros_like(geometry.positions), path, config
    )
    assert record["evaluation_error"] == {
        "type": "ContractError",
        "message": "retained initial evaluation failure",
    }
    with np.load(path / "failure.npz", allow_pickle=False) as raw:
        assert set(raw.files) == {"positions", "velocities", "numbers"}
    assert replay_trajectory(path)["completed_steps"] == 0


def test_initial_collision_is_preserved_before_force_evaluation(initial):
    geometry, path, config = initial
    positions = geometry.positions.copy()
    positions[1] = positions[0] + 0.1

    def must_not_run(positions, numbers):
        pytest.fail("collision reached force evaluator")

    record = simulate(
        must_not_run, geometry.numbers, positions, np.zeros_like(positions), path, config
    )
    assert record["reason"] == "atomic-collision"
    assert record["observation_kind"] == "geometry-rejection"
    replay_trajectory(path)


def test_initial_uncertainty_contract_error_is_preserved_as_raw_output(initial):
    geometry, path, _ = initial
    record = run(
        initial, ForceResult(0.0, np.zeros_like(geometry.positions), uncertainty_available=True)
    )
    assert record["reason"] == "invalid-uncertainty-contract"
    assert record["uncertainty_quantity"] == "unavailable"
    replay_trajectory(path)


def test_initial_negative_uncertainty_never_enters_valid_history(initial):
    geometry, path, _ = initial
    record = run(initial, ForceResult(0.0, np.zeros_like(geometry.positions), uncertainty=-1))
    assert record["reason"] == "invalid-uncertainty"
    assert not (path / "trajectory.npz").exists()
    replay_trajectory(path)


def test_initial_rejection_has_no_resumable_integration_state(initial):
    geometry, path, config = initial
    run(initial, ForceResult(float("nan"), np.zeros_like(geometry.positions)))
    with pytest.raises(IntegrityError, match="no valid integration checkpoint"):
        simulate(
            LearnedEvaluator,
            geometry.numbers,
            geometry.positions,
            np.zeros_like(geometry.positions),
            path,
            config,
            resume=True,
        )


def test_initial_rejection_replay_rejects_inflated_completed_work(initial):
    geometry, path, _ = initial
    run(initial, ForceResult(float("nan"), np.zeros_like(geometry.positions)))
    record = read_json(path / "trajectory.json")
    record["completed_steps"] = 1
    record["physical_horizon_fs"] = 0.1
    write_json(path / "trajectory.json", record)
    with pytest.raises(IntegrityError, match="invents valid frames"):
        replay_trajectory(path)


def test_initial_rejection_replay_rejects_changed_raw_observation(initial):
    geometry, path, _ = initial
    run(initial, ForceResult(float("nan"), np.zeros_like(geometry.positions)))
    with (path / "failure.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(IntegrityError, match="bytes changed"):
        replay_trajectory(path)


def test_initial_rejection_replay_rejects_fabricated_failure_reason(initial):
    geometry, path, _ = initial
    run(initial, ForceResult(float("nan"), np.zeros_like(geometry.positions)))
    record = read_json(path / "trajectory.json")
    record["reason"] = "extreme-force"
    write_json(path / "trajectory.json", record)
    with pytest.raises(IntegrityError, match="raw support"):
        replay_trajectory(path)


def test_initial_rejection_cannot_also_publish_a_valid_trajectory(initial):
    geometry, path, _ = initial
    run(initial, ForceResult(float("nan"), np.zeros_like(geometry.positions)))
    atomic_npz(path / "trajectory.npz", positions=geometry.positions)
    with pytest.raises(IntegrityError, match="valid trajectory"):
        replay_trajectory(path)


def test_finite_valid_force_cannot_be_claimed_as_initial_rejection(initial):
    geometry, path, config = initial
    path.mkdir()
    reject_initial(
        path,
        geometry.numbers,
        geometry.positions,
        np.zeros_like(geometry.positions),
        config,
        Detector(geometry.numbers, geometry.positions, ValidityPolicy()),
        "analytic",
        "nonfinite-energy-or-force",
        result=ForceResult(0.0, np.zeros_like(geometry.positions)),
    )
    with pytest.raises(IntegrityError, match="raw support"):
        replay_trajectory(path)


def test_saved_model_initial_resource_failure_is_independently_reproduced(model, initial):
    geometry, path, config = initial
    model.config = replace(model.config, max_message_elements=1)
    result = simulate(
        LearnedEvaluator(model),
        geometry.numbers,
        geometry.positions,
        np.zeros_like(geometry.positions),
        path,
        config,
    )
    assert result["observation_kind"] == "force-error"
    assert result["evaluation_error"]["type"] == "ResourceError"
    verify_rejected_model(path, model)


def test_forged_initial_nan_output_fails_saved_weight_replay(model, initial):
    geometry, path, _ = initial
    run(initial, ForceResult(float("nan"), np.zeros_like(geometry.positions)))
    with pytest.raises(IntegrityError, match="saved weights"):
        verify_rejected_model(path, model)


def test_rejected_initial_viewer_shows_unavailable_without_fabricated_energy(model, initial):
    geometry, path, _ = initial
    run(initial, ForceResult(float("nan"), np.zeros_like(geometry.positions)))
    energy, force = model(collate([geometry]))
    prediction = path.parent / "prediction.npz"
    atomic_npz(
        prediction,
        energy=energy.detach().numpy(),
        target_energy=energy.detach().numpy(),
        force=force.detach().numpy(),
        target_force=force.detach().numpy(),
        positions=geometry.positions,
        numbers=geometry.numbers,
        pointers=[0, len(geometry.numbers)],
        identities=[geometry.identity],
    )
    output = export_viewer(prediction, path, path.parent / "rejected.html", {})
    data = json.loads(re.search(r"const data=(.*?);\nconst", output.read_text()).group(1))
    assert data["frames"][0]["energy"] is data["frames"][0]["force"] is None
    assert data["history"]["potential"] == []
    assert data["lineage"]["rejected_initial_sha256"] == file_hash(path / "failure.npz")
