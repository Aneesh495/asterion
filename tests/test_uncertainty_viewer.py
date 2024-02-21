import json
import re

import numpy as np
import pytest

from asterion.artifacts.io import atomic_npz, file_hash, read_json, write_json
from asterion.artifacts.viewer import verify_viewer
from asterion.contracts.errors import IntegrityError
from asterion.dynamics.analytic import ForceResult, Harmonic
from asterion.dynamics.detectors import Detector, ValidityPolicy
from asterion.dynamics.integrator import SimulationConfig, simulate
from asterion.evaluation.replay import replay_trajectory
from asterion.viewer.export import export_viewer


class PerfectBound(Harmonic):
    def __call__(self, positions, numbers):
        result = super().__call__(positions, numbers)
        result.uncertainty_available = True
        result.uncertainty_quantity = "calibrated-mae-bound"
        return result


@pytest.fixture
def artifacts(tmp_path):
    numbers = np.array([6, 6])
    positions = np.array([[0.1, 0.0, 0.0], [3.1, 0.0, 0.0]])
    equilibrium = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    evaluator = PerfectBound(equilibrium)
    result = evaluator(positions, numbers)
    run = tmp_path / "trajectory"
    simulate(
        evaluator,
        numbers,
        positions,
        np.zeros_like(positions),
        run,
        SimulationConfig(steps=4),
        lineage="analytic-perfect-bound-fixture",
    )
    prediction = tmp_path / "predictions.npz"
    atomic_npz(
        prediction,
        energy=[result.energy],
        target_energy=[result.energy],
        force=result.forces,
        target_force=result.forces,
        positions=positions,
        numbers=numbers,
        pointers=[0, 2],
        identities=["analytic-fixture"],
    )
    return prediction, run, tmp_path / "viewer.html"


def payload(path):
    return json.loads(re.search(r"const data=(.*?);\nconst", path.read_text()).group(1))


def test_available_zero_bound_survives_original_integration(artifacts):
    _, run, _ = artifacts
    with np.load(run / "trajectory.npz") as raw:
        assert raw["uncertainty_available"].all()
        assert np.array_equal(raw["uncertainty"], np.zeros(5))
        assert np.array_equal(raw["uncertainty_quantity"], np.full(5, 2))
    assert read_json(run / "trajectory.json")["uncertainty_quantity"] == "calibrated-mae-bound"


def test_viewer_keeps_zero_value_distinct_from_unavailable_score(artifacts):
    prediction, run, output = artifacts
    export_viewer(prediction, run, output, {})
    data = payload(output)
    assert all(data["history"]["uncertainty_available"])
    assert all(frame["uncertainty_available"] for frame in data["frames"])
    assert data["trajectory"]["uncertainty_quantity"] == "calibrated-mae-bound"


def test_viewer_binds_actual_input_artifact_bytes(artifacts):
    prediction, run, output = artifacts
    export_viewer(prediction, run, output, {"predictions_sha256": "invented"})
    data = payload(output)
    assert data["lineage"]["predictions_sha256"] == file_hash(prediction)
    assert data["lineage"]["trajectory_sha256"] == file_hash(run / "trajectory.npz")


def test_viewer_verifier_checks_raw_heldout_and_complete_physical_history(artifacts):
    prediction, run, output = artifacts
    export_viewer(prediction, run, output, {})
    assert verify_viewer(output, prediction, run) == {
        "samples": 1,
        "frames": 5,
        "acquisition_rounds": 0,
    }


def change_viewer(output, intervention):
    data = payload(output)
    intervention(data)
    text = output.read_text()
    original = re.search(r"const data=(.*?);\nconst", text).group(1)
    output.write_text(text.replace(original, json.dumps(data)))


def test_viewer_verifier_rejects_changed_raw_force_in_embedded_payload(artifacts):
    prediction, run, output = artifacts
    export_viewer(prediction, run, output, {})
    change_viewer(output, lambda data: data["samples"][0]["force"][0].__setitem__(0, 5))
    with pytest.raises(IntegrityError, match="held-out observation"):
        verify_viewer(output, prediction, run)


def test_viewer_verifier_rejects_zero_bound_relabelled_unavailable(artifacts):
    prediction, run, output = artifacts
    export_viewer(prediction, run, output, {})
    change_viewer(
        output, lambda data: data["frames"][0].__setitem__("uncertainty_available", False)
    )
    with pytest.raises(IntegrityError, match="trajectory frame"):
        verify_viewer(output, prediction, run)


def test_viewer_verifier_rejects_modified_executable_template(artifacts):
    prediction, run, output = artifacts
    export_viewer(prediction, run, output, {})
    output.write_text(output.read_text().replace("function render()", "function altered()"))
    with pytest.raises(IntegrityError, match="executable markup"):
        verify_viewer(output, prediction, run)


def test_viewer_rejects_changed_physical_trajectory(artifacts):
    prediction, run, output = artifacts
    with (run / "trajectory.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(IntegrityError, match="bytes"):
        export_viewer(prediction, run, output, {})


def test_detector_rejects_score_available_without_a_physical_quantity():
    result = ForceResult(0.0, np.zeros((2, 3)), uncertainty_available=True)
    detector = Detector(
        np.array([6, 6]), np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]), ValidityPolicy()
    )
    assert detector.forces(result) == "invalid-uncertainty-contract"


def test_detector_rejects_calibrated_quantity_marked_unavailable():
    result = ForceResult(0.0, np.zeros((2, 3)), uncertainty_quantity="calibrated-mae-bound")
    detector = Detector(
        np.array([6, 6]), np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]), ValidityPolicy()
    )
    assert detector.forces(result) == "invalid-uncertainty-contract"


def test_detector_accepts_exact_zero_bound_when_available():
    result = ForceResult(
        0.0,
        np.zeros((2, 3)),
        uncertainty_available=True,
        uncertainty_quantity="calibrated-mae-bound",
    )
    detector = Detector(
        np.array([6, 6]),
        np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        ValidityPolicy(maximum_uncertainty=1),
    )
    assert detector.forces(result) is None


def corrupt(run, key, value):
    with np.load(run / "trajectory.npz") as raw:
        arrays = {name: raw[name] for name in raw.files}
    arrays[key] = value(arrays[key])
    atomic_npz(run / "trajectory.npz", **arrays)
    meta = read_json(run / "trajectory.json")
    meta["trajectory_sha256"] = file_hash(run / "trajectory.npz")
    write_json(run / "trajectory.json", meta)


def test_replay_rejects_rehashed_uncertainty_availability_intervention(artifacts):
    _, run, _ = artifacts
    corrupt(run, "uncertainty_available", lambda values: np.zeros_like(values))
    with pytest.raises(IntegrityError, match="availability"):
        replay_trajectory(run)


def test_replay_rejects_rehashed_uncertainty_quantity_intervention(artifacts):
    _, run, _ = artifacts
    corrupt(run, "uncertainty_quantity", lambda values: np.ones_like(values))
    with pytest.raises(IntegrityError, match="mislabels"):
        replay_trajectory(run)


def test_replay_rejects_fractional_quantity_code(artifacts):
    _, run, _ = artifacts
    corrupt(run, "uncertainty_quantity", lambda values: values.astype(float))
    with pytest.raises(IntegrityError, match="quantity"):
        replay_trajectory(run)


def test_replay_rejects_changed_uncertainty_kind_during_one_trajectory(artifacts):
    _, run, _ = artifacts

    def changed(values):
        values[-1] = 1
        return values

    corrupt(run, "uncertainty_quantity", changed)
    with pytest.raises(IntegrityError, match="mixes"):
        replay_trajectory(run)
