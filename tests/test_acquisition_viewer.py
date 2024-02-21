import shutil

import numpy as np
import pytest

from asterion.acquisition.campaign import acquisition_campaign
from asterion.artifacts.acquisition_analysis import verify_uncertainty_summary
from asterion.artifacts.io import atomic_npz, file_hash, read_json, write_json
from asterion.artifacts.source import project_root
from asterion.contracts.errors import ContractError, IntegrityError
from asterion.evaluation.metrics import uncertainty_metrics
from asterion.viewer.acquisition import AcquisitionArtifacts, acquisition_payload


def test_independent_uncertainty_ranking_handles_tied_observations():
    scores = np.array([1.0, 1.0, 2.0, 3.0, 3.0])
    errors = np.array([1.0, 2.0, 1.0, 4.0, 4.0])
    bounds = np.array([0.0, 0.0, 2.0, 4.0, 4.0])
    summary = uncertainty_metrics(scores, errors, bounds)
    verify_uncertainty_summary(summary, scores, errors, bounds)
    assert summary["coverage"] == 0.6 and 0 < summary["spearman"] < 1


def test_constant_uncertainty_ranking_is_unavailable():
    summary = uncertainty_metrics(np.zeros(3), np.arange(3), np.zeros(3))
    verify_uncertainty_summary(summary, np.zeros(3), np.arange(3), np.zeros(3))
    assert summary["spearman"] is None and summary["mean_bound"] == 0


def test_uncertainty_mean_bound_remains_finite_at_representable_scale():
    bounds = np.full(4, 1e308)
    summary = uncertainty_metrics(np.arange(4), np.arange(4), bounds)
    verify_uncertainty_summary(summary, np.arange(4), np.arange(4), bounds)
    assert summary["mean_bound"] == 1e308


def test_uncertainty_summary_rejects_nonfinite_bound():
    with pytest.raises(ContractError, match="finite nonnegative"):
        uncertainty_metrics(np.arange(3), np.arange(3), np.full(3, np.inf))


def test_independent_uncertainty_ranking_rejects_invented_unavailable_result():
    summary = uncertainty_metrics(np.arange(3), np.arange(3), np.arange(3))
    summary["spearman"] = None
    with pytest.raises(IntegrityError, match="ranking"):
        verify_uncertainty_summary(summary, np.arange(3), np.arange(3), np.arange(3))


def test_viewer_without_acquisition_preserves_unavailable_state():
    assert acquisition_payload(None) == {
        "status": "unavailable",
        "reason": "No acquisition observations attached",
        "rounds": [],
    }


@pytest.fixture(scope="module")
def acquired(tmp_path_factory):
    root = tmp_path_factory.mktemp("original-acquisition-viewer")
    author = root / "storage/rmd17"
    author.mkdir(parents=True)
    rng = np.random.default_rng(491)
    xyz = np.zeros((3000, 2, 3))
    xyz[:, 1, 0] = 1.1 + rng.normal(0, 0.04, 3000)
    force = np.zeros_like(xyz)
    force[:, 0, 0] = xyz[:, 1, 0] - 1.1
    force[:, 1, 0] = -force[:, 0, 0]
    atomic_npz(
        author / "rmd17_ethanol.npz",
        coords=xyz,
        nuclear_charges=[6, 1],
        energies=(xyz[:, 1, 0] - 1.1) ** 2 / 2,
        forces=force,
        old_indices=np.arange(3000),
    )
    np.savetxt(author / "index_train_01.csv", np.arange(1000), fmt="%d")
    np.savetxt(author / "index_test_01.csv", np.arange(1000, 2000), fmt="%d")
    config = read_json(project_root() / "configs/ci.json")
    config.update(
        epochs=1,
        acquisition_budgets=[2, 4],
        validation_size=8,
        calibration_size=10,
        evaluation_size=12,
    )
    config["architecture"].update(width=2, radial=2)
    acquisition_campaign(config, root / "storage", root / "acquisition")
    source = read_json(root / "acquisition/ethanol-1-uniform/round-0-member-1/SOURCE_INPUTS.json")
    inputs = AcquisitionArtifacts(
        root,
        "acquisition/acquisition-units.json",
        config,
        source,
        {"rmd17_ethanol.npz": author / "rmd17_ethanol.npz"},
    )
    return inputs, acquisition_payload(inputs)


def test_acquisition_viewer_reconstructs_original_selected_coordinates(acquired):
    inputs, payload = acquired
    assert payload["status"] == "verified observations" and len(payload["rounds"]) == 6
    with np.load(inputs.datasets["rmd17_ethanol.npz"]) as author:
        for row in payload["rounds"]:
            assert len(row["geometries"]) == row["budget"]
            assert sum(g["selected_this_round"] for g in row["geometries"]) == row["new_labels"]
            for geometry in row["geometries"]:
                index = int(geometry["identity"].rsplit(":", 1)[1])
                assert np.array_equal(geometry["positions"], author["coords"][index])
                assert geometry["force"] is None and geometry["energy"] is None


def test_acquisition_viewer_binds_three_saved_checkpoint_identities(acquired):
    inputs, payload = acquired
    assert payload["report_sha256"] == file_hash(inputs.root / inputs.report)
    for row in payload["rounds"]:
        assert row["lineage"]["source"] == inputs.source["digest"]
        checkpoints = row["lineage"]["checkpoints"]
        assert len({checkpoint["identity"] for checkpoint in checkpoints}) == 3


def copied_inputs(inputs, target):
    shutil.copytree(inputs.root / "acquisition", target / "acquisition")
    return AcquisitionArtifacts(
        target, inputs.report, inputs.config, inputs.source, inputs.datasets
    )


def test_acquisition_viewer_rejects_missing_round_population(acquired, tmp_path):
    inputs = copied_inputs(acquired[0], tmp_path)
    report = inputs.root / inputs.report
    outcomes = read_json(report)
    write_json(report, outcomes[:-1])
    with pytest.raises(IntegrityError, match="population"):
        acquisition_payload(inputs)


def test_acquisition_viewer_rejects_forged_ranking_with_unchanged_raw_arrays(acquired, tmp_path):
    inputs = copied_inputs(acquired[0], tmp_path)
    report = inputs.root / inputs.report
    outcomes = read_json(report)
    old = outcomes[0]["uncertainty"]["spearman"]
    outcomes[0]["uncertainty"]["spearman"] = 0.125 if old != 0.125 else 0.875
    write_json(report, outcomes)
    with pytest.raises(IntegrityError, match="ranking"):
        acquisition_payload(inputs)


def test_acquisition_viewer_rejects_forged_mean_bound_with_valid_coverage(acquired, tmp_path):
    inputs = copied_inputs(acquired[0], tmp_path)
    report = inputs.root / inputs.report
    outcomes = read_json(report)
    outcomes[0]["uncertainty"]["mean_bound"] += 1
    write_json(report, outcomes)
    with pytest.raises(IntegrityError, match="bound scale"):
        acquisition_payload(inputs)
