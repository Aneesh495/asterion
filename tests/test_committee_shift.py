import copy
import shutil

import numpy as np
import pytest
from test_acquisition_viewer import acquired

from asterion.acquisition.shift import temporal_slice
from asterion.artifacts.committee_shift import verify_committee_shift
from asterion.artifacts.io import atomic_npz, object_hash, read_json, write_json
from asterion.artifacts.verify import read_arrays
from asterion.contracts.errors import ContractError, IntegrityError
from asterion.contracts.geometry import Chemistry
from asterion.data.dataset import CoordinatePool
from asterion.data.splits import Split
from asterion.training.checkpoint import load_checkpoint


@pytest.fixture
def slice_inputs():
    xyz = np.zeros((100, 2, 3))
    xyz[:, 1, 0] = 1.1
    original = np.random.default_rng(5).permutation(100) * 17
    pool = CoordinatePool(np.array([6, 1]), xyz, original, "fixture-hash", "fixture", Chemistry())
    primary = Split(
        pool.dataset_hash,
        "fixture-primary",
        2,
        tuple(range(20)),
        tuple(range(20, 30)),
        tuple(range(30, 40)),
        tuple(range(40, 50)),
    )
    return pool, primary


def test_temporal_selector_excludes_every_primary_role_and_uses_original_order(slice_inputs):
    pool, primary = slice_inputs
    shifted = temporal_slice(pool, primary, 5)
    forbidden = set(
        primary.train_pool + primary.validation + primary.calibration + primary.evaluation
    )
    assert not set(shifted.evaluation) & forbidden
    threshold = sorted(pool.original_indices)[75]
    assert all(pool.original_indices[i] >= threshold for i in shifted.evaluation)
    assert len(set(shifted.evaluation)) == 5 and shifted.temporal_guard == 0
    assert temporal_slice(pool, primary, 5) == shifted


def test_temporal_selector_refuses_a_shortened_denominator(slice_inputs):
    pool, primary = slice_inputs
    with pytest.raises(ContractError, match="frozen distinct denominator"):
        temporal_slice(pool, primary, 26)


@pytest.mark.parametrize("count", [0, -1, True, 2.5])
def test_temporal_selector_rejects_invalid_geometry_count(slice_inputs, count):
    with pytest.raises(ContractError, match="positive geometry"):
        temporal_slice(*slice_inputs, count)


@pytest.fixture(scope="module")
def shift_fixture(tmp_path_factory):
    # Existing real tiny-training fixture. It never represents author-data/full-campaign evidence.
    inputs, payload = acquired.__wrapped__(tmp_path_factory)
    return inputs, payload


def check(inputs, row, directory=None):
    directory = directory or inputs.root / "acquisition" / row["path"]
    primary = read_json(directory / row["members"][0] / "split.json")
    fitted = read_json(directory / f"calibration-{row['round']}.json")
    models = []
    for name in row["members"]:
        member = directory / name
        selected = read_json(member / "selection.json")
        models.append(load_checkpoint(member / selected["checkpoint"])[0])
    return verify_committee_shift(
        directory,
        row,
        inputs.config,
        inputs.source,
        primary,
        fitted,
        models,
        inputs.datasets["rmd17_ethanol.npz"],
    )


def row_and_copy(shift_fixture, tmp_path):
    inputs, _ = shift_fixture
    row = copy.deepcopy(read_json(inputs.root / inputs.report)[0])
    directory = tmp_path / "round"
    shutil.copytree(inputs.root / "acquisition" / row["path"], directory)
    return inputs, row, directory


def test_temporal_committee_observations_replay_real_saved_members(shift_fixture):
    inputs, payload = shift_fixture
    rows = read_json(inputs.root / inputs.report)
    for row in rows:
        result = check(inputs, row)
        assert result["members"] == 3 and result["geometries"] == inputs.config["evaluation_size"]
        assert result["replayed_probes"] == 24
    assert all(
        row["shift"]["calibration"] == "unchanged-primary-calibration" for row in payload["rounds"]
    )


def test_shift_verifier_rejects_changed_author_targets_with_valid_archive(shift_fixture, tmp_path):
    inputs, row, directory = row_and_copy(shift_fixture, tmp_path)
    path = directory / row["shift"]["members"][0]
    raw = read_arrays(path)
    raw["target_force"] += 0.1
    atomic_npz(path, **raw)
    with pytest.raises(IntegrityError, match="targets changed"):
        check(inputs, row, directory)


def test_shift_verifier_rejects_changed_member_outputs_with_valid_archive(shift_fixture, tmp_path):
    inputs, row, directory = row_and_copy(shift_fixture, tmp_path)
    path = directory / row["shift"]["members"][0]
    raw = read_arrays(path)
    raw["force"] += 0.1
    atomic_npz(path, **raw)
    with pytest.raises(IntegrityError, match="another saved checkpoint"):
        check(inputs, row, directory)


def test_shift_verifier_rejects_refitted_bounds(shift_fixture, tmp_path):
    inputs, row, directory = row_and_copy(shift_fixture, tmp_path)
    path = directory / row["shift"]["observations"]
    raw = read_arrays(path)
    raw["bounds"] *= 2
    atomic_npz(path, **raw)
    with pytest.raises(IntegrityError, match="reused bounds"):
        check(inputs, row, directory)


def test_shift_verifier_rejects_forged_coherent_ranking_summary(shift_fixture, tmp_path):
    inputs, row, directory = row_and_copy(shift_fixture, tmp_path)
    row["shift"]["uncertainty"]["coverage"] = 0.123
    write_json(directory / f"shift-{row['round']}.json", row["shift"])
    with pytest.raises(IntegrityError, match="coverage"):
        check(inputs, row, directory)


def test_shift_verifier_rejects_primary_role_leak_even_with_updated_split_digest(
    shift_fixture, tmp_path
):
    inputs, row, directory = row_and_copy(shift_fixture, tmp_path)
    path = directory / "shift-split.json"
    split = read_json(path)
    split.pop("digest")
    split["evaluation"][0] = split["train_pool"][0]
    split["digest"] = object_hash({k: v for k, v in split.items() if k != "digest"})
    write_json(path, split)
    row["shift"]["split"] = split["digest"]
    write_json(directory / f"shift-{row['round']}.json", row["shift"])
    with pytest.raises(IntegrityError, match="crossed primary roles"):
        check(inputs, row, directory)


def test_shift_verifier_rejects_fabricated_guarded_training_claim(shift_fixture, tmp_path):
    inputs, row, directory = row_and_copy(shift_fixture, tmp_path)
    row["shift"]["temporal_guard"] = 100
    write_json(directory / f"shift-{row['round']}.json", row["shift"])
    with pytest.raises(IntegrityError, match="calibration claim"):
        check(inputs, row, directory)


def test_shift_verifier_refuses_missing_shift_observations(shift_fixture, tmp_path):
    inputs, row, directory = row_and_copy(shift_fixture, tmp_path)
    row.pop("shift")
    with pytest.raises(IntegrityError, match="slice is missing"):
        check(inputs, row, directory)
