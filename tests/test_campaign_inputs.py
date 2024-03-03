import subprocess
import sys

import pytest

from asterion.artifacts.inputs import CampaignInputs, verify_input_identities
from asterion.artifacts.io import file_hash, read_json, write_json
from asterion.contracts.errors import IntegrityError


@pytest.fixture
def inputs(tmp_path):
    storage = tmp_path / "storage"
    (storage / "rmd17").mkdir(parents=True)
    raw = storage / "rmd17/fixture"
    raw.write_bytes(b"author bytes fixture")
    guard = CampaignInputs(tmp_path / "run", {"epochs": 3}, storage, "source-fixture")
    guard.record["raw_files"] = {
        "rmd17/fixture": {"sha256": file_hash(raw), "size": raw.stat().st_size}
    }
    write_json(guard.identity, guard.record)
    return guard, raw


def test_managed_configuration_snapshot_is_independent_of_changed_external_file(tmp_path):
    external = tmp_path / "external.json"
    write_json(external, {"epochs": 3})
    guard = CampaignInputs(tmp_path / "run", read_json(external), tmp_path / "storage", "source")
    write_json(external, {"epochs": 30})
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json,sys;print(json.load(open(sys.argv[1])))",
            str(guard.configuration),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "'epochs': 3}" in result.stdout
    guard.check(require_data=False)


def test_frozen_configuration_cannot_change_between_stages(inputs):
    guard, _ = inputs
    write_json(guard.configuration, {"epochs": 30})
    with pytest.raises(IntegrityError, match="configuration changed"):
        guard.check()


def test_raw_bytes_cannot_change_between_stages_even_with_same_size(inputs):
    guard, raw = inputs
    original = raw.read_bytes()
    raw.write_bytes(b"x" * len(original))
    with pytest.raises(IntegrityError, match="author bytes changed"):
        guard.check()
    raw.write_bytes(original)
    guard.check()


def test_raw_identity_cannot_be_rewritten_to_accept_changed_inputs(inputs):
    guard, raw = inputs
    raw.write_bytes(b"other data")
    record = read_json(guard.identity)
    record["raw_files"]["rmd17/fixture"]["sha256"] = file_hash(raw)
    write_json(guard.identity, record)
    with pytest.raises(IntegrityError, match="identity record changed"):
        guard.check()


def test_continuation_preserves_same_consumed_configuration_and_raw_identity(inputs):
    guard, _ = inputs
    continued = CampaignInputs(guard.run, guard.config, guard.storage, "source-fixture")
    continued.check()
    assert continued.record == guard.record


def test_continuation_refuses_changed_configuration(inputs):
    guard, _ = inputs
    with pytest.raises(IntegrityError, match="resumed request"):
        CampaignInputs(guard.run, {"epochs": 4}, guard.storage, "source-fixture")


def test_input_guard_refuses_scientific_stage_before_author_identity_exists(tmp_path):
    guard = CampaignInputs(tmp_path / "run", {}, tmp_path / "storage", "source")
    with pytest.raises(IntegrityError, match="unavailable"):
        guard.check()


def test_independent_input_check_binds_consumed_stage_flags(inputs):
    guard, raw = inputs
    write_json(guard.run / "INPUT_IDENTITIES.json", guard.record)
    finals = {
        "train": {
            "argv": [
                "python",
                "--config",
                str(guard.configuration.resolve()),
                "--storage",
                str(guard.storage),
            ]
        }
    }
    result = verify_input_identities(
        guard.run, {"digest": "source-fixture"}, guard.config, {"fixture": raw}, finals=finals
    )
    assert result == guard.record
    finals["train"]["argv"][2] = "changed-external.json"
    with pytest.raises(IntegrityError, match="did not consume"):
        verify_input_identities(
            guard.run, {"digest": "source-fixture"}, guard.config, {"fixture": raw}, finals=finals
        )


def test_independent_input_check_rejects_missing_author_population(inputs):
    guard, raw = inputs
    record = read_json(guard.identity)
    record["raw_files"] = {}
    write_json(guard.identity, record)
    with pytest.raises(IntegrityError, match="population"):
        verify_input_identities(
            guard.run, {"digest": "source-fixture"}, guard.config, {"fixture": raw}
        )
