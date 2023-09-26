import hashlib
import platform
import sys

import pytest

from asterion.artifacts.bundle import copy_observations, promote_full
from asterion.artifacts.campaign import (
    CampaignOwner,
    Stage,
    StageRunner,
    output_inventory,
    research_stages,
)
from asterion.artifacts.io import read_json, write_json
from asterion.contracts.errors import IntegrityError, ResourceError


def test_stage_rejects_path_escape():
    with pytest.raises(IntegrityError, match="inside"):
        Stage("sample", ("program",), ("../escape",)).validate()


def test_stage_rejects_absolute_output():
    with pytest.raises(IntegrityError, match="inside"):
        Stage("sample", ("program",), ("/tmp/escape",)).validate()


def test_stage_rejects_unsafe_log_name():
    with pytest.raises(IntegrityError, match="name"):
        Stage("../../escape", ("program",), ()).validate()


def test_stage_rejects_nonfinite_timeout():
    with pytest.raises(IntegrityError, match="stage"):
        Stage("sample", ("program",), (), float("nan")).validate()


def test_stage_rejects_empty_command():
    with pytest.raises(IntegrityError, match="stage"):
        Stage("sample", (), ()).validate()


def test_inventory_hashes_actual_nested_bytes(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/value").write_bytes(b"actual output")
    expected = hashlib.sha256(b"actual output").hexdigest()
    assert output_inventory(tmp_path, ("nested",)) == {"nested/value": expected}


def test_inventory_rejects_missing_observation(tmp_path):
    with pytest.raises(IntegrityError, match="absent"):
        output_inventory(tmp_path, ("missing",))


def test_inventory_rejects_empty_directory(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(IntegrityError, match="empty"):
        output_inventory(tmp_path, ("empty",))


def test_inventory_rejects_linked_observation(tmp_path):
    (tmp_path / "value").write_bytes(b"data")
    (tmp_path / "link").symlink_to(tmp_path / "value")
    with pytest.raises(IntegrityError, match="symlink"):
        output_inventory(tmp_path, ("link",))


def test_observation_copy_refuses_linked_files(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "link").symlink_to(tmp_path / "outside")
    with pytest.raises(IntegrityError, match="symlink"):
        copy_observations(source, tmp_path / "copy")


def test_campaign_lock_has_single_live_owner(tmp_path):
    with CampaignOwner(tmp_path):
        with pytest.raises(IntegrityError, match="owner"):
            with CampaignOwner(tmp_path, resume=True):
                pass
    assert not (tmp_path / ".coordinator.lock").exists()


def test_campaign_owner_releases_after_failure(tmp_path):
    with pytest.raises(RuntimeError):
        with CampaignOwner(tmp_path):
            raise RuntimeError("interrupted action")
    assert not (tmp_path / ".coordinator.lock").exists()


def test_campaign_owner_does_not_delete_replaced_lock(tmp_path):
    with CampaignOwner(tmp_path):
        value = read_json(tmp_path / ".coordinator.lock")
        value["nonce"] = "a" * 32
        write_json(tmp_path / ".coordinator.lock", value)
    assert (tmp_path / ".coordinator.lock").exists()


def test_campaign_owner_rejects_malformed_identity(tmp_path):
    write_json(
        tmp_path / ".coordinator.lock", {"pid": -1, "host": platform.node(), "nonce": "../escape"}
    )
    with pytest.raises(IntegrityError, match="malformed"):
        with CampaignOwner(tmp_path, resume=True):
            pass


def test_stage_executes_and_retains_true_process_exit(tmp_path):
    stage = Stage(
        "sample",
        (sys.executable, "-c", "from pathlib import Path;Path('value').write_text('done')"),
        ("value",),
    )
    with CampaignOwner(tmp_path):
        result = StageRunner(tmp_path, "source", root=tmp_path).execute(stage)
    assert result["exit"] == 0 and result["status"] == "passed"
    assert (tmp_path / "value").read_text() == "done"
    assert result["outputs"]["value"] == hashlib.sha256(b"done").hexdigest()


def test_stage_nonzero_exit_is_preserved(tmp_path):
    with CampaignOwner(tmp_path):
        runner = StageRunner(tmp_path, "source", root=tmp_path)
        with pytest.raises(ResourceError, match="exit=7"):
            runner.execute(Stage("failure", (sys.executable, "-c", "raise SystemExit(7)"), ()))
    state = read_json(tmp_path / "STAGES.json")
    assert state["status"] == "failed" and state["stages"][0]["exit"] == 7


def test_stage_timeout_stops_owned_child(tmp_path):
    with CampaignOwner(tmp_path):
        runner = StageRunner(tmp_path, "source", root=tmp_path)
        stage = Stage("timeout", (sys.executable, "-c", "import time;time.sleep(60)"), (), 0.05)
        with pytest.raises(ResourceError, match="timed out"):
            runner.execute(stage)
        assert runner.child is None
    state = read_json(tmp_path / "STAGES.json")
    assert state["stages"][0]["exit"] != 0


def test_completed_stage_resume_does_not_repeat_work(tmp_path):
    stage = Stage(
        "once",
        (sys.executable, "-c", "from pathlib import Path;Path('value').write_bytes(b'once')"),
        ("value",),
    )
    with CampaignOwner(tmp_path):
        StageRunner(tmp_path, "source", root=tmp_path).execute(stage)
    with CampaignOwner(tmp_path, resume=True):
        resumed = StageRunner(tmp_path, "source", root=tmp_path, resume=True)
        result = resumed.execute(stage)
    assert len(resumed.state["stages"]) == 1 and result["status"] == "passed"


def test_resume_rejects_changed_source(tmp_path):
    write_json(
        tmp_path / "STAGES.json", {"identity": "original", "stages": [], "status": "running"}
    )
    with pytest.raises(IntegrityError, match="different source"):
        StageRunner(tmp_path, "changed", resume=True)


def test_resume_rejects_changed_completed_output(tmp_path):
    stage = Stage(
        "sample",
        (sys.executable, "-c", "from pathlib import Path;Path('value').write_bytes(b'good')"),
        ("value",),
    )
    with CampaignOwner(tmp_path):
        StageRunner(tmp_path, "source", root=tmp_path).execute(stage)
    (tmp_path / "value").write_bytes(b"altered")
    with CampaignOwner(tmp_path, resume=True):
        with pytest.raises(IntegrityError, match="outputs changed"):
            StageRunner(tmp_path, "source", root=tmp_path, resume=True).execute(stage)


def test_failed_stage_cannot_be_relabelled_as_completed(tmp_path):
    write_json(
        tmp_path / "STAGES.json",
        {
            "identity": "source",
            "status": "failed",
            "stages": [{"name": "train", "status": "failed"}],
        },
    )
    with pytest.raises(IntegrityError, match="failed stage retained"):
        StageRunner(tmp_path, "source", resume=True).execute(Stage("train", ("program",), ()))


def test_full_stage_plan_preserves_all_required_families(tmp_path):
    stages = research_stages(tmp_path / "full.json", tmp_path / "data", tmp_path / "campaign")
    assert [x.name for x in stages] == [
        "tests",
        "data",
        "train",
        "evaluate",
        "simulate",
        "acquire",
        "ablate",
        "reference",
        "secondary",
        "properties",
        "derivatives",
        "analytic",
        "stress",
        "bench",
        "demo",
    ]
    assert all(x.validate() for x in stages)


def test_resume_keeps_frozen_training_stage_definition(tmp_path):
    first = research_stages(tmp_path / "full.json", tmp_path / "data", tmp_path / "campaign")
    resumed = research_stages(
        tmp_path / "full.json", tmp_path / "data", tmp_path / "campaign", True
    )
    assert first == resumed


def test_diagnostic_verification_cannot_promote_full(tmp_path):
    with pytest.raises(IntegrityError, match="verified candidate"):
        promote_full(
            tmp_path / "candidate", tmp_path / "final", tmp_path / "controls", {"phase": "final"}
        )


def test_full_promotion_requires_unaccepted_controls(tmp_path):
    controls = tmp_path / "controls.json"
    write_json(
        controls,
        {
            "profile": "full",
            "phase": "candidate",
            "positive_before": {"exit": 0},
            "positive_after": {"exit": 0},
            "source_bundle_unchanged": True,
            "controls": [{"exit": 0, "stderr": "accepted", "semantic_rehashed": True}] * 12,
        },
    )
    with pytest.raises(IntegrityError, match="accepted negative controls"):
        promote_full(tmp_path / "candidate", tmp_path / "final", controls, {"phase": "candidate"})
