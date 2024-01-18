import os
import shutil
import subprocess
import sys
import time

import pytest

from asterion.artifacts.campaign import Stage, StageRunner
from asterion.artifacts.io import read_json, write_json
from asterion.artifacts.receipts import verify_stage_receipts
from asterion.contracts.errors import IntegrityError


@pytest.fixture(scope="module")
def lost_coordinator(tmp_path_factory):
    run = tmp_path_factory.mktemp("lost-coordinator")
    script = "from pathlib import Path; import time; Path('started').touch(); print('actual child observation', flush=True); time.sleep(.5); Path('value').write_bytes(b'observed work')"
    driver = (
        "from pathlib import Path; import sys; "
        "from asterion.artifacts.campaign import Stage,StageRunner; "
        f"stage=Stage('sample',(sys.executable,'-c',{script!r}),('value',),continuation='resume'); "
        "StageRunner(Path.cwd(),'source',root=Path.cwd()).execute(stage)"
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", driver], cwd=run, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    child_pid = None
    deadline = time.monotonic() + 10
    try:
        while time.monotonic() < deadline:
            if (run / "STAGES.json").exists() and (run / "started").exists():
                child_pid = read_json(run / "STAGES.json")["stages"][0].get("child_pid")
                if child_pid:
                    break
            time.sleep(0.01)
        assert child_pid is not None, "coordinator did not publish its real child identity"
        parent.kill()
        parent.wait(timeout=5)
        assert parent.returncode != 0
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.01)
        else:
            pytest.fail("owned orphan did not exit and get reaped")
        row = read_json(run / "STAGES.json")["stages"][0]
        assert row["status"] == "running" and row["exit"] is None
        assert "ended_ns" not in row and (run / "value").read_bytes() == b"observed work"
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        if child_pid:
            try:
                os.kill(child_pid, 15)
            except ProcessLookupError:
                pass
        parent.communicate(timeout=5)
    return run, Stage("sample", (sys.executable, "-c", script), ("value",), continuation="resume")


@pytest.fixture
def recovered(lost_coordinator, tmp_path):
    original, stage = lost_coordinator
    run = tmp_path / "run"
    shutil.copytree(original, run)
    final = StageRunner(run, "source", root=run, resume=True).execute(stage)
    write_json(
        run / "STAGE_OUTPUTS.json",
        {"value": {"path": "value", "sha256": final["outputs"]["value"]}},
    )
    return run


def test_real_killed_coordinator_preserves_unknown_exit_and_observed_logs(recovered):
    rows = read_json(recovered / "STAGES.json")["stages"]
    first, last = rows
    assert first["status"] == "interrupted" and first["exit_observed"] is False
    assert first["exit"] is first["ended_ns"] is first["elapsed_seconds"] is None
    assert first["continued_by_attempt"] == 2 and last["exit"] == 0
    assert (recovered / first["stdout"]).read_text() == "actual child observation\n"
    assert verify_stage_receipts(recovered, ("sample",))["sample"]["attempt"] == 2


def test_recovered_unknown_exit_cannot_be_changed_to_success(recovered):
    state = read_json(recovered / "STAGES.json")
    state["stages"][0]["exit"] = 0
    write_json(recovered / "STAGES.json", state)
    with pytest.raises(IntegrityError, match="unobserved exit"):
        verify_stage_receipts(recovered, ("sample",))


def test_recovered_unknown_duration_cannot_be_filled_with_estimate(recovered):
    state = read_json(recovered / "STAGES.json")
    state["stages"][0]["elapsed_seconds"] = 0.5
    state["stages"][0]["ended_ns"] = state["stages"][0]["started_ns"] + 500000000
    write_json(recovered / "STAGES.json", state)
    with pytest.raises(IntegrityError, match="unobserved exit or duration"):
        verify_stage_receipts(recovered, ("sample",))


def test_recovered_log_corruption_is_rejected(recovered):
    first = read_json(recovered / "STAGES.json")["stages"][0]
    (recovered / first["stdout"]).write_text("invented child execution\n")
    with pytest.raises(IntegrityError, match="log bytes"):
        verify_stage_receipts(recovered, ("sample",))


def test_recovered_attempt_cannot_hide_its_continuation(recovered):
    state = read_json(recovered / "STAGES.json")
    del state["stages"][0]["continued_by_attempt"]
    write_json(recovered / "STAGES.json", state)
    with pytest.raises(IntegrityError, match="without continuation"):
        verify_stage_receipts(recovered, ("sample",))


def test_running_orphan_is_never_duplicated(lost_coordinator, tmp_path):
    original, stage = lost_coordinator
    run = tmp_path / "run"
    shutil.copytree(original, run)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        state = read_json(run / "STAGES.json")
        state["stages"][0]["child_pid"] = child.pid
        write_json(run / "STAGES.json", state)
        with pytest.raises(IntegrityError, match="remains alive"):
            StageRunner(run, "source", root=run, resume=True).execute(stage)
        assert len(read_json(run / "STAGES.json")["stages"]) == 1
    finally:
        child.terminate()
        child.wait(timeout=5)


def test_unknown_child_launch_window_refuses_duplicate_work(lost_coordinator, tmp_path):
    original, stage = lost_coordinator
    run = tmp_path / "run"
    shutil.copytree(original, run)
    state = read_json(run / "STAGES.json")
    del state["stages"][0]["child_pid"]
    write_json(run / "STAGES.json", state)
    with pytest.raises(IntegrityError, match="launch identity is unknown"):
        StageRunner(run, "source", root=run, resume=True).execute(stage)
    assert len(read_json(run / "STAGES.json")["stages"]) == 1
