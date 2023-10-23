import os
import signal
import subprocess
import sys
import time
from dataclasses import replace

import numpy as np
import pytest
import torch

from asterion.artifacts.campaign import Stage, StageRunner, cancellation_signals
from asterion.artifacts.io import read_json, safe_member, write_json
from asterion.contracts.errors import (
    CancellationError,
    ContractError,
    IntegrityError,
    ResourceError,
)
from asterion.contracts.geometry import Chemistry
from asterion.data.dataset import CoordinatePool, LabelOracle
from asterion.data.splits import Split
from asterion.dynamics.analytic import HarmonicBond
from asterion.models.potential import ModelConfig
from asterion.training import campaign as training_campaign
from asterion.training import engine
from asterion.training import recovery as training_recovery
from asterion.training.checkpoint import load_checkpoint
from asterion.training.engine import TrainConfig, train
from asterion.training.recovery import checkpoint_history, reconcile_training, tree_digest


@pytest.fixture
def training_case():
    positions = np.zeros((24, 2, 3))
    positions[:, 1, 0] = np.linspace(0.8, 1.2, len(positions))
    pool = CoordinatePool(
        np.array([6, 1]), positions, np.arange(24), "bond", "bond", Chemistry(level="analytic")
    )
    values = [HarmonicBond()(xyz, pool.numbers) for xyz in positions]
    oracle = LabelOracle(
        pool, np.array([v.energy for v in values]), np.array([v.forces for v in values])
    )
    split = Split(
        pool.dataset_hash,
        "recovery",
        1,
        tuple(range(12)),
        tuple(range(12, 16)),
        tuple(range(16, 20)),
        tuple(range(20, 24)),
    )
    architecture = ModelConfig(kind="invariant", width=4, radial=4, depth=1)
    config = TrainConfig(budget=8, epochs=3, batch_size=4, checkpoint_every=1)
    return oracle, split, architecture, config


def compare_final(first, second):
    a, _ = load_checkpoint(first / "epoch-00003")
    b, _ = load_checkpoint(second / "epoch-00003")
    assert all(torch.equal(value, b.state_dict()[key]) for key, value in a.state_dict().items())
    selections = [read_json(path / "selection.json") for path in (first, second)]
    assert {k: v for k, v in selections[0].items() if k != "lineage"} == {
        k: v for k, v in selections[1].items() if k != "lineage"
    }
    for x, y in zip(
        read_json(first / "history.json"), read_json(second / "history.json"), strict=True
    ):
        assert {k: v for k, v in x.items() if k != "elapsed_seconds"} == {
            k: v for k, v in y.items() if k != "elapsed_seconds"
        }


def crash_at_pointer(case, directory, monkeypatch, name):
    oracle, split, architecture, config = case
    original = engine.write_json
    armed = True

    def publish(path, value):
        nonlocal armed
        if (
            path.name == name
            and armed
            and value.get("epoch") == (1 if name == "selection.json" else 2)
        ):
            armed = False
            raise OSError("injected publication interruption")
        return original(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(engine, "write_json", publish)
        with pytest.raises(OSError, match="injected"):
            train(oracle, split, directory, architecture, config)
    checkpoint = (
        directory / read_json(directory / "latest.json")["checkpoint"]
        if (directory / "latest.json").exists()
        else directory / "initial"
    )
    train(oracle, split, directory, architecture, config, resume=checkpoint)


def test_checkpoint_published_before_latest_recovers_exact_weights(
    training_case, tmp_path, monkeypatch
):
    oracle, split, architecture, config = training_case
    whole, resumed = tmp_path / "whole", tmp_path / "resumed"
    train(oracle, split, whole, architecture, config)
    crash_at_pointer(training_case, resumed, monkeypatch, "latest.json")
    compare_final(whole, resumed)
    archived = list((resumed / "recovery").glob("*/epoch-00002/checkpoint.json"))
    assert len(archived) == 1
    assert read_json(archived[0])["epoch"] == 2


def test_selection_publication_interruption_recovers_from_immutable_history(
    training_case, tmp_path, monkeypatch
):
    oracle, split, architecture, config = training_case
    whole, resumed = tmp_path / "whole", tmp_path / "resumed"
    train(oracle, split, whole, architecture, config)
    crash_at_pointer(training_case, resumed, monkeypatch, "selection.json")
    compare_final(whole, resumed)


def test_future_summary_selection_is_rolled_back(training_case, tmp_path):
    oracle, split, architecture, config = training_case
    train(oracle, split, tmp_path / "run", architecture, config)
    _, record = load_checkpoint(tmp_path / "run/epoch-00001")
    history, selection = reconcile_training(tmp_path / "run", tmp_path / "run/epoch-00001", record)
    assert len(history) == 1 and selection["epoch"] == 1
    assert not (tmp_path / "run/epoch-00002").exists()
    assert not (tmp_path / "run/completed.json").exists()


def test_initial_generation_can_restart_after_first_epoch_publication(training_case, tmp_path):
    oracle, split, architecture, config = training_case
    whole, resumed = tmp_path / "whole", tmp_path / "resumed"
    train(oracle, split, whole, architecture, config)
    train(oracle, split, resumed, architecture, config, stop_after=1)
    train(oracle, split, resumed, architecture, config, resume=resumed / "initial")
    compare_final(whole, resumed)


def test_checkpoint_loading_preserves_callers_rng(training_case, tmp_path):
    oracle, split, architecture, config = training_case
    train(oracle, split, tmp_path / "run", architecture, config, stop_after=1)
    before = torch.random.get_rng_state().clone()
    load_checkpoint(tmp_path / "run/epoch-00001")
    assert torch.equal(before, torch.random.get_rng_state())


def test_recovery_requires_local_generation(training_case, tmp_path):
    oracle, split, architecture, config = training_case
    train(oracle, split, tmp_path / "run", architecture, config, stop_after=1)
    _, record = load_checkpoint(tmp_path / "run/epoch-00001")
    with pytest.raises(IntegrityError, match="belong"):
        reconcile_training(tmp_path / "another", tmp_path / "run/epoch-00001", record)


def test_recovery_rejects_skipped_epoch_history(tmp_path):
    record = {
        "epoch": 1,
        "training_history": [{"epoch": 2, "validation_force_mae": 0.1}],
        "updates": 1,
    }
    with pytest.raises(IntegrityError, match="history"):
        checkpoint_history(tmp_path, record)


def test_recovery_rejects_nonfinite_validation_selection(tmp_path):
    record = {
        "epoch": 1,
        "training_history": [{"epoch": 1, "validation_force_mae": float("nan")}],
        "updates": 1,
    }
    with pytest.raises(IntegrityError, match="history"):
        checkpoint_history(tmp_path, record)


def test_recovery_rejects_forged_optimizer_count(tmp_path):
    record = {
        "epoch": 1,
        "training_history": [{"epoch": 1, "updates": 3, "validation_force_mae": 0.1}],
        "updates": 2,
    }
    with pytest.raises(IntegrityError, match="optimizer"):
        checkpoint_history(tmp_path, record)


def test_recovery_tree_refuses_linked_generations(tmp_path):
    (tmp_path / "original").write_bytes(b"evidence")
    (tmp_path / "link").symlink_to(tmp_path / "original")
    with pytest.raises(IntegrityError, match="linked"):
        tree_digest(tmp_path)


def test_recovery_archive_digest_changes_with_bytes(tmp_path):
    (tmp_path / "weights").write_bytes(b"one")
    original = tree_digest(tmp_path)
    (tmp_path / "weights").write_bytes(b"two")
    assert tree_digest(tmp_path) != original


def test_safe_member_refuses_link_to_inside_root(tmp_path):
    (tmp_path / "actual").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "actual")
    with pytest.raises(IntegrityError, match="linked"):
        safe_member(tmp_path, "link/observation")


def test_safe_member_refuses_absolute_inside_path(tmp_path):
    with pytest.raises(IntegrityError, match="escapes"):
        safe_member(tmp_path, str(tmp_path / "value"))


def test_stage_restart_preserves_failed_attempt_logs_and_partial_outputs(tmp_path):
    script = "from pathlib import Path; p=Path('marker'); Path('output').write_text('partial' if not p.exists() else 'completed'); existed=p.exists(); p.touch(); raise SystemExit(0 if existed else 7)"
    stage = Stage(
        "sample",
        (sys.executable, "-c", script),
        ("output",),
        continuation="restart",
        reset_paths=("output",),
    )
    with pytest.raises(ResourceError, match="exit=7"):
        StageRunner(tmp_path, "source", root=tmp_path).execute(stage)
    resumed = StageRunner(tmp_path, "source", root=tmp_path, resume=True)
    result = resumed.execute(stage)
    assert result["status"] == "passed" and result["attempt"] == 2
    assert (tmp_path / "stage-attempts/sample-0001/outputs/output").read_text() == "partial"
    assert (tmp_path / "output").read_text() == "completed"
    assert resumed.state["stages"][0]["exit"] == 7
    assert resumed.state["stages"][0]["continued_by_attempt"] == 2


def test_stage_continuation_cannot_change_command(tmp_path):
    stage = Stage(
        "sample", (sys.executable, "-c", "raise SystemExit(7)"), (), continuation="resume"
    )
    with pytest.raises(ResourceError):
        StageRunner(tmp_path, "source", root=tmp_path).execute(stage)
    with pytest.raises(IntegrityError, match="definition changed"):
        StageRunner(tmp_path, "source", root=tmp_path, resume=True).execute(
            replace(stage, argv=(sys.executable, "-c", "pass"))
        )


def test_stage_cannot_reset_coordinator_directory():
    with pytest.raises(IntegrityError, match="coordinator"):
        Stage(
            "sample", ("program",), (), continuation="restart", reset_paths=("stage-logs/nested",)
        ).validate()


def test_live_orphan_stage_is_not_duplicated(tmp_path):
    stage = Stage("sample", ("program",), (), continuation="resume")
    from dataclasses import asdict

    from asterion.artifacts.io import object_hash

    write_json(
        tmp_path / "STAGES.json",
        {
            "identity": "source",
            "stages": [
                {
                    "name": stage.name,
                    "definition": object_hash(asdict(stage)),
                    "status": "running",
                    "child_pid": os.getpid(),
                }
            ],
        },
    )
    with pytest.raises(IntegrityError, match="remains alive"):
        StageRunner(tmp_path, "source", root=tmp_path, resume=True).execute(stage)


def test_signal_handlers_restore_after_failure():
    original = signal.getsignal(signal.SIGTERM)
    with pytest.raises(CancellationError):
        with cancellation_signals():
            os.kill(os.getpid(), signal.SIGTERM)
    assert signal.getsignal(signal.SIGTERM) is original


@pytest.mark.parametrize("number", [signal.SIGINT, signal.SIGTERM])
def test_actual_parent_cancellation_terminates_owned_child_group(tmp_path, number):
    child = "import os,time;from pathlib import Path;Path('child-pid').write_text(str(os.getpid()));time.sleep(60)"
    code = f"from pathlib import Path\nfrom asterion.artifacts.campaign import Stage,StageRunner,cancellation_signals\nwith cancellation_signals():\n StageRunner(Path('.'), 'source', root=Path('.')).execute(Stage('child', ({sys.executable!r}, '-c', {child!r}), ()))\n"
    parent = subprocess.Popen(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 15
        while not (tmp_path / "child-pid").exists() and time.monotonic() < deadline:
            assert parent.poll() is None
            time.sleep(0.02)
        child_pid = int((tmp_path / "child-pid").read_text())
        parent.send_signal(number)
        assert parent.wait(timeout=10) != 0
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        state = read_json(tmp_path / "STAGES.json")
        assert state["stages"][0]["status"] == "interrupted"
        assert state["stages"][0]["exit"] == -signal.SIGTERM
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()


def test_training_rejects_nonfinite_learning_rate():
    with pytest.raises(ContractError, match="weight"):
        TrainConfig(learning_rate=float("nan")).validate()


def test_training_rejects_zero_clip_norm():
    with pytest.raises(ContractError, match="weight"):
        TrainConfig(clip_norm=0).validate()


def test_training_rejects_infinite_force_weight():
    with pytest.raises(ContractError, match="weight"):
        TrainConfig(force_weight=float("inf")).validate()


def tiny_campaign(training_case, monkeypatch):
    oracle, split, _, _ = training_case
    monkeypatch.setattr(training_campaign, "load_rmd17", lambda *unused: (oracle.pool, oracle))
    monkeypatch.setattr(training_campaign, "author_split", lambda *unused: split)
    return {
        "profile": "diagnostic",
        "molecules": ["ethanol"],
        "models": ["invariant"],
        "seeds": [1],
        "budgets": [8],
        "epochs": 3,
        "batch_size": 4,
        "architecture": {"width": 4, "depth": 1, "radial": 4},
        "training": {"learning_rate": 0.003, "checkpoint_every": 1},
        "simulation": {},
        "evaluation_size": 4,
        "validation_size": 4,
        "calibration_size": 4,
    }


def test_completed_campaign_resume_validates_full_training_lineage(
    training_case, tmp_path, monkeypatch
):
    config = tiny_campaign(training_case, monkeypatch)
    training_campaign.train_campaign(config, tmp_path, tmp_path / "run")
    config["training"]["learning_rate"] = 0.005
    with pytest.raises(IntegrityError, match="lineage mismatch"):
        training_campaign.train_campaign(config, tmp_path, tmp_path / "run", resume=True)


def test_completed_campaign_resume_retains_optimizer_work(training_case, tmp_path, monkeypatch):
    config = tiny_campaign(training_case, monkeypatch)
    first = training_campaign.train_campaign(config, tmp_path, tmp_path / "run")
    second = training_campaign.train_campaign(config, tmp_path, tmp_path / "run", resume=True)
    assert first == second and second[0]["result"]["updates"] == 6


def test_incomplete_initialization_is_preserved_before_restart(
    training_case, tmp_path, monkeypatch
):
    config = tiny_campaign(training_case, monkeypatch)
    partial = tmp_path / "run/invariant-ethanol-8-1"
    partial.mkdir(parents=True)
    (partial / "partial").write_bytes(b"interrupted initialization")
    result = training_campaign.train_campaign(config, tmp_path, tmp_path / "run", resume=True)
    assert result[0]["result"]["status"] == "passed"
    retained = list((tmp_path / "run/initialization-attempts").glob("*/partial"))
    assert len(retained) == 1 and retained[0].read_bytes() == b"interrupted initialization"


def test_recovery_resumes_interrupted_archive_transaction(training_case, tmp_path, monkeypatch):
    oracle, split, architecture, config = training_case
    directory = tmp_path / "run"
    train(oracle, split, directory, architecture, config)
    checkpoint = directory / "epoch-00001"
    _, record = load_checkpoint(checkpoint)

    def interrupt(root, plan_path, plan):
        move = plan["moves"][0]
        retained = root / move["retained"]
        retained.parent.mkdir(parents=True, exist_ok=True)
        (root / move["original"]).rename(retained)
        raise OSError("archive interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(training_recovery, "finish_archive", interrupt)
        with pytest.raises(OSError, match="archive interrupted"):
            reconcile_training(directory, checkpoint, record)
    history, selection = reconcile_training(directory, checkpoint, record)
    assert len(history) == selection["epoch"] == 1
    assert read_json(directory / "RECOVERY.json")["status"] == "restored"
    assert not (directory / "epoch-00003").exists()
