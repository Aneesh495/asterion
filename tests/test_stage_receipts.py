import hashlib
import sys
from copy import deepcopy

import pytest

from asterion.artifacts.bundle import retain_stage_receipts
from asterion.artifacts.campaign import Stage, StageRunner, output_inventory, research_stages
from asterion.artifacts.io import read_json, write_json
from asterion.artifacts.receipts import verify_command_receipts, verify_stage_receipts
from asterion.contracts.errors import IntegrityError, ResourceError


@pytest.fixture
def receipt(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    stage = Stage(
        "sample",
        (
            sys.executable,
            "-c",
            "from pathlib import Path; print('observed'); Path('value').write_bytes(b'actual observation')",
        ),
        ("value",),
    )
    record = StageRunner(run, "source", root=run).execute(stage)
    write_json(
        run / "STAGE_OUTPUTS.json",
        {"value": {"path": "value", "sha256": record["outputs"]["value"]}},
    )
    return run


def mutate(run, change):
    state = read_json(run / "STAGES.json")
    change(state["stages"][0])
    write_json(run / "STAGES.json", state)


def test_stage_receipt_reads_true_exit_and_actual_observation_bytes(receipt):
    final = verify_stage_receipts(receipt, ("sample",))
    assert final["sample"]["exit"] == 0
    assert final["sample"]["outputs"]["value"] == hashlib.sha256(b"actual observation").hexdigest()


def test_stage_receipt_rejects_changed_process_log(receipt):
    row = read_json(receipt / "STAGES.json")["stages"][0]
    (receipt / row["stdout"]).write_bytes(b"fabricated execution")
    with pytest.raises(IntegrityError, match="log bytes"):
        verify_stage_receipts(receipt, ("sample",))


def test_stage_receipt_rejects_changed_model_or_raw_observation(receipt):
    (receipt / "value").write_bytes(b"changed retained weights")
    with pytest.raises(IntegrityError, match="observation bytes"):
        verify_stage_receipts(receipt, ("sample",))


def test_stage_receipt_rejects_missing_output_retention(receipt):
    write_json(receipt / "STAGE_OUTPUTS.json", {})
    with pytest.raises(IntegrityError, match="retention"):
        verify_stage_receipts(receipt, ("sample",))


def test_stage_receipt_rejects_unexplained_extra_observation(receipt):
    mapping = read_json(receipt / "STAGE_OUTPUTS.json")
    mapping["extra"] = deepcopy(mapping["value"])
    write_json(receipt / "STAGE_OUTPUTS.json", mapping)
    with pytest.raises(IntegrityError, match="unexplained"):
        verify_stage_receipts(receipt, ("sample",))


def test_stage_receipt_rejects_false_successful_exit(receipt):
    mutate(receipt, lambda row: row.update(exit=7))
    with pytest.raises(IntegrityError, match="exit"):
        verify_stage_receipts(receipt, ("sample",))


def test_stage_receipt_rejects_nonpositive_completed_runtime(receipt):
    mutate(receipt, lambda row: row.update(elapsed_seconds=0))
    with pytest.raises(IntegrityError, match="elapsed"):
        verify_stage_receipts(receipt, ("sample",))


def test_stage_receipt_rejects_inverted_execution_clock(receipt):
    mutate(receipt, lambda row: row.update(ended_ns=row["started_ns"] - 1))
    with pytest.raises(IntegrityError, match="clock"):
        verify_stage_receipts(receipt, ("sample",))


def test_stage_receipt_rejects_failed_stage_presented_as_pending(receipt):
    mutate(receipt, lambda row: row.update(status="running"))
    with pytest.raises(IntegrityError, match="pending"):
        verify_stage_receipts(receipt, ("sample",))


def test_stage_receipt_rejects_hidden_required_family(receipt):
    with pytest.raises(IntegrityError, match="population"):
        verify_stage_receipts(receipt, ("sample", "missing"))


def test_stage_receipt_rejects_skipped_attempt_sequence(receipt):
    mutate(receipt, lambda row: row.update(attempt=2))
    with pytest.raises(IntegrityError, match="sequence"):
        verify_stage_receipts(receipt, ("sample",))


def test_stage_receipt_rejects_changed_command_without_matching_definition(receipt):
    mutate(receipt, lambda row: row.update(argv=["replacement"]))
    with pytest.raises(IntegrityError, match="command"):
        verify_stage_receipts(receipt, ("sample",))


def test_stage_receipt_rejects_empty_completed_outputs(receipt):
    mutate(receipt, lambda row: row.update(outputs={}))
    with pytest.raises(IntegrityError, match="useful"):
        verify_stage_receipts(receipt, ("sample",))


def test_actual_failed_stage_continuation_retains_both_exits(tmp_path):
    script = "from pathlib import Path; p=Path('marker'); existed=p.exists(); p.touch(); Path('value').write_bytes(b'completed'); raise SystemExit(0 if existed else 7)"
    stage = Stage("sample", (sys.executable, "-c", script), ("value",), continuation="resume")
    with pytest.raises(ResourceError, match="exit=7"):
        StageRunner(tmp_path, "source", root=tmp_path).execute(stage)
    row = StageRunner(tmp_path, "source", root=tmp_path, resume=True).execute(stage)
    write_json(
        tmp_path / "STAGE_OUTPUTS.json",
        {"value": {"path": "value", "sha256": row["outputs"]["value"]}},
    )
    assert verify_stage_receipts(tmp_path, ("sample",))["sample"]["attempt"] == 2
    assert [r["exit"] for r in read_json(tmp_path / "STAGES.json")["stages"]] == [7, 0]


def test_stage_inventory_hashes_every_checkpoint_generation(tmp_path):
    for epoch in (1, 2):
        directory = tmp_path / f"epoch-{epoch:05}"
        directory.mkdir()
        (directory / "weights").write_bytes(bytes([epoch]))
    inventory = output_inventory(tmp_path, ("epoch-[0-9][0-9][0-9][0-9][0-9]",))
    assert len(inventory) == 2
    assert inventory["epoch-00002/weights"] == hashlib.sha256(bytes([2])).hexdigest()


def test_stage_inventory_refuses_empty_checkpoint_pattern(tmp_path):
    with pytest.raises(IntegrityError, match="no observations"):
        output_inventory(tmp_path, ("epoch-*",))


def test_stage_plan_tracks_weights_beyond_training_index(tmp_path):
    path = tmp_path / "config.json"
    write_json(
        path, {"molecules": ["ethanol"], "seeds": [1], "models": ["equivariant"], "budgets": [100]}
    )
    stages = research_stages(path, tmp_path / "data", tmp_path / "run")
    train = next(s for s in stages if s.name == "train")
    assert "equivariant-ethanol-100-1/initial" in train.outputs
    assert "equivariant-ethanol-100-1/history.json" in train.outputs
    assert any("epoch-" in name for name in train.outputs)
    acquire = next(s for s in stages if s.name == "acquire")
    assert acquire.outputs == ("acquisition",)


def test_stage_bundle_receipts_retain_owned_raw_bytes(receipt, tmp_path):
    write_json(receipt / "training-units.json", [])
    evidence = tmp_path / "bundle"
    evidence.mkdir()
    result = retain_stage_receipts(receipt, evidence)
    assert (evidence / result["value"]["path"]).read_bytes() == b"actual observation"
    assert verify_stage_receipts(evidence, ("sample",))["sample"]["exit"] == 0


def test_stage_bundle_refuses_post_execution_changed_observation(receipt, tmp_path):
    write_json(receipt / "training-units.json", [])
    (receipt / "value").write_bytes(b"changed after successful execution")
    evidence = tmp_path / "bundle"
    evidence.mkdir()
    with pytest.raises(IntegrityError, match="changed before"):
        retain_stage_receipts(receipt, evidence)


def test_command_receipt_rejects_absent_required_stage(tmp_path):
    (tmp_path / "COMMANDS.jsonl").write_text(
        '{"stage":"train","started_ns":1,"ended_ns":2,"exit":0,"execution":"in-process-original-stage"}\n'
    )
    with pytest.raises(IntegrityError, match="absent"):
        verify_command_receipts(tmp_path, ("train", "evaluate"))


def test_command_receipt_rejects_unexplained_failed_invocation(tmp_path):
    (tmp_path / "COMMANDS.jsonl").write_text(
        '{"stage":"train","started_ns":1,"ended_ns":2,"exit":7,"error":"failed","execution":"in-process-original-stage"}\n'
    )
    with pytest.raises(IntegrityError, match="continuation"):
        verify_command_receipts(tmp_path, ("train",))


def stress_receipts(tmp_path, missing=None, command="stress"):
    from asterion.artifacts.io import canonical

    phases = ["simulation-recovery", "training-recovery", "faults", "memory-stress"]
    records = [
        {
            "stage": name,
            "started_ns": i * 10 + 1,
            "ended_ns": i * 10 + 2,
            "exit": 0,
            "execution": "in-process-original-stage",
            "argv": ["cli.py", command],
        }
        for i, name in enumerate(phases)
        if name != missing
    ]
    (tmp_path / "COMMANDS.jsonl").write_bytes(b"\n".join(canonical(row) for row in records) + b"\n")
    return {"stress": {"argv": [sys.executable, "-m", "asterion.cli", "stress"], "attempt": 1}}


def test_command_receipt_binds_all_stress_phases_to_managed_process(tmp_path):
    finals = stress_receipts(tmp_path)
    result = verify_command_receipts(tmp_path, ("stress",), finals)
    assert set(result) == {"simulation-recovery", "training-recovery", "faults", "memory-stress"}


@pytest.mark.parametrize(
    "phase", ["simulation-recovery", "training-recovery", "faults", "memory-stress"]
)
def test_command_receipt_rejects_omitted_stress_phase(tmp_path, phase):
    finals = stress_receipts(tmp_path, missing=phase)
    with pytest.raises(IntegrityError, match="absent"):
        verify_command_receipts(tmp_path, ("stress",), finals)


def test_command_receipt_rejects_substituted_stress_process(tmp_path):
    finals = stress_receipts(tmp_path, command="bench")
    with pytest.raises(IntegrityError, match="differs"):
        verify_command_receipts(tmp_path, ("stress",), finals)
