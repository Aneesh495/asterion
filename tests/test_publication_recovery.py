import os
import signal
import subprocess
import sys
import time

import pytest

from asterion.artifacts import bundle
from asterion.artifacts.io import file_hash, read_json, write_json
from asterion.artifacts.verify import check_manifest
from asterion.contracts.errors import IntegrityError


@pytest.fixture
def candidate(tmp_path):
    # This is a publication-protocol fixture, not scientific full-acceptance evidence.
    source = tmp_path / "candidate"
    write_json(
        source / "ACCEPTANCE.json",
        {
            "status": "candidate",
            "profile": "full",
            "counts": {"negative_controls": 0},
            "gates": {"fixture": {"status": "unverified"}},
            "reference": {"status": "candidate"},
            "secondary": {"status": "candidate"},
        },
    )
    write_json(source / "SOURCE_INPUTS.json", {"digest": "publication-fixture"})
    write_json(source / "GATE_OBSERVATIONS.json", {"controls": "gates/controls.json"})
    (source / "raw").write_bytes(b"publication payload fixture")
    bundle.refresh_manifest(source)
    digest = file_hash(source / "MANIFEST.json")
    controls = tmp_path / "controls.json"
    write_json(
        controls,
        {
            "profile": "full",
            "phase": "candidate",
            "candidate_manifest_sha256": digest,
            "positive_before": {"exit": 0},
            "positive_after": {"exit": 0},
            "source_bundle_unchanged": True,
            "controls": [
                {"exit": 2, "stderr": "protocol fixture rejection", "semantic_rehashed": True}
            ]
            * 12,
        },
    )
    verified = {"phase": "candidate", "source": "publication-fixture", "manifest_sha256": digest}
    return source, tmp_path / "published", controls, verified


def test_publication_commits_complete_manifest_and_retains_candidate(candidate):
    source, final, controls, verified = candidate
    original = file_hash(source / "MANIFEST.json")
    result = bundle.promote_full(source, final, controls, verified)
    check_manifest(final)
    assert (
        result["status"] == "passed"
        and (final / "raw").read_bytes() == (source / "raw").read_bytes()
    )
    assert file_hash(source / "MANIFEST.json") == original
    intent = next(final.with_name(f".{final.name}.publication").glob("attempt-*/PUBLICATION.json"))
    assert read_json(intent)["status"] == "published"


def test_publication_copy_failure_keeps_final_absent_and_partial_evidence(candidate, monkeypatch):
    source, final, controls, verified = candidate

    def fail_copy(original, destination):
        destination.mkdir()
        (destination / "partial").write_bytes(b"retained incomplete copy")
        raise OSError("injected copy failure")

    monkeypatch.setattr(bundle, "copy_observations", fail_copy)
    with pytest.raises(OSError, match="copy failure"):
        bundle.promote_full(source, final, controls, verified)
    assert not final.exists()
    parent = final.with_name(f".{final.name}.publication")
    assert len(list(parent.glob("attempt-*/bundle/partial"))) == 1
    assert read_json(next(parent.glob("attempt-*/PUBLICATION.json")))["status"] == "interrupted"


def test_publication_rejects_candidate_changed_during_copy(candidate, monkeypatch):
    source, final, controls, verified = candidate
    original = bundle.copy_observations

    def corrupt_copy(before, after):
        original(before, after)
        (after / "raw").write_bytes(b"changed after verification")

    monkeypatch.setattr(bundle, "copy_observations", corrupt_copy)
    with pytest.raises(IntegrityError, match="checksum"):
        bundle.promote_full(source, final, controls, verified)
    assert not final.exists()


def test_publication_rejects_controls_changed_between_validation_and_copy(candidate, monkeypatch):
    source, final, controls, verified = candidate
    original = bundle.copy_observations

    def altered_controls(before, after):
        original(before, after)
        record = read_json(controls)
        record["positive_after"]["exit"] = 2
        write_json(controls, record)

    monkeypatch.setattr(bundle, "copy_observations", altered_controls)
    with pytest.raises(IntegrityError, match="controls changed"):
        bundle.promote_full(source, final, controls, verified)
    assert not final.exists()


def test_publication_does_not_overwrite_existing_output(candidate):
    source, final, controls, verified = candidate
    final.mkdir()
    (final / "owned").write_text("original")
    with pytest.raises(IntegrityError, match="new output"):
        bundle.promote_full(source, final, controls, verified)
    assert (final / "owned").read_text() == "original"


def test_killed_publication_recovers_stale_owner_without_publishing_partial_copy(
    candidate, tmp_path
):
    source, final, controls, verified = candidate
    marker = tmp_path / "copied"
    program = """
import sys,time
from pathlib import Path
from asterion.artifacts import bundle
from asterion.artifacts.io import read_json
source,final,controls,proof,marker=map(Path,sys.argv[1:])
original=bundle.copy_observations
def pause(before,after):
    original(before,after)
    marker.write_text(str(after))
    time.sleep(600)
bundle.copy_observations=pause
bundle.promote_full(source,final,controls,read_json(proof))
"""
    proof = tmp_path / "verified.json"
    write_json(proof, verified)
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            program,
            str(source),
            str(final),
            str(controls),
            str(proof),
            str(marker),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
    )
    try:
        deadline = time.monotonic() + 20
        while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists(), (
            process.communicate(timeout=1)[1].decode()
            if process.poll() is not None
            else "publication did not pause"
        )
        process.send_signal(signal.SIGKILL)
        process.wait(timeout=5)
        assert not final.exists()
        assert (read_json(proof))["manifest_sha256"] == file_hash(source / "MANIFEST.json")
        bundle.promote_full(source, final, controls, verified)
        check_manifest(final)
        parent = final.with_name(f".{final.name}.publication")
        assert len(list(parent.glob("attempt-*/PUBLICATION.json"))) == 2
        assert len(list(parent.glob(".coordinator.stale-*"))) == 1
        assert not (parent / ".coordinator.lock").exists()
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)
