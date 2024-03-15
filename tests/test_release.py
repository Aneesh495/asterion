import gzip
import io
import tarfile

import pytest

from asterion.artifacts import release, verify
from asterion.artifacts.io import canonical, file_hash, read_json
from asterion.contracts.errors import IntegrityError, ResourceError


@pytest.fixture
def payload(tmp_path):
    source = tmp_path / "weights"
    source.write_bytes(b"container protocol fixture, not trained model evidence")
    return {"models/weights": source}, {"kind": "protocol-fixture"}


def test_release_container_is_byte_deterministic(payload, tmp_path):
    files, metadata = payload
    first, second = tmp_path / "one.gz", tmp_path / "two.gz"
    release.write_archive(first, files, metadata)
    release.write_archive(second, files, metadata)
    assert first.read_bytes() == second.read_bytes()
    checked = release.verify_release_archive(first)
    assert checked["members"] == 1 and checked["metadata"]["kind"] == "protocol-fixture"
    assert checked["status"] == "archive-integrity-verified"


def forged_archive(path, files, metadata, mode):
    manifest = {
        "format": "asterion-release-v1",
        "metadata": metadata,
        "files": release.release_inventory(files),
    }
    encoded = canonical(manifest)
    with tarfile.open(path, "w:gz") as archive:
        archive.addfile(release.tar_info("RELEASE.json", len(encoded)), io.BytesIO(encoded))
        for name, original in files.items():
            data = original.read_bytes()
            if mode == "missing":
                continue
            info = release.tar_info(name, len(data))
            if mode == "link":
                info.type = tarfile.SYMTYPE
                info.linkname = "../escape"
                info.size = 0
                archive.addfile(info)
            elif mode == "escape":
                info.name = "../escape"
                archive.addfile(info, io.BytesIO(data))
            elif mode == "corrupt":
                archive.addfile(info, io.BytesIO(bytes([data[0] ^ 1]) + data[1:]))
            else:
                archive.addfile(info, io.BytesIO(data))
                if mode == "repeat":
                    archive.addfile(info, io.BytesIO(data))
                if mode == "extra":
                    archive.addfile(release.tar_info("unexplained", 1), io.BytesIO(b"x"))


@pytest.mark.parametrize("mode", ["missing", "link", "escape", "corrupt", "repeat", "extra"])
def test_release_readback_rejects_invalid_members(payload, tmp_path, mode):
    files, metadata = payload
    path = tmp_path / "invalid.tar.gz"
    forged_archive(path, files, metadata, mode)
    with pytest.raises(IntegrityError):
        release.verify_release_archive(path)
    assert not (tmp_path / "escape").exists()


def test_release_rejects_unsafe_requested_name(payload, tmp_path):
    files, metadata = payload
    with pytest.raises(IntegrityError, match="unsafe"):
        release.write_archive(
            tmp_path / "bad.gz", {"../outside": next(iter(files.values()))}, metadata
        )


def test_release_rejects_linked_source(payload, tmp_path):
    files, metadata = payload
    link = tmp_path / "link"
    link.symlink_to(next(iter(files.values())))
    with pytest.raises(IntegrityError, match="linked"):
        release.write_archive(tmp_path / "bad.gz", {"weights": link}, metadata)


def test_release_rejects_truncated_archive(payload, tmp_path):
    files, metadata = payload
    path = tmp_path / "truncated.gz"
    release.write_archive(path, files, metadata)
    path.write_bytes(path.read_bytes()[:70])
    with pytest.raises(IntegrityError):
        release.verify_release_archive(path)


def test_release_atomic_publication_retains_complete_readback(payload, tmp_path):
    files, metadata = payload
    output = tmp_path / "release.tar.gz"
    checked = release.publish_archive(output, files, metadata)
    assert checked["sha256"] == file_hash(output)
    intent = next(tmp_path.glob(".release.tar.gz.publication/attempt-*/PUBLICATION.json"))
    assert read_json(intent)["status"] == "published"


def test_release_atomic_publication_preserves_existing_output(payload, tmp_path):
    files, metadata = payload
    output = tmp_path / "release.tar.gz"
    output.write_bytes(b"existing output")
    with pytest.raises(IntegrityError, match="already exists"):
        release.publish_archive(output, files, metadata)
    assert output.read_bytes() == b"existing output"


def test_release_failure_retains_partial_and_leaves_final_absent(payload, tmp_path, monkeypatch):
    files, metadata = payload
    output = tmp_path / "release.tar.gz"
    original = release.write_archive

    def interrupted(path, files, metadata):
        original(path, files, metadata)
        raise OSError("injected publication failure")

    monkeypatch.setattr(release, "write_archive", interrupted)
    with pytest.raises(OSError, match="injected"):
        release.publish_archive(output, files, metadata)
    assert not output.exists()
    parent = tmp_path / ".release.tar.gz.publication"
    assert len(list(parent.glob("attempt-*/release.tar.gz"))) == 1
    assert read_json(next(parent.glob("attempt-*/PUBLICATION.json")))["status"] == "interrupted"
    monkeypatch.setattr(release, "write_archive", original)
    release.publish_archive(output, files, metadata)
    assert len(list(parent.glob("attempt-*/PUBLICATION.json"))) == 2


def test_release_changed_input_fails_readback(payload, tmp_path, monkeypatch):
    files, metadata = payload
    original = release.release_inventory

    def changed(inputs):
        inventory = original(inputs)
        next(iter(inputs.values())).write_bytes(b"substituted after inventory")
        return inventory

    monkeypatch.setattr(release, "release_inventory", changed)
    with pytest.raises(IntegrityError, match="changed"):
        release.publish_archive(tmp_path / "release.gz", files, metadata)
    assert not (tmp_path / "release.gz").exists()


def test_release_rejects_dirty_source_before_any_packaging(tmp_path, monkeypatch):
    monkeypatch.setattr(release, "source_inputs", lambda: {"dirty": True})
    with pytest.raises(IntegrityError, match="committed"):
        release.build_release(tmp_path / "absent", tmp_path / "data", tmp_path / "release.gz")
    assert not (tmp_path / "release.gz").exists()


@pytest.mark.parametrize(
    "profile,phase,status",
    [("diagnostic", "final", "passed"), ("full", "candidate", "candidate-verified")],
)
def test_release_rejects_diagnostic_or_candidate_verification(
    tmp_path, monkeypatch, profile, phase, status
):
    monkeypatch.setattr(release, "source_inputs", lambda: {"dirty": False})
    monkeypatch.setattr(
        verify, "verify", lambda *a, **k: {"profile": profile, "phase": phase, "status": status}
    )
    with pytest.raises(IntegrityError, match="full final"):
        release.build_release(tmp_path / "diagnostic", tmp_path / "data", tmp_path / "release.gz")
    assert not (tmp_path / "release.gz").exists()


def test_release_preserves_actual_full_verifier_rejection(tmp_path, monkeypatch):
    monkeypatch.setattr(release, "source_inputs", lambda: {"dirty": False})

    def blocked(*args, **kwargs):
        raise IntegrityError("required full observations absent")

    monkeypatch.setattr(verify, "verify", blocked)
    with pytest.raises(IntegrityError, match="full observations"):
        release.build_release(tmp_path / "blocked", tmp_path / "data", tmp_path / "release.gz")
    assert not (tmp_path / "release.gz").exists()


def test_release_archive_size_bound_is_checked_before_writing(payload, tmp_path, monkeypatch):
    files, metadata = payload
    monkeypatch.setattr(release, "MAX_TOTAL_BYTES", 1)
    with pytest.raises(ResourceError, match="size bound"):
        release.write_archive(tmp_path / "big.gz", files, metadata)
    assert not (tmp_path / "big.gz").exists()


def test_release_rejects_bad_gzip_trailer(payload, tmp_path):
    files, metadata = payload
    path = tmp_path / "crc.gz"
    release.write_archive(path, files, metadata)
    data = bytearray(path.read_bytes())
    data[-8] ^= 1
    path.write_bytes(data)
    with pytest.raises(IntegrityError):
        release.verify_release_archive(path)
    with pytest.raises(gzip.BadGzipFile):
        gzip.decompress(path.read_bytes())
