import hashlib
from copy import deepcopy
from pathlib import Path

import pytest

from asterion.artifacts import verify as verifier
from asterion.artifacts.datasets import checked_md22, pinned_md22
from asterion.artifacts.io import read_json, write_json
from asterion.contracts.errors import IntegrityError


@pytest.fixture
def release(tmp_path, monkeypatch):
    names = [f"rmd17_{m}.npz" for m in ("aspirin", "ethanol", "malonaldehyde", "salicylic")]
    names += [f"index_{role}_{seed:02}.csv" for role in ("train", "test") for seed in (1, 2, 3)]
    names += ["readme.txt"]
    records = []
    for index, name in enumerate(names):
        payload = f"unit provenance fixture {name}".encode()
        path = tmp_path / name
        path.write_bytes(payload)
        records.append(
            {
                "name": name,
                "path": str(path),
                "id": index,
                "size": len(payload),
                "download_url": f"https://author.invalid/{index}",
                "supplied_md5": hashlib.md5(payload).hexdigest(),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    release = {
        "version": 4,
        "doi": "author-release-v4",
        "license": {"name": "CC0"},
        "files": records,
    }
    write_json(tmp_path / "configs/rmd17-release-v4.json", release)
    monkeypatch.setattr(verifier, "project_root", lambda: tmp_path)
    return {**deepcopy(release), "family": "rMD17", "level": "PBE/def2-SVP"}


def test_dataset_provenance_binds_all_eleven_files_to_source_release(release):
    assert len(verifier.check_dataset(release)) == 11


def test_dataset_provenance_rejects_wrong_release_version(release):
    release["version"] = 3
    with pytest.raises(IntegrityError, match="identity"):
        verifier.check_dataset(release)


def test_dataset_provenance_rejects_changed_electronic_level(release):
    release["level"] = "PBE+MBD"
    with pytest.raises(IntegrityError, match="level"):
        verifier.check_dataset(release)


def test_dataset_provenance_rejects_changed_license(release):
    release["license"] = {"name": "unverified"}
    with pytest.raises(IntegrityError, match="license"):
        verifier.check_dataset(release)


def test_dataset_provenance_rejects_repeated_file_hiding_missing_molecule(release):
    release["files"][0] = release["files"][1]
    with pytest.raises(IntegrityError, match="population"):
        verifier.check_dataset(release)


def test_dataset_provenance_rejects_substituted_bytes_even_with_updated_sha(release):
    record = release["files"][0]
    from pathlib import Path

    payload = b"substituted author data"
    Path(record["path"]).write_bytes(payload)
    record["sha256"] = hashlib.sha256(payload).hexdigest()
    with pytest.raises(IntegrityError, match="pinned author release"):
        verifier.check_dataset(release)


def moved_release(release, storage):
    directory = storage / "rmd17"
    directory.mkdir(parents=True)
    for record in release["files"]:
        Path(record["path"]).rename(directory / record["name"])


def test_dataset_verifier_resolves_relocated_bytes_without_rewriting_recorded_paths(
    release, tmp_path
):
    original = deepcopy(release)
    storage = tmp_path / "relocated"
    moved_release(release, storage)
    paths = verifier.check_dataset(release, storage)
    assert release == original
    assert all(path.parent == storage / "rmd17" for path in paths.values())
    assert all(not Path(record["path"]).exists() for record in release["files"])


def test_dataset_storage_override_does_not_fall_back_to_original_files(release, tmp_path):
    with pytest.raises(IntegrityError, match="external dataset missing"):
        verifier.check_dataset(release, tmp_path / "unavailable-storage")


def test_relocated_dataset_still_rejects_coherently_substituted_bytes(release, tmp_path):
    storage = tmp_path / "relocated"
    moved_release(release, storage)
    record = release["files"][0]
    payload = b"self-consistent substituted release"
    (storage / "rmd17" / record["name"]).write_bytes(payload)
    record["sha256"] = hashlib.sha256(payload).hexdigest()
    with pytest.raises(IntegrityError, match="pinned author release"):
        verifier.check_dataset(release, storage)


@pytest.fixture
def md22_release(tmp_path, monkeypatch):
    from asterion.artifacts import datasets

    pinned = read_json(datasets.project_root() / "configs/md22-release.json")
    path = tmp_path / pinned["filename"]
    payload = b"bounded MD22 provenance fixture"
    path.write_bytes(payload)
    pinned.update(sha256=hashlib.sha256(payload).hexdigest(), size=len(payload))
    write_json(tmp_path / "configs/md22-release.json", pinned)
    monkeypatch.setattr(datasets, "project_root", lambda: tmp_path)
    return {**pinned, "path": str(path)}


def test_secondary_verifier_binds_all_pinned_author_metadata(md22_release):
    assert checked_md22(md22_release) == Path(md22_release["path"])


def test_secondary_verifier_rejects_self_consistent_file_substitution(md22_release):
    payload = b"substituted secondary file with updated report"
    Path(md22_release["path"]).write_bytes(payload)
    md22_release.update(sha256=hashlib.sha256(payload).hexdigest(), size=len(payload))
    with pytest.raises(IntegrityError, match="pinned author release"):
        checked_md22(md22_release)


def test_secondary_verifier_rejects_invented_verified_license(md22_release):
    md22_release["license_status"] = "verified"
    with pytest.raises(IntegrityError, match="metadata"):
        checked_md22(md22_release)


def test_secondary_verifier_rejects_another_electronic_level(md22_release):
    md22_release["level"] = "PBE/def2-SVP"
    with pytest.raises(IntegrityError, match="metadata"):
        checked_md22(md22_release)


def test_secondary_verifier_relocates_pinned_bytes_without_changing_provenance(
    md22_release, tmp_path
):
    original = deepcopy(md22_release)
    storage = tmp_path / "relocated"
    (storage / "md22").mkdir(parents=True)
    target = storage / "md22" / md22_release["filename"]
    Path(md22_release["path"]).rename(target)
    assert checked_md22(md22_release, storage) == target
    assert pinned_md22(storage) == target and md22_release == original


def test_secondary_override_does_not_use_original_bytes(md22_release, tmp_path):
    with pytest.raises(IntegrityError, match="author bytes"):
        checked_md22(md22_release, tmp_path / "unavailable-storage")


def test_secondary_verifier_rejects_truncated_original_file(md22_release):
    Path(md22_release["path"]).write_bytes(b"truncated")
    with pytest.raises(IntegrityError, match="author bytes"):
        checked_md22(md22_release)
