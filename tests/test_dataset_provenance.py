import hashlib
from copy import deepcopy

import pytest

from asterion.artifacts import verify as verifier
from asterion.artifacts.io import write_json
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
