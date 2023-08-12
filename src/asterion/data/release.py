import os
import urllib.request
from pathlib import Path

from asterion.artifacts.io import file_hash, read_json, write_json
from asterion.contracts.errors import IntegrityError, ResourceError

PRIMARY = {
    "aspirin": "rmd17_aspirin.npz",
    "ethanol": "rmd17_ethanol.npz",
    "malonaldehyde": "rmd17_malonaldehyde.npz",
    "salicylic": "rmd17_salicylic.npz",
}


def fetch_file(record: dict, destination: Path, maximum_bytes: int = 1_000_000_000):
    expected = int(record["size"])
    if expected <= 0 or expected > maximum_bytes:
        raise ResourceError("download exceeds declared byte budget")
    if destination.exists():
        if destination.stat().st_size != expected:
            raise IntegrityError(f"existing dataset has wrong size: {destination}")
        if file_hash(destination, "md5") != record["computed_md5"]:
            raise IntegrityError(f"existing dataset checksum mismatch: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    consumed = 0
    try:
        with urllib.request.urlopen(record["download_url"], timeout=60) as response:
            with temporary.open("wb") as stream:
                while chunk := response.read(1 << 20):
                    consumed += len(chunk)
                    if consumed > expected:
                        raise ResourceError("server exceeded frozen download size")
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
        if consumed != expected or file_hash(temporary, "md5") != record["computed_md5"]:
            raise IntegrityError("incomplete or changed author file")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def acquire_rmd17(metadata: Path, storage: Path, molecules=tuple(PRIMARY), seeds=(1, 2, 3)):
    release = read_json(metadata)
    if release["id"] != 12672038 or release["version"] != 4:
        raise IntegrityError("rMD17 release identity mismatch")
    requested = [PRIMARY[molecule] for molecule in molecules]
    requested += [f"index_{role}_{seed:02}.csv" for seed in seeds for role in ("train", "test")]
    requested += ["readme.txt"]
    files = {record["name"]: record for record in release["files"]}
    records = []
    for name in requested:
        record = files[name]
        destination = storage / "rmd17" / name
        fetch_file(record, destination)
        records.append({**record, "sha256": file_hash(destination), "path": str(destination)})
    manifest = {
        "family": "rMD17",
        "version": 4,
        "doi": release["doi"],
        "license": release["license"],
        "level": "PBE/def2-SVP",
        "files": records,
    }
    write_json(storage / "rmd17" / "DATASETS.json", manifest)
    return manifest
