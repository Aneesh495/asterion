import os
import urllib.request
from pathlib import Path

from asterion.artifacts.io import file_hash, read_json, write_json
from asterion.contracts.errors import IntegrityError, ResourceError


def acquire_md22(manifest_path: Path, storage: Path):
    manifest = read_json(manifest_path)
    destination = storage / "md22" / manifest["filename"]
    if not destination.exists():
        temporary = destination.with_suffix(".tmp")
        destination.parent.mkdir(parents=True, exist_ok=True)
        total = 0
        try:
            with (
                urllib.request.urlopen(manifest["url"], timeout=60) as response,
                temporary.open("wb") as stream,
            ):
                while chunk := response.read(1 << 20):
                    total += len(chunk)
                    if total > manifest["size"]:
                        raise ResourceError("MD22 exceeds frozen bounded file size")
                    stream.write(chunk)
            if total != manifest["size"] or file_hash(temporary) != manifest["sha256"]:
                raise IntegrityError("MD22 release content changed")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    if file_hash(destination) != manifest["sha256"]:
        raise IntegrityError("MD22 existing file checksum mismatch")
    write_json(storage / "md22" / "DATASETS.json", {**manifest, "path": str(destination.resolve())})
    return destination, manifest
