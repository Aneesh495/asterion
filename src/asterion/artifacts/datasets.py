"""Resolve external author bytes while retaining the immutable recorded identities."""

from pathlib import Path

from asterion.artifacts.io import file_hash, read_json
from asterion.artifacts.source import project_root
from asterion.contracts.errors import IntegrityError


def checked_md22(record, storage: Path | None = None):
    pinned = read_json(project_root() / "configs/md22-release.json")
    if {key: value for key, value in record.items() if key != "path"} != pinned:
        raise IntegrityError("secondary metadata differs from the pinned author release")
    path = storage.resolve() / "md22" / pinned["filename"] if storage else Path(record["path"])
    if (
        not path.is_file()
        or path.stat().st_size != pinned["size"]
        or file_hash(path) != pinned["sha256"]
    ):
        raise IntegrityError("secondary author bytes differ from the pinned release")
    return path


def pinned_md22(storage: Path | None = None, original_path: Path | None = None):
    manifest = read_json(project_root() / "configs/md22-release.json")
    if storage is None and original_path is None:
        raise IntegrityError("secondary author data location is unavailable")
    return checked_md22({**manifest, "path": str(original_path)}, storage)
