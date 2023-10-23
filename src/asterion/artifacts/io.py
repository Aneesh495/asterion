import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from asterion.contracts.errors import IntegrityError


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def object_hash(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path: Path, value):
    atomic_bytes(path, canonical(value) + b"\n")


def read_json(path: Path):
    if path.stat().st_size > 64 * 1024 * 1024:
        raise IntegrityError("JSON artifact exceeds safety bound")
    return json.loads(
        path.read_text(),
        parse_constant=lambda x: (_ for _ in ()).throw(
            IntegrityError(f"nonfinite JSON constant {x}")
        ),
    )


def atomic_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile() as stream:
        np.savez_compressed(stream, **arrays)
        stream.seek(0)
        atomic_bytes(path, stream.read())


def safe_member(root: Path, relative: str) -> Path:
    member = Path(relative)
    if member.is_absolute() or ".." in member.parts:
        raise IntegrityError("artifact path escapes evidence root")
    path = root
    for part in member.parts:
        path /= part
        if path.is_symlink():
            raise IntegrityError("linked artifact path")
    path = path.resolve()
    if not path.is_relative_to(root.resolve()):
        raise IntegrityError("artifact path escapes evidence root")
    return path
