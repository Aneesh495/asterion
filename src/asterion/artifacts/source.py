import platform
import resource
import subprocess
import sys
import tomllib
from pathlib import Path

import e3nn
import numpy as np
import scipy
import torch

from asterion.artifacts.io import file_hash, object_hash
from asterion.contracts.errors import ContractError, IntegrityError


def project_root() -> Path:
    # Installed wheels have no repository alongside site-packages. Prefer the calling
    # checkout, then the editable source location, using an explicit project marker.
    for anchor in (Path.cwd(), Path(__file__).resolve().parent):
        for candidate in (anchor, *anchor.parents):
            marker = candidate / "pyproject.toml"
            if not marker.is_file():
                continue
            try:
                metadata = tomllib.loads(marker.read_text())
            except (OSError, tomllib.TOMLDecodeError):
                continue
            if (
                metadata.get("project", {}).get("name") == "asterion-md"
                and (candidate / "src/asterion/artifacts/source.py").is_file()
            ):
                return candidate
    raise ContractError("run repository commands from an Asterion source checkout")


def assert_runtime_source(root: Path):
    """A wheel must not record checkout source hashes while executing different bytes."""
    expected = root / "src/asterion"
    loaded = Path(__file__).resolve().parents[1]

    def inventory(directory):
        return {
            str(path.relative_to(directory)): file_hash(path)
            for path in directory.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        }

    reference = inventory(expected)
    if not reference or inventory(loaded) != reference:
        raise IntegrityError("installed Asterion code differs from the source checkout")


def source_inputs(root: Path | None = None):
    root = root or project_root()
    assert_runtime_source(root)
    paths: list[Path] = []
    for directory in ("src", "tests", "configs", "scripts", "requirements", "docs", ".github"):
        paths.extend(
            path
            for path in (root / directory).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        )
    paths.extend(
        root / name
        for name in (
            "pyproject.toml",
            "uv.lock",
            "Makefile",
            ".python-version",
            "README.md",
            "LICENSE",
        )
        if (root / name).exists()
    )
    hashes = {str(path.relative_to(root)): file_hash(path) for path in sorted(paths)}
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True, check=True
    ).stdout.strip()
    return {"commit": commit, "hashes": hashes, "digest": object_hash(hashes), "dirty": bool(dirty)}


def environment():
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "e3nn": e3nn.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "mps_available": torch.backends.mps.is_available(),
        "threads": torch.get_num_threads(),
        "peak_rss_native": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "rss_native_units": "bytes" if sys.platform == "darwin" else "KiB",
    }
