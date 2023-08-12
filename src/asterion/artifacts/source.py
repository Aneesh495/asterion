import platform
import resource
import subprocess
import sys
from pathlib import Path

import e3nn
import numpy as np
import scipy
import torch

from asterion.artifacts.io import file_hash, object_hash


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def source_inputs(root: Path | None = None):
    root = root or project_root()
    paths = []
    for directory in ("src", "tests", "configs", "scripts"):
        paths.extend(
            path
            for path in (root / directory).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        )
    paths.extend(
        root / name
        for name in ("pyproject.toml", "uv.lock", "Makefile", ".python-version")
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
