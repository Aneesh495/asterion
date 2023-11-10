"""Check the isolated comparison interpreter against its actual platform lock."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from asterion.artifacts.io import file_hash
from asterion.artifacts.source import project_root
from asterion.contracts.errors import IntegrityError, ResourceError


def normalize_name(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def locked_versions(path: Path):
    result = {}
    for line in path.read_text().splitlines():
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)\s*(?:\\)?", line)
        if match:
            name, version = normalize_name(match[1]), match[2]
            if name in result:
                raise IntegrityError("duplicate comparison dependency pin")
            result[name] = version
    if (
        result.get("mace-torch") != "0.3.14"
        or result.get("e3nn") != "0.4.4"
        or result.get("torch") != "2.8.0"
    ):
        raise IntegrityError("comparison lock is missing its fixed scientific stack")
    return result


def reference_process_environment(root: Path):
    cache = root / ".agent-local/reference-cache"
    matplotlib = root / ".agent-local/reference-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    matplotlib.mkdir(parents=True, exist_ok=True)
    return {
        **os.environ,
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "MPLCONFIGDIR": str(matplotlib.resolve()),
        "XDG_CACHE_HOME": str(cache.resolve()),
        # Only the isolated MACE process loads its own exported model or dependency constants.
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
    }


PROBE = """
import hashlib, importlib.metadata, json, pathlib, sys
import torch, e3nn, mace
expected = json.loads(sys.argv[1])
versions = {name: importlib.metadata.version(name) for name in expected}
root = pathlib.Path(mace.__file__).parent
modules = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
           for path in (root/'cli/run_train.py', root/'cli/eval_configs.py')}
print(json.dumps({'versions':versions, 'python':list(sys.version_info[:3]),
  'cuda_available':torch.cuda.is_available(), 'cuda_version':torch.version.cuda,
  'modules':modules, 'e3nn':e3nn.__version__}))
"""


def probe_reference(executable: Path, require_cuda=False, root: Path | None = None):
    root = root or project_root()
    platform = (
        "darwin" if sys.platform == "darwin" else "linux" if sys.platform == "linux" else None
    )
    if platform is None:
        raise ResourceError("comparison platform has no audited dependency lock")
    path = root / "requirements" / f"reference-{platform}.txt"
    expected = locked_versions(path)
    if not executable.is_file():
        raise ResourceError("separate comparison interpreter is absent")
    try:
        completed = subprocess.run(
            [str(executable.absolute()), "-c", PROBE, json.dumps(expected)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=root,
            env=reference_process_environment(root),
        )
    except subprocess.TimeoutExpired as failure:
        raise ResourceError("comparison environment probe timed out") from failure
    if completed.returncode != 0:
        raise ResourceError(f"comparison import failed: {completed.stderr[-2000:]}")
    try:
        observed = json.loads(completed.stdout)
    except ValueError as failure:
        raise ResourceError("comparison probe emitted malformed observations") from failure
    if observed["versions"] != expected or observed["python"][:2] != [3, 12]:
        raise ResourceError("comparison interpreter differs from the pinned platform stack")
    if require_cuda and not observed["cuda_available"]:
        raise ResourceError("comparison interpreter has no working CUDA runtime")
    return {
        **observed,
        "lock_sha256": file_hash(path),
        "lock": str(path.relative_to(root)),
        "executable": str(executable.absolute()),
    }
