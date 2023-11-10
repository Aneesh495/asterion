import json
import subprocess
from pathlib import Path

import pytest

from asterion.contracts.errors import IntegrityError, ResourceError
from asterion.training.reference_env import (
    locked_versions,
    normalize_name,
    probe_reference,
    reference_process_environment,
)


def stack(tmp_path):
    directory = tmp_path / "requirements"
    directory.mkdir()
    text = "mace-torch==0.3.14\ne3nn==0.4.4\ntorch==2.8.0\n"
    for platform in ("darwin", "linux"):
        (directory / f"reference-{platform}.txt").write_text(text)
    executable = tmp_path / "python"
    executable.touch()
    return executable, {
        "versions": {"mace-torch": "0.3.14", "e3nn": "0.4.4", "torch": "2.8.0"},
        "python": [3, 12, 14],
        "cuda_available": False,
    }


def test_comparison_lock_retains_exact_scientific_versions(tmp_path):
    executable, observation = stack(tmp_path)
    assert locked_versions(tmp_path / "requirements/reference-linux.txt") == observation["versions"]
    assert executable.exists()


def test_comparison_dependency_names_follow_distribution_normalization():
    assert normalize_name("Torch_EMA") == "torch-ema"
    assert normalize_name("opt.einsum") == "opt-einsum"


def test_comparison_lock_rejects_floating_scientific_version(tmp_path):
    path = tmp_path / "lock.txt"
    path.write_text("mace-torch>=0.3\ne3nn==0.4.4\ntorch==2.8.0\n")
    with pytest.raises(IntegrityError, match="scientific stack"):
        locked_versions(path)


def test_comparison_lock_rejects_duplicate_dependency(tmp_path):
    path = tmp_path / "lock.txt"
    path.write_text("mace-torch==0.3.14\ne3nn==0.4.4\ne3nn==0.4.4\ntorch==2.8.0\n")
    with pytest.raises(IntegrityError, match="duplicate"):
        locked_versions(path)


def test_comparison_cache_paths_remain_inside_project(tmp_path):
    environment = reference_process_environment(tmp_path)
    assert Path(environment["MPLCONFIGDIR"]).is_relative_to(tmp_path)
    assert Path(environment["XDG_CACHE_HOME"]).is_relative_to(tmp_path)
    assert environment["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"


def test_reference_probe_preserves_virtual_interpreter_symlink(tmp_path, monkeypatch):
    executable, observation = stack(tmp_path)
    actual = tmp_path / "native-python"
    actual.touch()
    executable.unlink()
    executable.symlink_to(actual)

    def run(argv, **unused):
        assert argv[0] == str(executable.absolute())
        assert argv[0] != str(executable.resolve())
        return subprocess.CompletedProcess(argv, 0, json.dumps(observation), "")

    monkeypatch.setattr(subprocess, "run", run)
    assert probe_reference(executable, root=tmp_path)["versions"] == observation["versions"]


def test_reference_probe_rejects_wrong_e3nn_stack(tmp_path, monkeypatch):
    executable, observation = stack(tmp_path)
    observation["versions"]["e3nn"] = "0.5.7"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, json.dumps(observation), ""),
    )
    with pytest.raises(ResourceError, match="pinned"):
        probe_reference(executable, root=tmp_path)


def test_reference_probe_rejects_cpu_runtime_for_full_campaign(tmp_path, monkeypatch):
    executable, observation = stack(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, json.dumps(observation), ""),
    )
    with pytest.raises(ResourceError, match="CUDA"):
        probe_reference(executable, require_cuda=True, root=tmp_path)


def test_reference_probe_retains_import_failure_diagnostic(tmp_path, monkeypatch):
    executable, _ = stack(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 3, "", "incompatible extension"),
    )
    with pytest.raises(ResourceError, match="incompatible extension"):
        probe_reference(executable, root=tmp_path)


def test_reference_probe_bounds_import_execution(tmp_path, monkeypatch):
    executable, _ = stack(tmp_path)

    def timeout(argv, **kwargs):
        assert kwargs["timeout"] == 60
        raise subprocess.TimeoutExpired(argv, 60)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(ResourceError, match="timed out"):
        probe_reference(executable, root=tmp_path)
