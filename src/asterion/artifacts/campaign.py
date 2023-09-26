"""Single-owner, source-frozen execution of the complete research stage graph."""

import ast
import math
import os
import platform
import re
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from asterion.artifacts.io import file_hash, object_hash, read_json, write_json
from asterion.artifacts.source import environment, project_root, source_inputs
from asterion.contracts.errors import IntegrityError, ResourceError


@dataclass(frozen=True)
class Stage:
    name: str
    argv: tuple[str, ...]
    outputs: tuple[str, ...]
    timeout_seconds: float = 72 * 3600

    def validate(self):
        if (
            not self.name
            or not self.argv
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise IntegrityError("invalid campaign stage")
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", self.name):
            raise IntegrityError("invalid campaign stage name")
        if any(Path(p).is_absolute() or ".." in Path(p).parts for p in self.outputs):
            raise IntegrityError("stage output must remain inside the run")
        return self


def output_inventory(root: Path, outputs):
    inventory = {}
    for relative in outputs:
        path = root / relative
        if not path.exists() or path.is_symlink():
            raise IntegrityError(f"declared stage output is absent or a symlink: {relative}")
        files = [path] if path.is_file() else list(path.rglob("*"))
        retained = 0
        for item in files:
            if item.is_symlink():
                raise IntegrityError("symlink in campaign output")
            if item.is_file():
                inventory[str(item.relative_to(root))] = file_hash(item)
                retained += 1
        if not retained:
            raise IntegrityError(f"declared stage output is empty: {relative}")
    return inventory


class CampaignOwner:
    def __init__(self, run: Path, resume=False):
        self.run = run
        self.resume = resume
        self.path = run / ".coordinator.lock"
        self.record = {
            "pid": os.getpid(),
            "host": platform.node(),
            "nonce": uuid.uuid4().hex,
            "created_ns": time.time_ns(),
        }

    def __enter__(self):
        self.run.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            previous = read_json(self.path)
            if (
                not isinstance(previous.get("pid"), int)
                or previous["pid"] <= 0
                or not isinstance(previous.get("host"), str)
                or not re.fullmatch(r"[0-9a-f]{32}", str(previous.get("nonce", "")))
            ):
                raise IntegrityError("malformed campaign ownership record")
            if not self.resume or previous["host"] != self.record["host"]:
                raise IntegrityError("campaign already has an owner")
            try:
                os.kill(int(previous["pid"]), 0)
            except ProcessLookupError:
                self.path.rename(self.run / f".coordinator.stale-{previous['nonce']}")
            else:
                raise IntegrityError("cannot replace a live campaign owner")
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as failure:
            raise IntegrityError("another coordinator acquired the run") from failure
        import json

        with os.fdopen(descriptor, "w") as stream:
            json.dump(self.record, stream)
            stream.flush()
            os.fsync(stream.fileno())
        return self

    def __exit__(self, *unused):
        if self.path.exists() and read_json(self.path).get("nonce") == self.record["nonce"]:
            self.path.unlink()


class StageRunner:
    def __init__(self, run: Path, identity, root: Path | None = None, resume=False):
        self.run = run
        self.root = root or project_root()
        self.identity = identity
        self.resume = resume
        self.state_path = run / "STAGES.json"
        if self.state_path.exists():
            if not resume:
                raise IntegrityError("campaign state exists; explicit resume required")
            self.state = read_json(self.state_path)
            if self.state["identity"] != identity:
                raise IntegrityError("cannot resume different source, data, or configuration")
        else:
            self.state = {"identity": identity, "stages": [], "status": "running"}
        self.child: subprocess.Popen | None = None

    def terminate_child(self):
        child = self.child
        if child is None or child.poll() is not None:
            return
        # The process group was created for this stage. No unrelated process is targeted.
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)

    def execute(self, stage: Stage):
        stage.validate()
        previous = next((r for r in self.state["stages"] if r["name"] == stage.name), None)
        definition = object_hash(asdict(stage))
        if previous and previous["status"] == "passed":
            if previous["definition"] != definition or previous["outputs"] != output_inventory(
                self.run, stage.outputs
            ):
                raise IntegrityError("completed campaign stage inputs or outputs changed")
            return previous
        if previous:
            raise IntegrityError("failed stage retained; use a new campaign identity after repair")
        logs = self.run / "stage-logs"
        logs.mkdir(exist_ok=True)
        record: dict[str, Any] = {
            "name": stage.name,
            "definition": definition,
            "argv": list(stage.argv),
            "status": "running",
            "started_ns": time.time_ns(),
            "exit": None,
            "stdout": str((logs / f"{stage.name}.stdout").relative_to(self.run)),
            "stderr": str((logs / f"{stage.name}.stderr").relative_to(self.run)),
        }
        self.state["stages"].append(record)
        write_json(self.state_path, self.state)
        print(f"campaign stage={stage.name}", flush=True)
        started = time.monotonic()
        try:
            with (
                (self.run / record["stdout"]).open("wb") as stdout,
                (self.run / record["stderr"]).open("wb") as stderr,
            ):
                self.child = subprocess.Popen(
                    stage.argv,
                    cwd=self.root,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                    env={**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"},
                )
                while self.child.poll() is None:
                    if time.monotonic() - started > stage.timeout_seconds:
                        raise ResourceError(f"campaign stage timed out: {stage.name}")
                    if max(stdout.tell(), stderr.tell()) > 128 * 1024**2:
                        raise ResourceError(f"campaign stage log budget exceeded: {stage.name}")
                    time.sleep(0.2)
                record["exit"] = self.child.returncode
            if record["exit"] != 0:
                raise ResourceError(f"campaign stage failed: {stage.name}, exit={record['exit']}")
            record["outputs"] = output_inventory(self.run, stage.outputs)
            record["status"] = "passed"
            return record
        except BaseException as failure:
            self.terminate_child()
            record["status"] = "failed"
            record["error"] = f"{type(failure).__name__}: {failure}"
            record["exit"] = self.child.returncode if self.child else None
            self.state["status"] = "failed"
            raise
        finally:
            record["ended_ns"] = time.time_ns()
            record["elapsed_seconds"] = time.monotonic() - started
            record["stdout_sha256"] = file_hash(self.run / record["stdout"])
            record["stderr_sha256"] = file_hash(self.run / record["stderr"])
            write_json(self.state_path, self.state)
            self.child = None


def full_preflight(config):
    source = source_inputs()
    authored = sum(
        isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        for path in (project_root() / "tests").rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text()))
    )
    reasons = []
    if source["dirty"]:
        reasons.append("uncommitted source inputs")
    if authored < config["requirements"]["semantic_tests"]:
        reasons.append("authored semantic test population is below the frozen requirement")
    if not torch.cuda.is_available():
        reasons.append("full profile requires a CUDA accelerator")
    elif torch.cuda.get_device_properties(0).total_memory < 16 * 1024**3:
        reasons.append("full profile requires at least 16 GiB accelerator memory")
    reference = Path(config["reference"].get("python", ".reference-venv/bin/python")).resolve()
    if not reference.is_file():
        reasons.append("separate pinned reference environment is absent")
    return {
        "source": source,
        "environment": environment(),
        "authored_tests": authored,
        "status": "blocked" if reasons else "ready",
        "reasons": reasons,
    }


def research_stages(config_path: Path, storage: Path, run: Path, resume=False):
    prefix = (sys.executable, "-m", "asterion.cli")

    def command(name):
        args = prefix + (
            name,
            "--config",
            str(config_path.resolve()),
            "--storage",
            str(storage.resolve()),
            "--run-id",
            run.name,
        )
        return args + (("--resume",) if name == "train" else ())

    return [
        Stage(
            "tests",
            (sys.executable, "-m", "pytest", f"--junitxml={run / 'tests.xml'}"),
            ("tests.xml",),
            3600,
        ),
        Stage("data", command("data"), (), 3600),
        Stage("train", command("train"), ("training-units.json",)),
        Stage("evaluate", command("evaluate"), ("evaluation-units.json",)),
        Stage("simulate", command("simulate"), ("simulation-units.json",)),
        Stage("acquire", command("acquire"), ("acquisition/acquisition-units.json",)),
        Stage("ablate", command("ablate"), ("ablations/ablation-units.json",)),
        Stage("reference", command("reference"), ("reference/reference-units.json",)),
        Stage("secondary", command("secondary"), ("secondary/secondary.json",)),
        Stage("properties", command("properties"), ("properties.json",), 3600),
        Stage("derivatives", command("derivatives"), ("derivatives/derivatives.json",), 3600),
        Stage("analytic", command("analytic"), ("analytic/analytic.json",), 3600),
        Stage(
            "stress",
            command("stress"),
            (
                "memory-stress.json",
                "faults/faults.json",
                "simulation-recovery/recovery.json",
                "training-recovery/recovery.json",
            ),
            7200,
        ),
        Stage("bench", command("bench"), ("benchmark.json",), 3600),
        Stage("demo", command("demo"), ("viewer.html", "failure-viewer.html"), 3600),
    ]


def execute_research(config, config_path, storage: Path, run: Path, evidence: Path, resume=False):
    from asterion.artifacts.bundle import build_full_candidate, promote_full
    from asterion.artifacts.controls import controls
    from asterion.artifacts.verify import verify

    preflight = full_preflight(config)
    if preflight["status"] != "ready":
        if evidence.exists():
            raise IntegrityError("full preflight evidence identity already exists")
        evidence.mkdir(parents=True)
        write_json(evidence / "PREFLIGHT.json", preflight)
        write_json(evidence / "SOURCE_INPUTS.json", preflight["source"])
        write_json(evidence / "CAMPAIGN.json", config)
        write_json(
            evidence / "ACCEPTANCE.json",
            {
                "profile": "full",
                "status": "blocked",
                "reasons": preflight["reasons"],
                "gates": {f"MA{i:02}": {"status": "unverified"} for i in range(1, 13)},
            },
        )
        raise ResourceError("; ".join(preflight["reasons"]))
    if evidence.exists() or (run.exists() and not resume):
        raise IntegrityError("full campaign identity already exists")
    identity = object_hash(
        {
            "source": preflight["source"]["digest"],
            "config": config,
            "storage": str(storage.resolve()),
        }
    )
    with CampaignOwner(run, resume):
        runner = StageRunner(run, identity, resume=resume)
        for stage in research_stages(config_path, storage, run, resume):
            if source_inputs()["digest"] != preflight["source"]["digest"]:
                raise IntegrityError("source changed during the research campaign")
            runner.execute(stage)
        candidate = evidence.with_name(evidence.name + ".candidate")
        build_full_candidate(config, storage, run, candidate)
        verified = verify(candidate, "full", candidate=True)
        control_path = run / "full-controls.json"
        controls(candidate, control_path, "full", candidate=True)
        promote_full(candidate, evidence, control_path, verified)
        result = verify(evidence, "full")
        runner.state["status"] = "passed"
        write_json(runner.state_path, runner.state)
        return result
