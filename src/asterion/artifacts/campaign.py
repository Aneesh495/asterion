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
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from asterion.artifacts.io import file_hash, object_hash, read_json, safe_member, write_json
from asterion.artifacts.source import environment, project_root, source_inputs
from asterion.contracts.errors import CancellationError, IntegrityError, ResourceError


@dataclass(frozen=True)
class Stage:
    name: str
    argv: tuple[str, ...]
    outputs: tuple[str, ...]
    timeout_seconds: float = 72 * 3600
    continuation: str = "fail"
    reset_paths: tuple[str, ...] = ()

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
        if any(
            not p or Path(p).is_absolute() or ".." in Path(p).parts or p == "."
            for p in (*self.outputs, *self.reset_paths)
        ):
            raise IntegrityError("stage output must remain inside the run")
        if self.continuation not in {"fail", "restart", "resume"}:
            raise IntegrityError("unknown stage continuation policy")
        if self.reset_paths and self.continuation != "restart":
            raise IntegrityError("only restart stages may archive owned outputs")
        if any(
            Path(p).parts[0] in {"STAGES.json", ".coordinator.lock", "stage-logs", "stage-attempts"}
            for p in self.reset_paths
        ):
            raise IntegrityError("stage cannot reset coordinator records")
        return self


@contextmanager
def cancellation_signals():
    """Convert cancellation into unwinding so the owner can stop its process group."""
    previous = {}

    def cancel(number, frame):
        raise CancellationError(f"campaign cancelled by signal {number}")

    try:
        for number in (signal.SIGINT, signal.SIGTERM):
            previous[number] = signal.signal(number, cancel)
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def output_inventory(root: Path, outputs):
    root = root.resolve()
    inventory = {}
    for relative in outputs:
        paths = (
            sorted(root.glob(relative))
            if any(mark in relative for mark in "*?[")
            else [root / relative]
        )
        if not paths:
            raise IntegrityError(f"declared stage output pattern has no observations: {relative}")
        files: list[Path] = []
        for path in paths:
            path = safe_member(root, str(path.relative_to(root)))
            if not path.exists():
                raise IntegrityError(f"declared stage output is absent: {relative}")
            files.extend([path] if path.is_file() else path.rglob("*"))
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
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            child.wait()
            return
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=5)

    def archive_partial(self, stage, attempt):
        archive = self.run / "stage-attempts" / f"{stage.name}-{attempt:04}"
        archive.mkdir(parents=True, exist_ok=True)
        plan_path = archive / "ARCHIVE.json"
        if plan_path.exists():
            plan = read_json(plan_path)
            if plan["definition"] != object_hash(asdict(stage)):
                raise IntegrityError("partial-stage archive definition changed")
        else:
            existing = tuple(p for p in stage.reset_paths if (self.run / p).exists())
            plan = {"definition": object_hash(asdict(stage)), "paths": list(existing)}
            # This intent is durable before the first rename. Repeated recovery finishes it.
            write_json(plan_path, plan)
        for relative in plan["paths"]:
            source = self.run / relative
            destination = archive / "outputs" / relative
            member = self.run
            for part in Path(relative).parts:
                member /= part
                if member.is_symlink():
                    raise IntegrityError("linked partial-stage observation")
            if source.exists() and destination.exists():
                raise IntegrityError("ambiguous partial-stage archive collision")
            if source.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                source.rename(destination)
            elif not destination.exists():
                raise IntegrityError("partial-stage observation disappeared during recovery")
        plan["completed"] = True
        write_json(plan_path, plan)

    def recover_lost_attempt(self, previous):
        """A dead orphan is observable; its exit and actual end time are unavailable."""
        pid = previous.get("child_pid")
        if type(pid) is not int or pid <= 0:
            raise IntegrityError("previous stage launch identity is unknown; cannot duplicate work")
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise IntegrityError("previous stage process remains alive; cannot duplicate work")
        previous.update(
            status="interrupted",
            exit=None,
            exit_observed=False,
            ended_ns=None,
            elapsed_seconds=None,
            recovered_ns=time.time_ns(),
            recovery="lost-coordinator-and-absent-child",
            error="coordinator-recovery: prior child is absent; exit and end time unavailable",
        )
        for channel in ("stdout", "stderr"):
            previous[f"{channel}_sha256"] = file_hash(safe_member(self.run, previous[channel]))
        write_json(self.state_path, self.state)

    def execute(self, stage: Stage):
        stage.validate()
        previous = next(
            (r for r in reversed(self.state["stages"]) if r["name"] == stage.name), None
        )
        definition = object_hash(asdict(stage))
        if previous and previous["status"] == "passed":
            if previous["definition"] != definition or previous["outputs"] != output_inventory(
                self.run, stage.outputs
            ):
                raise IntegrityError("completed campaign stage inputs or outputs changed")
            return previous
        if previous:
            if stage.continuation == "fail":
                raise IntegrityError(
                    "failed stage retained; use a new campaign identity after repair"
                )
            if previous.get("definition") != definition:
                raise IntegrityError("interrupted stage definition changed")
            if previous["status"] not in {"running", "interrupted", "failed"}:
                raise IntegrityError("invalid previous stage status")
            if previous["status"] == "running":
                self.recover_lost_attempt(previous)
            if stage.continuation == "restart":
                self.archive_partial(stage, previous.get("attempt", 1))
        logs = self.run / "stage-logs"
        logs.mkdir(exist_ok=True)
        record: dict[str, Any] = {
            "name": stage.name,
            "definition": definition,
            "stage": asdict(stage),
            "working_directory": str(self.root.resolve()),
            "attempt": 1 if previous is None else previous.get("attempt", 1) + 1,
            "argv": list(stage.argv),
            "status": "running",
            "started_ns": time.time_ns(),
            "exit": None,
        }
        for channel in ("stdout", "stderr"):
            record[channel] = str(
                (logs / f"{stage.name}-{record['attempt']:04}.{channel}").relative_to(self.run)
            )
        if previous is not None:
            previous["continued_by_attempt"] = record["attempt"]
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
                record["child_pid"] = self.child.pid
                write_json(self.state_path, self.state)
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
            record["status"] = (
                "interrupted"
                if isinstance(failure, (CancellationError, KeyboardInterrupt))
                else "failed"
            )
            record["error"] = f"{type(failure).__name__}: {failure}"
            record["exit"] = self.child.returncode if self.child else None
            self.state["status"] = record["status"]
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
    reference = Path(config["reference"].get("python", ".reference-venv/bin/python")).absolute()
    comparison = None
    if not reference.is_file():
        reasons.append("separate pinned reference environment is absent")
    else:
        from asterion.training.reference_env import probe_reference

        try:
            comparison = probe_reference(reference, require_cuda=True)
        except ResourceError as failure:
            reasons.append(str(failure))
    return {
        "source": source,
        "environment": environment(),
        "authored_tests": authored,
        "reference": comparison,
        "status": "blocked" if reasons else "ready",
        "reasons": reasons,
    }


def research_stages(config_path: Path, storage: Path, run: Path, resume=False):
    config = read_json(config_path)
    names = [
        f"{kind}-{molecule}-{budget}-{seed}"
        for molecule in config["molecules"]
        for seed in config["seeds"]
        for kind in config["models"]
        for budget in config["budgets"]
    ]
    generations = "epoch-[0-9][0-9][0-9][0-9][0-9]"
    training_outputs = ("training-units.json",) + tuple(
        f"{name}/{member}"
        for name in names
        for member in (
            "config.json",
            "split.json",
            "SOURCE_INPUTS.json",
            "normalization.json",
            "history.json",
            "selection.json",
            "latest.json",
            "completed.json",
            "initial",
            "reveal",
            generations,
        )
    )
    evaluation_outputs = ("evaluation-units.json",) + tuple(
        f"{name}/{member}"
        for name in names
        for member in ("evaluation*", "shift*", "learning-paired.json", "leakage.json")
    )
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
            "restart",
            ("tests.xml",),
        ),
        Stage("data", command("data"), (), 3600, "resume"),
        Stage("pilot", command("pilot"), ("pilot",), 3600, "restart", ("pilot",)),
        Stage("train", command("train"), training_outputs, continuation="resume"),
        Stage("evaluate", command("evaluate"), evaluation_outputs, continuation="resume"),
        Stage(
            "guarded-shift",
            command("guarded-shift"),
            ("guarded-shift",),
            continuation="restart",
            reset_paths=("guarded-shift",),
        ),
        Stage(
            "simulate",
            command("simulate"),
            ("simulation-units.json", "simulation"),
            continuation="restart",
            reset_paths=("simulation", "simulation-units.json"),
        ),
        Stage(
            "acquire",
            command("acquire"),
            ("acquisition",),
            continuation="restart",
            reset_paths=("acquisition",),
        ),
        Stage(
            "ablate",
            command("ablate"),
            ("ablations",),
            continuation="restart",
            reset_paths=("ablations",),
        ),
        Stage(
            "reference",
            command("reference"),
            ("reference",),
            continuation="restart",
            reset_paths=("reference",),
        ),
        Stage(
            "secondary",
            command("secondary"),
            ("secondary",),
            continuation="restart",
            reset_paths=("secondary",),
        ),
        Stage("properties", command("properties"), ("properties.json",), 3600, "resume"),
        Stage(
            "derivatives",
            command("derivatives"),
            ("derivatives",),
            3600,
            "restart",
            ("derivatives",),
        ),
        Stage(
            "analytic",
            command("analytic"),
            ("analytic",),
            3600,
            "restart",
            ("analytic",),
        ),
        Stage(
            "stress",
            command("stress"),
            (
                "memory-stress.json",
                "memory-stress.progress.json",
                "faults",
                "simulation-recovery",
                "training-recovery",
            ),
            7200,
            "restart",
            (
                "memory-stress.json",
                "memory-stress.progress.json",
                "faults",
                "simulation-recovery",
                "training-recovery",
            ),
        ),
        Stage(
            "bench",
            command("bench"),
            ("benchmark.json", "benchmark-dynamics"),
            3600,
            "restart",
            ("benchmark.json", "benchmark-dynamics"),
        ),
        Stage(
            "demo",
            command("demo"),
            ("viewer.html", "failure-viewer.html", "demo-failure"),
            3600,
            "restart",
            ("viewer.html", "failure-viewer.html", "demo-failure"),
        ),
        Stage("docs", command("docs"), ("docs.json",), 3600, "restart", ("docs.json",)),
    ]


def execute_research(config, config_path, storage: Path, run: Path, evidence: Path, resume=False):
    from asterion.artifacts.bundle import build_full_candidate, promote_full
    from asterion.artifacts.controls import controls
    from asterion.artifacts.verify import verify

    if resume and evidence.exists():
        if read_json(evidence / "CAMPAIGN.json") != config:
            raise IntegrityError("cannot resume published evidence with different configuration")
        with CampaignOwner(run, resume=True):
            result = verify(evidence, "full", storage=storage)
            state = read_json(run / "STAGES.json")
            if state["identity"] != read_json(evidence / "STAGES.json")["identity"]:
                raise IntegrityError("published evidence belongs to another coordinator")
            state["status"] = "passed"
            write_json(run / "STAGES.json", state)
        return result
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
    with CampaignOwner(run, resume), cancellation_signals():
        from asterion.artifacts.inputs import CampaignInputs

        inputs = CampaignInputs(run, config, storage, preflight["source"]["digest"])
        runner = StageRunner(run, identity, resume=resume)
        for stage in research_stages(inputs.configuration, storage, run, resume):
            if source_inputs()["digest"] != preflight["source"]["digest"]:
                raise IntegrityError("source changed during the research campaign")
            inputs.check(require_data=stage.name not in {"tests", "data"})
            runner.execute(stage)
            if stage.name == "data":
                inputs.freeze_data()
            inputs.check(require_data=stage.name != "tests")
            if source_inputs()["digest"] != preflight["source"]["digest"]:
                raise IntegrityError("source changed while executing the research stage")
        candidate = evidence.with_name(evidence.name + ".candidate")
        if candidate.exists():
            if not resume:
                raise IntegrityError("candidate exists; explicit resume required")
            try:
                verify(candidate, "full", candidate=True, storage=storage)
            except (IntegrityError, OSError, ValueError, KeyError) as error:
                retained = candidate.with_name(f"{candidate.name}.interrupted-{uuid.uuid4().hex}")
                candidate.rename(retained)
                write_json(
                    run / f"candidate-recovery-{retained.name}.json",
                    {"retained": str(retained), "error": f"{type(error).__name__}: {error}"},
                )
        if not candidate.exists():
            build_full_candidate(config, storage, run, candidate)
        verified = verify(candidate, "full", candidate=True, storage=storage)
        control_path = run / "full-controls.json"
        if control_path.exists():
            try:
                record = read_json(control_path)
                complete = (
                    record["profile"] == "full"
                    and record["phase"] == "candidate"
                    and record["candidate_manifest_sha256"] == verified["manifest_sha256"]
                    and record["positive_before"]["exit"] == record["positive_after"]["exit"] == 0
                    and record["source_bundle_unchanged"] is True
                    and len(record["controls"]) >= 12
                    and all(row["exit"] != 0 and row["stderr"] for row in record["controls"])
                    and sum(row["semantic_rehashed"] for row in record["controls"]) >= 4
                )
            except (IntegrityError, OSError, ValueError, KeyError, TypeError):
                complete = False
            if not complete:
                control_path.rename(run / f"full-controls.interrupted-{uuid.uuid4().hex}.json")
        if not control_path.exists():
            controls(candidate, control_path, "full", candidate=True, storage=storage)
        promote_full(candidate, evidence, control_path, verified)
        result = verify(evidence, "full", storage=storage)
        runner.state["status"] = "passed"
        write_json(runner.state_path, runner.state)
        return result
