import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from asterion.artifacts.controls import refresh_manifest
from asterion.artifacts.io import canonical, file_hash, read_json, safe_member, write_json
from asterion.artifacts.source import environment, source_inputs
from asterion.contracts.errors import IntegrityError


def copy_observations(source: Path, target: Path):
    if source.is_symlink() or any(p.is_symlink() for p in source.rglob("*")):
        raise IntegrityError("symlink in scientific observations")
    shutil.copytree(source, target)


def build_diagnostic(config, storage: Path, runs: Path, evidence: Path):
    source = source_inputs()
    if source["dirty"]:
        raise IntegrityError("commit executable inputs before generating source-bound evidence")
    if evidence.exists():
        raise IntegrityError("evidence run ID already exists")
    evidence.mkdir(parents=True)
    training = read_json(runs / "training-units.json")
    copied = []
    for entry in training:
        src = safe_member(runs, entry["path"])
        dst = evidence / "training" / entry["path"]
        copy_observations(src, dst)
        copied.append(
            {
                k: v
                for k, v in {**entry, "path": str(dst.relative_to(evidence))}.items()
                if k != "result"
            }
        )
    trajectories = read_json(runs / "simulation-units.json")
    retained = []
    for entry in trajectories:
        src = safe_member(runs, entry["path"])
        dst = evidence / "trajectories" / Path(entry["path"]).name
        copy_observations(src, dst)
        retained.append({**entry, "path": str(dst.relative_to(evidence))})
    dataset = read_json(storage / "rmd17" / "DATASETS.json")
    for record in dataset["files"]:
        record["path"] = str(Path(record["path"]).resolve())
    write_json(evidence / "SOURCE_INPUTS.json", source)
    write_json(evidence / "DATASETS.json", dataset)
    write_json(evidence / "ENVIRONMENT.json", environment())
    write_json(evidence / "CAMPAIGN.json", config)
    units: dict = {"training": copied, "trajectories": retained}
    if "dynamics_study" in config:
        copy_observations(runs / "simulation/studies", evidence / "trajectories/studies")
        units["dynamics_study"] = "trajectories/studies/study.json"
    write_json(evidence / "UNITS.json", units)
    steps = sum(
        read_json(evidence / entry["path"] / "trajectory.json")["completed_steps"]
        for entry in retained
    )
    report = {
        "profile": "diagnostic",
        "status": "passed",
        "source": source["digest"],
        "completed_training_runs": len(copied),
        "completed_learned_steps": steps,
        "full_acceptance": "unverified",
        "counts": {},
        "reference": {"status": "unverified"},
        "secondary": {"status": "unverified"},
    }
    write_json(evidence / "ACCEPTANCE.json", report)
    (evidence / "COMMANDS.jsonl").write_bytes((runs / "COMMANDS.jsonl").read_bytes())
    refresh_manifest(evidence)
    return report


def authored_test_count():
    import ast

    from asterion.artifacts.source import project_root

    return sum(
        isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        for path in (project_root() / "tests").rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text()))
    )


def retain_stage_receipts(runs: Path, evidence: Path):
    state = read_json(runs / "STAGES.json")
    units = read_json(runs / "training-units.json")
    primary = {entry["path"] for entry in units}
    mapping = {}
    prefixes = {
        "guarded-shift",
        "pilot",
        "acquisition",
        "ablations",
        "reference",
        "secondary",
        "derivatives",
        "analytic",
        "faults",
        "simulation-recovery",
        "training-recovery",
        "benchmark-dynamics",
    }
    for record in state["stages"]:
        if record["status"] != "passed":
            continue
        for relative, expected in record["outputs"].items():
            source = safe_member(runs, relative)
            if file_hash(source) != expected:
                raise IntegrityError("completed stage observation changed before bundle retention")
            path = Path(relative)
            first = path.parts[0]
            if first in primary:
                target = evidence / "training" / path
            elif first == "simulation":
                target = evidence / "trajectories" / Path(*path.parts[1:])
            elif relative == "analytic/analytic.json":
                target = evidence / "gates/analytic/original-analytic.json"
            elif first in prefixes:
                target = evidence / "gates" / path
            elif relative in {
                "properties.json",
                "benchmark.json",
                "memory-stress.json",
                "tests.xml",
            }:
                target = evidence / "gates" / path
            else:
                target = evidence / "stage-observations" / path
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            if file_hash(target) != expected:
                raise IntegrityError("bundle stage relocation changed observation bytes")
            mapping[relative] = {"path": str(target.relative_to(evidence)), "sha256": expected}
    copy_observations(runs / "stage-logs", evidence / "stage-logs")
    write_json(evidence / "STAGE_OUTPUTS.json", mapping)
    shutil.copyfile(runs / "STAGES.json", evidence / "STAGES.json")
    return mapping


def build_full_candidate(config, storage: Path, runs: Path, evidence: Path):
    """Copy all mandatory raw families without claiming they passed verification."""
    report = build_diagnostic(config, storage, runs, evidence)
    for name in ("FROZEN_CONFIGURATION.json", "INPUT_IDENTITIES.json"):
        shutil.copyfile(runs / name, evidence / name)
    families = {
        "resource_pilot": "pilot",
        "properties": "properties.json",
        "derivatives": "derivatives",
        "analytic": "analytic",
        "simulation_recovery": "simulation-recovery",
        "training_recovery": "training-recovery",
        "faults": "faults",
        "memory": "memory-stress.json",
        "benchmark": "benchmark.json",
        "ablations": "ablations",
        "acquisitions": "acquisition",
        "reference": "reference",
        "secondary": "secondary",
        "guarded_shift": "guarded-shift",
        "benchmark_dynamics": "benchmark-dynamics",
    }
    endpoints = {
        "resource_pilot": "pilot.json",
        "derivatives": "derivatives.json",
        "analytic": "analytic.json",
        "simulation_recovery": "recovery.json",
        "training_recovery": "recovery.json",
        "faults": "faults.json",
        "ablations": "ablation-units.json",
        "acquisitions": "acquisition-units.json",
        "reference": "reference-units.json",
        "secondary": "secondary.json",
        "guarded_shift": "guarded-units.json",
        "benchmark_dynamics": "",
    }
    index = {"junit": "gates/tests.xml", "controls": "gates/full-controls.json"}
    index.update(
        viewer="stage-observations/viewer.html",
        failure_viewer="stage-observations/failure-viewer.html",
        viewer_failure="stage-observations/demo-failure",
    )
    index["dynamics_study"] = read_json(evidence / "UNITS.json")["dynamics_study"]
    destination = evidence / "gates"
    destination.mkdir()
    shutil.copyfile(runs / "tests.xml", destination / "tests.xml")
    for family, relative in families.items():
        source = safe_member(runs, relative)
        target = destination / Path(relative).name
        if source.is_dir():
            copy_observations(source, target)
            index[family] = str((target / endpoints[family]).relative_to(evidence))
        else:
            if source.is_symlink():
                raise IntegrityError("symlink in gate observation")
            shutil.copyfile(source, target)
            index[family] = str(target.relative_to(evidence))
    path = evidence / index["analytic"]
    original = read_json(path)
    write_json(path.with_name("original-analytic.json"), original)
    relocation = {}
    for fixture in original["outcomes"]:
        for entry in fixture["runs"]:
            analytic_relative = (
                Path(entry["path"]).resolve().relative_to((runs / "analytic").resolve())
            )
            new = str((destination / "analytic" / analytic_relative).relative_to(evidence))
            relocation[entry["path"]] = new
            entry["path"] = new
    write_json(path, original)
    write_json(destination / "PATH_RELOCATION.json", relocation)
    retain_stage_receipts(runs, evidence)
    write_json(evidence / "GATE_OBSERVATIONS.json", index)
    properties = read_json(evidence / index["properties"])
    derivatives = read_json(evidence / index["derivatives"])
    analytic = read_json(evidence / index["analytic"])
    faults = read_json(evidence / index["faults"])
    memory = read_json(evidence / index["memory"])
    recovery = sum(
        len(read_json(evidence / index[name])["outcomes"])
        for name in ("training_recovery", "simulation_recovery")
    )
    report.update(
        profile="full",
        status="candidate",
        counts={
            "semantic_tests": authored_test_count(),
            "property_cases": len(properties["cases"]),
            "symmetry_trials": len(derivatives["symmetry"]),
            "force_checks": len(derivatives["force"]),
            "hessian_checks": len(derivatives["hessian"]),
            "analytic_dynamics": len(analytic["outcomes"]),
            "learned_steps": report["completed_learned_steps"],
            "recovery_cases": recovery,
            "fault_cases": len(faults["outcomes"]),
            "stress_seconds": memory["elapsed_seconds"],
            "negative_controls": 0,
        },
        gates={f"MA{i:02}": {"status": "candidate"} for i in range(1, 13)},
        reference={"status": "candidate"},
        secondary={"status": "candidate"},
    )
    report.pop("full_acceptance", None)
    write_json(evidence / "ACCEPTANCE.json", report)
    refresh_manifest(evidence)
    return report


def promote_full(candidate: Path, final: Path, controls_path: Path, verified):
    if final.exists() or verified.get("phase") != "candidate":
        raise IntegrityError("full promotion requires a verified candidate and new output path")
    control = read_json(controls_path)
    if (
        control["profile"] != "full"
        or control.get("phase") != "candidate"
        or control["positive_before"]["exit"] != 0
        or control["positive_after"]["exit"] != 0
        or not control["source_bundle_unchanged"]
    ):
        raise IntegrityError("candidate controls did not complete positively")
    if (
        len(control["controls"]) < 12
        or any(row["exit"] == 0 or not row["stderr"] for row in control["controls"])
        or sum(row["semantic_rehashed"] for row in control["controls"]) < 4
    ):
        raise IntegrityError("full promotion has missing or accepted negative controls")
    report = read_json(candidate / "ACCEPTANCE.json")
    source = read_json(candidate / "SOURCE_INPUTS.json")
    if report["status"] != "candidate" or verified["source"] != source["digest"]:
        raise IntegrityError("promotion verification belongs to another candidate")
    if control.get("candidate_manifest_sha256") != file_hash(candidate / "MANIFEST.json"):
        raise IntegrityError("controls belong to another candidate manifest")
    if verified.get("manifest_sha256") != file_hash(candidate / "MANIFEST.json"):
        raise IntegrityError("candidate changed after independent verification")
    from asterion.artifacts.campaign import CampaignOwner
    from asterion.artifacts.verify import check_manifest

    candidate_hash = verified["manifest_sha256"]
    control_hash = file_hash(controls_path)
    publication = final.with_name(f".{final.name}.publication")
    with CampaignOwner(publication, resume=True):
        if final.exists():
            raise IntegrityError("publication output appeared while acquiring ownership")
        holder = Path(tempfile.mkdtemp(prefix="attempt-", dir=publication))
        staged = holder / "bundle"
        intent = {
            "candidate_manifest_sha256": candidate_hash,
            "controls_sha256": control_hash,
            "source": source["digest"],
            "target": str(final.resolve()),
            "status": "copying",
        }
        write_json(holder / "PUBLICATION.json", intent)
        try:
            copy_observations(candidate, staged)
            if file_hash(staged / "MANIFEST.json") != candidate_hash:
                raise IntegrityError("candidate manifest changed during publication copy")
            check_manifest(staged)
            index = read_json(staged / "GATE_OBSERVATIONS.json")
            retained_controls = safe_member(staged, index["controls"])
            retained_controls.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(controls_path, retained_controls)
            if (
                file_hash(retained_controls) != control_hash
                or read_json(retained_controls) != control
            ):
                raise IntegrityError("controls changed during publication copy")
            report.update(status="passed", candidate_manifest_sha256=candidate_hash)
            report["counts"]["negative_controls"] = len(control["controls"])
            for key in report["gates"]:
                report["gates"][key]["status"] = "passed"
            report["reference"]["status"] = report["secondary"]["status"] = "passed"
            write_json(staged / "ACCEPTANCE.json", report)
            refresh_manifest(staged)
            check_manifest(staged)
            for path in staged.rglob("*"):
                if path.is_file():
                    with path.open("rb") as stream:
                        os.fsync(stream.fileno())
            for directory in sorted(
                (p for p in staged.rglob("*") if p.is_dir()),
                key=lambda p: len(p.parts),
                reverse=True,
            ) + [staged]:
                descriptor = os.open(directory, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            if final.exists():
                raise IntegrityError("publication output appeared during staging")
            os.rename(staged, final)
            descriptor = os.open(final.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            write_json(holder / "PUBLICATION.json", {**intent, "status": "published"})
        except BaseException as error:
            write_json(
                holder / "PUBLICATION.json",
                {**intent, "status": "interrupted", "error": f"{type(error).__name__}: {error}"},
            )
            raise
    return report


class CommandJournal:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def execute(self, label, action):
        started = time.time_ns()
        result = None
        status = 1
        error = None
        try:
            result = action()
            status = 0
            return result
        except Exception as failure:
            error = f"{type(failure).__name__}: {failure}"
            raise
        finally:
            record = {
                "stage": label,
                "argv": sys.argv,
                "started_ns": started,
                "ended_ns": time.time_ns(),
                "exit": status,
                "error": error,
                "execution": "in-process-original-stage",
            }
            with self.path.open("ab") as stream:
                stream.write(canonical(record) + b"\n")
