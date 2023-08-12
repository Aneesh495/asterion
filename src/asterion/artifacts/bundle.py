import shutil
import sys
import time
from pathlib import Path

from asterion.artifacts.controls import refresh_manifest
from asterion.artifacts.io import canonical, read_json, write_json
from asterion.artifacts.source import environment, source_inputs
from asterion.contracts.errors import IntegrityError


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
        src = runs / entry["path"]
        dst = evidence / "training" / entry["path"]
        shutil.copytree(src, dst)
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
        src = runs / entry["path"]
        dst = evidence / "trajectories" / Path(entry["path"]).name
        shutil.copytree(src, dst)
        retained.append({**entry, "path": str(dst.relative_to(evidence))})
    dataset = read_json(storage / "rmd17" / "DATASETS.json")
    for record in dataset["files"]:
        record["path"] = str(Path(record["path"]).resolve())
    write_json(evidence / "SOURCE_INPUTS.json", source)
    write_json(evidence / "DATASETS.json", dataset)
    write_json(evidence / "ENVIRONMENT.json", environment())
    write_json(evidence / "CAMPAIGN.json", config)
    write_json(evidence / "UNITS.json", {"training": copied, "trajectories": retained})
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
