"""Validate process receipts, continuation and exact retained output ownership."""

from pathlib import Path

import numpy as np

from asterion.artifacts.io import file_hash, object_hash, read_json, read_json_lines, safe_member
from asterion.contracts.errors import IntegrityError


def require(condition, message):
    if not condition:
        raise IntegrityError(message)


def verify_stage_receipts(root: Path, expected_names, scientific=False):
    state = read_json(root / "STAGES.json")
    mapping = read_json(root / "STAGE_OUTPUTS.json")
    records = state["stages"]
    require(
        records and {row["name"] for row in records} == set(expected_names),
        "campaign stage population is missing or unexplained",
    )
    finals = {}
    attempts: dict[str, int] = {}
    definitions: dict[str, str] = {}
    required_outputs = set()
    for row in records:
        name = row["name"]
        count = attempts.get(name, 0) + 1
        attempts[name] = count
        require(row["attempt"] == count, "campaign attempt sequence is missing or repeated")
        specification = row["stage"]
        require(name not in finals, "completed campaign stage was executed again")
        if name in definitions:
            require(
                definitions[name] == row["definition"],
                "campaign continuation changed the frozen stage definition",
            )
        definitions[name] = row["definition"]
        if scientific:
            argv = row["argv"]
            if name == "tests":
                require(
                    argv[1:3] == ["-m", "pytest"]
                    and len(argv) == 4
                    and argv[3].startswith("--junitxml="),
                    "test receipt does not invoke the declared suite",
                )
            else:
                require(
                    argv[1:4] == ["-m", "asterion.cli", name],
                    "scientific receipt does not invoke its original stage",
                )
                require(
                    all(
                        argv.count(option) == 1 for option in ("--config", "--storage", "--run-id")
                    ),
                    "scientific receipt has missing or repeated input flags",
                )
        require(
            specification["name"] == name
            and specification["argv"] == row["argv"]
            and object_hash(specification) == row["definition"],
            "campaign stage command or definition changed",
        )
        if row.get("exit_observed") is False:
            require(
                row["status"] == "interrupted"
                and row["exit"] is None
                and row["ended_ns"] is None
                and row["elapsed_seconds"] is None
                and row["recovery"] == "lost-coordinator-and-absent-child"
                and type(row["recovered_ns"]) is int
                and row["recovered_ns"] >= row["started_ns"]
                and type(row["child_pid"]) is int
                and row["child_pid"] > 0
                and row["error"]
                == "coordinator-recovery: prior child is absent; exit and end time unavailable",
                "recovered campaign attempt invents an unobserved exit or duration",
            )
        else:
            require(
                row["ended_ns"] >= row["started_ns"]
                and np.isfinite(row["elapsed_seconds"])
                and row["elapsed_seconds"] > 0,
                "invalid retained campaign clock or elapsed work",
            )
        for channel in ("stdout", "stderr"):
            path = safe_member(root, row[channel])
            require(
                file_hash(path) == row[f"{channel}_sha256"], "campaign process log bytes changed"
            )
        if row["status"] == "passed":
            require(
                row["exit"] == 0 and row["elapsed_seconds"] <= specification["timeout_seconds"] + 1,
                "campaign successful exit or time budget is false",
            )
            require(
                specification["outputs"] or name == "data",
                "completed stage has no declared observations",
            )
            for relative, digest in row["outputs"].items():
                require(
                    relative in mapping and mapping[relative]["sha256"] == digest,
                    "stage observation retention is missing or changes its hash",
                )
                require(
                    file_hash(safe_member(root, mapping[relative]["path"])) == digest,
                    "retained stage observation bytes changed",
                )
                required_outputs.add(relative)
            if specification["outputs"]:
                require(row["outputs"], "completed stage retained no useful observations")
            finals[name] = row
        else:
            require(
                row["status"] in {"failed", "interrupted"}
                and row["exit"] != 0
                and row.get("error"),
                "unexplained campaign failure or pending stage",
            )
            require(
                row.get("continued_by_attempt") == count + 1,
                "failed campaign attempt was hidden without continuation",
            )
    require(set(finals) == set(expected_names), "campaign has a required unfinished stage")
    require(list(finals) == list(expected_names), "campaign dependency execution order changed")
    require(
        set(mapping) == required_outputs, "stage observation map has missing or unexplained entries"
    )
    for name, row in finals.items():
        require(
            row["attempt"] == attempts[name],
            "last campaign attempt is not the retained successful one",
        )
    return finals


def verify_command_receipts(root, expected_names, finals=None):
    records = read_json_lines(root / "COMMANDS.jsonl")
    latest = {}
    for row in records:
        name = row["stage"]
        require(name in expected_names, "unexplained evidence command")
        require(
            row["ended_ns"] >= row["started_ns"]
            and row["execution"] == "in-process-original-stage",
            "invalid command execution receipt",
        )
        if finals is not None:
            argv = finals[name]["argv"]
            require(
                argv[:1] and argv[1:3] == ["-m", "asterion.cli"] and argv[3:] == row["argv"][1:],
                "inner command differs from the managed stage definition",
            )
        if row["exit"] != 0:
            require(
                finals is not None and row.get("error") and finals[name]["attempt"] > 1,
                "failed command has no retained successful continuation",
            )
        latest[name] = row
    require(
        set(latest) == set(expected_names) and all(row["exit"] == 0 for row in latest.values()),
        "required command is absent or its last invocation failed",
    )
    return latest
