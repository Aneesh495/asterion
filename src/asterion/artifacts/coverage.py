"""Bind each required capability to frozen source, executed tests and retained observations."""

import ast
import xml.etree.ElementTree as ET
from pathlib import Path

from asterion.artifacts.io import file_hash, object_hash, read_json, safe_member
from asterion.artifacts.source import project_root
from asterion.contracts.errors import IntegrityError


def require(condition, message):
    if not condition:
        raise IntegrityError(message)


def authored_tests(repository: Path):
    """Discover pytest functions and class methods without counting nested helper functions."""
    result = {}
    for path in sorted((repository / "tests").rglob("test_*.py")):
        module = str(path.relative_to(repository)).removesuffix(".py").replace("/", ".")
        names = []
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
                "test_"
            ):
                names.append((module, node.name))
            elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                names.extend(
                    (f"{module}.{node.name}", method.name)
                    for method in node.body
                    if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and method.name.startswith("test_")
                )
        require(
            names and len(names) == len(set(names)),
            f"empty or repeated authored tests: {path.name}",
        )
        result[str(path.relative_to(repository))] = names
    require(result, "authored test inventory is empty")
    return result


def executed_tests(path: Path, definitions):
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as error:
        raise IntegrityError("invalid retained test invocation") from error
    require(
        not any(root.findall(f".//{tag}") for tag in ("failure", "error", "skipped")),
        "required capability test failed or skipped",
    )
    known = {key for names in definitions.values() for key in names}
    observed: dict[tuple[str, str], list[str]] = {}
    identities = set()
    cases = root.findall(".//testcase")
    for case in cases:
        class_name, name = case.get("classname", ""), case.get("name", "")
        key = (class_name, name.split("[", 1)[0])
        require(key in known, "test invocation contains an unexplained authored identity")
        identity = (class_name, name)
        require(identity not in identities, "test invocation repeats an executed identity")
        identities.add(identity)
        observed.setdefault(key, []).append(name)
    require(set(observed) == known, "test invocation omits authored test functions")
    suites = root.findall(".//testsuite") if root.tag != "testsuite" else [root]
    require(
        suites and sum(int(suite.get("tests", "-1")) for suite in suites) == len(cases),
        "test suite counters disagree with retained cases",
    )
    return observed


def artifact_inventory(root: Path, relative: str):
    path = safe_member(root, relative)
    require(path.exists(), f"capability observation is missing: {relative}")
    paths = [path] if path.is_file() else sorted(path.rglob("*"))
    require(not any(p.is_symlink() for p in paths), "linked capability observation")
    hashes = {str(p.relative_to(root)): file_hash(p) for p in paths if p.is_file()}
    require(hashes, "capability observation is empty")
    return hashes


def feature_coverage(root: Path, source, finals, repository: Path | None = None):
    repository = repository or project_root()
    registry_path = repository / "configs/features.json"
    require(
        source["hashes"].get("configs/features.json") == file_hash(registry_path),
        "capability register differs from frozen source",
    )
    registry = read_json(registry_path)
    require(registry.get("format") == "asterion-features-v1", "unknown capability register")
    rows = registry["features"]
    require(
        len(rows) == 32 and [r["id"] for r in rows] == [f"M{i:02}" for i in range(1, 33)],
        "required capability population is missing or repeated",
    )
    definitions = authored_tests(repository)
    index = read_json(root / "GATE_OBSERVATIONS.json")
    units = read_json(root / "UNITS.json")
    families = {
        **{k: [v] for k, v in index.items() if k != "controls"},
        "datasets": ["DATASETS.json"],
        "training": [row["path"] for row in units["training"]],
        "trajectories": [row["path"] for row in units["trajectories"]],
        "receipts": ["STAGES.json", "STAGE_OUTPUTS.json", "COMMANDS.jsonl"],
    }
    for family in (
        "resource_pilot",
        "derivatives",
        "analytic",
        "simulation_recovery",
        "training_recovery",
        "faults",
        "ablations",
        "acquisitions",
        "reference",
        "secondary",
        "guarded_shift",
        "dynamics_study",
    ):
        if family in index:
            families[family] = [str(Path(index[family]).parent)]
    observed = executed_tests(safe_member(root, index["junit"]), definitions)
    used_tests, used_source, used_gates = set(), set(), set()
    result = []
    inventories = {}
    for row in rows:
        require(
            set(row) == {"id", "mechanism", "source", "tests", "stages", "artifacts", "gates"}
            and row["mechanism"],
            "invalid capability register fields",
        )
        for field in ("source", "tests", "stages", "artifacts", "gates"):
            require(
                isinstance(row[field], list)
                and row[field]
                and len(row[field]) == len(set(row[field])),
                f"empty or repeated capability {field}",
            )
        require(set(row["stages"]) <= finals.keys(), "capability requires an unfinished command")
        require(
            all(
                finals[name].get("status") == "passed" and finals[name].get("exit") == 0
                for name in row["stages"]
            ),
            "capability command failed or is unfinished",
        )
        require(set(row["gates"]) <= {f"MA{i:02}" for i in range(1, 13)}, "unknown capability gate")
        pointers = {}
        for path in (*row["source"], *row["tests"]):
            value = safe_member(repository, path)
            require(
                value.is_file() and source["hashes"].get(path) == file_hash(value),
                f"capability pointer is outside frozen source: {path}",
            )
            pointers[path] = source["hashes"][path]
        require(
            set(row["tests"]) <= definitions.keys(), "capability points to unknown authored tests"
        )
        test_rows = [
            {"classname": key[0], "function": key[1], "executed_cases": sorted(observed[key])}
            for path in row["tests"]
            for key in definitions[path]
        ]
        observations = {}
        for family in row["artifacts"]:
            require(
                family in families and families[family],
                "unknown or empty capability artifact family",
            )
            if family not in inventories:
                inventories[family] = {
                    name: digest
                    for path in families[family]
                    for name, digest in artifact_inventory(root, path).items()
                }
            observations[family] = {
                "files": len(inventories[family]),
                "inventory_sha256": object_hash(inventories[family]),
            }
        result.append(
            {
                **row,
                "status": "source-test-artifact-bound",
                "input_sha256": pointers,
                "test_observations": test_rows,
                "observations": observations,
                "commands": {
                    name: {
                        "argv": finals[name]["argv"],
                        "attempt": finals[name]["attempt"],
                        "exit": finals[name]["exit"],
                        "definition": finals[name]["definition"],
                    }
                    for name in row["stages"]
                },
            }
        )
        used_source.update(row["source"])
        used_tests.update(row["tests"])
        used_gates.update(row["gates"])
    require(used_tests == set(definitions), "capability register leaves authored tests unmapped")
    production = {
        p for p in source["hashes"] if p.startswith("src/") and not p.endswith("__init__.py")
    }
    require(production <= used_source, "capability register leaves production files unmapped")
    require(
        used_gates == {f"MA{i:02}" for i in range(1, 13)},
        "capability register leaves gates unmapped",
    )
    return {
        "format": "asterion-feature-coverage-v1",
        "source": source["digest"],
        "status": "source-test-artifact-bound",
        "scientific_acceptance": "separately-verified",
        "registry_sha256": file_hash(registry_path),
        "junit_sha256": file_hash(safe_member(root, index["junit"])),
        "authored_functions": sum(map(len, definitions.values())),
        "executed_cases": sum(map(len, observed.values())),
        "features": result,
    }


def verify_feature_coverage(root: Path, source, finals):
    actual = feature_coverage(root, source, finals)
    require(
        read_json(root / "FEATURE_COVERAGE.json") == actual,
        "capability coverage disagrees with retained execution",
    )
    return actual
