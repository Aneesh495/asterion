import copy
import xml.etree.ElementTree as ET

import pytest

from asterion.artifacts.coverage import authored_tests, executed_tests, feature_coverage
from asterion.artifacts.io import file_hash, object_hash, read_json, write_json
from asterion.artifacts.source import project_root, source_inputs
from asterion.contracts.errors import IntegrityError


@pytest.fixture
def coverage_fixture(tmp_path):
    # Only a coverage protocol fixture; none of these bytes certify scientific acceptance.
    repo, evidence = tmp_path / "repo", tmp_path / "evidence"
    (repo / "tests").mkdir(parents=True)
    (repo / "src").mkdir()
    (repo / "tests/test_small.py").write_text(
        "def test_actual():\n    pass\nclass TestBoundary:\n    def test_method(self):\n        pass\n"
    )
    (repo / "src/mechanism.py").write_text("def compute(x):\n    return x * 2\n")
    features = [
        {
            "id": f"M{i:02}",
            "mechanism": "fixture protocol",
            "source": ["src/mechanism.py"],
            "tests": ["tests/test_small.py"],
            "stages": ["tests"],
            "artifacts": ["raw"],
            "gates": [f"MA{(i - 1) % 12 + 1:02}"],
        }
        for i in range(1, 33)
    ]
    write_json(
        repo / "configs/features.json", {"format": "asterion-features-v1", "features": features}
    )
    write_json(evidence / "GATE_OBSERVATIONS.json", {"junit": "tests.xml", "raw": "raw.json"})
    write_json(evidence / "UNITS.json", {"training": [], "trajectories": []})
    write_json(evidence / "raw.json", {"observation": 7})
    suite = ET.Element("testsuite", tests="3")
    for class_name, name in [
        ("tests.test_small", "test_actual[one]"),
        ("tests.test_small", "test_actual[two]"),
        ("tests.test_small.TestBoundary", "test_method"),
    ]:
        ET.SubElement(suite, "testcase", classname=class_name, name=name)
    ET.ElementTree(suite).write(evidence / "tests.xml")
    paths = ["configs/features.json", "src/mechanism.py", "tests/test_small.py"]
    hashes = {p: file_hash(repo / p) for p in paths}
    source = {"hashes": hashes, "digest": object_hash(hashes)}
    finals = {
        "tests": {
            "argv": ["python", "-m", "pytest", "--junitxml=tests.xml"],
            "attempt": 1,
            "status": "passed",
            "exit": 0,
            "definition": "protocol-fixture",
        }
    }
    return repo, evidence, source, finals


def produce(fixture):
    repo, root, source, finals = fixture
    return feature_coverage(root, source, finals, repo)


def rewrite_registry(fixture, change):
    repo, _, source, _ = fixture
    registry = read_json(repo / "configs/features.json")
    change(registry["features"])
    write_json(repo / "configs/features.json", registry)
    source["hashes"]["configs/features.json"] = file_hash(repo / "configs/features.json")
    source["digest"] = object_hash(source["hashes"])


def test_coverage_binds_class_methods_and_parameterized_observations(coverage_fixture):
    result = produce(coverage_fixture)
    assert result["authored_functions"] == 2 and result["executed_cases"] == 3
    assert len(result["features"]) == 32
    assert result["scientific_acceptance"] == "separately-verified"
    assert result["features"][0]["test_observations"][1]["classname"].endswith("TestBoundary")


def test_coverage_rejects_missing_required_capability(coverage_fixture):
    rewrite_registry(coverage_fixture, lambda rows: rows.pop())
    with pytest.raises(IntegrityError, match="population"):
        produce(coverage_fixture)


def test_coverage_rejects_repeated_capability(coverage_fixture):
    rewrite_registry(coverage_fixture, lambda rows: rows.__setitem__(1, copy.deepcopy(rows[0])))
    with pytest.raises(IntegrityError, match="population"):
        produce(coverage_fixture)


def test_coverage_rejects_edited_unfrozen_register(coverage_fixture):
    repo, *_ = coverage_fixture
    (repo / "configs/features.json").write_text("{}")
    with pytest.raises(IntegrityError, match="frozen"):
        produce(coverage_fixture)


def test_coverage_rejects_missing_authored_test(coverage_fixture):
    _, root, *_ = coverage_fixture
    tree = ET.parse(root / "tests.xml")
    tree.getroot().remove(tree.getroot()[-1])
    tree.getroot().set("tests", "2")
    tree.write(root / "tests.xml")
    with pytest.raises(IntegrityError, match="omits"):
        produce(coverage_fixture)


@pytest.mark.parametrize("tag", ["failure", "error", "skipped"])
def test_coverage_rejects_failed_or_skipped_observations(coverage_fixture, tag):
    _, root, *_ = coverage_fixture
    tree = ET.parse(root / "tests.xml")
    ET.SubElement(tree.getroot()[0], tag)
    tree.write(root / "tests.xml")
    with pytest.raises(IntegrityError, match="failed or skipped"):
        produce(coverage_fixture)


def test_coverage_rejects_repeated_executed_case(coverage_fixture):
    _, root, *_ = coverage_fixture
    tree = ET.parse(root / "tests.xml")
    tree.getroot().append(copy.deepcopy(tree.getroot()[0]))
    tree.getroot().set("tests", "4")
    tree.write(root / "tests.xml")
    with pytest.raises(IntegrityError, match="repeats"):
        produce(coverage_fixture)


def test_coverage_rejects_test_counter_inflation(coverage_fixture):
    _, root, *_ = coverage_fixture
    tree = ET.parse(root / "tests.xml")
    tree.getroot().set("tests", "400")
    tree.write(root / "tests.xml")
    with pytest.raises(IntegrityError, match="counters"):
        produce(coverage_fixture)


def test_coverage_rejects_unknown_executed_identity(coverage_fixture):
    _, root, *_ = coverage_fixture
    tree = ET.parse(root / "tests.xml")
    tree.getroot()[0].set("name", "test_fabricated")
    tree.write(root / "tests.xml")
    with pytest.raises(IntegrityError, match="unexplained"):
        produce(coverage_fixture)


def test_coverage_rejects_missing_raw_family(coverage_fixture):
    _, root, *_ = coverage_fixture
    (root / "raw.json").unlink()
    with pytest.raises(IntegrityError, match="missing"):
        produce(coverage_fixture)


def test_coverage_observation_digest_changes_with_actual_bytes(coverage_fixture):
    before = produce(coverage_fixture)
    _, root, *_ = coverage_fixture
    write_json(root / "raw.json", {"observation": 8})
    after = produce(coverage_fixture)
    assert before["features"][0]["observations"] != after["features"][0]["observations"]


def test_coverage_rejects_unfinished_command(coverage_fixture):
    coverage_fixture[-1].clear()
    with pytest.raises(IntegrityError, match="unfinished"):
        produce(coverage_fixture)


def test_coverage_rejects_source_pointer_outside_frozen_inputs(coverage_fixture):
    rewrite_registry(coverage_fixture, lambda rows: rows[0].__setitem__("source", ["../outside"]))
    with pytest.raises(IntegrityError, match="escapes"):
        produce(coverage_fixture)


def test_coverage_rejects_unmapped_production_file(coverage_fixture):
    repo, _, source, _ = coverage_fixture
    (repo / "src/forgotten.py").write_text("def unseen():\n    return 1\n")
    source["hashes"]["src/forgotten.py"] = file_hash(repo / "src/forgotten.py")
    with pytest.raises(IntegrityError, match="production files unmapped"):
        produce(coverage_fixture)


def test_coverage_rejects_unknown_gate(coverage_fixture):
    rewrite_registry(coverage_fixture, lambda rows: rows[0].__setitem__("gates", ["MA99"]))
    with pytest.raises(IntegrityError, match="unknown capability gate"):
        produce(coverage_fixture)


def test_real_register_covers_actual_tests_and_production_inputs():
    root = project_root()
    rows = read_json(root / "configs/features.json")["features"]
    tests = {p for row in rows for p in row["tests"]}
    production = {p for row in rows for p in row["source"]}
    assert tests == set(authored_tests(root))
    source = source_inputs()
    assert {
        p for p in source["hashes"] if p.startswith("src/") and not p.endswith("__init__.py")
    } <= production
    assert all((root / p).is_file() for row in rows for p in (*row["source"], *row["tests"]))


def test_authored_inventory_does_not_count_nested_helpers(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_nested.py").write_text(
        "def test_outer():\n    def test_helper():\n        pass\n"
    )
    assert authored_tests(tmp_path) == {
        "tests/test_nested.py": [("tests.test_nested", "test_outer")]
    }


def test_suite_function_binding_rejects_new_unexecuted_function(coverage_fixture):
    repo, evidence, *_ = coverage_fixture
    path = repo / "tests/test_small.py"
    path.write_text(path.read_text() + "\ndef test_added():\n    pass\n")
    with pytest.raises(IntegrityError, match="omits"):
        executed_tests(evidence / "tests.xml", authored_tests(repo))


def test_coverage_rejects_failed_command_even_with_existing_artifacts(coverage_fixture):
    coverage_fixture[-1]["tests"].update(status="failed", exit=1)
    with pytest.raises(IntegrityError, match="failed or is unfinished"):
        produce(coverage_fixture)
