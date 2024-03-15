import pytest

from asterion.artifacts.io import file_hash
from asterion.artifacts.source import project_root
from asterion.contracts.errors import IntegrityError
from asterion.evaluation.documentation import check_docs, render_dot_subset


@pytest.fixture
def document_fixture(tmp_path):
    (tmp_path / "README.md").write_text("# Fixture\n")
    diagrams = tmp_path / "docs/diagrams"
    decisions = tmp_path / "docs/adr"
    diagrams.mkdir(parents=True)
    decisions.mkdir()
    for i in range(10):
        source = diagrams / f"{i:02}.dot"
        source.write_text('digraph {a [label="first"]; b [label="second"]; a -> b;}')
        render_dot_subset(source)
    for i in range(8):
        (decisions / f"{i:02}-fixture.md").write_text("# Fixture decision\n")
    return tmp_path


def test_document_validation_never_rewrites_frozen_inputs():
    root = project_root()
    paths = [root / "README.md", *(root / "docs").rglob("*")]
    before = {str(p): file_hash(p) for p in paths if p.is_file()}
    result = check_docs(root)
    assert result["validated_diagrams"] >= 10 and result["architecture_decisions"] >= 8
    assert before == {str(p): file_hash(p) for p in paths if p.is_file()}


def test_document_validation_rejects_missing_render(document_fixture):
    (document_fixture / "docs/diagrams/00.svg").unlink()
    with pytest.raises(IntegrityError, match="export"):
        check_docs(document_fixture)


def test_document_validation_rejects_stale_valid_svg(document_fixture):
    path = document_fixture / "docs/diagrams/00.dot"
    path.write_text(path.read_text().replace("first", "changed source"))
    with pytest.raises(IntegrityError, match="labels differ"):
        check_docs(document_fixture)


def test_document_validation_rejects_broken_markdown_link(document_fixture):
    (document_fixture / "README.md").write_text("[missing](docs/not-here.md)\n")
    with pytest.raises(IntegrityError, match="broken local"):
        check_docs(document_fixture)


def test_explicit_portable_render_repairs_only_requested_exports(document_fixture):
    path = document_fixture / "docs/diagrams/00.dot"
    path.write_text(path.read_text().replace("first", "updated"))
    before = path.read_bytes()
    check_docs(document_fixture, render=True)
    assert path.read_bytes() == before and "updated" in path.with_suffix(".svg").read_text()


def test_document_validation_rejects_missing_decision(document_fixture):
    (document_fixture / "docs/adr/00-fixture.md").unlink()
    with pytest.raises(IntegrityError, match="decisions"):
        check_docs(document_fixture)


def test_portable_renderer_rejects_cycle(tmp_path):
    path = tmp_path / "cycle.dot"
    path.write_text('digraph {a [label="a"]; b [label="b"]; a -> b; b -> a;}')
    with pytest.raises(IntegrityError, match="acyclic"):
        render_dot_subset(path)
    assert not path.with_suffix(".svg").exists()
