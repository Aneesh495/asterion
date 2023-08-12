import re
import subprocess
from pathlib import Path

from asterion.contracts.errors import IntegrityError, ResourceError


def check_docs(root: Path):
    paths = [root / "README.md", *(root / "docs").rglob("*.md")]
    errors = []
    for path in paths:
        text = path.read_text()
        if "\u2014" in text:
            errors.append(f"em dash in {path.relative_to(root)}")
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
            if target.startswith(("https://", "http://", "#")):
                continue
            relative = target.split("#", 1)[0]
            if not (path.parent / relative).exists():
                errors.append(f"broken local link {path.name}: {target}")
    diagrams = list((root / "docs" / "diagrams").glob("*.dot"))
    if len(diagrams) < 10:
        errors.append("fewer than ten editable domain diagrams")
    for source in diagrams:
        try:
            subprocess.run(
                ["dot", "-Tsvg", str(source), "-o", str(source.with_suffix(".svg"))],
                check=True,
                capture_output=True,
            )
        except FileNotFoundError as error:
            raise ResourceError(
                "Graphviz dot is required to render editable architecture diagrams"
            ) from error
    if errors:
        raise IntegrityError("; ".join(errors))
    return {"status": "passed", "markdown_files": len(paths), "rendered_diagrams": len(diagrams)}
