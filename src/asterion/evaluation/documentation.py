import html
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from asterion.artifacts.io import file_hash
from asterion.contracts.errors import IntegrityError, ResourceError


def render_dot_subset(source: Path):
    """Render the documented acyclic label/edge DOT subset without an external executable."""
    text = source.read_text()
    nodes = dict(re.findall(r'(\w+)\s*\[label="([^"]+)"\]', text))
    edges = re.findall(r"(\w+)\s*->\s*(\w+)\s*;", text)
    if not nodes or any(a not in nodes or b not in nodes for a, b in edges):
        raise IntegrityError("invalid diagram node or edge")
    depths: dict[str, int] = {}
    for _ in range(len(nodes)):
        for node in nodes:
            parents = [a for a, b in edges if b == node]
            if all(parent in depths for parent in parents):
                depths[node] = max([depths[p] + 1 for p in parents], default=0)
    if len(depths) != len(nodes):
        raise IntegrityError("portable diagram renderer requires an acyclic graph")
    ranks = {
        level: [node for node in nodes if depths[node] == level] for level in set(depths.values())
    }
    positions = {
        node: (30 + level * 240, 30 + row * 115)
        for level, values in ranks.items()
        for row, node in enumerate(values)
    }
    width = (max(ranks) + 1) * 240 + 20
    height = max(map(len, ranks.values())) * 115 + 20
    result = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#4b718c"/></marker></defs>',
        '<rect width="100%" height="100%" fill="#f8fbfd"/>',
    ]
    for first, second in edges:
        x, y = positions[first]
        tx, ty = positions[second]
        result.append(
            f'<path d="M{x + 200},{y + 36} C{x + 220},{y + 36} {tx - 20},{ty + 36} {tx},{ty + 36}" stroke="#4b718c" fill="none" stroke-width="2" marker-end="url(#arrow)"/>'
        )
    for node, label in nodes.items():
        x, y = positions[node]
        result.append(
            f'<rect x="{x}" y="{y}" width="200" height="72" rx="8" fill="#e6f0f5" stroke="#4b718c"/>'
        )
        for i, line in enumerate(label.split(r"\n")):
            result.append(
                f'<text x="{x + 100}" y="{y + 28 + i * 18}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#182e40">{html.escape(line)}</text>'
            )
    result.append("</svg>")
    source.with_suffix(".svg").write_text("\n".join(result))


def check_docs(root: Path, render=False):
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
            linked = path.parent / relative
            if not linked.exists():
                errors.append(f"broken local link {path.name}: {target}")
    diagrams = list((root / "docs" / "diagrams").glob("*.dot"))
    if len(diagrams) < 10:
        errors.append("fewer than ten editable domain diagrams")
    for source in diagrams:
        if render:
            if shutil.which("dot") is None:
                render_dot_subset(source)
            else:
                try:
                    subprocess.run(
                        ["dot", "-Tsvg", str(source), "-o", str(source.with_suffix(".svg"))],
                        check=True,
                        capture_output=True,
                    )
                except (OSError, subprocess.CalledProcessError) as error:
                    raise ResourceError("diagram rendering failed") from error
        try:
            tree = ET.parse(source.with_suffix(".svg"))
            if tree.getroot().tag != "{http://www.w3.org/2000/svg}svg":
                raise ValueError("missing SVG root")
            labels = [
                line
                for label in re.findall(r'\w+\s*\[label="([^"]+)"\]', source.read_text())
                for line in label.split(r"\n")
            ]
            exported = [
                "".join(node.itertext()) for node in tree.iter("{http://www.w3.org/2000/svg}text")
            ]
            if not labels or any(label not in exported for label in labels):
                raise ValueError("diagram labels differ from editable source")
        except (OSError, ET.ParseError, ValueError) as error:
            errors.append(f"invalid diagram export {source.name}: {error}")
    decisions = sorted((root / "docs/adr").glob("[0-9][0-9]-*.md"))
    if len(decisions) < 8:
        errors.append("fewer than eight architecture decisions")
    if errors:
        raise IntegrityError("; ".join(errors))
    retained = sorted({*paths, *diagrams, *(p.with_suffix(".svg") for p in diagrams)})
    return {
        "status": "passed",
        "markdown_files": len(paths),
        "validated_diagrams": len(diagrams),
        "architecture_decisions": len(decisions),
        "files": {str(p.relative_to(root)): file_hash(p) for p in retained},
    }
