"""Render the reviewed HEST hierarchy with depth-aware cell counts."""

from __future__ import annotations

import argparse
import subprocess
from collections import defaultdict
from pathlib import Path

import pandas as pd

COLORS = {
    "epithelial": "#4C78A8",
    "hematopoietic": "#E45756",
    "stromal": "#59A14F",
    "vascular": "#B279A2",
}


def aggregate_node_counts(usage: pd.DataFrame) -> dict[tuple[int, str, str], int]:
    """Sum cells along each valid hierarchy path, stopping partial paths safely."""

    status_column = "mapping_status" if "mapping_status" in usage else "status"
    required = {status_column, "hierarchy_path", "n_cells"}
    if not required <= set(usage):
        raise ValueError(f"canonical usage requires columns {sorted(required)}")
    counts: defaultdict[tuple[int, str, str], int] = defaultdict(int)
    usable = usage.loc[usage[status_column].isin(["full", "partial"])]
    for row in usable.itertuples(index=False):
        path = str(row.hierarchy_path).strip()
        if not path:
            continue
        nodes = [node.strip() for node in path.split(">")]
        for level, node in enumerate(nodes):
            parent = "" if level == 0 else nodes[level - 1]
            counts[(level, parent, node)] += int(row.n_cells)
    return dict(counts)


def _quoted(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def _parents(nodes: pd.DataFrame) -> tuple[dict[int, int], dict[int, str]]:
    parents: dict[int, int] = {}
    branches: dict[int, str] = {}
    for index, row in nodes.iterrows():
        level = int(row["level"])
        if level == 0:
            branches[index] = ""
            continue
        candidates = nodes.index[
            (nodes["level"].astype(int) == level - 1)
            & nodes["node"].eq(row["parent"])
        ].tolist()
        if len(candidates) != 1:
            raise ValueError(f"ambiguous parent for level {level} node {row['node']!r}")
        parent = candidates[0]
        parents[index] = parent
        branches[index] = str(row["node"]) if level == 1 else branches[parent]
    return parents, branches


def build_dot(nodes: pd.DataFrame, usage: pd.DataFrame) -> str:
    """Build Graphviz source using the existing tree style plus node counts."""

    required = {"level", "parent", "node", "cl_id"}
    if not required <= set(nodes):
        raise ValueError(f"hierarchy nodes require columns {sorted(required)}")
    nodes = nodes.fillna("").reset_index(drop=True)
    counts = aggregate_node_counts(usage)
    parents, branches = _parents(nodes)
    lines = [
        "digraph hierarchy {",
        "  rankdir=LR;",
        '  graph [ranksep=1.5, nodesep=0.10, bgcolor=white];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", '
        'fontsize=11, color="#00000022", penwidth=1, margin="0.11,0.05"];',
        '  edge [color="#999999cc", arrowsize=0.5, penwidth=0.9];',
    ]
    for index, row in nodes.iterrows():
        level = int(row["level"])
        key = (level, str(row["parent"]), str(row["node"]))
        count = counts.get(key, 0)
        if count <= 0:
            raise ValueError(f"hierarchy node has no cells: {key}")
        label = [str(row["node"])]
        if row["cl_id"]:
            label.append(str(row["cl_id"]))
        label.append(f"n={count:,}")
        if level == 0:
            attributes = {
                "fillcolor": "#555555",
                "fontcolor": "white",
                "fontsize": "16",
                "fontname": "Helvetica-Bold",
            }
        else:
            color = COLORS[branches[index]]
            attributes = {
                "fillcolor": color if level == 1 else color + ("66" if level == 2 else "22"),
                "fontcolor": "white" if level == 1 else "#1a1a1a",
                "fontsize": "13" if level == 1 else ("12" if level == 2 else "11"),
                "fontname": "Helvetica-Bold" if level < 3 else "Helvetica",
            }
        rendered = ", ".join(
            [f"label={_quoted(chr(10).join(label))}"]
            + [f"{name}={_quoted(value)}" for name, value in attributes.items()]
        )
        lines.append(f"  n{index} [{rendered}];")
    lines.extend(f"  n{parent} -> n{child};" for child, parent in parents.items())
    title = (
        "HEST label_test hierarchy  —  cell → compartment → lineage → "
        "Cell Ontology identity\nNode labels show depth-aware cell counts"
    )
    lines.append(
        f'  labelloc="t"; fontsize=20; fontname="Helvetica-Bold"; label={_quoted(title)};'
    )
    lines.append("}")
    return "\n".join(lines) + "\n"


def render_tree(artifact_dir: str | Path, *, dpi: int = 150) -> tuple[Path, Path]:
    artifact_dir = Path(artifact_dir)
    nodes = pd.read_csv(artifact_dir / "hierarchy_nodes.csv", keep_default_na=False)
    usage = pd.read_csv(artifact_dir / "canonical_usage.csv", keep_default_na=False)
    dot_path = artifact_dir / "hierarchy_tree.dot"
    png_path = artifact_dir / "hierarchy_tree.png"
    dot_path.write_text(build_dot(nodes, usage))
    subprocess.run(
        ["dot", "-Tpng", f"-Gdpi={int(dpi)}", str(dot_path), "-o", str(png_path)],
        check=True,
    )
    return dot_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    _, output = render_tree(args.artifact_dir, dpi=args.dpi)
    print(f"wrote hierarchy tree -> {output}")


if __name__ == "__main__":
    main()
