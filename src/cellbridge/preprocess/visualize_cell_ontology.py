"""Render a cell-label mapping YAML as a Graphviz hierarchy.

Example:
    PYTHONPATH=src python -m cellbridge.preprocess.visualize_cell_ontology \
      configs/preprocess/cell_ontology/phenocell.yaml \
      --output phenocell.svg
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from cellbridge.config import load_config

_OUTPUT_FORMATS = {".pdf": "pdf", ".png": "png", ".svg": "svg"}


def _quote(value) -> str:
    value = str(value)
    value = value.replace("\\", "\\\\").replace('"', '\\"')
    value = value.replace("\n", r"\n")
    return f'"{value}"'


def _attributes(**values) -> str:
    return ", ".join(f"{key}={_quote(value)}" for key, value in values.items())


def _label_text(normalized: str, entry: dict) -> str:
    lines = [normalized, f"n={entry['n_cells']:,}", "source:"]
    lines.extend(map(str, entry["source_labels"]))
    return "\n".join(lines)


def mapping_to_dot(config: dict) -> str:
    """Return a DOT graph for the mappings and direct parents in a config."""
    root = config["ontology"]["root"]
    terms = {root["id"]: root["name"]}
    mapped_terms = set()
    ontology_edges = set()
    mapped_labels = []
    excluded_labels = []

    def add_term(term_id: str, name: str) -> None:
        previous = terms.setdefault(term_id, name)
        if previous != name:
            raise ValueError(f"conflicting names for {term_id}: {previous!r}, {name!r}")

    for normalized, entry in config["labels"].items():
        term_id = entry.get("ontology_id")
        if term_id is None:
            excluded_labels.append((normalized, entry))
            continue

        add_term(term_id, entry["ontology_name"])
        mapped_terms.add(term_id)
        mapped_labels.append((normalized, entry, term_id))
        for parent in entry["parents"]:
            add_term(parent["id"], parent["name"])
            ontology_edges.add((term_id, parent["id"]))

    dataset = str(config["dataset"])
    if config.get("sample_id"):
        dataset += f" / {config['sample_id']}"
    title = (
        f"{dataset}\nCell Ontology {config['ontology']['release']}\n"
        "normalized labels → CL terms → asserted direct parents"
    )

    lines = [
        "digraph cell_ontology_mapping {",
        "  graph ["
        + _attributes(
            rankdir="LR",
            bgcolor="white",
            label=title,
            labelloc="t",
            fontsize="18",
            fontname="Helvetica",
            nodesep="0.35",
            ranksep="0.65",
        )
        + "];",
        "  node [fontname=\"Helvetica\", fontsize=\"10\"];",
        "  edge [fontname=\"Helvetica\", fontsize=\"9\", arrowsize=\"0.7\"];",
    ]

    for term_id, name in sorted(terms.items()):
        attributes = {
            "label": f"{name}\n{term_id}",
            "shape": "ellipse",
            "style": "filled",
            "color": "#2563eb",
            "fillcolor": "#dbeafe",
        }
        if term_id in mapped_terms:
            attributes["fillcolor"] = "#bfdbfe"
            attributes["penwidth"] = "2"
        if term_id == root["id"]:
            attributes.update(fillcolor="#1d4ed8", fontcolor="white", penwidth="2")
        lines.append(f"  {_quote(f'term:{term_id}')} [{_attributes(**attributes)}];")

    for child, parent in sorted(ontology_edges):
        lines.append(
            f"  {_quote(f'term:{child}')} -> {_quote(f'term:{parent}')} "
            f"[{_attributes(label='is_a', color='#64748b')}];"
        )

    for index, (normalized, entry, term_id) in enumerate(mapped_labels):
        node_id = f"label:{index}"
        attributes = {
            "label": _label_text(normalized, entry),
            "shape": "box",
            "style": "rounded,filled",
            "color": "#15803d",
            "fillcolor": "#dcfce7",
        }
        if entry.get("note"):
            attributes["tooltip"] = entry["note"]
        lines.append(f"  {_quote(node_id)} [{_attributes(**attributes)}];")
        lines.append(
            f"  {_quote(node_id)} -> {_quote(f'term:{term_id}')} "
            f"[{_attributes(label='maps to', color='#15803d', style='dashed')}];"
        )

    if excluded_labels:
        lines.append(
            "  \"excluded\" ["
            + _attributes(
                label="excluded from Cell Ontology",
                shape="ellipse",
                style="dashed,filled",
                color="#b91c1c",
                fillcolor="#fee2e2",
            )
            + "];"
        )
        for index, (normalized, entry) in enumerate(excluded_labels):
            node_id = f"excluded:{index}"
            attributes = {
                "label": _label_text(normalized, entry),
                "shape": "box",
                "style": "rounded,filled",
                "color": "#b91c1c",
                "fillcolor": "#fee2e2",
                "tooltip": entry["note"],
            }
            lines.append(f"  {_quote(node_id)} [{_attributes(**attributes)}];")
            lines.append(
                f"  {_quote(node_id)} -> \"excluded\" "
                f"[{_attributes(color='#b91c1c', style='dashed')}];"
            )

    lines.append("}")
    return "\n".join(lines) + "\n"


def render_mapping(config_path, output_path) -> Path:
    """Render one mapping YAML to DOT, SVG, PNG, or PDF."""
    config = load_config(config_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dot_source = mapping_to_dot(config)

    if output_path.suffix.lower() == ".dot":
        output_path.write_text(dot_source)
        return output_path

    output_format = _OUTPUT_FORMATS.get(output_path.suffix.lower())
    if output_format is None:
        supported = ", ".join([".dot", *_OUTPUT_FORMATS])
        raise ValueError(f"output must use one of: {supported}")

    executable = shutil.which("dot")
    if executable is None:
        raise RuntimeError("Graphviz 'dot' is required to render this output format")
    try:
        subprocess.run(
            [executable, f"-T{output_format}", "-o", str(output_path)],
            input=dot_source,
            text=True,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(error.stderr.strip()) from error
    return output_path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="cell ontology mapping YAML")
    parser.add_argument("--output", required=True, help=".dot, .svg, .png, or .pdf")
    args = parser.parse_args(argv)
    output = render_mapping(args.config, args.output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
