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
from cellbridge.ontology import Ontology, load_mapping, load_ontology

_OUTPUT_FORMATS = {".pdf": "pdf", ".png": "png", ".svg": "svg"}

# Pango parses a trailing "Roman" as a style, so the family needs a second
# entry to survive as a family name.
_FONT = "Latin Modern Roman, serif"

_TERM_COLOR = "#287DC5"
_TERM_FILL = "#9CBDDF"
_MAPPED_COLOR = "#8859AB"
_MAPPED_FILL = "#C0ADD2"
_LABEL_COLOR = "#88C2AB"
_LABEL_FILL = "#C9E0D6"
_EXCLUDED_COLOR = "#CB722F"
_EXCLUDED_FILL = "#DEBA9B"
_EDGE_COLOR = "#000000"

_DATASET_NAMES = {
    "hest": "HEST",
    "lizard": "Lizard",
    "nucls": "NuCLS",
    "pannuke": "PanNuke",
    "phenocell": "PhenoCell",
}


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


def _ancestor_graph(mapping: dict, ontology: Ontology):
    """Return mapped terms and their complete ancestor subgraph."""
    root = {"id": ontology.root, "name": ontology.names[ontology.root]}
    if mapping["ontology"]["root"] != root:
        raise ValueError("mapping and ontology root disagree")

    mapped_terms = {
        entry["ontology_id"]
        for entry in mapping["labels"].values()
        if entry.get("ontology_id") is not None
    }
    nodes = {ontology.root}
    for term_id in mapped_terms:
        nodes.update(ontology.ancestors(term_id))
    names = {term_id: ontology.names[term_id] for term_id in nodes}
    edges = {
        (term_id, parent)
        for term_id in nodes
        for parent in ontology.parents[term_id]
        if parent in nodes
    }
    return root, names, edges, mapped_terms


def mapping_to_dot(config: dict, ontology: Ontology) -> str:
    """Return a DOT graph through every asserted ancestor to the root."""
    _, terms, ontology_edges, mapped_terms = _ancestor_graph(config, ontology)
    mapped_labels = []
    excluded_labels = []

    for normalized, entry in config["labels"].items():
        term_id = entry.get("ontology_id")
        if term_id is None:
            excluded_labels.append((normalized, entry))
            continue

        if terms[term_id] != entry["ontology_name"]:
            raise ValueError(f"mapping and ontology name disagree for {term_id}")
        mapped_labels.append((normalized, entry, term_id))

    dataset = _DATASET_NAMES.get(config["dataset"], str(config["dataset"]))
    if config.get("sample_id"):
        dataset += f" \u2014 {config['sample_id']}"
    title = f"Cell Ontology Tree for {dataset}"

    lines = [
        "digraph cell_ontology_mapping {",
        "  graph ["
        + _attributes(
            rankdir="BT",
            bgcolor="white",
            label=title,
            labelloc="t",
            fontsize="28",
            fontname=_FONT,
            nodesep="0.35",
            ranksep="0.65",
        )
        + "];",
        f'  node [fontname={_quote(_FONT)}, fontsize="14"];',
        f'  edge [fontname={_quote(_FONT)}, fontsize="12", arrowsize="0.7"];',
    ]

    for term_id, name in sorted(terms.items()):
        attributes = {
            "label": f"{name}\n{term_id}",
            "shape": "ellipse",
            "style": "filled",
            "color": _MAPPED_COLOR if term_id in mapped_terms else _TERM_COLOR,
            "fillcolor": _MAPPED_FILL if term_id in mapped_terms else _TERM_FILL,
        }
        lines.append(f"  {_quote(f'term:{term_id}')} [{_attributes(**attributes)}];")

    for child, parent in sorted(ontology_edges):
        lines.append(
            f"  {_quote(f'term:{child}')} -> {_quote(f'term:{parent}')} "
            f"[{_attributes(color=_EDGE_COLOR, tooltip='is_a')}];"
        )

    for index, (normalized, entry, term_id) in enumerate(mapped_labels):
        node_id = f"label:{index}"
        attributes = {
            "label": _label_text(normalized, entry),
            "shape": "box",
            "style": "rounded,filled",
            "color": _LABEL_COLOR,
            "fillcolor": _LABEL_FILL,
        }
        if entry.get("note"):
            attributes["tooltip"] = entry["note"]
        lines.append(f"  {_quote(node_id)} [{_attributes(**attributes)}];")
        lines.append(
            f"  {_quote(node_id)} -> {_quote(f'term:{term_id}')} "
            f"[{_attributes(label='maps to', color=_EDGE_COLOR, style='dashed')}];"
        )

    if excluded_labels:
        lines.append(
            "  \"excluded\" ["
            + _attributes(
                label="excluded from Cell Ontology",
                shape="ellipse",
                style="dashed,filled",
                color=_EXCLUDED_COLOR,
                fillcolor=_EXCLUDED_FILL,
            )
            + "];"
        )
        for index, (normalized, entry) in enumerate(excluded_labels):
            node_id = f"excluded:{index}"
            attributes = {
                "label": _label_text(normalized, entry),
                "shape": "box",
                "style": "rounded,filled",
                "color": _EXCLUDED_COLOR,
                "fillcolor": _EXCLUDED_FILL,
                "tooltip": entry["note"],
            }
            lines.append(f"  {_quote(node_id)} [{_attributes(**attributes)}];")
            lines.append(
                f"  {_quote(node_id)} -> \"excluded\" "
                f"[{_attributes(color=_EDGE_COLOR, style='dashed')}];"
            )

    lines.append("}")
    return "\n".join(lines) + "\n"


def render_mapping(config_path, output_path) -> Path:
    """Render one mapping YAML to DOT, SVG, PNG, or PDF."""
    config = load_config(config_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ontology = load_ontology(config["ontology"]["snapshot"])
    load_mapping(config_path).validate_ontology(ontology)
    dot_source = mapping_to_dot(config, ontology)

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
