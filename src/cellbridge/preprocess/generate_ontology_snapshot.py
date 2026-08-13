"""Build the compact ontology topology consumed by inference and visualization.

Example:
    PYTHONPATH=src python -m \
      cellbridge.preprocess.generate_ontology_snapshot cl-base.json \
      configs/preprocess/cell_ontology/immunocto.yaml \
      configs/preprocess/cell_ontology/lizard.yaml \
      configs/preprocess/cell_ontology/nucls.yaml \
      configs/preprocess/cell_ontology/pannuke.yaml \
      configs/preprocess/cell_ontology/phenocell.yaml \
      configs/preprocess/cell_ontology/hest/*.yaml \
      --output configs/preprocess/cell_ontology/cl-2026-06-08.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

ROOT = "CL:0000000"


def _cl_id(iri: str) -> str | None:
    marker = "/obo/CL_"
    if marker not in iri:
        return None
    return f"CL:{iri.rsplit(marker, 1)[1]}"


def _read_mapping(path: Path) -> dict:
    with path.open() as handle:
        mapping = yaml.safe_load(handle)
    if not isinstance(mapping, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return mapping


def _mapping_metadata(paths: list[Path]):
    mappings = [_read_mapping(path) for path in paths]
    ontology_entries = [mapping["ontology"] for mapping in mappings]

    def one(key):
        values = {str(entry[key]) for entry in ontology_entries}
        if len(values) != 1:
            raise ValueError(f"mappings disagree on ontology.{key}: {sorted(values)}")
        return values.pop()

    expected_names: dict[str, set[str]] = {}
    for mapping in mappings:
        for entry in mapping["labels"].values():
            term = entry.get("ontology_id")
            if term is not None:
                expected_names.setdefault(str(term), set()).add(
                    str(entry["ontology_name"])
                )
    roots = {
        (str(entry["root"]["id"]), str(entry["root"]["name"]))
        for entry in ontology_entries
    }
    if len(roots) != 1:
        raise ValueError(f"mappings disagree on ontology.root: {sorted(roots)}")
    return one("release"), one("source"), one("sha256"), roots.pop(), expected_names


def _load_ontology(path: Path):
    with path.open() as handle:
        graph = json.load(handle)["graphs"][0]

    names = {}
    parents = {}
    for node in graph["nodes"]:
        term = _cl_id(node["id"])
        if term is not None and node.get("lbl"):
            names[term] = str(node["lbl"])
            parents[term] = set()
    for edge in graph["edges"]:
        child = _cl_id(edge["sub"])
        parent = _cl_id(edge["obj"])
        if edge["pred"] == "is_a" and child in names and parent in names:
            parents[child].add(parent)

    version_iri = graph["meta"]["version"]
    release = version_iri.split("/releases/", 1)[-1].split("/", 1)[0]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return release, digest, names, parents


def _ancestors(term: str, parents: dict[str, set[str]]) -> set[str]:
    found = set()
    pending = [term]
    while pending:
        node = pending.pop()
        if node not in parents:
            raise ValueError(f"mapped term is absent from Cell Ontology: {node}")
        if node not in found:
            found.add(node)
            pending.extend(parents[node])
    return found


def generate(ontology_path, mapping_paths, output_path) -> None:
    mapping_paths = [Path(path) for path in mapping_paths]
    if not mapping_paths:
        raise ValueError("at least one mapping is required")
    expected_release, source, expected_sha256, expected_root, expected_names = (
        _mapping_metadata(mapping_paths)
    )
    release, sha256, names, parents = _load_ontology(Path(ontology_path))
    if release != expected_release:
        raise ValueError(f"ontology release is {release}, not {expected_release}")
    if sha256 != expected_sha256:
        raise ValueError(f"ontology SHA-256 is {sha256}, not {expected_sha256}")
    actual_root = (ROOT, names[ROOT])
    if expected_root != actual_root:
        raise ValueError(f"mapping ontology root is {expected_root}, not {actual_root}")

    mismatched = sorted(
        term
        for term, configured_names in expected_names.items()
        if names.get(term) not in configured_names or len(configured_names) != 1
    )
    if mismatched:
        raise ValueError(f"mapping ontology IDs and names disagree: {mismatched}")

    closure = {ROOT}
    for term in expected_names:
        ancestors = _ancestors(term, parents)
        if ROOT not in ancestors:
            raise ValueError(f"{term} is not a descendant of {ROOT}")
        closure.update(ancestors)

    config = {
        "release": release,
        "source": source,
        "sha256": sha256,
        "root": {"id": ROOT, "name": names[ROOT]},
        "terms": {
            term: {
                "name": names[term],
                "parents": sorted(parents[term] & closure),
            }
            for term in sorted(closure)
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = "# Generated from the pinned Cell Ontology release. Do not edit by hand.\n"
    output_path.write_text(header + yaml.safe_dump(config, sort_keys=False, width=100))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ontology", help="downloaded cl-base.json")
    parser.add_argument("mappings", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    generate(args.ontology, args.mappings, args.output)


if __name__ == "__main__":
    main()
