"""Harmonize dataset labels and apply reviewed Cell Ontology mappings."""

from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from cellbridge.config import load_config

_CLUSTER_PREFIX = re.compile(r"^\d+_")
_TRAILING_CLUSTER = re.compile(r"\s+\d+$")
_SPACE = re.compile(r"\s+")
_MARKER_POSITIVE = re.compile(r"\b([a-z][a-z0-9-]*)\+(?=\s|$)")
_MARKER_NEGATIVE = re.compile(r"\b([a-z][a-z0-9]*)-(?=\s|$)")

_PLURALS = {
    "adipocytes": "adipocyte",
    "cafs": "caf",
    "cells": "cell",
    "fibroblasts": "fibroblast",
    "granulocytes": "granulocyte",
    "lymphocytes": "lymphocyte",
    "macrophages": "macrophage",
    "monocytes": "monocyte",
    "nerves": "nerve",
    "pericytes": "pericyte",
}

_ALIASES = {
    "acinar": "acinar cell",
    "activated lymphoid cell": "activated lymphocyte",
    "b lymphocyte": "b cell",
    "bcell": "b cell",
    "cd4": "cd4-positive t cell",
    "cd8": "cd8-positive t cell",
    "connective tissue": "connective tissue cell",
    "cxcl14-positive caf": "cxcl14-positive cancer-associated fibroblast",
    "ductal": "ductal cell",
    "ductal epithelium": "duct epithelial cell",
    "endocrine": "endocrine cell",
    "endothelial": "endothelial cell",
    "epithelial": "epithelial cell",
    "effector t lymphocyte": "effector t cell",
    "fallopian tube epithelium": "fallopian tube epithelial cell",
    "inflammatory": "inflammatory cell",
    "mmp11-positive caf": "mmp11-positive cancer-associated fibroblast",
    "myoepithelial": "myoepithelial cell",
    "myoepithelium": "myoepithelial cell",
    "nk cell": "natural killer cell",
    "non classical monocyte": "non-classical monocyte",
    "other": "other cell",
    "plasma": "plasma cell",
    "proliferative tumor cell": "proliferating tumor cell",
    "smooth muscle": "smooth muscle cell",
    "stroma": "stromal cell",
    "t and nk cell": "t/nk cell",
    "t lymphocyte": "t cell",
    "tumor": "tumor cell",
    "vascular endothelium": "endothelial cell",
    "vasculature/lymphatics": "vascular/lymphatic endothelial cell",
}


def normalize_label(label: str) -> str:
    """Return a lowercase, singular label without source cluster numbers."""
    value = _CLUSTER_PREFIX.sub("", str(label).strip()).replace("_", " ")
    value = _TRAILING_CLUSTER.sub("", value).replace("&", "and")
    value = value.replace("αβ", "alpha-beta").casefold()
    value = _MARKER_POSITIVE.sub(r"\1-positive", value)
    value = _MARKER_NEGATIVE.sub(r"\1-negative", value)
    value = _SPACE.sub(" ", value).strip()
    value = re.sub(
        r"\b(" + "|".join(map(re.escape, _PLURALS)) + r")\b",
        lambda match: _PLURALS[match.group()],
        value,
    )
    return _ALIASES.get(value, value)


def _read_table(path: Path, columns=None) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, usecols=columns)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path, columns=columns)
    raise ValueError(f"unsupported label file: {path}")


def _label_lookup(config: dict) -> dict[str, tuple[str, dict]]:
    lookup = {}
    for normalized, entry in config["labels"].items():
        if normalized != normalize_label(normalized):
            raise ValueError(f"normalized label is not canonical: {normalized!r}")
        for source_label in entry["source_labels"]:
            if source_label in lookup:
                raise ValueError(f"source label is mapped twice: {source_label!r}")
            if normalize_label(source_label) != normalized:
                observed = normalize_label(source_label)
                raise ValueError(
                    f"{source_label!r} normalizes to {observed!r}, "
                    f"not {normalized!r}"
                )
            lookup[str(source_label)] = (normalized, entry)
    return lookup


def check_mapping(config_path, ontology_path=None) -> dict[str, int]:
    """Validate one mapping against its source labels and optional CL graph."""
    config = load_config(config_path)
    source = config["source"]
    columns = [source["label_column"]]
    if source.get("sample_column"):
        columns.append(source["sample_column"])
    table = _read_table(Path(source["path"]), columns)
    labels = table[source["label_column"]].astype(str)
    lookup = _label_lookup(config)

    observed = set(labels)
    configured = set(lookup)
    if observed != configured:
        missing = sorted(observed - configured)
        extra = sorted(configured - observed)
        raise ValueError(f"unmapped labels={missing}; unobserved labels={extra}")

    counts = labels.value_counts()
    for normalized, entry in config["labels"].items():
        observed_count = int(counts.reindex(entry["source_labels"]).sum())
        if entry.get("n_cells") != observed_count:
            raise ValueError(
                f"{normalized!r}: expected n_cells={observed_count}, "
                f"found {entry.get('n_cells')!r}"
            )
        excluded = bool(entry.get("excluded", False))
        if excluded != (entry.get("ontology_id") is None):
            raise ValueError(f"{normalized!r}: excluded and ontology_id disagree")
        if excluded and not entry.get("note"):
            raise ValueError(f"{normalized!r}: excluded labels require a note")

    if ontology_path:
        _check_ontology(config, Path(ontology_path))

    samples = (
        table[source["sample_column"]].nunique()
        if source.get("sample_column")
        else 1
    )
    return {
        "cells": len(table),
        "samples": int(samples),
        "source_labels": len(observed),
        "normalized_labels": len(config["labels"]),
    }


@lru_cache(maxsize=None)
def _ontology_graph(path: Path):
    with path.open() as handle:
        graph = json.load(handle)["graphs"][0]

    def cl_id(iri):
        marker = "/obo/CL_"
        return f"CL:{iri.rsplit(marker, 1)[1]}" if marker in iri else None

    names = {
        cl_id(node["id"]): node.get("lbl")
        for node in graph["nodes"]
        if cl_id(node["id"])
    }
    parents = {term_id: set() for term_id in names}
    for edge in graph["edges"]:
        child, parent = cl_id(edge["sub"]), cl_id(edge["obj"])
        if edge["pred"] == "is_a" and child in names and parent in names:
            parents[child].add(parent)
    version_iri = graph["meta"]["version"]
    version = version_iri.split("/releases/", 1)[-1].split("/", 1)[0]
    return version, names, parents


def _check_ontology(config: dict, path: Path) -> None:
    version, names, parents = _ontology_graph(path)
    if config["ontology"]["release"] != version:
        configured = config["ontology"]["release"]
        raise ValueError(f"ontology release is {version}, not {configured}")

    root = config["ontology"]["root"]
    if names.get(root["id"]) != root["name"]:
        raise ValueError("ontology root ID and name disagree")
    if parents[root["id"]]:
        raise ValueError(f"configured ontology root {root['id']} has a parent")

    for normalized, entry in config["labels"].items():
        term_id = entry.get("ontology_id")
        if term_id is None:
            continue
        if names.get(term_id) != entry.get("ontology_name"):
            raise ValueError(f"{normalized!r}: ontology ID and name disagree")
        ancestors = {term_id}
        pending = [term_id]
        while pending:
            current = pending.pop()
            unseen = parents[current] - ancestors
            ancestors.update(unseen)
            pending.extend(unseen)
        if root["id"] not in ancestors:
            raise ValueError(f"{normalized!r}: ontology term does not reach root")
        configured_parents = {
            (parent["id"], parent["name"]) for parent in entry["parents"]
        }
        expected_parents = {(parent, names[parent]) for parent in parents[term_id]}
        if configured_parents != expected_parents:
            raise ValueError(f"{normalized!r}: direct ontology parents disagree")


def apply_mapping(config_path, output_path) -> None:
    """Write a source table with its reviewed normalized and ontology labels."""
    config = load_config(config_path)
    source = config["source"]
    table = _read_table(Path(source["path"]))
    source_labels = table[source["label_column"]].astype(str)
    lookup = _label_lookup(config)
    unknown = sorted(set(source_labels) - set(lookup))
    if unknown:
        raise ValueError(f"unmapped source labels: {unknown}")

    table["source_label"] = source_labels
    table["normalized_label"] = source_labels.map(
        {raw: normalized for raw, (normalized, _) in lookup.items()}
    )
    for column in ("ontology_id", "ontology_name"):
        table[column] = source_labels.map(
            {raw: entry.get(column) for raw, (_, entry) in lookup.items()}
        )
    table["excluded_from_ontology"] = source_labels.map(
        {raw: bool(entry.get("excluded", False)) for raw, (_, entry) in lookup.items()}
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        table.to_csv(output_path, index=False)
    elif output_path.suffix.lower() in {".parquet", ".pq"}:
        table.to_parquet(output_path, index=False)
    else:
        raise ValueError("output must be CSV or Parquet")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check")
    check.add_argument("configs", nargs="+")
    check.add_argument("--ontology", help="official cl-base.json release")
    apply = commands.add_parser("apply")
    apply.add_argument("config")
    apply.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.command == "check":
        for config_path in args.configs:
            summary = check_mapping(config_path, args.ontology)
            values = ", ".join(f"{key}={value:,}" for key, value in summary.items())
            print(f"{config_path}: {values}")
    else:
        apply_mapping(args.config, args.output)


if __name__ == "__main__":
    main()
