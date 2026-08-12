"""Inspect source cell labels before Cell Ontology mapping.

PYTHONPATH=src python -m cellbridge.preprocess.cell_ontology_labels apply \
  configs/preprocess/immunocto.yaml \
  --output immunocto_mapped_labels.parquet

"""

from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path

import pandas as pd
import yaml

from cellbridge.config import load_config


def _read_table(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, usecols=columns)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path, columns=columns)
    raise ValueError(f"unsupported label file: {path}")


def _source_paths(source: dict) -> list[tuple[Path, str | None]]:
    if "samples" not in source:
        paths = sorted(glob(str(source["path"])))
        return [(Path(path), None) for path in paths]

    sample_column = str(source["sample_column"])
    split_column = str(source["split_column"])
    samples = pd.read_csv(source["samples"], usecols=[sample_column, split_column])
    samples = samples.loc[samples[split_column].eq(source["split"]), sample_column]
    return [
        (Path(str(source["path"]).format(sample_id=sample_id)), str(sample_id))
        for sample_id in samples
    ]


def inspect_labels(config_path: str | Path) -> pd.DataFrame:
    """Return source-label counts and sample coverage for one dataset."""
    config = load_config(config_path)
    dataset = str(config["dataset"])
    source = config["source"]
    label_column = str(source["label_column"])
    sample_column = str(source["sample_column"])
    paths = _source_paths(source)
    if not paths:
        raise FileNotFoundError(f"{config_path}: no files match {source['path']}")

    parts = []
    for path, sample_id in paths:
        columns = (
            [label_column] if sample_id is not None else [label_column, sample_column]
        )
        table = _read_table(path, columns)
        table = table.rename(columns={label_column: "source_label"})
        if sample_id is None:
            table = table.rename(columns={sample_column: "sample_id"})
        else:
            table["sample_id"] = sample_id
        parts.append(table)

    labels = pd.concat(parts, ignore_index=True)
    labels["source_label"] = labels["source_label"].astype("string")
    labels["sample_id"] = labels["sample_id"].astype("string")
    if labels[["source_label", "sample_id"]].isna().any().any():
        raise ValueError(f"{config_path}: source labels or sample IDs contain nulls")

    inventory = (
        labels.groupby("source_label", as_index=False)
        .agg(n_cells=("source_label", "size"), n_samples=("sample_id", "nunique"))
        .sort_values("source_label")
        .reset_index(drop=True)
    )
    inventory.insert(0, "dataset", dataset)
    return inventory


def update_inventory(config_path: str | Path) -> None:
    """Add observed source labels to a dataset YAML for review."""
    config_path = Path(config_path)
    with open(config_path) as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{config_path}: configuration must be a YAML mapping")

    existing = config.get("labels") or {}
    if not isinstance(existing, dict):
        raise ValueError(f"{config_path}: labels must be a YAML mapping")

    labels = {}
    inventory = inspect_labels(config_path)
    for row in inventory.itertuples(index=False):
        source_label = str(row.source_label)
        previous = existing.get(source_label) or {}
        if not isinstance(previous, dict):
            raise ValueError(
                f"{config_path}: mapping for {source_label!r} must be a mapping"
            )
        entry = {
            "n_cells": int(row.n_cells),
            "n_samples": int(row.n_samples),
            "cleaned_label": None,
            "grouped_label": None,
            "cell_ontology_id": None,
            "review_status": "pending",
            "notes": None,
        }
        entry.update(previous)
        entry["n_cells"] = int(row.n_cells)
        entry["n_samples"] = int(row.n_samples)
        labels[source_label] = entry

    for source_label, previous in existing.items():
        if str(source_label) not in labels:
            labels[str(source_label)] = {
                "n_cells": 0,
                "n_samples": 0,
                **(previous or {}),
            }

    config["labels"] = labels
    with open(config_path, "w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    print(f"updated {len(inventory)} source labels -> {config_path}")


def apply_mapping(config_path: str | Path, output_path: str | Path) -> None:
    """Apply an approved YAML mapping and write cell-level labels."""
    config = load_config(config_path)
    dataset = str(config["dataset"])
    source = config["source"]
    label_column = str(source["label_column"])
    sample_column = str(source["sample_column"])
    mapping = config.get("labels") or {}

    incomplete = [
        label
        for label, entry in mapping.items()
        if not isinstance(entry, dict)
        or not entry.get("cleaned_label")
        or entry.get("review_status") != "approved"
    ]
    if incomplete:
        shown = ", ".join(map(repr, incomplete[:5]))
        raise ValueError(
            f"{config_path}: {len(incomplete)} labels are incomplete or "
            f"unapproved: {shown}"
        )

    parts = []
    for path, sample_id in _source_paths(source):
        table = _read_table(path)
        if label_column not in table:
            raise ValueError(f"{path}: missing label column {label_column!r}")
        if sample_id is None:
            if sample_column not in table:
                raise ValueError(f"{path}: missing sample column {sample_column!r}")
        else:
            table[sample_column] = sample_id
        table["source_label"] = table[label_column].astype("string")
        parts.append(table)

    cells = pd.concat(parts, ignore_index=True)
    observed = set(cells["source_label"])
    missing = sorted(observed - set(mapping))
    if missing:
        raise ValueError(f"{config_path}: source labels are not mapped: {missing}")

    cells.insert(0, "dataset", dataset)
    for field in ("cleaned_label", "grouped_label", "cell_ontology_id"):
        values = {label: entry.get(field) for label, entry in mapping.items()}
        cells[field] = cells["source_label"].map(values).astype("string")

    output_path = Path(output_path)
    if output_path.suffix.lower() not in {".parquet", ".pq"}:
        raise ValueError("mapped cell labels must be written to a Parquet file")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cells.to_parquet(output_path, index=False)
    print(f"wrote {len(cells)} mapped cells -> {output_path}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    inventory_parser = commands.add_parser(
        "inventory", help="add observed source labels to dataset YAML files"
    )
    inventory_parser.add_argument("configs", nargs="+", help="dataset label YAML files")

    apply_parser = commands.add_parser(
        "apply", help="apply one approved mapping to cell-level labels"
    )
    apply_parser.add_argument("config", help="dataset label YAML file")
    apply_parser.add_argument("--output", required=True, help="output Parquet file")
    args = parser.parse_args(argv)

    if args.command == "inventory":
        for config_path in args.configs:
            update_inventory(config_path)
    else:
        apply_mapping(args.config, args.output)


if __name__ == "__main__":
    main()
