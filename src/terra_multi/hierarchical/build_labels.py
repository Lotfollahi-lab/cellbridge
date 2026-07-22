"""Build reviewed hierarchical labels for the annotated HEST cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from terra_multi.cohorts import samples_in_split
from terra_multi.config import load_config
from terra_multi.hierarchical.mapping import RULES, map_label, split_source_label
from terra_multi.hierarchical.taxonomy import CL_RELEASE, CL_SHA256


def _label_path(config: dict, sample_id: str) -> Path:
    return Path(
        config["labels_pattern"].format(
            data_root=config["data_root"], sample_id=sample_id
        )
    )


def inventory_labels(config: dict, samples: list[str]) -> pd.DataFrame:
    """Count raw labels per sample without loading all samples at once."""
    label_column = config.get("label_column", "group")
    rows = []
    for sample_id in samples:
        path = _label_path(config, sample_id)
        labels = pd.read_csv(path, usecols=[label_column])[label_column]
        if labels.isna().any():
            raise ValueError(f"{path}: {label_column!r} contains null values")
        for raw_label, count in labels.astype(str).value_counts(sort=False).items():
            rows.append(
                {"sample_id": sample_id, "raw_label": raw_label, "n_cells": int(count)}
            )
    return pd.DataFrame(rows).sort_values(["sample_id", "raw_label"])


def build_mapping(inventory: pd.DataFrame) -> pd.DataFrame:
    """Map every exact source label and attach its observed usage."""
    rows = []
    for raw_label, group in inventory.groupby("raw_label", sort=True):
        record = map_label(raw_label)
        record.update(
            n_cells=int(group["n_cells"].sum()),
            n_samples=int(group["sample_id"].nunique()),
            samples=";".join(sorted(group["sample_id"].unique())),
        )
        rows.append(record)
    mapping = pd.DataFrame(rows)
    return mapping.drop(columns="term").sort_values(["mapping_status", "raw_label"])


def validate_inventory(config: dict, samples: list[str], inventory: pd.DataFrame) -> None:
    normalized = {split_source_label(label)[1] for label in inventory["raw_label"]}
    unknown = sorted(normalized - set(RULES))
    if unknown:
        raise ValueError(f"unreviewed normalized labels: {unknown}")
    expected = {
        "samples": len(samples),
        "source_labels": inventory["raw_label"].nunique(),
        "normalized_labels": len(normalized),
        "cells": int(inventory["n_cells"].sum()),
    }
    for name, actual in expected.items():
        configured = config.get(f"expected_{name}")
        if configured is not None and int(configured) != actual:
            raise ValueError(f"found {actual:,} {name}; expected {int(configured):,}")


def summarize_usage(mapping: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "mapping_status",
        "canonical_identity",
        "cl_id",
        "hierarchy_path",
        "state",
        "malignancy",
        "context",
    ]
    usage = (
        mapping.groupby(columns, dropna=False)
        .agg(
            n_source_labels=("raw_label", "nunique"),
            n_normalized_labels=("normalized_label", "nunique"),
            n_cells=("n_cells", "sum"),
        )
        .reset_index()
    )
    usage["fraction_of_cells"] = usage["n_cells"] / mapping["n_cells"].sum()
    return usage.sort_values(["mapping_status", "n_cells"], ascending=[True, False])


def hierarchy_nodes(mapping: pd.DataFrame) -> pd.DataFrame:
    """Derive the task-tree nodes and edges from all usable paths."""
    nodes = {(0, "", "cell", "CL:0000000")}
    for row in mapping.loc[mapping["use_in_hierarchy"]].itertuples():
        path = [row.level_0, row.level_1_compartment, row.level_2_lineage]
        if row.level_3_identity:
            path.append(row.level_3_identity)
        for level, node in enumerate(path[1:], start=1):
            cl_id = row.cl_id if level == 3 else ""
            nodes.add((level, path[level - 1], node, cl_id))
    return pd.DataFrame(
        sorted(nodes), columns=["level", "parent", "node", "cl_id"]
    )


def _write_parquet(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_parquet(temporary, index=False)
    temporary.replace(path)


def materialize_cells(config: dict, samples: list[str], mapping: pd.DataFrame) -> None:
    label_column = config.get("label_column", "group")
    id_column = config.get("label_id_column", "cell_id")
    annotations = mapping.drop(columns=["n_cells", "n_samples", "samples"])
    for sample_id in samples:
        path = _label_path(config, sample_id)
        cells = pd.read_csv(path, usecols=[id_column, label_column]).rename(
            columns={id_column: "cell_id", label_column: "raw_label"}
        )
        if cells["cell_id"].isna().any() or cells["raw_label"].isna().any():
            raise ValueError(f"{path}: cell IDs and labels must be non-null")
        if cells["cell_id"].duplicated().any():
            raise ValueError(f"{path}: cell IDs must be unique")
        cells.insert(0, "sample_id", sample_id)
        cells["raw_label"] = cells["raw_label"].astype(str)
        cells = cells.merge(annotations, on="raw_label", how="left", validate="many_to_one")
        if cells["mapping_status"].isna().any():
            missing = cells.loc[cells["mapping_status"].isna(), "raw_label"].unique()
            raise ValueError(f"{path}: unmapped labels after merge: {missing[:5].tolist()}")
        _write_parquet(cells, Path(config["output_dir"]) / "cells" / f"{sample_id}.parquet")


def _markdown(table: pd.DataFrame) -> list[str]:
    shown = table.fillna("").astype(str)
    escape = lambda value: value.replace("|", "\\|").replace("\n", " ")
    rows = [[escape(value) for value in row] for row in shown.to_numpy()]
    header = [escape(str(column)) for column in shown.columns]
    return [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def write_report(
    path: Path, samples: list[str], inventory: pd.DataFrame, mapping: pd.DataFrame, usage: pd.DataFrame
) -> None:
    total = int(mapping["n_cells"].sum())
    status = (
        mapping.groupby("mapping_status")
        .agg(source_labels=("raw_label", "size"), cells=("n_cells", "sum"))
        .reset_index()
    )
    status["cell_fraction"] = status["cells"].map(lambda value: f"{value / total:.2%}")
    overview = [
        "# HEST test-label hierarchy",
        "",
        f"Mapped **{mapping['raw_label'].nunique()} exact source labels** from "
        f"**{len(samples)} samples** and **{total:,} cells** into "
        f"**{mapping['canonical_identity'].replace('', pd.NA).nunique()} canonical identities**.",
        "",
        "Numeric prefixes are source cluster IDs only. For example, `0_Tumor_Cells` and "
        "`3_Tumor_Cells` share one biological identity but remain separate rows for auditability.",
        "States, malignancy, and tissue context are metadata, not hierarchy nodes. `partial` labels "
        "participate only to their known depth; `excluded` labels never form contrastive positives.",
        "",
        f"Cell Ontology release: `{CL_RELEASE}`; OBO SHA-256: `{CL_SHA256}`.",
        "",
        "## Coverage",
        "",
        *_markdown(status),
        "",
        "## Canonical usage",
        "",
        *_markdown(
            usage[
                ["mapping_status", "canonical_identity", "cl_id", "hierarchy_path", "state", "n_source_labels", "n_cells"]
            ]
        ),
        "",
        "## All exact source labels",
        "",
        *_markdown(
            mapping[
                ["raw_label", "n_cells", "mapping_status", "canonical_identity", "cl_id", "hierarchy_path", "state", "malignancy", "context", "samples"]
            ]
        ),
        "",
        "The machine-readable source of truth is `source_label_mapping.csv`; per-cell labels are "
        "stored in `cells/<sample_id>.parquet`.",
    ]
    path.write_text("\n".join(overview) + "\n")


def run(config: dict) -> dict:
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    samples = samples_in_split(config["splits_path"], config.get("split", "label_test"))
    inventory = inventory_labels(config, samples)
    validate_inventory(config, samples, inventory)
    mapping = build_mapping(inventory)
    usage = summarize_usage(mapping)
    nodes = hierarchy_nodes(mapping)

    inventory.to_csv(output / "sample_label_counts.csv", index=False)
    mapping.to_csv(output / "source_label_mapping.csv", index=False)
    usage.to_csv(output / "canonical_usage.csv", index=False)
    nodes.to_csv(output / "hierarchy_nodes.csv", index=False)
    metadata = {
        "samples": len(samples),
        "source_labels": int(mapping["raw_label"].nunique()),
        "normalized_labels": int(mapping["normalized_label"].nunique()),
        "canonical_identities": int(mapping["canonical_identity"].replace("", pd.NA).nunique()),
        "cells": int(mapping["n_cells"].sum()),
        "ontology_release": CL_RELEASE,
        "ontology_sha256": CL_SHA256,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    if config.get("materialize_cells", True):
        materialize_cells(config, samples, mapping)
    write_report(output / "REPORT.md", samples, inventory, mapping, usage)
    print((output / "REPORT.md").read_text())
    print(f"Artifacts written to {output}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hierarchical labels for HEST.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overlay")
    parser.add_argument("--mapping-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, overlay=args.overlay)
    if args.mapping_only:
        config["materialize_cells"] = False
    run(config)


if __name__ == "__main__":
    main()
