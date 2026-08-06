"""Aligned frozen HEST features with reviewed Cell Ontology leaf labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cellbridge.evaluation.artifacts import load_array_artifact
from cellbridge.lora.features import FORMAT_VERSION

from .hierarchy import Hierarchy


@dataclass(frozen=True)
class FrozenHestData:
    features: np.ndarray
    labels: np.ndarray
    row_ids: np.ndarray
    sample_ids: np.ndarray
    split: np.ndarray
    train_indices: np.ndarray
    calibration_indices: np.ndarray
    test_indices: np.ndarray


def _manifest(path, expected_samples):
    path = Path(path)
    table = pd.read_csv(path)
    required = {"sample_id", "path", "signature"}
    if not required <= set(table):
        raise ValueError(f"{path}: missing columns {sorted(required - set(table))}")
    table["sample_id"] = table["sample_id"].astype(str)
    observed = table["sample_id"].tolist()
    expected = list(map(str, expected_samples))
    if observed != expected:
        raise ValueError(f"{path}: samples {observed} do not equal expected {expected}")
    if table["sample_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate samples")
    return table


def _sample_labels(labels_dir, sample_id):
    path = Path(labels_dir) / f"{sample_id}.parquet"
    columns = ["cell_id", "status", "hierarchy_depth", "level_3_identity"]
    table = pd.read_parquet(path, columns=columns)
    keep = (
        table["status"].astype(str).eq("full")
        & table["hierarchy_depth"].eq(3)
        & table["level_3_identity"].notna()
    )
    table = table.loc[keep, ["cell_id", "level_3_identity"]].copy()
    table["cell_id"] = table["cell_id"].astype(str)
    table["level_3_identity"] = table["level_3_identity"].astype(str)
    if table["cell_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate full-label cell IDs")
    return table.set_index("cell_id")["level_3_identity"]


def load_frozen_hest_representation(
    manifests,
    samples_by_split,
    labels_dir,
    arm,
):
    """Load one arm and retain only reviewed, full depth-three identities."""
    feature_parts = []
    label_parts = []
    row_parts = []
    sample_parts = []
    split_parts = []
    for split_name in ("train", "calibration", "test"):
        expected = samples_by_split[split_name]
        manifest = _manifest(manifests[split_name], expected)
        for record in manifest.itertuples(index=False):
            sample_id = str(record.sample_id)
            arrays, cell_ids, _ = load_array_artifact(
                record.path,
                format_version=FORMAT_VERSION,
                signature=str(record.signature),
                array_dims={arm: None},
            )
            cell_ids = np.asarray(cell_ids).astype(str)
            labels = _sample_labels(labels_dir, sample_id).reindex(cell_ids)
            keep = labels.notna().to_numpy()
            if not keep.any():
                raise ValueError(f"{sample_id}: no exported cells have full ontology labels")
            kept_ids = cell_ids[keep]
            feature_parts.append(np.asarray(arrays[arm])[keep])
            label_parts.append(labels.to_numpy()[keep].astype(str))
            row_parts.append(np.char.add(f"{sample_id}:", kept_ids))
            sample_parts.append(np.repeat(sample_id, keep.sum()))
            split_parts.append(np.repeat(split_name, keep.sum()))

    features = np.concatenate(feature_parts)
    labels = np.concatenate(label_parts)
    row_ids = np.concatenate(row_parts)
    sample_ids = np.concatenate(sample_parts)
    split = np.concatenate(split_parts)
    if len(np.unique(row_ids)) != len(row_ids):
        raise ValueError("sample-qualified HEST row IDs must be unique")
    train = np.flatnonzero(split == "train")
    calibration = np.flatnonzero(split == "calibration")
    test = np.flatnonzero(split == "test")
    train_classes = set(labels[train])
    for name, indices in (("calibration", calibration), ("test", test)):
        unseen = sorted(set(labels[indices]) - train_classes)
        if unseen:
            raise ValueError(f"{name} contains identities absent from training: {unseen}")
    return FrozenHestData(
        features,
        labels,
        row_ids,
        sample_ids,
        split,
        train,
        calibration,
        test,
    )


def load_pruned_ontology_hierarchy(nodes_path, leaf_classes):
    """Build a collision-free task tree for the observed training leaves.

    Compartment and lineage names occasionally coincide (for example,
    ``epithelial``), so internal node IDs carry their level explicitly.
    """
    table = pd.read_csv(nodes_path, keep_default_na=False)
    required = {"level", "parent", "node"}
    if not required <= set(table):
        raise ValueError(
            f"{nodes_path}: missing columns {sorted(required - set(table))}"
        )
    known_compartments = set(
        table.loc[table["level"].eq(1), "node"].astype(str)
    )
    lineages = table.loc[table["level"].eq(2), ["parent", "node"]].copy()
    lineage_parent = dict(
        zip(lineages["node"].astype(str), lineages["parent"].astype(str))
    )
    leaves = table.loc[table["level"].eq(3), ["parent", "node"]].copy()
    leaf_lineage = dict(
        zip(leaves["node"].astype(str), leaves["parent"].astype(str))
    )

    edges = set()
    for leaf in map(str, leaf_classes):
        if leaf not in leaf_lineage:
            raise ValueError(f"ontology hierarchy has no depth-three identity {leaf!r}")
        lineage = leaf_lineage[leaf]
        if lineage not in lineage_parent:
            raise ValueError(f"ontology hierarchy has no compartment for lineage {lineage!r}")
        compartment = lineage_parent[lineage]
        if compartment not in known_compartments:
            raise ValueError(f"ontology hierarchy has no node for {compartment!r}")
        compartment_node = f"compartment::{compartment}"
        lineage_node = f"lineage::{lineage}"
        edges.update(
            {
                ("cell", compartment_node),
                (compartment_node, lineage_node),
                (lineage_node, leaf),
            }
        )
    return Hierarchy.from_edges("cell", sorted(edges))
