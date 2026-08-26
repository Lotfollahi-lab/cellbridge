"""Per-sample embedding artifacts and per-arm manifests."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cellbridge.inference.embed.methods import arms as method_arms
from cellbridge.inference.embed.runner import EmbeddingResult

MANIFEST_COLUMNS = (
    "method",
    "arm",
    "sample_id",
    "path",
    "n_cells",
    "n_features",
)


@dataclass(frozen=True)
class ArmEmbeddings:
    """One embedding arm restored in manifest order."""

    method: str
    arm: str
    sample_ids: np.ndarray
    cell_ids: np.ndarray
    row_ids: np.ndarray
    embeddings: np.ndarray


def save_sample(
    output_dir,
    method,
    sample_id,
    row_ids,
    result: EmbeddingResult,
    *,
    storage_dtype="float16",
) -> list[dict]:
    """Write one artifact for every arm emitted for a sample."""
    arms = method_arms(method)
    if set(result.embeddings) != set(arms):
        raise ValueError(f"{method} returned unexpected embedding arms")

    cell_ids = np.asarray(result.cell_ids).astype(str)
    row_ids = np.asarray(row_ids)
    if cell_ids.ndim != 1 or row_ids.ndim != 1 or len(cell_ids) != len(row_ids):
        raise ValueError("cell_ids and row_ids must be aligned one-dimensional arrays")
    if len(np.unique(cell_ids)) != len(cell_ids) or len(np.unique(row_ids)) != len(
        row_ids
    ):
        raise ValueError("cell_ids and row_ids must be unique within a sample")

    dtype = np.dtype(storage_dtype)
    if dtype.kind != "f":
        raise ValueError("storage_dtype must be floating point")

    records = []
    for arm in arms:
        embeddings = np.asarray(result.embeddings[arm])
        if embeddings.ndim != 2 or len(embeddings) != len(cell_ids):
            raise ValueError(f"{arm} embeddings are not aligned with the cells")
        embeddings = embeddings.astype(dtype, copy=False)
        if not np.isfinite(embeddings).all():
            raise ValueError(f"{arm} contains non-finite embeddings")

        directory = Path(output_dir) / arm
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{sample_id}.npz"
        np.savez(
            path,
            cell_ids=cell_ids,
            row_ids=row_ids,
            embeddings=embeddings,
        )
        records.append(
            {
                "method": method,
                "arm": arm,
                "sample_id": str(sample_id),
                "path": path.name,
                "n_cells": len(cell_ids),
                "n_features": embeddings.shape[1],
            }
        )
    return records


def write_manifests(output_dir, method, records) -> dict[str, Path]:
    """Write one manifest per arm after all samples are complete."""
    table = pd.DataFrame(records)
    missing = set(MANIFEST_COLUMNS) - set(table)
    if table.empty or missing:
        raise ValueError(f"manifest records are empty or missing {sorted(missing)}")
    arms = method_arms(method)
    if set(table["method"]) != {method} or set(table["arm"]) != set(arms):
        raise ValueError("manifest records do not match the requested method")
    if table.duplicated(["arm", "sample_id"]).any():
        raise ValueError("manifest records contain duplicate samples")

    paths = {}
    sample_order = None
    for arm in arms:
        selected = table.loc[table["arm"] == arm, list(MANIFEST_COLUMNS)]
        current_order = selected["sample_id"].astype(str).tolist()
        if sample_order is None:
            sample_order = current_order
        elif current_order != sample_order:
            raise ValueError(
                "every arm must contain the same samples in the same order"
            )

        directory = Path(output_dir) / arm
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "manifest.csv"
        selected.to_csv(path, index=False)
        paths[arm] = path
    return paths


def _read_manifest(path):
    path = Path(path)
    table = pd.read_csv(
        path,
        dtype={"method": str, "arm": str, "sample_id": str, "path": str},
    )
    missing = set(MANIFEST_COLUMNS) - set(table)
    if table.empty or missing:
        raise ValueError(f"{path} is empty or missing {sorted(missing)}")
    if table["method"].nunique() != 1 or table["arm"].nunique() != 1:
        raise ValueError(f"{path} must describe exactly one method and arm")
    return path, table


def _load_record(path, row):
    with np.load(path.parent / row.path, allow_pickle=False) as data:
        if not {"cell_ids", "row_ids", "embeddings"} <= set(data.files):
            raise ValueError(f"{row.path} is missing artifact arrays")
        sample_cells = data["cell_ids"].astype(str)
        sample_rows = np.asarray(data["row_ids"])
        values = np.asarray(data["embeddings"])
    if (
        sample_cells.ndim != 1
        or sample_rows.ndim != 1
        or values.shape != (row.n_cells, row.n_features)
        or len(sample_cells) != row.n_cells
        or len(sample_rows) != row.n_cells
        or not np.isfinite(values).all()
    ):
        raise ValueError(f"{row.path} does not match its manifest")
    return sample_cells, sample_rows, values.astype(np.float32)


def load_manifest(path) -> ArmEmbeddings:
    """Load and concatenate one arm in its recorded sample order."""
    path, table = _read_manifest(path)

    sample_ids, cell_ids, row_ids, embeddings = [], [], [], []
    for row in table.itertuples(index=False):
        sample_cells, sample_rows, values = _load_record(path, row)
        sample_ids.append(np.repeat(row.sample_id, row.n_cells))
        cell_ids.append(sample_cells)
        row_ids.append(sample_rows)
        embeddings.append(values)

    loaded = ArmEmbeddings(
        method=table["method"].iloc[0],
        arm=table["arm"].iloc[0],
        sample_ids=np.concatenate(sample_ids),
        cell_ids=np.concatenate(cell_ids),
        row_ids=np.concatenate(row_ids),
        embeddings=np.concatenate(embeddings),
    )
    if len(np.unique(loaded.cell_ids)) != len(loaded.cell_ids):
        raise ValueError(f"{path} contains duplicate cell IDs")
    if len(np.unique(loaded.row_ids)) != len(loaded.row_ids):
        raise ValueError(f"{path} contains duplicate row IDs")
    return loaded


def load_manifest_sample(path, sample_id) -> ArmEmbeddings:
    """Load one sample from an arm manifest without concatenating the cohort."""
    path, table = _read_manifest(path)
    selected = table.loc[table["sample_id"] == str(sample_id)]
    if len(selected) != 1:
        raise ValueError(
            f"{path} contains {len(selected)} rows for sample {sample_id!r}"
        )
    row = next(selected.itertuples(index=False))
    cell_ids, row_ids, embeddings = _load_record(path, row)
    loaded = ArmEmbeddings(
        method=row.method,
        arm=row.arm,
        sample_ids=np.repeat(row.sample_id, row.n_cells),
        cell_ids=cell_ids,
        row_ids=row_ids,
        embeddings=embeddings,
    )
    if len(np.unique(loaded.cell_ids)) != len(loaded.cell_ids):
        raise ValueError(f"{path} contains duplicate cell IDs for {sample_id}")
    if len(np.unique(loaded.row_ids)) != len(loaded.row_ids):
        raise ValueError(f"{path} contains duplicate row IDs for {sample_id}")
    return loaded
