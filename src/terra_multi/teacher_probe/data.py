"""Strict loading and cell-ID alignment for the teacher-only probe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from terra_multi.teacher_probe.neighborhood import neighborhood_expression
from terra_multi.teacher_probe.store import read_token_store_header


@dataclass(frozen=True)
class ProbeData:
    cell_ids: np.ndarray
    labels: np.ndarray
    coords_um: np.ndarray
    cell_emb: np.ndarray
    expression: np.ndarray
    presence: np.ndarray
    genes: np.ndarray
    store_rows: np.ndarray
    token_store_path: Path
    token_store_signature: str
    alignment_report: dict


def _unique_ids(values, source):
    values = np.asarray(values).astype(str)
    if values.ndim != 1 or np.any(values == ""):
        raise ValueError(f"{source}: cell IDs must be a nonempty 1D array")
    duplicated = pd.Index(values)[pd.Index(values).duplicated()].unique()
    if len(duplicated):
        raise ValueError(
            f"{source}: duplicate cell IDs including {duplicated[:5].tolist()}"
        )
    return values


def _positions(reference, requested, source):
    position = {cell_id: row for row, cell_id in enumerate(reference)}
    missing = [cell_id for cell_id in requested if cell_id not in position]
    if missing:
        raise KeyError(
            f"{source}: missing {len(missing)} cell IDs including {missing[:5]}"
        )
    return np.asarray([position[cell_id] for cell_id in requested], dtype=int)


def _load_labels(path, id_column, label_column):
    path = Path(path)
    table = (
        pd.read_parquet(path)
        if path.suffix in {".parquet", ".pq"}
        else pd.read_csv(path)
    )
    missing = [name for name in (id_column, label_column) if name not in table]
    if missing:
        raise KeyError(f"{path}: missing columns {missing}")
    if table[[id_column, label_column]].isna().any().any():
        raise ValueError(f"{path}: null cell IDs or labels")
    ids = _unique_ids(table[id_column].astype(str), path)
    return ids, table[label_column].astype(str).to_numpy()


def _selected_expression(adata, rows, genes):
    var_names = _unique_ids(adata.var_names, "AnnData var_names")
    columns = _positions(var_names, genes, "AnnData genes")
    values = adata.X[rows][:, columns]
    values = values.toarray() if sp.issparse(values) else np.asarray(values)
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("selected raw counts must be finite and nonnegative")
    if not np.allclose(values, np.rint(values)):
        raise ValueError("AnnData.X does not contain raw integer counts")
    if "total_counts" in adata.obs:
        totals = adata.obs["total_counts"].to_numpy(dtype=np.float64)[rows]
    else:
        full = adata.X[rows]
        totals = np.asarray(full.sum(axis=1)).ravel()
    if not np.isfinite(totals).all() or np.any(totals <= 0):
        raise ValueError("selected cells have invalid total counts")
    expression = np.log1p(values * (10_000.0 / totals[:, None])).astype(np.float32)
    return values, expression


def load_probe_data(cfg):
    """Align pooled embeddings, counts, labels, coordinates, and gene tokens."""
    token_path = Path(cfg["active_gene_tokens_path"])
    header = read_token_store_header(token_path)
    token_ids = _unique_ids(header["cell_ids"], token_path)
    genes = np.asarray(cfg["genes"])
    if not np.array_equal(header["genes"], genes):
        raise ValueError("token-store genes do not exactly match configured gene order")
    if len(token_ids) != int(cfg["active_max_cells"]):
        raise ValueError(
            f"token store has {len(token_ids)} cells, expected {cfg['active_max_cells']}"
        )

    with np.load(cfg["terra_embeddings"], allow_pickle=False) as terra:
        required = {"cell_ids", "cell_emb", "he_x", "he_y"}
        if not required.issubset(terra.files):
            raise KeyError(
                f"{cfg['terra_embeddings']}: missing {sorted(required - set(terra.files))}"
            )
        terra_ids = _unique_ids(terra["cell_ids"], cfg["terra_embeddings"])
        terra_rows = _positions(terra_ids, token_ids, "pooled TERRA embeddings")
        cell_emb = np.asarray(terra["cell_emb"][terra_rows], dtype=np.float32)
        pooled_coords = np.column_stack(
            (terra["he_x"][terra_rows], terra["he_y"][terra_rows])
        )
    if cell_emb.ndim != 2 or not np.isfinite(cell_emb).all():
        raise ValueError("pooled TERRA cell embeddings must be a finite matrix")

    adata = ad.read_h5ad(cfg["adata_path"])
    adata_ids = _unique_ids(adata.obs_names, cfg["adata_path"])
    if "cell_id" in adata.obs and not np.array_equal(
        adata.obs["cell_id"].astype(str).to_numpy(), adata_ids
    ):
        raise ValueError("AnnData obs cell_id does not match obs_names")
    adata_rows = _positions(adata_ids, token_ids, "raw-count AnnData")
    if "spatial" not in adata.obsm:
        raise KeyError(f"{cfg['adata_path']}: obsm['spatial'] is missing")
    # obsm["spatial"] is already in microns; it is a rigid rotation of the
    # H&E-pixel he_x/he_y that TERRA carries, so the two frames agree on
    # distances but never on values. Only their row alignment is checkable here.
    coords = np.asarray(adata.obsm["spatial"][adata_rows], dtype=np.float64)
    if coords.shape != pooled_coords.shape:
        raise ValueError("AnnData and pooled TERRA coordinates differ in shape")
    raw_expression, expression = _selected_expression(adata, adata_rows, genes)
    neighborhood_report = None
    if cfg["eval"].get("expression_source", "focal") == "neighborhood":
        universe_rows = _positions(adata_ids, terra_ids, "neighborhood universe")
        _, universe_expression = _selected_expression(adata, universe_rows, genes)
        universe_coords = np.asarray(
            adata.obsm["spatial"][universe_rows], dtype=np.float64
        )
        expression, neighborhood_report = neighborhood_expression(
            terra_ids,
            universe_coords,
            universe_expression,
            genes,
            cfg["neighborhood_expression"]["n_neighbors"],
            cfg["neighborhood_expression"]["cache_path"],
        )
        expression = expression[terra_rows]

    eval_cfg = cfg["eval"]
    label_ids, label_values = _load_labels(
        cfg["labels_path"],
        eval_cfg.get("label_id_column", "cell_id"),
        eval_cfg.get("label_column", "group"),
    )
    label_rows = _positions(label_ids, token_ids, "phenotype labels")
    labels = label_values[label_rows]

    presence = np.asarray(header["presence"], dtype=bool)
    if presence.shape != raw_expression.shape:
        raise ValueError("gene-token presence and selected expression shapes differ")
    invalid_presence = presence & (raw_expression <= 0)
    if invalid_presence.any():
        row, column = np.argwhere(invalid_presence)[0]
        raise ValueError(
            f"token {genes[column]} is present with zero raw count at {token_ids[row]}"
        )

    counts = pd.Series(labels).value_counts()
    minimum = int(eval_cfg.get("min_class_count", 25))
    keep_classes = set(counts.index[counts >= minimum])
    keep = np.asarray([label in keep_classes for label in labels])
    if len(keep_classes) < 2 or keep.sum() < 2:
        raise ValueError("fewer than two phenotype classes survive filtering")

    report = {
        "sample_id": cfg["sample_id"],
        "n_token_cells": int(len(token_ids)),
        "n_pooled_cells": int(len(terra_ids)),
        "n_count_cells": int(len(adata_ids)),
        "n_label_cells": int(len(label_ids)),
        "n_pooled_cells_not_selected": int(len(set(terra_ids) - set(token_ids))),
        "n_count_cells_not_selected": int(len(set(adata_ids) - set(token_ids))),
        "n_label_cells_not_selected": int(len(set(label_ids) - set(token_ids))),
        "n_evaluation_cells": int(keep.sum()),
        "n_classes": int(len(keep_classes)),
        "dropped_rare_class_cells": int((~keep).sum()),
        "expression_source": eval_cfg.get("expression_source", "focal"),
        "neighborhood_expression": neighborhood_report,
        "token_presence_by_gene": {
            gene: int(presence[keep, column].sum()) for column, gene in enumerate(genes)
        },
        "expressed_but_truncated_by_gene": {
            gene: int(
                ((raw_expression[:, column] > 0) & ~presence[:, column])[keep].sum()
            )
            for column, gene in enumerate(genes)
        },
    }
    return ProbeData(
        cell_ids=token_ids[keep],
        labels=labels[keep],
        coords_um=coords[keep],
        cell_emb=cell_emb[keep],
        expression=expression[keep],
        presence=presence[keep],
        genes=genes,
        store_rows=np.flatnonzero(keep),
        token_store_path=token_path,
        token_store_signature=header["signature"],
        alignment_report=report,
    )
