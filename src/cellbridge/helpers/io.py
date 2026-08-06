"""
I/O and cell-ID alignment helpers for the linear-probe experiment.
"""

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp


def load_embeddings(path, key):
    """Load an embedding ``.npz`` written by the notebooks.

    Returns ``(embeddings, cell_ids)`` where ``embeddings`` is a float32
    ``(N, D)`` array and ``cell_ids`` an ``(N,)`` array of *strings*.
    """
    with np.load(path, allow_pickle=True) as data:
        emb = np.asarray(data[key], dtype=np.float32)
        cell_ids = np.asarray(data["cell_ids"]).astype(str)
    return emb, cell_ids


def align_by_cell_id(pairs):
    id_arrays = []
    id_sets = []
    for i, (emb, cell_ids) in enumerate(pairs):
        cell_ids = np.asarray(cell_ids).astype(str)
        if emb.shape[0] != cell_ids.shape[0]:
            raise ValueError(
                f"pair {i}: {emb.shape[0]} rows but {cell_ids.shape[0]} cell_ids"
            )
        if len(np.unique(cell_ids)) != cell_ids.shape[0]:
            raise ValueError(f"pair {i}: duplicate cell_ids")
        id_arrays.append(cell_ids)
        id_sets.append(set(cell_ids.tolist()))

    shared = set.intersection(*id_sets)
    if not shared:
        raise ValueError("align_by_cell_id: cell_ids have an empty intersection")
    shared_ids = np.array(sorted(shared))

    aligned = []
    for (emb, _), cell_ids in zip(pairs, id_arrays):
        position = {cid: row for row, cid in enumerate(cell_ids)}
        rows = np.array([position[cid] for cid in shared_ids], dtype=int)
        aligned.append(np.asarray(emb)[rows])
    return aligned, shared_ids


# TODO: this selects HVGs over all cells, including the held-out test region, so
# the choice of target genes leaks test information into the evaluation (mild)
def select_hvgs(adata_path, n_top):
    """Top-``n_top`` highly variable genes via ``seurat_v3`` on raw counts."""
    adata = ad.read_h5ad(adata_path)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=int(n_top))
    return adata.var_names[adata.var["highly_variable"]].tolist()


def load_targets(adata_path, cell_ids, gene_list):
    """log1p-normalized expression, rows ordered to match ``cell_ids``.

    Returns a dense float32 ``(len(cell_ids), len(gene_list))`` array. Library-size
    normalization is computed over the full gene panel *before* subsetting to
    ``gene_list``, so it does not depend on which genes the probe predicts.
    """
    adata = ad.read_h5ad(adata_path)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    var_index = {g: j for j, g in enumerate(adata.var_names.astype(str))}
    missing_genes = [g for g in gene_list if g not in var_index]
    if missing_genes:
        raise KeyError(
            f"{len(missing_genes)} gene(s) not in {adata_path}: {missing_genes[:5]}"
        )
    cols = np.array([var_index[g] for g in gene_list], dtype=int)

    obs_index = {c: i for i, c in enumerate(adata.obs_names.astype(str))}
    cell_ids = np.asarray(cell_ids).astype(str)
    missing_cells = [c for c in cell_ids if c not in obs_index]
    if missing_cells:
        raise KeyError(
            f"{len(missing_cells)} cell_id(s) not in {adata_path}: {missing_cells[:5]}"
        )
    rows = np.array([obs_index[c] for c in cell_ids], dtype=int)

    X = adata.X
    X = X.tocsr() if sp.issparse(X) else np.asarray(X)
    Y = X[rows][:, cols]
    Y = Y.toarray() if sp.issparse(Y) else np.asarray(Y)
    return Y.astype(np.float32)


def load_coords(adata_path, key="spatial"):
    """Return ``(coords, cell_ids)`` from ``adata.obsm[key]`` for the spatial split."""
    adata = ad.read_h5ad(adata_path)
    if key not in adata.obsm:
        raise KeyError(f"{adata_path}: obsm['{key}'] missing (have {list(adata.obsm)})")
    coords = np.asarray(adata.obsm[key], dtype=np.float64)
    cell_ids = np.asarray(adata.obs_names).astype(str)
    return coords, cell_ids
