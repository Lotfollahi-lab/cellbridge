"""
I/O and cell-ID alignment helpers for the linear-probe experiment.
"""

import numpy as np


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


def load_coords(adata_path, key="spatial"):
    """Return ``(coords, cell_ids)`` from ``adata.obsm[key]`` for the spatial split."""
    import anndata as ad

    adata = ad.read_h5ad(adata_path, backed="r")
    try:
        if key not in adata.obsm:
            raise KeyError(
                f"{adata_path}: obsm['{key}'] missing (have {list(adata.obsm)})"
            )
        coords = np.asarray(adata.obsm[key], dtype=np.float64)
        cell_ids = np.asarray(adata.obs_names).astype(str)
    finally:
        adata.file.close()
    return coords, cell_ids
