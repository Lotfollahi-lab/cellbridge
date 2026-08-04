"""Cached selected-gene expression over TERRA's spatial neighborhood."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
from sklearn.neighbors import NearestNeighbors


def _signature(cell_ids, coords, expression, genes, n_neighbors):
    digest = hashlib.sha256()
    digest.update(np.asarray(cell_ids).astype("S").tobytes())
    digest.update(np.asarray(coords, dtype=np.float64).tobytes())
    digest.update(np.asarray(expression, dtype=np.float32).tobytes())
    digest.update(np.asarray(genes).astype("S").tobytes())
    digest.update(str(int(n_neighbors)).encode())
    return digest.hexdigest()


def _load_cache(path, signature, cell_ids, genes):
    if not Path(path).exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as cache:
            if str(cache["signature"].item()) != signature:
                return None
            if not np.array_equal(cache["cell_ids"].astype(str), cell_ids):
                return None
            if not np.array_equal(cache["genes"].astype(str), genes):
                return None
            expression = np.asarray(cache["expression"], dtype=np.float32)
            diagnostics = {
                "n_neighbors": int(cache["n_neighbors"].item()),
                "mean_neighbor_distance_um": float(cache["mean_distance"].item()),
                "p95_neighbor_distance_um": float(cache["p95_distance"].item()),
                "max_neighbor_distance_um": float(cache["max_distance"].item()),
                "cache_reused": True,
            }
    except (OSError, ValueError, KeyError):
        return None
    return expression, diagnostics


def _save_cache(path, signature, cell_ids, genes, expression, distances, n_neighbors):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid4().hex}.npz")
    try:
        np.savez_compressed(
            temporary,
            signature=np.asarray(signature),
            cell_ids=np.asarray(cell_ids).astype("S"),
            genes=np.asarray(genes).astype("S"),
            expression=np.asarray(expression, dtype=np.float32),
            n_neighbors=np.asarray(n_neighbors),
            mean_distance=np.asarray(distances.mean()),
            p95_distance=np.asarray(np.quantile(distances, 0.95)),
            max_distance=np.asarray(distances.max()),
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def neighborhood_expression(
    cell_ids, coords_um, expression, genes, n_neighbors, cache_path
):
    """Average expression over the exact kNN convention used by TERRA."""
    cell_ids = np.asarray(cell_ids).astype(str)
    coords = np.asarray(coords_um, dtype=np.float64)
    expression = np.asarray(expression, dtype=np.float32)
    genes = np.asarray(genes).astype(str)
    n_neighbors = int(n_neighbors)
    expected = (len(cell_ids), len(genes))
    if coords.shape != (len(cell_ids), 2) or not np.isfinite(coords).all():
        raise ValueError("neighborhood coordinates must be a finite (N, 2) matrix")
    if expression.shape != expected or not np.isfinite(expression).all():
        raise ValueError(f"neighborhood expression must have shape {expected}")
    if n_neighbors < 1 or len(cell_ids) <= n_neighbors:
        raise ValueError("invalid neighborhood size")

    signature = _signature(cell_ids, coords, expression, genes, n_neighbors)
    cached = _load_cache(cache_path, signature, cell_ids, genes)
    if cached is not None:
        values, diagnostics = cached
        if values.shape != expected or not np.isfinite(values).all():
            raise ValueError(f"{cache_path}: invalid cached neighborhood expression")
        return values, diagnostics

    # Squidpy's generic KNNBuilder, used by the pinned TERRA tokenizer, calls
    # sklearn exactly this way. kneighbors(X=None) excludes each training row
    # from its own neighbors.
    model = NearestNeighbors(n_neighbors=n_neighbors, radius=1, metric="euclidean")
    model.fit(coords)
    distances, neighbors = model.kneighbors()
    if neighbors.shape != (len(cell_ids), n_neighbors):
        raise RuntimeError("unexpected spatial-neighbor shape")
    if np.any(neighbors == np.arange(len(cell_ids))[:, None]):
        raise RuntimeError("focal cells leaked into their own neighborhoods")
    if np.any(np.diff(np.sort(neighbors, axis=1), axis=1) == 0):
        raise RuntimeError("a spatial neighborhood contains duplicate cells")

    values = expression[neighbors].mean(axis=1, dtype=np.float32)
    _save_cache(
        cache_path,
        signature,
        cell_ids,
        genes,
        values,
        distances,
        n_neighbors,
    )
    diagnostics = {
        "n_neighbors": n_neighbors,
        "mean_neighbor_distance_um": float(distances.mean()),
        "p95_neighbor_distance_um": float(np.quantile(distances, 0.95)),
        "max_neighbor_distance_um": float(distances.max()),
        "cache_reused": False,
    }
    return values, diagnostics
