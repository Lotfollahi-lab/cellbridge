"""Sparse on-disk storage for selected contextual gene-token embeddings."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import h5py
import numpy as np

FORMAT = "terra_contextual_gene_tokens_v1"


def _strings(values):
    return np.asarray(values).astype(str)


def token_store_signature(cell_ids, genes, metadata):
    digest = hashlib.sha256()
    for value in _strings(cell_ids):
        digest.update(value.encode())
        digest.update(b"\0")
    digest.update(json.dumps(list(genes), separators=(",", ":")).encode())
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def write_token_store(
    path,
    *,
    cell_ids,
    genes,
    ensembl_ids,
    presence,
    spatial_embeddings,
    metadata,
):
    """Atomically write dense API outputs as sparse per-gene HDF5 datasets."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cell_ids, genes, ensembl_ids = map(_strings, (cell_ids, genes, ensembl_ids))
    presence = np.asarray(presence, dtype=bool)
    if presence.shape != (len(cell_ids), len(genes)):
        raise ValueError(f"invalid presence shape {presence.shape}")
    if len(np.unique(cell_ids)) != len(cell_ids):
        raise ValueError("token store cell IDs are not unique")
    signature = token_store_signature(cell_ids, genes, metadata)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    string_type = h5py.string_dtype("utf-8")
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["format"] = FORMAT
            handle.attrs["signature"] = signature
            handle.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
            handle.create_dataset(
                "cell_ids", data=cell_ids.astype(object), dtype=string_type
            )
            handle.create_dataset("genes", data=genes.astype(object), dtype=string_type)
            handle.create_dataset(
                "ensembl_ids", data=ensembl_ids.astype(object), dtype=string_type
            )
            handle.create_dataset("presence", data=presence, compression="gzip")
            token_group = handle.create_group("tokens")
            for column, gene in enumerate(genes):
                values = np.asarray(spatial_embeddings[str(gene)], dtype=np.float32)
                if values.ndim != 2 or values.shape[0] != len(cell_ids):
                    raise ValueError(f"{gene}: invalid token shape {values.shape}")
                if not np.isfinite(values).all():
                    raise ValueError(f"{gene}: non-finite token embeddings")
                rows = np.flatnonzero(presence[:, column])
                absent = ~presence[:, column]
                if np.any(values[absent] != 0):
                    raise ValueError(f"{gene}: absent token rows must be exactly zero")
                if len(rows) and np.any(np.linalg.norm(values[rows], axis=1) == 0):
                    raise ValueError(f"{gene}: present token rows must be nonzero")
                group = token_group.create_group(str(gene))
                group.create_dataset("rows", data=rows, compression="gzip")
                group.create_dataset(
                    "embeddings", data=values[rows], compression="gzip", shuffle=True
                )
            handle.flush()
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return signature


def read_token_store_header(path):
    """Load validated identifiers, presence, metadata, and signature."""
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("format") != FORMAT:
            raise ValueError(f"{path}: unsupported token-store format")
        header = {
            "cell_ids": handle["cell_ids"].asstr()[:],
            "genes": handle["genes"].asstr()[:],
            "ensembl_ids": handle["ensembl_ids"].asstr()[:],
            "presence": handle["presence"][:].astype(bool),
            "metadata": json.loads(handle.attrs["metadata_json"]),
            "signature": str(handle.attrs["signature"]),
        }
    if header["presence"].shape != (len(header["cell_ids"]), len(header["genes"])):
        raise ValueError(f"{path}: token presence has an invalid shape")
    return header


def read_gene_embeddings(path, gene, selected_store_rows):
    """Return selected-row indices and embeddings for one present gene token."""
    selected_store_rows = np.asarray(selected_store_rows, dtype=int)
    with h5py.File(path, "r") as handle:
        if gene not in handle["tokens"]:
            raise KeyError(f"{path}: gene {gene!r} is missing")
        rows = handle[f"tokens/{gene}/rows"][:]
        embeddings = handle[f"tokens/{gene}/embeddings"][:].astype(np.float32)
        n_store = len(handle["cell_ids"])
    inverse = np.full(n_store, -1, dtype=int)
    inverse[selected_store_rows] = np.arange(len(selected_store_rows))
    keep = inverse[rows] >= 0
    return inverse[rows[keep]], embeddings[keep]
