"""Exact installed-package and model-bundle provenance for token extraction."""

import hashlib
import importlib.metadata
from pathlib import Path


def file_sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def model_metadata(model_dir, cfg, terra_module, api_keys):
    files = ("model_checkpoint.pt", "model_config.yaml", "token_dictionary.pkl")
    return {
        "terra_version": importlib.metadata.version("terra-st"),
        "terra_module": str(Path(terra_module.__file__).resolve()),
        "model_name": cfg["extraction"]["model_name"],
        "model_revision": cfg["extraction"]["model_revision"],
        "resolved_model_dir": str(Path(model_dir).resolve()),
        "file_sha256": {name: file_sha256(Path(model_dir) / name) for name in files},
        "harmonization": {
            "min_genes_per_cell": int(cfg["extraction"]["min_genes_per_cell"]),
            "min_cells_per_gene": int(cfg["extraction"]["min_cells_per_gene"]),
        },
        "embedding": {
            "api": "terra.get_gene_embed",
            "api_output_keys": sorted(api_keys),
            "context": "spatial_cell_emb_gene<ENSEMBL_ID>",
            "batch_size": int(cfg["extraction"]["batch_size"]),
            "tokenize_workers": int(cfg["extraction"]["effective_nproc"]),
            "embedding_workers": int(cfg["extraction"]["effective_num_workers"]),
        },
        "sample_id": cfg["sample_id"],
        "selection": {
            "method": "blake2b_cell_id",
            "seed": int(cfg["selection"].get("seed", 0)),
            "max_cells": int(cfg["active_max_cells"]),
            "smoke": bool(cfg["smoke_mode"]),
        },
    }
