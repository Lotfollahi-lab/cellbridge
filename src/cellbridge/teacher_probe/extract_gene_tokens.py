"""Extract selected contextual gene-token embeddings with installed TERRA."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pickle
import shutil
from pathlib import Path

import numpy as np

from cellbridge.teacher_probe.config import load_teacher_probe_config
from cellbridge.teacher_probe.provenance import model_metadata
from cellbridge.teacher_probe.store import read_token_store_header, write_token_store


def stable_cell_subset(cell_ids, maximum, seed):
    """Select cells without using labels, while retaining source row order."""
    cell_ids = np.asarray(cell_ids).astype(str)
    if maximum >= len(cell_ids):
        return np.arange(len(cell_ids), dtype=int)
    scores = np.fromiter(
        (
            int.from_bytes(
                hashlib.blake2b(f"{seed}:{cell_id}".encode(), digest_size=8).digest(),
                "little",
            )
            for cell_id in cell_ids
        ),
        dtype=np.uint64,
        count=len(cell_ids),
    )
    return np.sort(np.argpartition(scores, maximum - 1)[:maximum])


def token_presence(dataset, token_ids, seq_len_cell, batch_size=4096):
    """Read the actual tokenized cell segment and return an explicit mask."""
    token_ids = np.asarray(token_ids)
    presence = np.zeros((len(dataset), len(token_ids)), dtype=bool)
    if "gene_tokens" not in dataset.column_names:
        raise KeyError(
            f"tokenized dataset has no gene_tokens; keys={dataset.column_names}"
        )
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        values = dataset[start:stop]["gene_tokens"]
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        cell_tokens = np.asarray(values)[:, :seq_len_cell]
        presence[start:stop] = (cell_tokens[:, :, None] == token_ids).any(axis=1)
    return presence


def extract(cfg, *, overwrite=False):
    output_path = Path(cfg["active_gene_tokens_path"])
    if output_path.exists() and not overwrite:
        header = read_token_store_header(output_path)
        if header["genes"].tolist() != cfg["genes"]:
            raise ValueError(f"{output_path}: cached genes differ from config")
        metadata = header["metadata"]
        expected = {
            "cell count": (len(header["cell_ids"]), int(cfg["active_max_cells"])),
            "model revision": (
                metadata.get("model_revision"),
                cfg["extraction"]["model_revision"],
            ),
            "TERRA version": (
                metadata.get("terra_version"),
                str(cfg["extraction"]["terra_version"]),
            ),
            # Both of these change which cells the store holds, or which counts
            # they were tokenized from, while leaving the cell count identical,
            # so omitting them would reuse a store that answers a different
            # question than the config now asks.
            "selection seed": (
                (metadata.get("selection") or {}).get("seed"),
                int(cfg["selection"].get("seed", 0)),
            ),
            "harmonization": (
                metadata.get("harmonization"),
                {
                    "min_genes_per_cell": int(cfg["extraction"]["min_genes_per_cell"]),
                    "min_cells_per_gene": int(cfg["extraction"]["min_cells_per_gene"]),
                },
            ),
        }
        for name, (observed, wanted) in expected.items():
            if observed != wanted:
                raise ValueError(
                    f"{output_path}: cached {name} {observed!r}, expected {wanted!r}"
                )
        print(f"reusing {output_path} ({len(header['cell_ids']):,} cells)")
        return output_path

    import anndata as ad
    import yaml
    from datasets import load_from_disk
    import terra
    from terra import (
        download_pretrained,
        get_gene_embed,
        harmonize_adata,
        tokenize_adata,
    )

    available_workers = len(os.sched_getaffinity(0))
    tokenize_workers = min(int(cfg["extraction"]["nproc"]), available_workers)
    loader_workers = min(int(cfg["extraction"]["num_workers"]), available_workers)
    cfg["extraction"]["effective_nproc"] = tokenize_workers
    cfg["extraction"]["effective_num_workers"] = loader_workers

    expected_version = str(cfg["extraction"]["terra_version"])
    actual_version = importlib.metadata.version("terra-st")
    if actual_version != expected_version:
        raise RuntimeError(
            f"TERRA version {actual_version}, expected {expected_version}"
        )
    model_dir = Path(
        download_pretrained(
            cfg["extraction"]["model_name"],
            revision=cfg["extraction"]["model_revision"],
        )
    )
    resolved_revision = model_dir.resolve().name
    if resolved_revision != cfg["extraction"]["model_revision"]:
        raise RuntimeError(
            f"resolved model revision {resolved_revision}, "
            f"expected {cfg['extraction']['model_revision']}"
        )
    with open(model_dir / "ensembl_dictionary.pkl", "rb") as handle:
        ensembl_dictionary = pickle.load(handle)
    with open(model_dir / "token_dictionary.pkl", "rb") as handle:
        token_dictionary = pickle.load(handle)
    missing = [gene for gene in cfg["genes"] if gene not in ensembl_dictionary]
    if missing:
        raise KeyError(f"selected genes missing from Ensembl mapping: {missing}")
    ensembl_ids = [ensembl_dictionary[gene] for gene in cfg["genes"]]
    missing = [
        gene
        for gene, ensembl in zip(cfg["genes"], ensembl_ids)
        if ensembl not in token_dictionary
    ]
    if missing:
        raise KeyError(f"selected genes missing from TERRA token dictionary: {missing}")

    tokenized_path = Path(cfg["extraction"]["tokenized_path"])
    if tokenized_path.exists():
        dataset = load_from_disk(str(tokenized_path))
    else:
        adata = ad.read_h5ad(cfg["adata_path"])
        harmonized = harmonize_adata(
            adata,
            gene_mapping_dict_file_path=str(model_dir / "ensembl_dictionary.pkl"),
            gene_occurrence_count_file_path=str(
                model_dir / "gene_count_dictionary.pkl"
            ),
            min_cells_per_gene=int(cfg["extraction"]["min_cells_per_gene"]),
            min_genes_per_cell=int(cfg["extraction"]["min_genes_per_cell"]),
        )
        tokenized_path.parent.mkdir(parents=True, exist_ok=True)
        dataset = tokenize_adata(
            adata=harmonized,
            model_folder_path=str(model_dir),
            cache_directory_path=str(
                tokenized_path.with_name(tokenized_path.name + "_cache")
            ),
            nproc=tokenize_workers,
        )
        dataset.save_to_disk(str(tokenized_path))

    full_ids = np.asarray(dataset["cell_id"]).astype(str)
    if len(np.unique(full_ids)) != len(full_ids):
        raise ValueError("tokenized TERRA dataset has duplicate cell IDs")
    with np.load(cfg["terra_embeddings"], allow_pickle=False) as pooled:
        pooled_ids = np.asarray(pooled["cell_ids"]).astype(str)
    if set(full_ids) != set(pooled_ids):
        raise ValueError("new tokenization and cached pooled TERRA cells differ")
    seed = int(cfg["selection"].get("seed", 0))
    selected = stable_cell_subset(full_ids, int(cfg["active_max_cells"]), seed)
    dataset = dataset.select(selected.tolist())
    cell_ids = full_ids[selected]

    with open(model_dir / "model_config.yaml") as handle:
        model_config = yaml.safe_load(handle)
    presence = token_presence(
        dataset,
        [token_dictionary[ensembl] for ensembl in ensembl_ids],
        int(model_config["data"]["seq_len_cell"]),
    )
    outputs = get_gene_embed(
        dataset=dataset,
        model_folder_path=str(model_dir),
        cell_gene_ensembl_id=ensembl_ids,
        neighborhood_gene_ensembl_id=[],
        batch_size=int(cfg["extraction"]["batch_size"]),
        pin_memory=bool(cfg["extraction"].get("pin_memory", True)),
        num_workers=loader_workers,
        include_spatial_cell_emb=True,
    )
    expected_keys = {
        f"{prefix}_gene{ensembl}"
        for ensembl in ensembl_ids
        for prefix in ("cell_emb", "spatial_cell_emb")
    }
    if set(outputs) != expected_keys:
        raise KeyError(f"unexpected TERRA output keys: {sorted(outputs)}")
    spatial = {
        gene: outputs[f"spatial_cell_emb_gene{ensembl}"]
        for gene, ensembl in zip(cfg["genes"], ensembl_ids)
    }
    for column, ensembl in enumerate(ensembl_ids):
        plain = np.asarray(outputs[f"cell_emb_gene{ensembl}"])
        if np.any(plain[~presence[:, column]] != 0):
            raise ValueError(f"{ensembl}: TERRA cell-token absence mask mismatch")

    metadata = model_metadata(model_dir, cfg, terra, outputs.keys())
    metadata["model_config"] = model_config
    write_token_store(
        output_path,
        cell_ids=cell_ids,
        genes=cfg["genes"],
        ensembl_ids=ensembl_ids,
        presence=presence,
        spatial_embeddings=spatial,
        metadata=metadata,
    )
    if not bool(cfg["extraction"].get("keep_tokenized", True)):
        shutil.rmtree(tokenized_path, ignore_errors=True)
        shutil.rmtree(
            tokenized_path.with_name(tokenized_path.name + "_cache"), ignore_errors=True
        )
    print(json.dumps({"path": str(output_path), "cells": len(cell_ids)}, indent=2))
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_teacher_probe_config(args.config, smoke=args.smoke)
    extract(cfg, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
