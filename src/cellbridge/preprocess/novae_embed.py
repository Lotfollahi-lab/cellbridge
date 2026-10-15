"""Extract frozen NOVAE teacher vectors for the existing distillation cohort."""

import argparse
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import anndata as ad
import numpy as np
import pandas as pd

from cellbridge.config import load_config
from cellbridge.helpers.cohorts import load_split_table
from cellbridge.train.lora.data import canonical_id


def sample_ids(config):
    splits = load_split_table(
        config["splits_path"], required_splits=("distill_train", "distill_val")
    )
    return splits.loc[
        splits["split"].isin(("distill_train", "distill_val")), "sample_id"
    ].tolist()


def input_paths(config, sample_id):
    if sample_id not in sample_ids(config):
        raise ValueError(f"{sample_id} is not a distillation train/validation sample")
    manifest = pd.read_csv(config["manifest_path"], dtype={"id": str}).set_index("id")
    if not manifest.index.is_unique:
        raise ValueError("manifest has duplicate sample IDs")
    row = manifest.loc[sample_id]
    directory = row.get("output_dir")
    directory = (
        Path(config["data_root"]) / sample_id
        if directory is None or pd.isna(directory)
        else Path(directory)
    )
    return directory / f"{sample_id}.h5ad", directory / f"{sample_id}_terra.npz"


def prepare_input(counts_path, reference_path):
    """Use baseline cells/order and micron coordinates; never read image features."""
    with np.load(reference_path, allow_pickle=True) as data:
        ids = np.asarray([canonical_id(value) for value in data["cell_ids"]])
        centers = np.column_stack((data["he_x"], data["he_y"])).astype(np.float64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("reference cell IDs are not unique")
    if centers.shape != (len(ids), 2) or not np.isfinite(centers).all():
        raise ValueError("reference H&E centers are invalid")
    counts = ad.read_h5ad(counts_path)
    counts.obs_names = [canonical_id(value) for value in counts.obs_names]
    if not counts.obs_names.is_unique:
        raise ValueError("reconstructed counts contain duplicate cell IDs")
    positions = counts.obs_names.get_indexer(ids)
    if (positions < 0).any():
        raise ValueError(f"counts are missing {int((positions < 0).sum())} reference cells")
    spatial = np.asarray(counts.obsm["spatial"])[positions].astype(np.float64)
    if spatial.shape != (len(ids), 2) or not np.isfinite(spatial).all():
        raise ValueError("transcriptomics coordinates must be finite micron coordinates")
    adata = ad.AnnData(
        X=counts.X[positions].copy(),
        obs=pd.DataFrame(index=pd.Index(ids, name="cell_id")),
        var=counts.var.copy(),
        obsm={"spatial": spatial},
    )
    return adata, ids, centers


def cached_model_path(config):
    from huggingface_hub import snapshot_download

    model = config["model"]
    return Path(snapshot_download(
        model["name"], revision=model["revision"], local_files_only=True,
    ))


def load_model(config):
    import novae
    import torch

    if version("novae") != "1.1.1":
        raise ValueError("this experiment is pinned to novae==1.1.1")
    seed = int(config["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    novae.settings.disable_multimodal = True
    novae.settings.scale_to_microns = 1.0
    model = novae.Novae.from_pretrained(
        str(cached_model_path(config)), batch_size=int(config["model"]["batch_size"]),
    )
    if model.hparams.output_size != int(config["model"]["output_dim"]):
        raise ValueError("cached NOVAE output dimension does not match configuration")
    model.requires_grad_(False)
    model.eval()
    return model


def embed(model, adata, config, device):
    import novae
    import scanpy as sc

    # Make raw-count normalization explicit instead of relying on a max-value heuristic.
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    novae.spatial_neighbors(adata, radius=float(config["radius_um"]))
    # Both modes use the same frozen encoder. False skips zero-shot k-means;
    # prototype/domain assignments are not used as distillation targets.
    model.compute_representations(
        adata, zero_shot=False, accelerator=device,
        num_workers=int(config["num_workers"]),
    )


def export_sample(output, sample_id, adata, reference_ids, centers, config):
    ids = np.asarray(adata.obs_names).astype(str)
    if not np.array_equal(ids, reference_ids):
        raise ValueError("NOVAE changed cell identity or row order")
    values = np.asarray(adata.obsm["novae_latent"], dtype=np.float32)
    if values.shape != (len(ids), int(config["model"]["output_dim"])):
        raise ValueError(f"unexpected NOVAE shape: {values.shape}")
    valid = adata.obs["neighborhood_valid"].to_numpy(dtype=bool)
    if not valid.any():
        raise ValueError("NOVAE produced no valid neighborhoods")
    if not np.isfinite(values[valid]).all() or np.any(np.linalg.norm(values[valid], axis=1) <= 0):
        raise ValueError("valid NOVAE vectors must be finite and nonzero")
    summary = {
        "sample_id": sample_id,
        "teacher": config["model"]["name"],
        "revision": config["model"]["revision"],
        "novae_version": version("novae"),
        "radius_um": config["radius_um"],
        "n_reference_cells": len(ids),
        "n_embedded_cells": int(valid.sum()),
        "n_invalid_neighborhoods": int((~valid).sum()),
        "n_input_genes": adata.n_vars,
        "reference_cell_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.npz")
    try:
        np.savez_compressed(
            temporary, cell_ids=ids[valid], cell_emb=values[valid],
            he_x=centers[valid, 0], he_y=centers[valid, 1],
            reference_cell_ids=ids, excluded_cell_ids=ids[~valid],
            provenance=np.asarray(json.dumps(summary, sort_keys=True)),
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    config = load_config(args.config)
    counts_path, reference_path = input_paths(config, args.sample_id)
    output = Path(config["output_dir"]) / args.sample_id / f"{args.sample_id}_novae.npz"
    if output.exists():
        raise FileExistsError(f"teacher artifact already exists: {output}")
    adata, ids, centers = prepare_input(counts_path, reference_path)
    model = load_model(config)
    embed(model, adata, config, args.device)
    export_sample(output, args.sample_id, adata, ids, centers, config)


if __name__ == "__main__":
    main()
