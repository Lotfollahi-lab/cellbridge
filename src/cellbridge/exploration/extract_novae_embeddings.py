"""Extract frozen NOVAE spatial representations for one complete HEST section."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import anndata as ad
import numpy as np

from cellbridge.config import load_config


def _output_path(cfg, sample_id):
    directory = Path(cfg["novae"]["embedding_dir"])
    pattern = cfg["novae"].get("embedding_pattern", "{sample_id}_novae.npz")
    return directory / pattern.format(sample_id=sample_id)


def _validate_existing(path, sample_id):
    with np.load(path, allow_pickle=False) as data:
        required = {"cell_ids", "novae_latent", "neighborhood_valid"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path}: incomplete existing output; missing {missing}")
        if len(data["cell_ids"]) != len(data["novae_latent"]):
            raise ValueError(f"{path}: cell ID and embedding counts differ")
    print(f"{sample_id}: validated existing embeddings -> {path}")


def run(cfg, sample_id, *, accelerator=None, force=False):
    if sample_id not in cfg["sample_ids"]:
        raise ValueError(f"sample {sample_id!r} is not in the configured cohort")
    source = Path(cfg["data_root"]) / sample_id / f"{sample_id}.h5ad"
    output = _output_path(cfg, sample_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        _validate_existing(output, sample_id)
        return output

    adata = ad.read_h5ad(source)
    if adata.obs_names.has_duplicates:
        raise ValueError(f"{sample_id}: duplicate AnnData cell IDs")
    if "spatial" not in adata.obsm or adata.obsm["spatial"].shape != (adata.n_obs, 2):
        raise ValueError(f"{sample_id}: expected two spatial coordinates per cell")

    import novae
    import torch

    seed = int(cfg["novae"].get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    novae.settings.disable_multimodal = True
    novae.settings.scale_to_microns = float(
        cfg["novae"].get("scale_to_microns", 1.0)
    )
    graph = cfg["novae"].get("graph", {})
    novae.spatial_neighbors(
        adata,
        radius=graph.get("radius"),
        percentile=graph.get("percentile"),
    )

    model_name = cfg["novae"]["model_name"]
    model = novae.Novae.from_pretrained(model_name)
    inference = cfg["novae"].get("inference", {})
    accelerator = accelerator or inference.get("accelerator", "auto")
    num_workers = int(inference.get("num_workers", 4))
    zero_shot = bool(inference.get("zero_shot", True))
    model.compute_representations(
        adata,
        zero_shot=zero_shot,
        accelerator=accelerator,
        num_workers=num_workers,
    )

    latent = np.asarray(adata.obsm["novae_latent"], dtype=np.float32)
    valid = np.asarray(adata.obs["neighborhood_valid"], dtype=bool)
    if latent.ndim != 2 or latent.shape[0] != adata.n_obs:
        raise ValueError(f"{sample_id}: invalid NOVAE output shape {latent.shape}")
    if not np.isfinite(latent[valid]).all():
        raise ValueError(f"{sample_id}: NOVAE produced non-finite representations")
    if np.any(np.linalg.norm(latent[valid], axis=1) == 0):
        raise ValueError(f"{sample_id}: NOVAE produced zero-norm valid representations")

    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            cell_ids=np.asarray(adata.obs_names, dtype=str),
            novae_latent=latent,
            neighborhood_valid=valid,
        )
    temporary.replace(output)

    distances = np.asarray(adata.obsp["spatial_distances"].data)
    distances = distances[distances > 0]
    metadata = {
        "sample_id": sample_id,
        "source_h5ad": str(source),
        "output_npz": str(output),
        "model_name": model_name,
        "novae_version": importlib.metadata.version("novae"),
        "zero_shot": zero_shot,
        "accelerator": accelerator,
        "seed": seed,
        "scale_to_microns": novae.settings.scale_to_microns,
        "graph": graph,
        "n_cells": adata.n_obs,
        "n_genes": adata.n_vars,
        "n_valid_neighborhoods": int(valid.sum()),
        "embedding_dimension": latent.shape[1],
        "median_neighbor_distance_microns": float(
            np.median(distances) * novae.settings.scale_to_microns
        ),
    }
    metadata_path = output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"{sample_id}: embedded {adata.n_obs:,} cells "
        f"({valid.mean():.2%} valid) -> {output}"
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--accelerator")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(
        load_config(args.config),
        args.sample_id,
        accelerator=args.accelerator,
        force=args.force,
    )


if __name__ == "__main__":
    main()
