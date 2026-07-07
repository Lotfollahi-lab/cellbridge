"""TERRA embeddings for one sample (harmonize -> tokenize -> embed).

    PYTHONPATH=src python -m terra_multi.pipeline.terra_embed \\
        --config configs/pipeline/pipeline.yaml --sample-id TENX99

Reads the step-2 QC h5ad, maps genes onto TERRA's vocabulary with
``harmonize_adata`` (which also drops cells with <10 genes and genes in <10
cells), tokenizes every surviving cell's k=10 neighborhood, and runs
``embed_dataset`` over ALL of them. Writes ``{output_dir}/{id}_terra.npz`` with
cell_emb, neighborhood_emb, cell_ids and he_x/he_y in embedding row order.

The tokenized dataset is written to ``{output_dir}/{id}_tokenized`` as a
resumable scratch artifact and removed once the npz is saved, unless
``keep_tokenized`` is set. The embedding pass needs a GPU.
"""

import argparse
import os
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml

MODEL_NAME = "Lotfollahi-lab/TERRA-112M"
MIN_GENES_PER_CELL = 10
MIN_CELLS_PER_GENE = 10
EMBED_BATCH_SIZE = 32


def aligned_he(dataset_cell_ids, harmonized_obs):
    ids = [str(c) for c in dataset_cell_ids]
    he = harmonized_obs.loc[ids, ["he_x", "he_y"]]
    assert list(he.index) == ids, "he centroid row order mismatch"
    return he["he_x"].to_numpy(), he["he_y"].to_numpy()


def main():
    parser = argparse.ArgumentParser(description="TERRA-embed one sample's cells.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-id", required=True)
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    keep_tokenized = bool(cfg.get("keep_tokenized", False))

    manifest = pd.read_csv(Path(cfg["output_root"]) / "manifest.csv").set_index("id")
    out_dir = Path(manifest.loc[args.sample_id, "output_dir"])
    adata = ad.read_h5ad(out_dir / f"{args.sample_id}_qc.h5ad")

    from datasets import load_from_disk
    from terra import (
        download_pretrained,
        embed_dataset,
        harmonize_adata,
        tokenize_adata,
    )

    model_dir = Path(download_pretrained(MODEL_NAME))
    harmonized = harmonize_adata(
        adata,
        gene_mapping_dict_file_path=str(model_dir / "ensembl_dictionary.pkl"),
        gene_occurrence_count_file_path=str(model_dir / "gene_count_dictionary.pkl"),
        min_cells_per_gene=MIN_CELLS_PER_GENE,
        min_genes_per_cell=MIN_GENES_PER_CELL,
    )

    # Tokenize all cells, reusing an existing tokenized dataset if a previous run got that far
    tokenized_path = out_dir / f"{args.sample_id}_tokenized"
    cache_dir = out_dir / ".tokenize_cache"
    if tokenized_path.exists():
        dataset = load_from_disk(str(tokenized_path))
    else:
        dataset = tokenize_adata(
            adata=harmonized,
            model_folder_path=str(model_dir),
            cache_directory_path=str(cache_dir),
            nproc=len(os.sched_getaffinity(0)),
        )
        dataset.save_to_disk(str(tokenized_path))

    embeddings = embed_dataset(
        dataset=dataset, model_folder_path=str(model_dir), batch_size=EMBED_BATCH_SIZE
    )

    cell_ids = np.array(dataset["cell_id"])
    he_x, he_y = aligned_he(cell_ids, harmonized.obs)

    out_path = out_dir / f"{args.sample_id}_terra.npz"
    np.savez(
        out_path,
        cell_ids=cell_ids,
        cell_emb=embeddings["cell_emb"],
        neighborhood_emb=embeddings["neighborhood_emb"],
        he_x=he_x,
        he_y=he_y,
    )

    if not keep_tokenized:
        shutil.rmtree(tokenized_path, ignore_errors=True)
        shutil.rmtree(cache_dir, ignore_errors=True)

    print(f"{args.sample_id}: embedded {len(cell_ids):,} cells -> {out_path}")


if __name__ == "__main__":
    main()
