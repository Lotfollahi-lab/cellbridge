"""Memory-safe streaming variant of 02_quality_check.

    PYTHONPATH=src python3 src/cellbridge/preprocess/02_quality_check_streaming.py \\
        --config configs/preprocess/preprocess.yaml --sample-id TENX160
"""

import argparse
import importlib
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as pa_ds
import scanpy as sc
import scipy.sparse as sp

from cellbridge.config import load_config

# Reuse the exact filter constants + helpers from the standard step
_qc = importlib.import_module("cellbridge.preprocess.02_quality_check")
_to_str = _qc._to_str
_CONTROL_PREFIXES = _qc._CONTROL_PREFIXES
_UNASSIGNED = _qc._UNASSIGNED
_COLUMNS = _qc._COLUMNS

_BATCH_ROWS = 5_000_000


def _kept(batch_df, qv_threshold):
    df = batch_df[batch_df["qv"] >= qv_threshold]
    cell_id = _to_str(df["cell_id"])
    gene = _to_str(df["feature_name"])
    keep = ~cell_id.isin(_UNASSIGNED) & ~gene.map(
        lambda s: s.startswith(_CONTROL_PREFIXES)
    )
    coords = df.loc[keep, ["x_location", "y_location", "he_x", "he_y"]].to_numpy(
        dtype=float
    )
    return cell_id[keep].to_numpy(), gene[keep].to_numpy(), coords


def reconstruct_cells_streaming(open_batches, qv_threshold):
    # --- pass 1: sorted cell / gene vocabularies -------------------------------
    cell_vocab, gene_vocab = set(), set()
    for batch in open_batches():
        cell_id, gene, _ = _kept(batch, qv_threshold)
        cell_vocab.update(cell_id.tolist())
        gene_vocab.update(gene.tolist())
    if not cell_vocab:
        raise ValueError("no transcripts survived filtering")
    cell_cats = np.array(sorted(cell_vocab))
    gene_cats = np.array(sorted(gene_vocab))
    cell_code = {c: i for i, c in enumerate(cell_cats)}
    gene_code = {g: i for i, g in enumerate(gene_cats)}
    n_cells, n_genes = len(cell_cats), len(gene_cats)

    # --- pass 2: counts + centroid sums, accumulated batch by batch ------------
    counts = sp.csr_matrix((n_cells, n_genes), dtype=np.float32)
    coord_sums = np.zeros((n_cells, 4), dtype=np.float64)
    cell_counts = np.zeros(n_cells, dtype=np.float64)
    for batch in open_batches():
        cell_id, gene, coords = _kept(batch, qv_threshold)
        if len(cell_id) == 0:
            continue
        rows = np.fromiter(
            (cell_code[c] for c in cell_id), dtype=np.int64, count=len(cell_id)
        )
        cols = np.fromiter(
            (gene_code[g] for g in gene), dtype=np.int64, count=len(gene)
        )
        batch_coo = sp.coo_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(n_cells, n_genes),
        )
        counts = counts + batch_coo.tocsr()  # duplicate (cell, gene) pairs summed
        for j in range(4):
            coord_sums[:, j] += np.bincount(
                rows, weights=coords[:, j], minlength=n_cells
            )
        cell_counts += np.bincount(rows, minlength=n_cells)

    centroids = coord_sums / cell_counts[:, None]  # mean over each cell's kept rows

    adata = ad.AnnData(
        X=counts,
        obs=pd.DataFrame(index=pd.Index(cell_cats, name="cell_id")),
        var=pd.DataFrame(index=pd.Index(gene_cats, name="gene")),
    )
    adata.obs["cell_id"] = adata.obs_names
    adata.obsm["spatial"] = centroids[:, [0, 1]]
    adata.obs["he_x"] = centroids[:, 2]
    adata.obs["he_y"] = centroids[:, 3]
    sc.pp.calculate_qc_metrics(adata, inplace=True, percent_top=None)
    return adata


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct per-cell QC counts (streaming, low-memory)."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--batch-rows", type=int, default=_BATCH_ROWS)
    args = parser.parse_args()

    cfg = load_config(args.config)
    qv_threshold = float(cfg.get("qv_threshold", 20))

    manifest = pd.read_csv(Path(cfg["output_root"]) / "manifest.csv").set_index("id")
    row = manifest.loc[args.sample_id]

    dataset = pa_ds.dataset(row["transcripts"], format="parquet")
    scan_filter = pc.field("qv") >= qv_threshold

    def open_batches():
        for rb in dataset.to_batches(
            columns=_COLUMNS, filter=scan_filter, batch_size=args.batch_rows
        ):
            if rb.num_rows:
                yield rb.to_pandas()

    adata = reconstruct_cells_streaming(open_batches, qv_threshold)

    out_dir = Path(row["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.sample_id}.h5ad"
    adata.write_h5ad(out_path)
    print(
        f"{args.sample_id}: {adata.n_obs:,} cells x {adata.n_vars} genes "
        f"-> {out_path} (streaming)"
    )


if __name__ == "__main__":
    main()
