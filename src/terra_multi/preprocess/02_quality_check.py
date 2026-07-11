"""Reconstruct per-cell counts from Xenium transcripts."""

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as pa_ds
import scanpy as sc
import scipy.sparse as sp
import yaml

_CONTROL_PREFIXES = ("NegControlProbe_", "NegControlCodeword_", "BLANK_")
_COLUMNS = ["cell_id", "feature_name", "x_location", "y_location", "he_x", "he_y", "qv"]
_UNASSIGNED = {"-1", "UNASSIGNED"}


def _to_str(series):
    if len(series) and isinstance(series.iloc[0], bytes):
        return series.str.decode("utf-8")
    return series.astype(str)


def reconstruct_cells(transcripts, qv_threshold):
    """Return raw counts plus native and H&E centroids."""
    df = transcripts[transcripts["qv"] >= qv_threshold]
    cell_id = _to_str(df["cell_id"])
    gene = _to_str(df["feature_name"])
    keep = ~cell_id.isin(_UNASSIGNED) & ~gene.map(
        lambda s: s.startswith(_CONTROL_PREFIXES)
    )

    cell_id, gene = cell_id[keep].to_numpy(), gene[keep].to_numpy()
    coords = df.loc[keep, ["x_location", "y_location", "he_x", "he_y"]]

    cell_cat = pd.Categorical(cell_id)
    gene_cat = pd.Categorical(gene)
    counts = sp.coo_matrix(
        (np.ones(len(cell_id), dtype=np.float32), (cell_cat.codes, gene_cat.codes)),
        shape=(len(cell_cat.categories), len(gene_cat.categories)),
    ).tocsr()

    centroids = coords.groupby(cell_id).mean().loc[cell_cat.categories]

    adata = ad.AnnData(
        X=counts,
        obs=pd.DataFrame(index=pd.Index(cell_cat.categories, name="cell_id")),
        var=pd.DataFrame(index=pd.Index(gene_cat.categories, name="gene")),
    )
    adata.obs["cell_id"] = adata.obs_names
    adata.obsm["spatial"] = centroids[["x_location", "y_location"]].to_numpy()
    adata.obs["he_x"] = centroids["he_x"].to_numpy()
    adata.obs["he_y"] = centroids["he_y"].to_numpy()
    sc.pp.calculate_qc_metrics(adata, inplace=True, percent_top=None)
    return adata


def main():
    parser = argparse.ArgumentParser(description="Reconstruct per-cell QC counts.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-id", required=True)
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    qv_threshold = float(cfg.get("qv_threshold", 20))

    manifest = pd.read_csv(Path(cfg["output_root"]) / "manifest.csv").set_index("id")
    row = manifest.loc[args.sample_id]

    dataset = pa_ds.dataset(row["transcripts"], format="parquet")
    table = dataset.to_table(columns=_COLUMNS, filter=pc.field("qv") >= qv_threshold)
    adata = reconstruct_cells(table.to_pandas(), qv_threshold)

    out_dir = Path(row["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.sample_id}.h5ad"
    adata.write_h5ad(out_path)
    print(
        f"{args.sample_id}: {adata.n_obs:,} cells x {adata.n_vars} genes -> {out_path}"
    )


if __name__ == "__main__":
    main()
