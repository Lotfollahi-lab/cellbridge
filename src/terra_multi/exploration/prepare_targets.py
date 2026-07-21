import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as pa_ds
import scipy.sparse as sp

from terra_multi.config import load_config

from terra_multi.helpers.io import load_embeddings

_CONTROL_PREFIXES = ("NegControlProbe_", "NegControlCodeword_", "BLANK_")


def main():
    parser = argparse.ArgumentParser(description="Rebuild per-cell target expression.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    transcripts_path = cfg["transcripts_path"]
    adata_path = Path(cfg["adata_path"])
    qv_threshold = float(cfg.get("qv_threshold", 20))

    # The cells to reconstruct are exactly those that were embedded.
    _, cell_ids = load_embeddings(cfg["terra_embeddings"], key=cfg.get("terra_key"))
    cell_ids = np.asarray(cell_ids).astype(str)
    target_int_ids = [int(c) for c in cell_ids]
    print(f"Reconstructing expression for {len(cell_ids):,} embedded cells")

    # Read only the needed columns and push the cell + qv filters into the
    # parquet scan, so we never materialize the full ~127M-row transcript file.
    dataset = pa_ds.dataset(transcripts_path, format="parquet")
    scan_filter = pc.field("cell_id").isin(target_int_ids) & (
        pc.field("qv") >= qv_threshold
    )
    table = dataset.to_table(
        columns=["cell_id", "feature_name", "x_location", "y_location", "qv"],
        filter=scan_filter,
    )
    df = table.to_pandas()
    print(f"Transcripts after cell + qv>={qv_threshold:g} filter: {len(df):,}")

    # Drop Xenium negative-control probes.
    feature = df["feature_name"].astype(str)
    is_control = feature.map(lambda s: s.startswith(_CONTROL_PREFIXES))
    df = df.loc[~is_control].copy()
    df["feature_name"] = df["feature_name"].astype(str)
    print(f"After dropping control probes: {len(df):,}")

    # Cell centroids in native Xenium microns
    centroids = df.groupby("cell_id")[["x_location", "y_location"]].mean()

    # Aggregate transcript counts into a sparse cells x genes matrix; duplicate
    # (cell, gene) coordinates are summed on COO -> CSR conversion.
    id_str = df["cell_id"].astype(str)
    cell_cat = pd.Categorical(id_str)
    gene_cat = pd.Categorical(df["feature_name"])
    counts = sp.coo_matrix(
        (np.ones(len(df), dtype=np.float32), (cell_cat.codes, gene_cat.codes)),
        shape=(len(cell_cat.categories), len(gene_cat.categories)),
    ).tocsr()

    adata = ad.AnnData(
        X=counts,
        obs=pd.DataFrame(
            index=pd.Index(cell_cat.categories.astype(str), name="cell_id")
        ),
        var=pd.DataFrame(index=pd.Index(gene_cat.categories.astype(str), name="gene")),
    )

    # Every embedded cell must be present; a missing one means the transcript
    # file and the embeddings disagree -- surface it rather than zero-filling.
    missing = [c for c in cell_ids if c not in set(adata.obs_names)]
    if missing:
        raise RuntimeError(
            f"{len(missing)} embedded cells have no surviving transcripts "
            f"(e.g. {missing[:5]}); transcript file and embeddings disagree"
        )

    # Order rows to match the embedding row order and attach native-micron coords.
    adata = adata[cell_ids].copy()
    cent = centroids.reindex([int(c) for c in cell_ids])
    adata.obsm["spatial"] = cent[["x_location", "y_location"]].to_numpy()
    if not np.isfinite(adata.obsm["spatial"]).all():
        raise RuntimeError("non-finite centroid(s) after reindex")

    adata_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(adata_path)
    print(
        f"Wrote {adata.n_obs:,} cells x {adata.n_vars} genes (raw counts, "
        f"obsm['spatial'] in microns) -> {adata_path}"
    )


if __name__ == "__main__":
    main()
