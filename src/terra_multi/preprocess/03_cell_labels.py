"""Transfer CellViT labels to reconstructed cells."""

import argparse
from pathlib import Path

import anndata as ad
import pandas as pd
import yaml

from terra_multi.helpers.labels import (
    load_cell_polygons,
    load_cellvit_points,
    transfer_labels,
)


def main():
    parser = argparse.ArgumentParser(description="Transfer CellViT cell-type labels.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-id", required=True)
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    fallback_max_um = cfg.get("fallback_max_um", 20)

    manifest = pd.read_csv(Path(cfg["output_root"]) / "manifest.csv").set_index("id")
    row = manifest.loc[args.sample_id]
    out_dir = Path(row["output_dir"])

    adata = ad.read_h5ad(out_dir / f"{args.sample_id}.h5ad")
    cell_ids = adata.obs_names.to_numpy()
    he = adata.obs[["he_x", "he_y"]].to_numpy(dtype=float)

    xen = load_cell_polygons(row["xenium_cell_seg"], cell_ids)
    cv_pts = load_cellvit_points(row["cellvit_seg"])
    labels = transfer_labels(
        cell_ids, he, xen, cv_pts, fallback_max_um, px_um=float(row["pixel_size_um"])
    )

    out_path = out_dir / f"{args.sample_id}_cell_labels.parquet"
    labels.to_parquet(out_path, index=False)
    print(
        f"{args.sample_id}: labelled {len(labels):,} cells "
        f"({labels['method'].value_counts().to_dict()}) -> {out_path}"
    )


if __name__ == "__main__":
    main()
