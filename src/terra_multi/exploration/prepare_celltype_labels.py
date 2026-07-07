"""Assign a CellViT cell-type label to each embedded TENX99 cell.

CellViT and Xenium segment the same tissue independently, so CellViT nuclei
aren't keyed by Xenium ``cell_id``. Instead: for each Xenium cell, take the
majority ``class`` among the CellViT nuclei whose centroid falls inside its
polygon. Cells with no nucleus inside fall back to the nearest centroid.
Writes ``cell_id -> cell_type``.

Run once, on the login/dev node (not inside the probe jobs):

    PYTHONPATH=src python -m terra_multi.exploration.prepare_celltype_labels \\
        --config configs/exploration/celltype.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import yaml

from terra_multi.helpers.labels import (
    load_cell_polygons,
    load_cellvit_points,
    transfer_labels,
)


def main():
    parser = argparse.ArgumentParser(description="Build CellViT cell-type labels.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    # Our cells + their H&E-pixel centroids (embedding row order), for the fallback.
    terra = np.load(cfg["terra_embeddings"], allow_pickle=True)
    cell_ids = terra["cell_ids"].astype(str)
    he = np.c_[terra["he_x"], terra["he_y"]].astype(float)
    print(f"labelling {len(cell_ids):,} cells")

    xen = load_cell_polygons(cfg["xenium_cell_seg"], cell_ids)
    print(f"matched {len(xen):,} Xenium cell polygons")
    cv_pts = load_cellvit_points(cfg["cellvit_seg"])
    print(
        f"loaded {len(cv_pts):,} CellViT nuclei; "
        f"classes {sorted(cv_pts['cellvit_class'].unique())}"
    )

    labels = transfer_labels(cell_ids, he, xen, cv_pts, cfg.get("fallback_max_um", 20))

    print("\n=== label distribution ===")
    print(labels["cell_type"].value_counts(dropna=False).to_string())
    print("\n=== assignment method ===")
    print(labels["method"].value_counts().to_string())

    labels_path = Path(cfg["labels_path"])
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(labels_path, index=False)
    print(f"\nwrote {labels_path}")


if __name__ == "__main__":
    main()
