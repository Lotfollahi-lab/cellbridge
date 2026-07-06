"""Assign a CellViT cell-type label to each embedded TENX99 cell.

CellViT and Xenium segment the same tissue independently, so CellViT nuclei
aren't keyed by Xenium ``cell_id``. Instead: for each Xenium cell, take the
majority ``class`` among the CellViT nuclei whose centroid falls inside its
polygon. Cells with no nucleus inside fall back to the nearest centroid.
Writes ``cell_id -> cell_type``.

Run once, on the login/dev node (not inside the probe jobs):

    PYTHONPATH=src python -m terra_multi.experiments.prepare_celltype_labels \\
        --config configs/celltype.yaml
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

PX_UM = 0.2125  # WSI level-0 micron/pixel (native Xenium)


def load_cell_polygons(xenium_cell_seg_path, cell_ids):
    """Xenium cell polygons for ``cell_ids``, as a GeoDataFrame with a cell_id column.

    The Xenium cell_id is stored as the parquet index (``__index_level_0__``), so
    pull it back out to a column before filtering to the cells we care about.
    """
    keep = set(np.asarray(cell_ids).astype(str))
    xen = gpd.read_parquet(xenium_cell_seg_path).reset_index(names="cell_id")
    xen["cell_id"] = xen["cell_id"].astype(str)
    xen = xen[xen["cell_id"].isin(keep)][["cell_id", "geometry"]]
    return xen.set_crs(None, allow_override=True)


def load_cellvit_points(cellvit_seg_path):
    """CellViT nucleus centroids + class, as a GeoDataFrame of points."""
    cv = gpd.read_parquet(cellvit_seg_path)
    return gpd.GeoDataFrame(
        {"cellvit_class": cv["class"].to_numpy()},
        geometry=cv.geometry.centroid,
    ).set_crs(None, allow_override=True)


def transfer_labels(
    cell_ids, he_xy, cell_polygons, cellvit_points, fallback_max_um=20.0
):
    """Transfer a CellViT class onto each Xenium cell.

    For each Xenium cell, take the majority ``class`` among the CellViT nucleus
    centroids that fall inside its polygon; cells whose polygon contains no
    nucleus fall back to the nearest CellViT centroid, and those farther than
    ``fallback_max_um`` are marked ``"Unassigned"``. ``he_xy`` is the ``(N, 2)``
    array of H&E-pixel cell centroids used for the fallback. Returns a DataFrame
    (cell_id, cell_type, n_nuclei, method, fallback_dist_um) in ``cell_ids`` order.
    """
    cell_ids = np.asarray(cell_ids).astype(str)
    he_xy = np.asarray(he_xy, dtype=float)
    fallback_max_px = float(fallback_max_um) / PX_UM

    # Polygon containment -> majority CellViT class per Xenium cell.
    joined = gpd.sjoin(cellvit_points, cell_polygons, predicate="within", how="inner")
    agg = joined.groupby("cell_id")["cellvit_class"].agg(
        cell_type=lambda s: s.value_counts().idxmax(),
        n_nuclei="size",
    )

    labels = pd.DataFrame({"cell_id": cell_ids})
    labels = labels.merge(agg.reset_index(), on="cell_id", how="left")
    labels["method"] = np.where(labels["cell_type"].notna(), "polygon", "fallback")
    labels["n_nuclei"] = labels["n_nuclei"].fillna(0).astype(int)

    # Nearest-centroid fallback for cells whose polygon contained no nucleus.
    need = labels["cell_type"].isna().to_numpy()
    labels["fallback_dist_um"] = np.nan
    if need.any():
        centroids = np.c_[cellvit_points.geometry.x, cellvit_points.geometry.y]
        dist, idx = cKDTree(centroids).query(he_xy[need], k=1)
        cls = cellvit_points["cellvit_class"].to_numpy()[idx]
        cls[dist > fallback_max_px] = "Unassigned"
        labels.loc[need, "cell_type"] = cls
        labels.loc[need, "fallback_dist_um"] = dist * PX_UM
    return labels


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
