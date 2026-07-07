"""Spatial CellViT-to-Xenium label transfer."""

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

PX_UM = 0.2125


def load_cell_polygons(xenium_cell_seg_path, cell_ids):
    """Load Xenium polygons for the requested cells."""
    keep = set(np.asarray(cell_ids).astype(str))

    xen = gpd.read_parquet(xenium_cell_seg_path)
    if "cell_id" not in xen.columns:
        index_name = xen.index.name or "index"
        xen = xen.reset_index().rename(columns={index_name: "cell_id"})
    else:
        xen = xen.copy()

    xen["cell_id"] = xen["cell_id"].astype(str)
    xen = xen[xen["cell_id"].isin(keep)][["cell_id", "geometry"]]
    return xen.set_crs(None, allow_override=True)


def load_cellvit_points(cellvit_seg_path):
    """Load CellViT nucleus centroids and classes."""
    cv = gpd.read_parquet(cellvit_seg_path)
    return gpd.GeoDataFrame(
        {"cellvit_class": cv["class"].to_numpy()},
        geometry=cv.geometry.centroid,
    ).set_crs(None, allow_override=True)


def transfer_labels(
    cell_ids, he_xy, cell_polygons, cellvit_points, fallback_max_um=20.0, px_um=PX_UM
):
    """Return one transferred label per input cell."""
    cell_ids = np.asarray(cell_ids).astype(str)
    he_xy = np.asarray(he_xy, dtype=float)
    fallback_max_px = float(fallback_max_um) / px_um

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
        labels.loc[need, "fallback_dist_um"] = dist * px_um
    return labels
