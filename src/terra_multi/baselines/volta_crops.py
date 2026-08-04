"""Adaptive cell boxes for VOLTA extraction."""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage

from terra_multi.evaluation.ood_data import tile_geometries


def square_bounds(bounds, scale=2.0):
    x0, y0, x1, y1 = map(float, bounds)
    if not np.isfinite([x0, y0, x1, y1]).all() or x1 <= x0 or y1 <= y0:
        raise ValueError(f"invalid cell bounds {bounds}")
    if float(scale) <= 0:
        raise ValueError("crop scale must be positive")
    side = max(1, int(np.ceil(max(x1 - x0, y1 - y0) * float(scale))))
    left = int(round((x0 + x1 - side) / 2))
    top = int(round((y0 + y1 - side) / 2))
    return left, top, left + side, top + side


def crop_box(image, bounds, scale=2.0, fill=255):
    """Read a square box from an HWC image or lazy WSI, padding in white."""
    x0, y0, x1, y1 = square_bounds(bounds, scale)
    height, width = map(int, image.shape[:2])
    vx0, vy0, vx1, vy1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
    crop = np.full((y1 - y0, x1 - x0, 3), fill, dtype=np.uint8)
    if vx1 > vx0 and vy1 > vy0:
        region = np.asarray(image[vy0:vy1, vx0:vx1])
        if region.ndim == 2:
            region = np.repeat(region[..., None], 3, axis=-1)
        crop[vy0 - y0 : vy1 - y0, vx0 - x0 : vx1 - x0] = region[..., :3]
    return Image.fromarray(crop)


def instance_bounds(mask):
    """Return half-open local boxes keyed by positive instance ID."""
    mask = np.asarray(mask, dtype=np.int64)
    if mask.ndim != 2 or np.any(mask < 0):
        raise ValueError("instance mask must be a non-negative 2D array")
    boxes = {}
    for instance_id, slices in enumerate(ndimage.find_objects(mask), start=1):
        if slices is None:
            continue
        y, x = slices
        boxes[instance_id] = (x.start, y.start, x.stop, y.stop)
    return boxes


def locate_instance_boxes(
    metadata,
    entries,
    image_shape=(1440, 1920),
    tile_size=224,
    *,
    skip_missing=False,
):
    """Attach field-level instance boxes to OOD metadata rows."""
    geometries = {item.index: item for item in tile_geometries(image_shape, tile_size)}
    located = metadata.copy()
    for name in ("x0", "y0", "x1", "y1"):
        located[name] = np.nan

    for index, entry in entries.items():
        geometry = geometries[index]
        rows = located["tile_id"].astype(str) == str(entry["tile_name"])
        if not rows.any():
            continue
        boxes = instance_bounds(entry["instance_mask"])
        for row_index, instance_id in located.loc[rows, "cell_id"].astype(int).items():
            if instance_id not in boxes:
                if skip_missing:
                    continue
                raise ValueError(
                    f"{entry['tile_name']}: instance {instance_id} is missing"
                )
            x0, y0, x1, y1 = boxes[instance_id]
            located.loc[row_index, ["x0", "y0", "x1", "y1"]] = (
                geometry.x0 + x0 - geometry.pad_left,
                geometry.y0 + y0 - geometry.pad_top,
                geometry.x0 + x1 - geometry.pad_left,
                geometry.y0 + y1 - geometry.pad_top,
            )

    missing = located[["x0", "y0", "x1", "y1"]].isna().any(axis=1)
    if missing.any() and not skip_missing:
        raise ValueError(f"could not locate {int(missing.sum())} cell instances")
    return located.loc[~missing].copy()
