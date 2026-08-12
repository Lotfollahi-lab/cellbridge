"""Reconstructed OOD images and centered cell crops for embedding inference."""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pandas as pd
from PIL import Image

from cellbridge.inference.embed.tiles import (
    TileLayout,
    fixed_layout,
    infer_layout,
)

CROP_SIZE = 224
_TILE_PATTERN = re.compile(r"_TILE_(\d+)$")


@dataclass(frozen=True)
class OodSample:
    """One reconstructed image and its ordered, located cells."""

    sample_id: str
    image: np.ndarray
    cells: pd.DataFrame
    layout: TileLayout

    @property
    def cell_ids(self) -> np.ndarray:
        """Return globally unique IDs for tile-local cell instances."""
        return (
            self.cells["sample_id"].astype(str)
            + "/"
            + self.cells["tile_id"].astype(str)
            + "/"
            + self.cells["cell_id"].astype(str)
        ).to_numpy(dtype=str)


def _tile_index(tile_name) -> int:
    match = _TILE_PATTERN.search(str(tile_name))
    if match is None:
        raise ValueError(f"cannot parse tile index from {tile_name!r}")
    return int(match.group(1))


def iter_lmdb_samples(path) -> Iterator[tuple[str, dict[int, dict]]]:
    """Yield consecutive LMDB entries grouped by sample."""
    try:
        import lmdb
    except ImportError as error:
        raise ImportError("reading OOD data requires the lmdb package") from error

    environment = lmdb.open(
        str(Path(path)), readonly=True, lock=False, readahead=False, meminit=False
    )
    try:
        with environment.begin(write=False) as transaction:
            sample_id, entries = None, {}
            for _, value in transaction.cursor():
                entry = pickle.loads(value)
                current = str(entry["sample_name"])
                if sample_id is not None and current != sample_id:
                    yield sample_id, entries
                    entries = {}
                sample_id = current
                index = _tile_index(entry["tile_name"])
                if index in entries:
                    raise ValueError(f"{sample_id}: duplicate tile index {index}")
                entries[index] = entry
            if sample_id is not None:
                yield sample_id, entries
    finally:
        environment.close()


def _layout(entries, image_shape, tile_size) -> TileLayout:
    if image_shape is not None:
        return fixed_layout(tuple(image_shape), tile_size)
    tiles = {index: entry["image"] for index, entry in entries.items()}
    return infer_layout(tiles)


def _reconstruct(entries, layout: TileLayout, tile_size: int) -> np.ndarray:
    geometries = {geometry.index: geometry for geometry in layout.tiles}
    if set(entries) != set(geometries):
        raise ValueError("tile set does not match the resolved image layout")

    first = np.asarray(next(iter(entries.values()))["image"])
    canvas = np.empty((3, *layout.image_shape), dtype=first.dtype)
    for index, entry in entries.items():
        tile = np.asarray(entry["image"])
        if tile.shape != (3, tile_size, tile_size):
            raise ValueError(f"tile {index} has unexpected shape {tile.shape}")
        geometry = geometries[index]
        pixels = tile[
            :,
            geometry.pad_top : geometry.pad_top + geometry.height,
            geometry.pad_left : geometry.pad_left + geometry.width,
        ]
        canvas[:, geometry.top : geometry.bottom, geometry.left : geometry.right] = pixels
    return canvas


def _centroids(mask) -> dict[int, tuple[float, float]]:
    mask = np.asarray(mask, dtype=np.int64)
    y, x = np.indices(mask.shape)
    counts = np.bincount(mask.ravel())
    x_sums = np.bincount(mask.ravel(), weights=x.ravel(), minlength=len(counts))
    y_sums = np.bincount(mask.ravel(), weights=y.ravel(), minlength=len(counts))
    return {
        instance: (x_sums[instance] / counts[instance], y_sums[instance] / counts[instance])
        for instance in np.flatnonzero(counts)[1:]
    }


def _locate_cells(metadata, entries, layout, *, skip_missing) -> pd.DataFrame:
    geometries = {geometry.index: geometry for geometry in layout.tiles}
    cells = metadata.copy()
    cells[["x", "y"]] = np.nan
    for index, entry in entries.items():
        geometry = geometries[index]
        rows = cells["tile_id"].astype(str) == str(entry["tile_name"])
        centers = _centroids(entry["instance_mask"])
        for row, instance in cells.loc[rows, "cell_id"].astype(int).items():
            if instance not in centers:
                if skip_missing:
                    continue
                raise ValueError(f"{entry['tile_name']}: instance {instance} is missing")
            x, y = centers[instance]
            cells.at[row, "x"] = geometry.left + x - geometry.pad_left
            cells.at[row, "y"] = geometry.top + y - geometry.pad_top

    missing = cells[["x", "y"]].isna().any(axis=1)
    if missing.any() and not skip_missing:
        raise ValueError(f"could not locate {int(missing.sum())} cells")
    return cells.loc[~missing].copy()


def load_sample(
    sample_id,
    entries,
    metadata,
    *,
    image_shape=None,
    tile_size: int = 224,
    value_scale: float = 250,
    skip_missing: bool = False,
) -> OodSample:
    """Reconstruct one tiled image and locate its labeled cells."""
    sample_id = str(sample_id)
    required = {"sample_id", "tile_id", "cell_id"}
    if not required <= set(metadata):
        raise ValueError(f"metadata is missing {sorted(required - set(metadata))}")
    cells = metadata.loc[metadata["sample_id"].astype(str) == sample_id].copy()
    if cells.empty:
        raise ValueError(f"{sample_id}: no metadata rows")
    if not np.isfinite(value_scale) or value_scale <= 0:
        raise ValueError("value_scale must be positive")

    layout = _layout(entries, image_shape, tile_size)
    cells = _locate_cells(cells, entries, layout, skip_missing=skip_missing)
    chw = _reconstruct(entries, layout, tile_size).astype(np.float32) / value_scale
    image = np.clip(np.rint(chw), 0, 255).astype(np.uint8).transpose(1, 2, 0)
    return OodSample(sample_id, image, cells, layout)


def center_crop(image, x, y, size: int = CROP_SIZE, fill: int = 255) -> Image.Image:
    """Return a centered source-pixel crop with constant boundary padding."""
    x0, y0 = round(x) - size // 2, round(y) - size // 2
    x1, y1 = x0 + size, y0 + size
    height, width = image.shape[:2]
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x1, width), min(y1, height)
    crop = np.full((size, size, 3), fill, dtype=np.uint8)
    if sx1 > sx0 and sy1 > sy0:
        crop[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = image[sy0:sy1, sx0:sx1]
    return Image.fromarray(crop)


class OodCrops:
    """Map-style dataset of centered crops from one OOD image."""

    def __init__(
        self,
        sample: OodSample,
        transform: Callable | None = None,
        *,
        crop_size: int = CROP_SIZE,
    ):
        if crop_size != CROP_SIZE:
            raise ValueError(f"OOD crop_size must be {CROP_SIZE}")
        self.sample = sample
        self.transform = transform
        self.crop_size = crop_size

    def __len__(self) -> int:
        return len(self.sample.cells)

    def __getitem__(self, index: int) -> dict:
        if not -len(self) <= index < len(self):
            raise IndexError(index)
        index %= len(self)
        row = self.sample.cells.iloc[index]
        image = center_crop(self.sample.image, row["x"], row["y"], self.crop_size)
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "sample_id": self.sample.sample_id,
            "cell_id": (
                f"{row['sample_id']}/{row['tile_id']}/{row['cell_id']}"
            ),
        }
