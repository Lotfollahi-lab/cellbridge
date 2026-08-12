"""Reconstructed OOD images and located cells for embedding inference."""

from __future__ import annotations

import pickle
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path

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


def load_metadata(
    metadata_path,
    split_path,
    *,
    require_all_samples: bool = True,
) -> pd.DataFrame:
    """Load cells in CSV order and attach their normalized official split."""
    metadata_path = Path(metadata_path)
    metadata = pd.read_csv(metadata_path)
    required = {"sample_id", "tile_id", "cell_id", "label"}
    if not required <= set(metadata):
        raise ValueError(
            f"{metadata_path} is missing {sorted(required - set(metadata))}"
        )
    if metadata[list(required)].isna().any().any():
        raise ValueError(f"{metadata_path} contains null cell metadata")
    if metadata.duplicated(["sample_id", "tile_id", "cell_id"]).any():
        raise ValueError(f"{metadata_path} contains duplicate cell IDs")

    if "row_id" in metadata:
        if metadata["row_id"].isna().any() or metadata["row_id"].duplicated().any():
            raise ValueError(f"{metadata_path} contains invalid row IDs")
    else:
        metadata.insert(0, "row_id", np.arange(len(metadata), dtype=np.int64))

    split_path = Path(split_path)
    official = pd.read_csv(split_path)
    split_columns = {"sample_name", "train_test_val_split"}
    if not split_columns <= set(official):
        raise ValueError(
            f"{split_path} is missing {sorted(split_columns - set(official))}"
        )
    if official[list(split_columns)].isna().any().any():
        raise ValueError(f"{split_path} contains null sample metadata")
    official = official[list(split_columns)].copy()
    official["sample_name"] = official["sample_name"].astype(str)
    official["train_test_val_split"] = official["train_test_val_split"].replace(
        {"valid": "val"}
    )
    if official["sample_name"].duplicated().any():
        raise ValueError(f"{split_path} contains duplicate samples")
    unknown_splits = set(official["train_test_val_split"]) - {"train", "val", "test"}
    if unknown_splits:
        raise ValueError(
            f"{split_path} contains unknown splits: {sorted(unknown_splits)}"
        )

    split_by_sample = dict(
        official[["sample_name", "train_test_val_split"]].itertuples(
            index=False, name=None
        )
    )
    sample_ids = metadata["sample_id"].astype(str)
    observed = set(sample_ids)
    expected = set(split_by_sample)
    metadata_only = observed - expected
    split_only = expected - observed
    if metadata_only or (require_all_samples and split_only):
        raise ValueError(
            "metadata and official split contain different samples: "
            f"metadata_only={sorted(metadata_only)[:5]}, "
            f"split_only={sorted(split_only)[:5]}"
        )

    resolved = sample_ids.map(split_by_sample)
    for column in ("split", "split_id"):
        if column in metadata:
            recorded = metadata[column].replace({"valid": "val"}).astype(str)
            if not recorded.equals(resolved):
                raise ValueError(f"{metadata_path} disagrees with the official split")
    metadata["split"] = resolved
    return metadata


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
        canvas[:, geometry.top : geometry.bottom, geometry.left : geometry.right] = (
            pixels
        )
    return canvas


def _instance_geometry(mask) -> dict[int, tuple[float, float, int, int, int, int]]:
    """Return centroid and exclusive bounds for each non-background instance."""
    mask = np.asarray(mask, dtype=np.int64)
    if mask.ndim != 2 or np.any(mask < 0):
        raise ValueError(
            f"instance mask must be a non-negative 2D array, got {mask.shape}"
        )
    y, x = np.indices(mask.shape)
    counts = np.bincount(mask.ravel())
    x_sums = np.bincount(mask.ravel(), weights=x.ravel(), minlength=len(counts))
    y_sums = np.bincount(mask.ravel(), weights=y.ravel(), minlength=len(counts))
    x0 = np.full(len(counts), mask.shape[1], dtype=int)
    y0 = np.full(len(counts), mask.shape[0], dtype=int)
    x1 = np.zeros(len(counts), dtype=int)
    y1 = np.zeros(len(counts), dtype=int)
    np.minimum.at(x0, mask.ravel(), x.ravel())
    np.minimum.at(y0, mask.ravel(), y.ravel())
    np.maximum.at(x1, mask.ravel(), x.ravel() + 1)
    np.maximum.at(y1, mask.ravel(), y.ravel() + 1)
    return {
        instance: (
            x_sums[instance] / counts[instance],
            y_sums[instance] / counts[instance],
            x0[instance],
            y0[instance],
            x1[instance],
            y1[instance],
        )
        for instance in np.flatnonzero(counts)[1:]
    }


def _locate_cells(metadata, entries, layout, *, skip_missing) -> pd.DataFrame:
    geometries = {geometry.index: geometry for geometry in layout.tiles}
    cells = metadata.copy()
    columns = ["x", "y", "x0", "y0", "x1", "y1"]
    cells[columns] = np.nan
    for index, entry in entries.items():
        geometry = geometries[index]
        rows = cells["tile_id"].astype(str) == str(entry["tile_name"])
        mask = np.asarray(entry["instance_mask"])
        tile_shape = np.asarray(entry["image"]).shape[-2:]
        if mask.shape != tile_shape:
            raise ValueError(f"tile {index} image and instance mask shapes differ")
        mask = mask[
            geometry.pad_top : geometry.pad_top + geometry.height,
            geometry.pad_left : geometry.pad_left + geometry.width,
        ]
        instances = _instance_geometry(mask)
        for row, instance in cells.loc[rows, "cell_id"].astype(int).items():
            if instance not in instances:
                if skip_missing:
                    continue
                raise ValueError(
                    f"{entry['tile_name']}: instance {instance} is missing"
                )
            x, y, x0, y0, x1, y1 = instances[instance]
            cells.loc[row, columns] = (
                geometry.left + x,
                geometry.top + y,
                geometry.left + x0,
                geometry.top + y0,
                geometry.left + x1,
                geometry.top + y1,
            )

    missing = cells[columns].isna().any(axis=1)
    if missing.any() and not skip_missing:
        raise ValueError(f"could not locate {int(missing.sum())} cells")
    cells = cells.loc[~missing].copy()
    cells[["x0", "y0", "x1", "y1"]] = cells[["x0", "y0", "x1", "y1"]].astype(int)
    return cells


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


def iter_samples(
    lmdb_path,
    metadata_path,
    split_path,
    *,
    image_shape=None,
    tile_size: int = 224,
    value_scale: float = 1,
    skip_missing: bool = False,
    allow_unlabeled_samples: bool = False,
) -> Iterator[OodSample]:
    """Yield fully reconstructed, officially annotated samples one at a time."""
    metadata = load_metadata(
        metadata_path,
        split_path,
        require_all_samples=not allow_unlabeled_samples,
    )
    expected = set(metadata["sample_id"].astype(str))
    official = set(pd.read_csv(split_path)["sample_name"].astype(str))
    seen = set()
    for sample_id, entries in iter_lmdb_samples(lmdb_path):
        sample_id = str(sample_id)
        if sample_id in seen:
            raise ValueError(f"LMDB contains non-consecutive entries for {sample_id!r}")
        if sample_id not in official:
            raise ValueError(f"LMDB contains unknown sample {sample_id!r}")
        seen.add(sample_id)
        if sample_id not in expected:
            if allow_unlabeled_samples:
                continue
            raise ValueError(f"{sample_id}: no cell metadata")
        yield load_sample(
            sample_id,
            entries,
            metadata,
            image_shape=image_shape,
            tile_size=tile_size,
            value_scale=value_scale,
            skip_missing=skip_missing,
        )
    missing = expected - seen
    if missing:
        raise ValueError(f"LMDB is missing samples: {sorted(missing)[:5]}")


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


Cropper = Callable[[np.ndarray, pd.Series], Image.Image]


def center_cell_crop(
    image,
    cell,
    *,
    size: int = CROP_SIZE,
    fill: int = 255,
) -> Image.Image:
    """Crop one located cell around its centroid."""
    return center_crop(image, cell["x"], cell["y"], size=size, fill=fill)


class OodCrops:
    """Map-style dataset of method-specific crops from one OOD image."""

    def __init__(
        self,
        sample: OodSample,
        transform: Callable | None = None,
        *,
        cropper: Cropper | None = None,
        crop_size: int = CROP_SIZE,
    ):
        if crop_size < 1:
            raise ValueError("crop_size must be positive")
        self.sample = sample
        self.transform = transform
        self.cropper = cropper or partial(center_cell_crop, size=crop_size)

    def __len__(self) -> int:
        return len(self.sample.cells)

    def __getitem__(self, index: int) -> dict:
        if not -len(self) <= index < len(self):
            raise IndexError(index)
        index %= len(self)
        row = self.sample.cells.iloc[index]
        image = self.cropper(self.sample.image, row)
        if not isinstance(image, Image.Image):
            raise TypeError("cropper must return a PIL image")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "sample_id": self.sample.sample_id,
            "cell_id": (f"{row['sample_id']}/{row['tile_id']}/{row['cell_id']}"),
        }
