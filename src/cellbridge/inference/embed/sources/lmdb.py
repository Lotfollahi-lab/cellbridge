"""Reconstruct tiled-LMDB samples for shared embedding inference."""

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

from cellbridge.inference.embed.crops import center_cell_crop, spatial_cropper
from cellbridge.inference.embed.sources import SourceSample
from cellbridge.inference.embed.sources.metadata import load_tiled_metadata

CROP_SIZE = 224
_TILE_PATTERN = re.compile(r"_TILE_(\d+)$")
make_cropper = spatial_cropper


@dataclass(frozen=True)
class Tile:
    index: int
    top: int
    left: int
    height: int
    width: int
    pad_top: int
    pad_left: int

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def right(self) -> int:
        return self.left + self.width


@dataclass(frozen=True)
class TileLayout:
    tiles: tuple[Tile, ...]
    image_shape: tuple[int, int]
    columns: int
    mode: str
    seam_score: float | None = None


def _fixed_layout(image_shape, tile_size: int) -> TileLayout:
    height, width = map(int, image_shape)
    rows = int(np.ceil(height / tile_size))
    columns = int(np.ceil(width / tile_size))
    y_edges = np.linspace(0, height, rows + 1, dtype=int)
    x_edges = np.linspace(0, width, columns + 1, dtype=int)
    tiles = []
    for row in range(rows):
        for column in range(columns):
            top, bottom = y_edges[row], y_edges[row + 1]
            left, right = x_edges[column], x_edges[column + 1]
            tile_height, tile_width = int(bottom - top), int(right - left)
            tiles.append(
                Tile(
                    index=row * columns + column,
                    top=int(top),
                    left=int(left),
                    height=tile_height,
                    width=tile_width,
                    pad_top=(tile_size - tile_height) // 2,
                    pad_left=(tile_size - tile_width) // 2,
                )
            )
    return TileLayout(tuple(tiles), (height, width), columns, "configured")


def _content_box(tile) -> tuple[int, int, int, int]:
    tile = np.asarray(tile)
    if tile.ndim != 3:
        raise ValueError(f"expected a CHW tile, got {tile.shape}")
    occupied = np.any(tile != 0, axis=0)
    rows = np.flatnonzero(occupied.any(axis=1))
    columns = np.flatnonzero(occupied.any(axis=0))
    if not len(rows) or not len(columns):
        raise ValueError("cannot infer a layout from an empty tile")
    return int(rows[0]), int(rows[-1] + 1), int(columns[0]), int(columns[-1] + 1)


def _edge_error(first, second, horizontal: bool, strip: int = 3) -> float:
    if horizontal:
        length = min(first.shape[1], second.shape[1])
        a = first[:, :length, -strip:].mean(axis=2)
        b = second[:, :length, :strip].mean(axis=2)
    else:
        length = min(first.shape[2], second.shape[2])
        a = first[:, -strip:, :length].mean(axis=1)
        b = second[:, :strip, :length].mean(axis=1)
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def _seam_score(tiles, columns: int) -> float:
    errors = []
    for index, tile in enumerate(tiles):
        if index % columns:
            errors.append(_edge_error(tiles[index - 1], tile, horizontal=True))
        if index >= columns:
            errors.append(_edge_error(tiles[index - columns], tile, horizontal=False))
    return float(np.mean(errors))


def _infer_layout(tile_arrays: dict[int, np.ndarray]) -> TileLayout:
    count = len(tile_arrays)
    if set(tile_arrays) != set(range(count)):
        raise ValueError("tile indices must be consecutive and start at zero")
    arrays = [np.asarray(tile_arrays[index]) for index in range(count)]
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("tiles have inconsistent shapes")

    _, tile_height, tile_width = arrays[0].shape
    boxes = [_content_box(tile) for tile in arrays]
    for top, bottom, left, right in boxes:
        if (
            abs(top - (tile_height - bottom)) > 1
            or abs(left - (tile_width - right)) > 1
        ):
            raise ValueError("tiles must use central zero padding")
    content = [
        tile[:, top:bottom, left:right]
        for tile, (top, bottom, left, right) in zip(arrays, boxes)
    ]

    candidates = [columns for columns in range(2, count) if count % columns == 0]
    if not candidates:
        raise ValueError(f"cannot infer a rectangular grid from {count} tiles")
    scores = {columns: _seam_score(content, columns) for columns in candidates}
    columns = min(scores, key=scores.get)
    rows = count // columns

    row_heights = []
    for row in range(rows):
        heights = {
            content[row * columns + column].shape[1] for column in range(columns)
        }
        if len(heights) != 1:
            raise ValueError(f"row {row} has inconsistent tile heights")
        row_heights.append(heights.pop())
    column_widths = []
    for column in range(columns):
        widths = {content[row * columns + column].shape[2] for row in range(rows)}
        if len(widths) != 1:
            raise ValueError(f"column {column} has inconsistent tile widths")
        column_widths.append(widths.pop())

    y_edges = np.concatenate(([0], np.cumsum(row_heights)))
    x_edges = np.concatenate(([0], np.cumsum(column_widths)))
    tiles = []
    for index, (pad_top, _, pad_left, _) in enumerate(boxes):
        row, column = divmod(index, columns)
        tiles.append(
            Tile(
                index=index,
                top=int(y_edges[row]),
                left=int(x_edges[column]),
                height=row_heights[row],
                width=column_widths[column],
                pad_top=pad_top,
                pad_left=pad_left,
            )
        )
    return TileLayout(
        tuple(tiles),
        (int(y_edges[-1]), int(x_edges[-1])),
        columns,
        "inferred",
        scores[columns],
    )


@dataclass(frozen=True)
class OodSample:
    sample_id: str
    image: np.ndarray
    cells: pd.DataFrame
    layout: TileLayout


def _tile_index(tile_name) -> int:
    match = _TILE_PATTERN.search(str(tile_name))
    if match is None:
        raise ValueError(f"cannot parse tile index from {tile_name!r}")
    return int(match.group(1))


def _iter_entries(path) -> Iterator[tuple[str, dict[int, dict]]]:
    try:
        import lmdb
    except ImportError as error:
        raise ImportError("reading tiled data requires the lmdb package") from error

    environment = lmdb.open(
        str(Path(path)),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
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
        return _fixed_layout(tuple(image_shape), tile_size)
    return _infer_layout({index: entry["image"] for index, entry in entries.items()})


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
        if mask.shape != np.asarray(entry["image"]).shape[-2:]:
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


def _load_sample(
    sample_id,
    entries,
    metadata,
    *,
    image_shape,
    tile_size,
    value_scale,
    skip_missing,
) -> OodSample:
    sample_id = str(sample_id)
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


Cropper = Callable[[np.ndarray, pd.Series], Image.Image]


class OodCrops:
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
        row = self.sample.cells.iloc[index % len(self)]
        image = self.cropper(self.sample.image, row)
        if not isinstance(image, Image.Image):
            raise TypeError("cropper must return a PIL image")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "sample_id": self.sample.sample_id,
            "cell_id": f"{row['sample_id']}/{row['tile_id']}/{row['cell_id']}",
        }


def load_metadata(data):
    """Load normalized metadata for a tiled-LMDB source."""
    return load_tiled_metadata(
        data["metadata"],
        data["split"],
        require_all_samples=not data.get("allow_unlabeled_samples", False),
    )


def check(data):
    """Validate source paths and return normalized metadata."""
    for key in ("lmdb", "metadata", "split"):
        if not Path(data[key]).exists():
            raise FileNotFoundError(f"data.{key} does not exist: {data[key]}")
    return load_metadata(data)


def iter_samples(data, *, transform, cropper):
    """Yield aligned crop datasets in existing LMDB sample order."""
    metadata = load_metadata(data)
    expected = set(metadata["sample_id"].astype(str))
    official = set(pd.read_csv(data["split"])["sample_name"].astype(str))
    allow_unlabeled = bool(data.get("allow_unlabeled_samples", False))
    seen = set()

    for sample_id, entries in _iter_entries(data["lmdb"]):
        sample_id = str(sample_id)
        if sample_id in seen:
            raise ValueError(f"LMDB contains non-consecutive entries for {sample_id!r}")
        if sample_id not in official:
            raise ValueError(f"LMDB contains unknown sample {sample_id!r}")
        seen.add(sample_id)
        if sample_id not in expected:
            if allow_unlabeled:
                continue
            raise ValueError(f"{sample_id}: no cell metadata")

        sample = _load_sample(
            sample_id,
            entries,
            metadata,
            image_shape=data.get("image_shape"),
            tile_size=int(data.get("tile_size", 224)),
            value_scale=float(data.get("value_scale", 1)),
            skip_missing=bool(data.get("skip_missing_cells", False)),
        )
        yield SourceSample(
            sample_id=sample.sample_id,
            row_ids=sample.cells["row_id"].to_numpy(),
            dataset=OodCrops(sample, transform=transform, cropper=cropper),
        )

    missing = expected - seen
    if missing:
        raise ValueError(f"LMDB is missing samples: {sorted(missing)[:5]}")
