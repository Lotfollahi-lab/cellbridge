"""Read tiled LMDB samples and recover cell-centred image crops."""

import pickle
import re
from dataclasses import dataclass

import numpy as np

TILE_PATTERN = re.compile(r"_TILE_(\d+)$")


@dataclass(frozen=True)
class TileGeometry:
    index: int
    y0: int
    y1: int
    x0: int
    x1: int
    pad_top: int
    pad_left: int

    @property
    def height(self):
        return self.y1 - self.y0

    @property
    def width(self):
        return self.x1 - self.x0


def tile_index(tile_name):
    match = TILE_PATTERN.search(str(tile_name))
    if match is None:
        raise ValueError(f"cannot parse tile index from {tile_name!r}")
    return int(match.group(1))


def tile_geometries(image_shape=(1440, 1920), tile_size=224):
    """Reproduce the benchmark's row-major, evenly spaced tiling."""
    height, width = map(int, image_shape)
    rows = int(np.ceil(height / tile_size))
    columns = int(np.ceil(width / tile_size))
    y_edges = np.linspace(0, height, rows + 1, dtype=int)
    x_edges = np.linspace(0, width, columns + 1, dtype=int)

    geometries = []
    for row in range(rows):
        for column in range(columns):
            y0, y1 = int(y_edges[row]), int(y_edges[row + 1])
            x0, x1 = int(x_edges[column]), int(x_edges[column + 1])
            geometries.append(
                TileGeometry(
                    index=row * columns + column,
                    y0=y0,
                    y1=y1,
                    x0=x0,
                    x1=x1,
                    pad_top=(tile_size - (y1 - y0)) // 2,
                    pad_left=(tile_size - (x1 - x0)) // 2,
                )
            )
    return geometries


def reconstruct_image(tiles, image_shape=(1440, 1920), tile_size=224):
    """Undo PhenoCell's per-tile zero padding and rebuild the RGB field."""
    height, width = map(int, image_shape)
    canvas = np.empty((3, height, width), dtype=np.asarray(next(iter(tiles.values()))).dtype)
    geometries = tile_geometries(image_shape, tile_size)
    if set(tiles) != {geometry.index for geometry in geometries}:
        raise ValueError("tile set does not match the configured field-of-view shape")

    for geometry in geometries:
        tile = np.asarray(tiles[geometry.index])
        if tile.shape != (3, tile_size, tile_size):
            raise ValueError(f"tile {geometry.index} has unexpected shape {tile.shape}")
        source = tile[
            :,
            geometry.pad_top : geometry.pad_top + geometry.height,
            geometry.pad_left : geometry.pad_left + geometry.width,
        ]
        canvas[:, geometry.y0 : geometry.y1, geometry.x0 : geometry.x1] = source
    return canvas


def instance_centroids(mask):
    """Return local (x, y) centroids indexed by non-zero instance ID."""
    mask = np.asarray(mask, dtype=np.int64)
    y, x = np.indices(mask.shape)
    flat = mask.ravel()
    counts = np.bincount(flat)
    x_sums = np.bincount(flat, weights=x.ravel(), minlength=len(counts))
    y_sums = np.bincount(flat, weights=y.ravel(), minlength=len(counts))
    return {
        instance_id: (x_sums[instance_id] / counts[instance_id], y_sums[instance_id] / counts[instance_id])
        for instance_id in np.flatnonzero(counts)[1:]
    }


def locate_cells(metadata, entries, image_shape=(1440, 1920), tile_size=224, *, skip_missing=False):
    """Add field-level x/y coordinates to one sample's metadata rows.

    With ``skip_missing`` set, cells whose instance is absent from its tile mask
    are dropped instead of raising; some benchmarks (e.g. PanNuke) list boundary
    cells whose instance carries no pixels in the tiled mask.
    """
    geometries = {geometry.index: geometry for geometry in tile_geometries(image_shape, tile_size)}
    located = metadata.copy()
    located["x"] = np.nan
    located["y"] = np.nan

    for index, entry in entries.items():
        geometry = geometries[index]
        rows = located["tile_id"].astype(str) == str(entry["tile_name"])
        if not rows.any():
            continue
        centroids = instance_centroids(entry["instance_mask"])
        for row_index, instance_id in located.loc[rows, "cell_id"].astype(int).items():
            if instance_id not in centroids:
                if skip_missing:
                    continue
                raise ValueError(f"{entry['tile_name']}: instance {instance_id} is missing from its mask")
            local_x, local_y = centroids[instance_id]
            located.at[row_index, "x"] = geometry.x0 + local_x - geometry.pad_left
            located.at[row_index, "y"] = geometry.y0 + local_y - geometry.pad_top

    missing = located[["x", "y"]].isna().any(axis=1)
    if missing.any():
        if skip_missing:
            return located.loc[~missing].copy()
        names = located.loc[missing, "tile_id"].astype(str).unique()[:5]
        raise ValueError(f"could not locate {missing.sum()} cells; tiles include {names.tolist()}")
    return located


def centered_crop(image, x, y, size, fill=255):
    """Crop an RGB context window, padding outside the field if necessary."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HWC RGB image, got {image.shape}")
    size = int(size)
    if size < 1:
        raise ValueError("crop size must be positive")

    x0 = int(round(x)) - size // 2
    y0 = int(round(y)) - size // 2
    x1, y1 = x0 + size, y0 + size
    height, width = image.shape[:2]
    source_x0, source_y0 = max(x0, 0), max(y0, 0)
    source_x1, source_y1 = min(x1, width), min(y1, height)
    crop = np.full((size, size, 3), fill, dtype=image.dtype)
    if source_x1 > source_x0 and source_y1 > source_y0:
        crop[
            source_y0 - y0 : source_y1 - y0,
            source_x0 - x0 : source_x1 - x0,
        ] = image[source_y0:source_y1, source_x0:source_x1]
    return crop


def to_uint8_rgb(image, value_scale=250):
    """Convert the benchmark's CHW uint16 encoding back to RGB uint8."""
    image = np.asarray(image, dtype=np.float32) / float(value_scale)
    return np.clip(np.rint(image), 0, 255).astype(np.uint8).transpose(1, 2, 0)


def iter_lmdb_samples(path):
    """Yield consecutive ``(sample_name, {tile_index: entry})`` groups."""
    try:
        import lmdb
    except ImportError as exc:
        raise ImportError(
            "LMDB reading requires lmdb; run through the supplied uv --with lmdb launcher"
        ) from exc

    environment = lmdb.open(str(path), readonly=True, lock=False, readahead=False, meminit=False)
    try:
        with environment.begin(write=False) as transaction:
            sample_name, entries = None, {}
            for _, value in transaction.cursor():
                entry = pickle.loads(value)
                current = str(entry["sample_name"])
                if sample_name is not None and current != sample_name:
                    yield sample_name, entries
                    entries = {}
                sample_name = current
                index = tile_index(entry["tile_name"])
                if index in entries:
                    raise ValueError(f"{sample_name}: duplicate tile index {index}")
                entries[index] = entry
            if sample_name is not None:
                yield sample_name, entries
    finally:
        environment.close()
