"""Resolve fixed and content-inferred layouts for padded image tiles."""

from dataclasses import dataclass

import numpy as np


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


@dataclass(frozen=True)
class TileLayout:
    """Resolved row-major tile geometry for one image field."""

    geometries: tuple[TileGeometry, ...]
    image_shape: tuple[int, int]
    grid_width: int
    mode: str
    seam_score: float | None = None


def fixed_tile_layout(image_shape=(1440, 1920), tile_size=224):
    """Reproduce the benchmark's evenly spaced, centrally padded tiling."""
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
    return TileLayout(tuple(geometries), (height, width), columns, "configured")


def _content_box(tile):
    tile = np.asarray(tile)
    if tile.ndim != 3:
        raise ValueError(f"expected CHW tile, got {tile.shape}")
    occupied = np.any(tile != 0, axis=0)
    rows = np.flatnonzero(occupied.any(axis=1))
    columns = np.flatnonzero(occupied.any(axis=0))
    if not len(rows) or not len(columns):
        raise ValueError("cannot infer padding from an empty tile")
    return int(rows[0]), int(rows[-1] + 1), int(columns[0]), int(columns[-1] + 1)


def _candidate_widths(n_tiles):
    candidates = [width for width in range(2, n_tiles) if n_tiles % width == 0]
    if not candidates:
        raise ValueError(f"cannot infer a rectangular grid from {n_tiles} tiles")
    return candidates


def _edge_difference(first, second, axis, strip=3):
    """Mean RGB discontinuity across one proposed tile boundary."""
    if axis == 1:
        length = min(first.shape[1], second.shape[1])
        a = first[:, :length, -strip:].mean(axis=2)
        b = second[:, :length, :strip].mean(axis=2)
    else:
        length = min(first.shape[2], second.shape[2])
        a = first[:, -strip:, :length].mean(axis=1)
        b = second[:, :strip, :length].mean(axis=1)
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def _seam_score(content, width):
    horizontal, vertical = [], []
    for index, tile in enumerate(content):
        if index % width:
            horizontal.append(_edge_difference(content[index - 1], tile, axis=1))
        if index >= width:
            vertical.append(_edge_difference(content[index - width], tile, axis=0))
    return float(np.mean([np.mean(horizontal), np.mean(vertical)]))


def infer_tile_layout(tiles):
    """Infer a rectangular row-major layout from zero-padded tile content."""
    if not tiles or set(tiles) != set(range(len(tiles))):
        raise ValueError("inferred layouts require consecutive tile indices from zero")
    arrays = [np.asarray(tiles[index]) for index in range(len(tiles))]
    tile_shapes = {array.shape for array in arrays}
    if len(tile_shapes) != 1:
        raise ValueError(f"tiles have inconsistent shapes: {sorted(tile_shapes)}")
    _, tile_height, tile_width = arrays[0].shape
    boxes = [_content_box(tile) for tile in arrays]
    for top, bottom, left, right in boxes:
        if (
            abs(top - (tile_height - bottom)) > 1
            or abs(left - (tile_width - right)) > 1
        ):
            raise ValueError("tiles are not centrally zero-padded")
    content = [
        tile[:, top:bottom, left:right]
        for tile, (top, bottom, left, right) in zip(arrays, boxes)
    ]
    scores = {
        width: _seam_score(content, width) for width in _candidate_widths(len(tiles))
    }
    grid_width = min(scores, key=scores.get)
    rows = len(tiles) // grid_width

    row_heights = []
    for row in range(rows):
        heights = [
            content[row * grid_width + column].shape[1] for column in range(grid_width)
        ]
        if max(heights) != min(heights):
            raise ValueError(f"inconsistent content heights in inferred row {row}")
        row_heights.append(heights[0])
    column_widths = []
    for column in range(grid_width):
        widths = [content[row * grid_width + column].shape[2] for row in range(rows)]
        if max(widths) != min(widths):
            raise ValueError(f"inconsistent content widths in inferred column {column}")
        column_widths.append(widths[0])

    y_edges = np.concatenate(([0], np.cumsum(row_heights)))
    x_edges = np.concatenate(([0], np.cumsum(column_widths)))
    geometries = []
    for index, (top, _, left, _) in enumerate(boxes):
        row, column = divmod(index, grid_width)
        geometries.append(
            TileGeometry(
                index=index,
                y0=int(y_edges[row]),
                y1=int(y_edges[row + 1]),
                x0=int(x_edges[column]),
                x1=int(x_edges[column + 1]),
                pad_top=top,
                pad_left=left,
            )
        )
    return TileLayout(
        tuple(geometries),
        (int(y_edges[-1]), int(x_edges[-1])),
        grid_width,
        "inferred_content",
        scores[grid_width],
    )
