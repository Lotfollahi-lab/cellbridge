"""Describe how padded image tiles fit together."""

from dataclasses import dataclass

import numpy as np


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


def fixed_layout(image_shape, tile_size: int) -> TileLayout:
    """Build the known row-major layout for an image of a fixed size."""
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
    """Return top, bottom, left, and right bounds of nonzero tile content."""
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


def infer_layout(tile_arrays: dict[int, np.ndarray]) -> TileLayout:
    """Infer a rectangular layout from central padding and matching seams."""
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
        heights = {content[row * columns + column].shape[1] for column in range(columns)}
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
