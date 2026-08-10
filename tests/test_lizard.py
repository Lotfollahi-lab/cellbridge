import numpy as np
import pandas as pd

from cellbridge.evaluation.ood_data import (
    locate_cells,
    reconstruct_image,
    resolve_tile_layout,
)


def _padded_tiles(source, rows, columns, tile_size):
    _, height, width = source.shape
    tile_height = height // rows
    tile_width = width // columns
    tiles = {}
    for row in range(rows):
        for column in range(columns):
            index = row * columns + column
            content = source[
                :,
                row * tile_height : (row + 1) * tile_height,
                column * tile_width : (column + 1) * tile_width,
            ]
            tile = np.zeros((3, tile_size, tile_size), dtype=source.dtype)
            top = (tile_size - tile_height) // 2
            left = (tile_size - tile_width) // 2
            tile[:, top : top + tile_height, left : left + tile_width] = content
            tiles[index] = tile
    return tiles


def test_lizard_layout_uses_padding_and_seam_continuity():
    rows, columns = 3, 2
    content_size, tile_size = 5, 7
    height, width = rows * content_size, columns * content_size
    y, x = np.indices((height, width))
    source = np.stack(
        [
            10 + 2 * x + y,
            20 + x + 3 * y,
            30 + 2 * x + 2 * y,
        ]
    ).astype(np.uint16)
    tiles = _padded_tiles(source, rows, columns, tile_size)

    layout = resolve_tile_layout(tiles, image_shape=None, tile_size=tile_size)
    assert layout.grid_width == columns
    assert layout.image_shape == (height, width)
    assert all(item.pad_top == 1 and item.pad_left == 1 for item in layout.geometries)

    image = reconstruct_image(
        tiles, image_shape=None, tile_size=tile_size, layout=layout
    )
    np.testing.assert_array_equal(image, source)

    mask = np.zeros((tile_size, tile_size), dtype=np.int8)
    local_x, local_y = 2, 3
    mask[1 + local_y, 1 + local_x] = 1
    entries = {
        index: {
            "image": tile,
            "tile_name": f"sample_TILE_{index}",
            "instance_mask": mask if index == 5 else np.zeros_like(mask),
        }
        for index, tile in tiles.items()
    }
    metadata = pd.DataFrame({"tile_id": ["sample_TILE_5"], "cell_id": [1]})
    located = locate_cells(
        metadata,
        entries,
        image_shape=None,
        tile_size=tile_size,
        layout=layout,
    )
    assert located.loc[0, ["x", "y"]].tolist() == [
        content_size + local_x,
        2 * content_size + local_y,
    ]
