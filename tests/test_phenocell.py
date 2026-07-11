import numpy as np
import pandas as pd

from terra_multi.evaluation.phenocell_data import (
    centered_crop,
    instance_centroids,
    locate_cells,
    reconstruct_image,
    tile_geometries,
)


def split_like_phenocell(image, tile_size):
    tiles = {}
    for geometry in tile_geometries(image.shape[1:], tile_size):
        tile = np.zeros((3, tile_size, tile_size), dtype=image.dtype)
        tile[
            :,
            geometry.pad_top : geometry.pad_top + geometry.height,
            geometry.pad_left : geometry.pad_left + geometry.width,
        ] = image[:, geometry.y0 : geometry.y1, geometry.x0 : geometry.x1]
        tiles[geometry.index] = tile
    return tiles


def test_reconstruct_image_undoes_even_tiling_and_padding():
    image = np.arange(3 * 5 * 7, dtype=np.uint16).reshape(3, 5, 7)
    tiles = split_like_phenocell(image, tile_size=4)
    assert np.array_equal(reconstruct_image(tiles, image.shape[1:], 4), image)


def test_instance_centroids_ignore_background():
    mask = np.array([[0, 1, 1], [0, 0, 2], [0, 0, 2]])
    assert instance_centroids(mask) == {1: (1.5, 0.0), 2: (2.0, 1.5)}


def test_locate_cells_removes_tile_padding():
    entries = {}
    for geometry in tile_geometries((5, 7), 4):
        mask = np.zeros((4, 4), dtype=np.uint8)
        entries[geometry.index] = {
            "tile_name": f"sample_TILE_{geometry.index}",
            "instance_mask": mask,
        }
    entries[1]["instance_mask"][2, 1] = 1
    metadata = pd.DataFrame(
        {"tile_id": ["sample_TILE_1"], "cell_id": [1], "label": ["T cells"]}
    )
    located = locate_cells(metadata, entries, (5, 7), 4)
    assert located.loc[0, "x"] == 4
    assert located.loc[0, "y"] == 1


def test_centered_crop_pads_field_edge_in_white():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    crop = centered_crop(image, 0, 0, 4)
    assert np.all(crop[:2] == 255)
    assert np.all(crop[:, :2] == 255)
    assert np.all(crop[2:, 2:] == 0)
