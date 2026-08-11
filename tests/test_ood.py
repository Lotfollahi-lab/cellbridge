import numpy as np
import pandas as pd

from cellbridge.evaluation.embed_ood import (
    complete_manifest,
    crop_valid_fraction,
    embedding_signature,
    resolve_crop_spec,
    select_development_subset,
    validate_existing_signature,
)
from cellbridge.evaluation.ood import validate_official_split
from cellbridge.evaluation.ood_data import (
    centered_crop,
    instance_centroids,
    locate_cells,
    reconstruct_image,
    tile_geometries,
)


def split_into_tiles(image, tile_size):
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
    tiles = split_into_tiles(image, tile_size=4)
    assert np.array_equal(reconstruct_image(tiles, image.shape[1:], 4), image)


def test_pannuke_field_splits_into_four_tiles():
    """PanNuke's 256x256 field tiles into a 2x2 grid at tile_size 224."""
    geometries = tile_geometries((256, 256), 224)
    assert [g.index for g in geometries] == [0, 1, 2, 3]
    image = np.arange(3 * 256 * 256, dtype=np.uint16).reshape(3, 256, 256)
    tiles = split_into_tiles(image, tile_size=224)
    assert np.array_equal(reconstruct_image(tiles, (256, 256), 224), image)


def _split_file(tmp_path, rows):
    path = tmp_path / "split.csv"
    pd.DataFrame(rows, columns=["sample_name", "train_test_val_split"]).to_csv(
        path, index=False
    )
    return path


def test_unlabeled_samples_relax_split_coverage(tmp_path):
    # "D" is an official sample with no labelled cells, so features never cover it.
    split_path = _split_file(
        tmp_path,
        [("A", "train"), ("B", "valid"), ("C", "test"), ("D", "train")],
    )
    table = pd.DataFrame(
        {"sample_id": ["A", "B", "C"], "split": ["train", "val", "test"]}
    )
    validate_official_split(table, split_path, require_full_coverage=False)
    with np.testing.assert_raises_regex(ValueError, "cover every official sample"):
        validate_official_split(table, split_path)


def test_split_validation_rejects_unknown_samples(tmp_path):
    split_path = _split_file(tmp_path, [("A", "train"), ("B", "test")])
    table = pd.DataFrame({"sample_id": ["A", "Z"], "split": ["train", "test"]})
    with np.testing.assert_raises_regex(ValueError, "absent from the official split"):
        validate_official_split(table, split_path, require_full_coverage=False)


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


def test_locate_cells_skips_missing_instances_when_allowed():
    entries = {}
    for geometry in tile_geometries((5, 7), 4):
        entries[geometry.index] = {
            "tile_name": f"sample_TILE_{geometry.index}",
            "instance_mask": np.zeros((4, 4), dtype=np.uint8),
        }
    entries[1]["instance_mask"][2, 1] = 1  # only instance 1 exists; 2 is absent
    metadata = pd.DataFrame(
        {
            "tile_id": ["sample_TILE_1", "sample_TILE_1"],
            "cell_id": [1, 2],
            "label": ["T cells", "B cells"],
        }
    )
    with np.testing.assert_raises_regex(ValueError, "instance 2 is missing"):
        locate_cells(metadata, entries, (5, 7), 4)

    located = locate_cells(metadata, entries, (5, 7), 4, skip_missing=True)
    assert located["cell_id"].tolist() == [1]
    assert located.loc[located.index[0], ["x", "y"]].tolist() == [4, 1]


def test_centered_crop_pads_field_edge_in_white():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    crop = centered_crop(image, 0, 0, 4)
    assert np.all(crop[:2] == 255)
    assert np.all(crop[:, :2] == 255)
    assert np.all(crop[2:, 2:] == 0)


def test_physical_crop_resolves_source_pixels_and_actual_fov():
    phenocell = resolve_crop_spec(
        {
            "data": {"source_mpp": 0.377},
            "embedding": {"fov_um": 112.0, "output_dir": "unused"},
        }
    )
    assert phenocell.source_size_px == 297
    assert phenocell.target_size_px == 224
    assert phenocell.actual_fov_um == 297 * 0.377

    pannuke = resolve_crop_spec(
        {
            "data": {"source_mpp": 0.25},
            "embedding": {"fov_um": 56.0, "output_dir": "unused"},
        }
    )
    assert pannuke.source_size_px == 224
    assert pannuke.actual_fov_um == 56.0

    legacy = resolve_crop_spec(
        {
            "data": {"source_mpp": 0.377},
            "embedding": {
                "fov_um": None,
                "crop_size": 96,
                "output_dir": "unused",
            },
        }
    )
    assert legacy.mode == "legacy_pixels"
    assert legacy.source_size_px == 96
    assert legacy.actual_fov_um == 96 * 0.377


def test_physical_crop_requires_source_mpp():
    with np.testing.assert_raises_regex(ValueError, "source_mpp"):
        resolve_crop_spec(
            {
                "data": {},
                "embedding": {"fov_um": 112.0, "output_dir": "unused"},
            }
        )


def test_development_subset_is_stratified_reproducible_and_excludes_test():
    metadata = pd.DataFrame(
        {
            "row_id": np.arange(18),
            "split_id": np.repeat(["train", "valid", "test"], 6),
            "label": np.tile(np.repeat(["A", "B"], 3), 3),
        }
    )
    settings = {
        "max_cells_per_class_per_split": 2,
        "seed": 7,
        "splits": ["train", "val"],
    }
    first = select_development_subset(metadata, settings)
    second = select_development_subset(metadata, settings)

    assert first["row_id"].tolist() == second["row_id"].tolist()
    assert set(first["split_id"]) == {"train", "valid"}
    assert first.groupby(["split_id", "label"]).size().eq(2).all()


def test_crop_valid_fraction_reports_padding():
    locations = pd.DataFrame({"x": [0, 2], "y": [0, 2]})
    fractions = crop_valid_fraction((4, 4, 3), locations, crop_size=4)
    np.testing.assert_allclose(fractions, [0.25, 1.0])


def test_complete_uni_manifest_requires_every_expected_cell(tmp_path):
    sample = tmp_path / "sample.npz"
    np.savez(sample, uni_emb=np.ones((2, 3)))
    pd.DataFrame(
        {"sample_id": ["A"], "n_cells": [2], "path": [str(sample)]}
    ).to_csv(tmp_path / "manifest.csv", index=False)
    metadata = pd.DataFrame({"sample_id": ["A", "A"]})
    assert complete_manifest(tmp_path, metadata) is not None

    metadata.loc[2] = "A"
    assert complete_manifest(tmp_path, metadata) is None
    # With unlabeled tolerance, a lower embedded count (skipped cells) is complete,
    # but a missing sample or an impossible over-count is not.
    assert complete_manifest(tmp_path, metadata, allow_unlabeled=True) is not None
    over = pd.DataFrame({"sample_id": ["A"]})
    assert complete_manifest(tmp_path, over, allow_unlabeled=True) is None


def test_uni_cache_rejects_a_different_crop_size(tmp_path):
    cfg = {
        "data": {"metadata": "cells.csv"},
        "embedding": {"encoder": "uni_v2", "crop_size": 96, "output_dir": str(tmp_path)},
    }
    with open(tmp_path / "embedding_config.yaml", "w") as handle:
        import yaml

        yaml.safe_dump(embedding_signature(cfg), handle)
    validate_existing_signature(tmp_path, cfg)

    cfg["embedding"]["crop_size"] = 128
    with np.testing.assert_raises_regex(ValueError, "different encoder, crop, or dataset"):
        validate_existing_signature(tmp_path, cfg)
