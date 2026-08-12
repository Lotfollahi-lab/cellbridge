import numpy as np
import pandas as pd
import pytest

from cellbridge.inference.embed import ood


def _entry(sample_id, mask):
    size = mask.shape[0]
    return {
        "sample_name": sample_id,
        "tile_name": f"{sample_id}_TILE_0",
        "image": np.full((3, size, size), 250, dtype=np.uint16),
        "instance_mask": mask,
    }


def _write_tables(tmp_path):
    metadata_path = tmp_path / "cells.csv"
    split_path = tmp_path / "split.csv"
    pd.DataFrame(
        {
            "sample_id": ["B", "A"],
            "tile_id": ["B_TILE_0", "A_TILE_0"],
            "cell_id": [1, 1],
            "label": ["T", "B"],
            "split_id": ["valid", "train"],
            "area": [2, 4],
        }
    ).to_csv(metadata_path, index=False)
    pd.DataFrame(
        {
            "sample_name": ["A", "B"],
            "train_test_val_split": ["train", "valid"],
        }
    ).to_csv(split_path, index=False)
    return metadata_path, split_path


def test_load_metadata_preserves_rows_and_attaches_official_split(tmp_path):
    metadata_path, split_path = _write_tables(tmp_path)

    metadata = ood.load_metadata(metadata_path, split_path)

    assert metadata["row_id"].tolist() == [0, 1]
    assert metadata["sample_id"].tolist() == ["B", "A"]
    assert metadata["area"].tolist() == [2, 4]
    assert metadata["split_id"].tolist() == ["valid", "train"]
    assert metadata["split"].tolist() == ["val", "train"]


def test_load_metadata_rejects_split_disagreement(tmp_path):
    metadata_path, split_path = _write_tables(tmp_path)
    metadata = pd.read_csv(metadata_path)
    metadata.loc[0, "split_id"] = "test"
    metadata.to_csv(metadata_path, index=False)

    with pytest.raises(ValueError, match="disagrees with the official split"):
        ood.load_metadata(metadata_path, split_path)


def test_load_sample_adds_global_centroids_and_exclusive_bounds():
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 2:4] = 1
    entry = _entry("A", mask)
    metadata = pd.DataFrame(
        {
            "row_id": [7],
            "sample_id": ["A"],
            "tile_id": ["A_TILE_0"],
            "cell_id": [1],
            "label": ["B"],
            "split": ["train"],
        }
    )

    sample = ood.load_sample("A", {0: entry}, metadata, image_shape=(4, 4), tile_size=4)

    assert sample.cells.loc[0, "row_id"] == 7
    assert sample.cells.loc[0, ["x", "y"]].tolist() == [2.5, 1.5]
    assert sample.cells.loc[0, ["x0", "y0", "x1", "y1"]].tolist() == [2, 1, 4, 3]


def test_iter_samples_keeps_lmdb_order_and_metadata_rows(tmp_path, monkeypatch):
    metadata_path, split_path = _write_tables(tmp_path)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 1

    monkeypatch.setattr(
        ood,
        "iter_lmdb_samples",
        lambda _: iter(
            [
                ("A", {0: _entry("A", mask)}),
                ("B", {0: _entry("B", mask)}),
            ]
        ),
    )

    samples = list(
        ood.iter_samples(
            "unused.lmdb",
            metadata_path,
            split_path,
            image_shape=(4, 4),
            tile_size=4,
        )
    )

    assert [sample.sample_id for sample in samples] == ["A", "B"]
    assert [sample.cells.iloc[0]["row_id"] for sample in samples] == [1, 0]
    assert [sample.cells.iloc[0]["split"] for sample in samples] == ["train", "val"]
