from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from terra_multi.baselines.uni_selective_ft import validate_config
from terra_multi.baselines.uni_selective_ft_data import (
    CachedOodCropDataset,
    GroupedBatchSampler,
    stratified_validation_split,
)
from terra_multi.baselines.uni_selective_ft_model import (
    SelectiveUNIClassifier,
    make_checkpoint,
)


class FakeBackbone(nn.Module):
    def __init__(self, width=4, depth=3):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)

    def forward(self, inputs):
        for block in self.blocks:
            inputs = block(inputs)
        return self.norm(inputs)


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeBackbone()
        self.precision = torch.bfloat16
        self.eval_transforms = object()

    def forward(self, inputs):
        return self.model(inputs)


def test_selective_classifier_unfreezes_only_last_block_norm_and_head():
    model = SelectiveUNIClassifier(FakeEncoder(), 3, image_dim=4, trainable_blocks=1)
    trainable = {name for name, value in model.named_parameters() if value.requires_grad}

    assert trainable == {
        "encoder.model.blocks.2.weight",
        "encoder.model.blocks.2.bias",
        "encoder.model.norm.weight",
        "encoder.model.norm.bias",
        "head.weight",
        "head.bias",
    }
    assert model(torch.ones(2, 4)).shape == (2, 3)
    assert set(model.trainable_state_dict()) == trainable


def test_selective_checkpoint_records_original_uni_and_cellbridge_seed():
    model = SelectiveUNIClassifier(FakeEncoder(), 2, image_dim=4)
    checkpoint = make_checkpoint(
        model,
        classes=["a", "b"],
        best_epoch=3,
        seed=0,
        config={"seed": 0},
    )

    assert checkpoint["initialization"] == "original_uni2_h"
    assert checkpoint["seed"] == 0
    assert checkpoint["best_epoch"] == 3
    assert checkpoint["classes"] == ["a", "b"]
    assert set(checkpoint["trainable_state"]) == set(model.trainable_state_dict())


def test_stratified_validation_split_is_deterministic_and_test_free():
    labels = np.asarray(["a"] * 10 + ["b"] * 10 + ["c"] * 5)
    spatial_train = np.arange(20)
    train, val = stratified_validation_split(spatial_train, labels, 0.2, seed=0)
    repeated = stratified_validation_split(spatial_train, labels, 0.2, seed=0)

    assert np.array_equal(train, repeated[0])
    assert np.array_equal(val, repeated[1])
    assert not set(train) & set(val)
    assert set(train) | set(val) == set(spatial_train)
    assert not (set(train) | set(val)) & set(range(20, 25))
    assert set(labels[val]) == {"a", "b"}


def _write_cached_ood(tmp_path: Path):
    sample_paths = []
    for number in range(2):
        path = tmp_path / f"sample_{number}.npz"
        np.savez_compressed(
            path,
            crops=np.full((3, 96, 96, 3), 30 + number, dtype=np.uint8),
        )
        sample_paths.append(path)
    pd.DataFrame(
        {
            "sample_number": [0, 1],
            "sample_id": ["s0", "s1"],
            "n_cells": [3, 3],
            "path": list(map(str, sample_paths)),
        }
    ).to_csv(tmp_path / "manifest.csv", index=False)
    pd.DataFrame(
        {
            "row_id": np.arange(6),
            "sample_id": ["s0"] * 3 + ["s1"] * 3,
            "tile_id": ["t"] * 6,
            "cell_id": np.arange(6),
            "label": ["a", "b", "a", "b", "a", "b"],
            "class_id": [0, 1, 0, 1, 0, 1],
            "split": ["train", "val", "test", "train", "val", "test"],
            "sample_number": [0] * 3 + [1] * 3,
            "local_index": [0, 1, 2] * 2,
        }
    ).to_csv(tmp_path / "cells.csv", index=False)


def test_cached_ood_keeps_original_classes_splits_and_sample_local_batches(tmp_path):
    _write_cached_ood(tmp_path)
    dataset = CachedOodCropDataset(
        {"cache_dir": str(tmp_path), "crop": {"target_size": 224}},
        transform=lambda image: torch.from_numpy(np.asarray(image).copy()),
    )

    assert dataset.classes.tolist() == ["a", "b"]
    assert len(dataset.train_indices) == len(dataset.val_indices) == len(dataset.test_indices) == 2
    assert dataset[0]["image"].shape == (224, 224, 3)

    sampler = GroupedBatchSampler(
        dataset, np.arange(len(dataset)), batch_size=2, seed=0, shuffle=True
    )
    for batch in sampler:
        sample_numbers = dataset.records.iloc[batch]["sample_number"].unique()
        assert len(sample_numbers) == 1


def _minimal_config():
    dataset = {"crop": {"source_size": 96, "target_size": 224}}
    return {
        "baseline": {"name": "uni_selective_ft"},
        "model": {"encoder": "uni_v2", "trainable_blocks": 1},
        "training": {},
        "seed": 0,
        "output_dir": "out",
        "hest": {
            "crop": {"target_size": 224, "target_pixel_size_um": 0.5}
        },
        "phenocell": dataset,
        "pannuke": dataset,
    }


def test_config_enforces_one_seed_and_one_unfrozen_block():
    validate_config(_minimal_config())
    wrong_seed = _minimal_config()
    wrong_seed["seed"] = 1
    with pytest.raises(ValueError, match="seed: 0"):
        validate_config(wrong_seed)
    wrong_depth = _minimal_config()
    wrong_depth["model"] = {"encoder": "uni_v2", "trainable_blocks": 2}
    with pytest.raises(ValueError, match="exactly the last block"):
        validate_config(wrong_depth)
