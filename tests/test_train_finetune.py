"""Focused contracts for selective UNI fine-tuning."""

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from cellbridge.train.finetune.data import LabeledSubset, prepare_task
from cellbridge.train.finetune.model import (
    SelectiveUNIClassifier,
    load_trainable_state_dict,
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
        self.eval_transforms = object()
        self.precision = torch.bfloat16

    def forward(self, inputs):
        return self.model(inputs)


def test_classifier_unfreezes_only_last_block_norm_and_head():
    model = SelectiveUNIClassifier(FakeEncoder(), 3, feature_dim=4)

    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable == {
        "encoder.model.blocks.2.weight",
        "encoder.model.blocks.2.bias",
        "encoder.model.norm.weight",
        "encoder.model.norm.bias",
        "head.weight",
        "head.bias",
    }
    assert model(torch.ones(2, 4)).shape == (2, 3)


def test_checkpoint_contains_only_trainable_state_and_round_trips():
    source = SelectiveUNIClassifier(FakeEncoder(), 2, feature_dim=4)
    checkpoint = make_checkpoint(source, classes=("a", "b"), epochs=3, seed=0)

    assert checkpoint["initialization"] == "original_uni2_h"
    assert checkpoint["unfreezing"] == "last_block+final_norm+head"
    assert set(checkpoint["trainable_state"]) == {
        name for name, parameter in source.named_parameters() if parameter.requires_grad
    }

    target = SelectiveUNIClassifier(FakeEncoder(), 2, feature_dim=4)
    load_trainable_state_dict(target, checkpoint["trainable_state"])
    for name, value in source.trainable_state_dict().items():
        assert torch.equal(value, target.state_dict()[name])


class FakeSource:
    @staticmethod
    def load_metadata(_data):
        raise AssertionError("metadata was supplied explicitly")


def _metadata():
    return pd.DataFrame(
        {
            "row_id": np.arange(8),
            "sample_id": ["train"] * 4 + ["val"] * 2 + ["test"] * 2,
            "label": ["B cells", "T cells"] * 4,
            "split": ["train"] * 4 + ["val"] * 2 + ["test"] * 2,
        }
    )


def test_prepare_task_preserves_official_splits_and_training_classes():
    task = prepare_task(
        {},
        {"method": "uni", "crop": {}},
        metadata=_metadata(),
        source=FakeSource,
    )

    assert task.classes == ("B cells", "T cells")
    assert task.counts == {"train": 4, "val": 2, "test": 2}

    unsupported = _metadata()
    unsupported.loc[unsupported["split"] == "test", "label"] = "new"
    with pytest.raises(ValueError, match="absent from training"):
        prepare_task(
            {},
            {"method": "uni", "crop": {}},
            metadata=unsupported,
            source=FakeSource,
        )


def test_labeled_subset_keeps_crop_alignment():
    dataset = [
        {"image": torch.tensor([index]), "cell_id": f"c{index}"} for index in range(4)
    ]
    subset = LabeledSubset(
        dataset,
        positions=np.asarray([3, 1]),
        row_ids=np.asarray([13, 11]),
        targets=np.asarray([1, 0]),
    )

    assert subset[0] == {
        "image": torch.tensor([3]),
        "cell_id": "c3",
        "target": 1,
        "row_id": "13",
    }
