"""CPU smoke tests for the selective UNI training loop."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn

from cellbridge.train.finetune.data import prepare_task
from cellbridge.train.finetune.model import SelectiveUNIClassifier
from cellbridge.train.finetune.train import (
    _class_weights,
    _evaluate_prediction,
    _loader,
    _optimizer,
    _predict,
    _train_epoch,
    validate_config,
)


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.blocks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.model.norm = nn.LayerNorm(4)
        self.eval_transforms = lambda image: image
        self.precision = torch.float32

    def forward(self, inputs):
        for block in self.model.blocks:
            inputs = block(inputs)
        return self.model.norm(inputs)


class CropDataset:
    def __init__(self, row_ids, transform):
        self.row_ids = list(row_ids)
        self.transform = transform

    def __len__(self):
        return len(self.row_ids)

    def __getitem__(self, index):
        row_id = self.row_ids[index]
        image = torch.tensor([row_id, row_id % 2, 1.0, -1.0], dtype=torch.float32)
        return {"image": self.transform(image), "cell_id": f"cell-{row_id}"}


class FakeSource:
    def __init__(self, metadata):
        self.metadata = metadata

    @staticmethod
    def make_cropper(_profile, _data):
        return None

    def iter_samples(self, _data, *, transform, cropper):
        assert cropper is None
        for sample_id, rows in self.metadata.groupby("sample_id", sort=False):
            row_ids = rows["row_id"].to_numpy()
            yield SimpleNamespace(
                sample_id=sample_id,
                row_ids=row_ids,
                dataset=CropDataset(row_ids, transform),
            )


def _task():
    metadata = pd.DataFrame(
        {
            "row_id": np.arange(8),
            "sample_id": ["train-a"] * 2
            + ["train-b"] * 2
            + ["val"] * 2
            + ["test"] * 2,
            "label": ["a", "b"] * 4,
            "split": ["train"] * 4 + ["val"] * 2 + ["test"] * 2,
        }
    )
    source = FakeSource(metadata)
    return prepare_task(
        {},
        {"method": "uni"},
        metadata=metadata,
        source=source,
    )


def _config():
    return {
        "profile": "uni.yaml",
        "training": {
            "batch_size": 4,
            "max_epochs": 1,
            "patience": 1,
            "backbone_lr": 1e-3,
            "head_lr": 1e-3,
            "weight_decay": 0.0,
            "num_workers": 0,
            "seed": 0,
        },
        "output_dir": "out",
    }


def test_loader_combines_sample_local_datasets_into_full_batches():
    task = _task()
    config = validate_config(_config())
    loader = _loader(
        task,
        {"train"},
        lambda image: image,
        config["training"],
        torch.device("cpu"),
        shuffle=False,
        seed=0,
    )

    batch = next(iter(loader))
    assert batch["row_id"] == ["0", "1", "2", "3"]
    assert len(loader) == 1


def test_one_cpu_epoch_and_prediction_cover_the_requested_rows():
    task = _task()
    config = validate_config(_config())
    device = torch.device("cpu")
    model = SelectiveUNIClassifier(FakeEncoder(), 2, feature_dim=4)
    optimizer = _optimizer(model, config["training"])
    weights = _class_weights(task, {"train"}, device)

    loss = _train_epoch(
        model,
        task,
        {"train"},
        optimizer,
        weights,
        config["training"],
        device,
        epoch=1,
    )
    prediction = _predict(model, task, "test", config, device)
    scores = _evaluate_prediction(prediction, task.classes).metrics

    assert np.isfinite(loss)
    assert prediction["row_id"].tolist() == ["6", "7"]
    assert prediction["probability"].shape == (2, 2)
    assert set(scores) == {
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_auroc",
    }
