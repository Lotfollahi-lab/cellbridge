import math
from copy import deepcopy

import pytest
import torch
import yaml
from torch.utils.data import Dataset

from cellbridge.jepa.model import SpatialJEPA, TargetNormalizer, load_model_state
from cellbridge.jepa.train import _batch_metrics, fit, load_config


def minimal_config(tmp_path):
    return {
        "jepa_version": "spatial_block_v1",
        "data_root": str(tmp_path / "data"),
        "manifest_path": str(tmp_path / "manifest.csv"),
        "splits_path": str(tmp_path / "splits.csv"),
        "output_dir": str(tmp_path / "output"),
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "model": {
            "image_dim": 6,
            "target_dim": 4,
            "model_dim": 8,
            "num_heads": 2,
            "context_depth": 1,
            "predictor_depth": 1,
            "dropout": 0.0,
        },
        "training": {
            "batch_size": 2,
            "max_epochs": 2,
            "patience": 2,
            "learning_rate": 1e-2,
            "weight_decay": 0.0,
            "precision": "fp32",
            "num_workers": 0,
            "seed": 0,
        },
    }


def write_config(tmp_path, config):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


class TinyWindowDataset(Dataset):
    def __init__(self, seed):
        generator = torch.Generator().manual_seed(seed)
        self.context = torch.randn(4, 2, 3, 6, generator=generator)
        self.context_coords = torch.randn(4, 2, 3, 2, generator=generator)
        self.target_coords = torch.randn(4, 2, 1, 2, generator=generator)
        signal = self.context.mean(dim=2)[..., :4]
        self.targets = signal[:, :, None, :]
        self.epoch = None

    def __len__(self):
        return len(self.context)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        return {
            "context_image": self.context[index],
            "context_coords": self.context_coords[index],
            "target_terra": self.targets[index],
            "target_coords": self.target_coords[index],
            "sample_id": "section",
            "anchor_cell_id": str(index),
            "context_cell_ids": (
                ("c0", "c1", "c2"),
                ("c3", "c4", "c5"),
            ),
            "target_cell_ids": ("t0", "t1"),
            "coordinate_normalization": "fixed_radius",
        }


def test_load_config_applies_defaults_and_validates_geometry(tmp_path):
    config = minimal_config(tmp_path)
    resolved = load_config(write_config(tmp_path, config))

    assert resolved["uni_key"] == "uni_emb"
    assert resolved["terra_key"] == "cell_emb"
    assert resolved["data"]["target_cells"] == 8
    assert resolved["data"]["target_sampling"] == "anchor_plus_nearest"
    assert resolved["data"]["context_cells"] == 64
    assert resolved["data"]["context_sampling"] == "uniform_within_radius"
    assert resolved["data"]["coordinate_normalization"] == "fixed_radius"
    assert resolved["data"]["max_anchor_cells_per_section"] == 100_000
    assert "max_cells_per_section" not in resolved["data"]
    assert resolved["wandb"]["enabled"] is False

    invalid = deepcopy(config)
    invalid["data"] = {
        "guard_radius_um": 300.0,
        "context_radius_um": 300.0,
    }
    with pytest.raises(ValueError, match="smaller than context_radius_um"):
        load_config(write_config(tmp_path, invalid))

    legacy = deepcopy(config)
    legacy["data"] = {"max_cells_per_section": 10}
    with pytest.raises(
        ValueError, match="renamed to data.max_anchor_cells_per_section"
    ):
        load_config(write_config(tmp_path, legacy))


def test_batch_retrieval_compares_singleton_targets_across_batch():
    normalizer = TargetNormalizer(torch.zeros(2), torch.ones(2))
    targets = torch.tensor(
        [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
            [[-1.0, 0.0]],
        ]
    )

    _, perfect_retrieval = _batch_metrics(targets, targets, normalizer)
    _, shifted_retrieval = _batch_metrics(targets.roll(1, dims=0), targets, normalizer)

    assert perfect_retrieval.item() == 1.0
    assert shifted_retrieval.item() == 0.0


def test_fit_optimizes_on_cpu_and_writes_compatible_checkpoint(tmp_path):
    torch.manual_seed(0)
    cfg = load_config(write_config(tmp_path, minimal_config(tmp_path)))
    (tmp_path / "output").mkdir()
    checkpoint = tmp_path / "checkpoints" / "best.pt"
    model = SpatialJEPA(**cfg["model"])
    train_dataset = TinyWindowDataset(1)
    val_dataset = TinyWindowDataset(2)
    targets = torch.cat((train_dataset.targets, val_dataset.targets)).reshape(-1, 4)
    normalizer = TargetNormalizer.fit(targets)
    initial = model.context_encoder.image_projection[-1].weight.detach().clone()

    best_epoch, best_loss, rows = fit(
        model,
        normalizer,
        train_dataset,
        val_dataset,
        cfg,
        torch.device("cpu"),
        checkpoint,
        run=None,
    )

    assert best_epoch in {1, 2}
    assert math.isfinite(best_loss)
    assert len(rows) == 2
    assert checkpoint.exists()
    assert train_dataset.epoch == 2
    assert val_dataset.epoch == 0
    assert not torch.equal(initial, model.context_encoder.image_projection[-1].weight)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restored = load_model_state(payload).eval()
    restored_normalizer = TargetNormalizer(
        payload["target_normalizer"]["mean"],
        payload["target_normalizer"]["scale"],
    )
    assert payload["config"]["data"]["coordinate_normalization"] == "fixed_radius"
    assert restored.model_config == model.model_config
    torch.testing.assert_close(restored_normalizer.mean, normalizer.mean)
    torch.testing.assert_close(restored_normalizer.scale, normalizer.scale)


def test_fit_rejects_mismatched_target_statistics(tmp_path):
    cfg = load_config(write_config(tmp_path, minimal_config(tmp_path)))
    model = SpatialJEPA(**cfg["model"])
    normalizer = TargetNormalizer.fit(torch.randn(8, 3))

    with pytest.raises(ValueError, match="dimensions do not match"):
        fit(
            model,
            normalizer,
            TinyWindowDataset(1),
            TinyWindowDataset(2),
            cfg,
            torch.device("cpu"),
            tmp_path / "best.pt",
            run=None,
        )
