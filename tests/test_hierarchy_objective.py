from copy import deepcopy
from pathlib import Path

import pytest
import torch

from cellbridge.hierarchical.objective import compute_hierarchy_objective
from cellbridge.lora.model import compute_loss
from cellbridge.lora.train import load_config


def _objective():
    return {
        "mode": "contrastive",
        "targets": "exact",
        "temperature": 0.2,
        "lambda_contrastive": 1.0,
        "bidirectional": True,
    }


def _hierarchy(downweight=False, ranking=False):
    return {
        "negative_downweighting": {
            "enabled": downweight,
            "weights": {"identity": 0.25, "lineage": 0.5},
        },
        "ranking": {
            "enabled": ranking,
            "lambda_hier": 0.1,
            "levels": ["match", "identity", "lineage", "unrelated"],
            "margins": {
                "match>identity": 0.05,
                "identity>lineage": 0.05,
                "lineage>unrelated": 0.05,
            },
            "temperature": 1.0,
            "cross_sample": "prefer",
            "min_triplets": 1,
            "bidirectional": True,
        },
    }


def test_disabled_hierarchy_is_exactly_the_existing_paired_objective():
    generator = torch.Generator().manual_seed(4)
    image = torch.randn(5, 4, generator=generator)
    terra = torch.randn(5, 4, generator=generator)
    batch = {
        "hierarchy_levels": torch.tensor([[0, 1, 2, index] for index in range(5)]),
        "hierarchy_depth": torch.full((5,), 3),
        "hierarchy_usable": torch.ones(5, dtype=torch.bool),
        "sample_id": ["A", "B", "A", "B", "A"],
    }

    expected = compute_loss({"projected": image}, terra, **_objective())
    actual = compute_hierarchy_objective(
        {"projected": image}, terra, batch, _objective(), _hierarchy()
    )

    assert torch.equal(actual["loss_pair"], expected["loss_contrastive"])
    assert torch.equal(actual["loss_total"], expected["loss_total"])
    assert actual["loss_hier"] == 0


def test_excluded_batch_cannot_change_either_training_loss():
    image = torch.eye(3)
    terra = torch.eye(3)
    batch = {
        "hierarchy_levels": torch.full((3, 4), -1),
        "hierarchy_depth": torch.zeros(3, dtype=torch.long),
        "hierarchy_usable": torch.zeros(3, dtype=torch.bool),
        "sample_id": ["A", "B", "C"],
    }
    expected = compute_loss({"projected": image}, terra, **_objective())
    actual = compute_hierarchy_objective(
        {"projected": image},
        terra,
        batch,
        _objective(),
        _hierarchy(downweight=True, ranking=True),
    )

    assert torch.equal(actual["loss_pair"], expected["loss_contrastive"])
    assert torch.equal(actual["loss_total"], expected["loss_total"])
    assert actual["hierarchy_triplets"] == 0
    assert actual["hierarchy_sufficient"] == 0


def test_four_ablation_configs_differ_only_in_loss_switches_and_run_identity():
    root = Path(__file__).resolve().parents[1]
    names = ["baseline", "negative_downweighting", "hierarchy_ranking", "combined"]
    configs = {
        name: load_config(root / "configs" / "hierarchical" / f"lora_{name}.yaml")
        for name in names
    }

    observed = {
        name: (
            cfg["hierarchy"]["negative_downweighting"]["enabled"],
            cfg["hierarchy"]["ranking"]["enabled"],
        )
        for name, cfg in configs.items()
    }
    assert observed == {
        "baseline": (False, False),
        "negative_downweighting": (True, False),
        "hierarchy_ranking": (False, True),
        "combined": (True, True),
    }
    comparable = {}
    for name, cfg in configs.items():
        cfg = deepcopy(cfg)
        for key in ("run_name", "output_dir", "checkpoint_dir"):
            cfg.pop(key)
        cfg["wandb"].pop("name")
        cfg["hierarchy"]["negative_downweighting"].pop("enabled")
        cfg["hierarchy"]["ranking"].pop("enabled")
        comparable[name] = cfg
    reference = comparable["baseline"]
    assert all(cfg == reference for cfg in comparable.values())

    for cfg in configs.values():
        assert cfg["objective"]["targets"] == "exact"
        assert cfg["max_cells_per_sample"] == 5000
        assert cfg["training"]["selection_metric"] == "breast_val/loss_total"
        assert cfg["hierarchy"]["sampling"]["strategy"] == "balanced"
        assert cfg["hierarchy"]["expected_counts"] == {
            "breast_train": 6,
            "breast_val": 3,
            "breast_test": 3,
            "cross_organ_test": 3,
        }


def test_config_rejects_soft_targets_with_negative_downweighting(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "distill_version: lora\n"
        "objective: {targets: terra_similarity}\n"
        "hierarchy:\n"
        "  enabled: true\n"
        "  split_manifest: split.csv\n"
        "  labels_dir: cells\n"
        "  negative_downweighting: {enabled: true}\n"
    )
    with pytest.raises(ValueError, match="requires objective.targets: exact"):
        load_config(path)
