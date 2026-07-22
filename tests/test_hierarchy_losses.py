import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from terra_multi.hierarchical.losses import (
    negative_weight_matrix,
    relation_masks,
    symmetric_hierarchy_ranking_loss,
    weighted_symmetric_pair_loss,
)
from terra_multi.hierarchical.splits import load_phenotype_split
from terra_multi.lora.model import symmetric_contrastive_loss


def _hierarchy_batch():
    levels = torch.tensor(
        [
            [0, 1, 10, 100],
            [0, 1, 10, 100],
            [0, 1, 10, -1],
            [0, 1, 11, 110],
            [0, 2, 20, 200],
            [-1, -1, -1, -1],
        ]
    )
    depth = torch.tensor([3, 3, 2, 3, 3, 0])
    usable = torch.tensor([True, True, True, True, True, False])
    return levels, depth, usable


def test_relation_masks_respect_partial_and_excluded_labels():
    relations = relation_masks(*_hierarchy_batch())

    assert relations["identity"][0, 1]
    assert relations["lineage"][0, 2]
    assert not relations["identity"][0, 2]
    assert relations["compartment"][0, 3]
    assert relations["unrelated"][0, 4]
    assert not any(mask[0, 5] for mask in relations.values())
    assert all(torch.equal(mask, mask.T) for mask in relations.values())


def test_downweighting_softens_known_negatives_without_adding_positives():
    image = F.normalize(torch.tensor([[1.0, 0.0], [0.95, 0.05], [-1.0, 0.0]]), dim=1)
    terra = image.clone()
    levels = torch.tensor([[0, 1, 10, 100], [0, 1, 10, 100], [0, 2, 20, 200]])
    relations = relation_masks(levels, torch.tensor([3, 3, 3]), torch.ones(3, dtype=torch.bool))
    weights = negative_weight_matrix(relations, {"identity": 0.1}, dtype=image.dtype)

    baseline = symmetric_contrastive_loss(image, terra, temperature=0.2)[
        "loss_contrastive"
    ]
    softened = weighted_symmetric_pair_loss(image, terra, weights, temperature=0.2)

    assert torch.equal(weights.diagonal(), torch.ones(3))
    assert weights[0, 1] == pytest.approx(0.1)
    assert softened < baseline


def test_symmetric_ranking_penalizes_hierarchy_ordering_violations():
    terra = F.normalize(
        torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.2, 0.8], [-1.0, 0.0]]), dim=1
    )
    levels = torch.tensor(
        [[0, 1, 10, 100], [0, 1, 10, 100], [0, 1, 10, 101], [0, 2, 20, 200]]
    )
    relations = relation_masks(
        levels, torch.full((4,), 3), torch.ones(4, dtype=torch.bool)
    )
    settings = {
        "levels": ["match", "identity", "lineage", "unrelated"],
        "margins": {
            "match>identity": 0.01,
            "identity>lineage": 0.01,
            "lineage>unrelated": 0.01,
        },
        "cross_sample": "prefer",
    }

    ordered = symmetric_hierarchy_ranking_loss(
        terra, terra, relations, ["a", "b", "c", "d"], **settings
    )
    violated = symmetric_hierarchy_ranking_loss(
        terra.flip(0), terra, relations, ["a", "b", "c", "d"], **settings
    )

    assert ordered["hierarchy_triplets"] > 0
    assert ordered["hierarchy_cross_sample_fraction"] == 1
    assert violated["loss_hier"] > ordered["loss_hier"]


def _write_split_inputs(tmp_path):
    base = tmp_path / "base.csv"
    pd.DataFrame(
        {
            "sample_id": ["base_train", "base_val", "A", "B", "C", "X"],
            "split": [
                "distill_train",
                "distill_val",
                "label_test",
                "label_test",
                "label_test",
                "label_test",
            ],
        }
    ).to_csv(base, index=False)
    manifest = tmp_path / "phenotype.csv"
    pd.DataFrame(
        {
            "sample_id": ["A", "B", "C", "X"],
            "organ": ["breast", "breast", "breast", "ovary"],
            "patient_id": ["p1", "p2", "p3", ""],
            "role": [
                "breast_train",
                "breast_val",
                "breast_test",
                "cross_organ_test",
            ],
        }
    ).to_csv(manifest, index=False)
    return base, manifest


def test_phenotype_split_is_disjoint_and_keeps_cross_organ_held_out(tmp_path):
    base, manifest = _write_split_inputs(tmp_path)
    split = load_phenotype_split(manifest, base)

    assert split.training == ("base_train", "A")
    assert split.validation == ("base_val", "B")
    assert split.breast_test == ("C",)
    assert split.cross_organ_test == ("X",)


def test_phenotype_split_rejects_patient_leakage(tmp_path):
    base, manifest = _write_split_inputs(tmp_path)
    table = pd.read_csv(manifest, keep_default_na=False)
    table.loc[table["sample_id"].eq("B"), "patient_id"] = "p1"
    table.to_csv(manifest, index=False)

    with pytest.raises(ValueError, match="patients cross roles"):
        load_phenotype_split(manifest, base)
