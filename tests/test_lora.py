import pytest
import torch
import torch.nn.functional as F

from cellbridge.lora.model import (
    compute_loss,
    soft_target_contrastive_loss,
    symmetric_contrastive_loss,
    terra_similarity_targets,
)
from cellbridge.lora.train import load_config


def test_terra_similarity_targets_are_normalized_and_keep_exact_match_highest():
    terra = torch.tensor(
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [-1.0, 0.0],
            [1.0, 0.0],
        ],
        requires_grad=True,
    )
    targets = terra_similarity_targets(
        terra,
        teacher_temperature=0.1,
        exact_match_boost=0.05,
    )

    assert not targets.requires_grad
    assert torch.allclose(targets.sum(dim=1), torch.ones(len(terra)))
    off_diagonal = targets.masked_fill(torch.eye(len(terra), dtype=torch.bool), -1)
    assert torch.all(targets.diagonal() > off_diagonal.max(dim=1).values)
    assert targets[0, 1] > targets[0, 2]


def test_soft_target_loss_matches_directional_kl():
    image = F.normalize(torch.tensor([[1.0, 1.0], [1.0, -1.0]]), dim=-1)
    terra = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    temperature = 0.2
    targets = terra_similarity_targets(terra, 0.3, 0.05)
    logits = F.normalize(image, dim=-1) @ F.normalize(terra, dim=-1).T
    logits = logits / temperature
    image_to_terra = F.kl_div(
        F.log_softmax(logits, dim=-1), targets, reduction="batchmean"
    )
    terra_to_image = F.kl_div(
        F.log_softmax(logits.T, dim=-1), targets, reduction="batchmean"
    )

    one_way = soft_target_contrastive_loss(
        image,
        terra,
        temperature,
        teacher_temperature=0.3,
        exact_match_boost=0.05,
        bidirectional=False,
    )
    both_ways = soft_target_contrastive_loss(
        image,
        terra,
        temperature,
        teacher_temperature=0.3,
        exact_match_boost=0.05,
        bidirectional=True,
    )
    assert torch.allclose(one_way["loss_contrastive"], image_to_terra)
    assert torch.allclose(
        both_ways["loss_contrastive"], 0.5 * (image_to_terra + terra_to_image)
    )


def test_soft_target_loss_prefers_semantically_aligned_images():
    terra = torch.eye(3)
    aligned = soft_target_contrastive_loss(
        terra,
        terra,
        temperature=0.2,
        teacher_temperature=0.2,
    )["loss_contrastive"]
    permuted = soft_target_contrastive_loss(
        terra.flip(0),
        terra,
        temperature=0.2,
        teacher_temperature=0.2,
    )["loss_contrastive"]
    assert aligned < permuted


def test_compute_loss_defaults_to_legacy_exact_pair_objective():
    generator = torch.Generator().manual_seed(0)
    image = torch.randn(5, 4, generator=generator)
    terra = torch.randn(5, 4, generator=generator)
    expected = symmetric_contrastive_loss(image, terra)
    actual = compute_loss({"projected": image}, terra)

    assert actual.keys() == {*expected, "loss_total"}
    for name, value in expected.items():
        assert torch.equal(actual[name], value)
    assert torch.equal(actual["loss_total"], expected["loss_contrastive"])


def test_soft_targets_compose_with_gex_objective():
    outputs = {
        "projected": torch.eye(3),
        "gex_prediction": torch.ones(3, 2),
    }
    losses = compute_loss(
        outputs,
        torch.eye(3),
        gex_target=torch.zeros(3, 2),
        mode="contrastive_gex",
        targets="terra_similarity",
        lambda_gex=0.5,
    )
    assert losses.keys() == {
        "loss_contrastive",
        "contrastive_top1",
        "contrastive_top5",
        "loss_gex",
        "loss_total",
    }
    assert torch.allclose(
        losses["loss_total"], losses["loss_contrastive"] + 0.5 * losses["loss_gex"]
    )


def test_lora_config_defaults_to_exact_targets(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("distill_version: lora\n")
    cfg = load_config(config_path)
    assert cfg["objective"] == {
        "mode": "contrastive",
        "lambda_contrastive": 1.0,
        "lambda_gex": 1.0,
        "temperature": 0.07,
        "gex_loss": "smooth_l1",
        "targets": "exact",
        "teacher_temperature": 0.1,
        "exact_match_boost": 0.05,
        "bidirectional": True,
    }


def test_lora_config_rejects_non_positive_exact_match_boost(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "distill_version: lora\nobjective:\n  exact_match_boost: 0\n"
    )
    with pytest.raises(ValueError, match="exact_match_boost"):
        load_config(config_path)
