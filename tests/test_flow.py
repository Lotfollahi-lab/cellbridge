import numpy as np
import torch

from cellbridge.flow.compatibility import (
    false_negative_mask,
    softrepa_logits,
    softrepa_loss,
    summarize_error_matrices,
)
from cellbridge.flow.data import CellNeighbourhoodDataset, PilotArrays
from cellbridge.flow.model import CellFlow, SoftPromptTokens
from cellbridge.flow.path import linear_path


def test_linear_path_uses_stflow_endpoint_interpolation():
    target = torch.arange(12, dtype=torch.float32).view(3, 4)
    state, time, noise = linear_path(target, generator=torch.Generator().manual_seed(4))
    expected = (1 - time[:, None]) * noise + time[:, None] * target
    torch.testing.assert_close(state, expected)


def test_cell_flow_and_frozen_soft_prompts_smoke():
    model = CellFlow(
        image_dim=8,
        target_dim=4,
        model_dim=8,
        num_heads=2,
        context_depth=1,
        predictor_depth=2,
        dropout=0.0,
    )
    prompts = SoftPromptTokens(2, 3, 8)
    state = torch.randn(3, 4)
    time = torch.rand(3)
    condition = torch.randn(3, 5, 8)
    coordinates = torch.randn(3, 5, 2)
    assert model(state, time, condition, coordinates).shape == (3, 4)

    model.requires_grad_(False)
    output = model(state, time, condition, coordinates, prompts=prompts)
    output.square().mean().backward()
    assert prompts.tokens.grad is not None
    assert all(parameter.grad is None for parameter in model.parameters())


def test_softrepa_uses_exponential_error_logits():
    errors = torch.tensor([[0.1, 0.3], [0.4, 0.2]])
    expected = 4.0 * torch.exp(-errors / 0.07)
    torch.testing.assert_close(
        softrepa_logits(errors, scale=4.0, temperature=0.07), expected
    )
    loss = softrepa_loss(errors, scale=4.0, temperature=0.07)
    assert torch.isfinite(loss)


def test_false_negative_mask_keeps_matches_and_excludes_neighbours():
    positions = torch.tensor([0, 1, 2])
    coordinates = torch.tensor([[0.0, 0.0], [1.0, 0.0], [20.0, 0.0]])
    nearest = torch.tensor([[1], [0], [1]])
    mask = false_negative_mask(
        positions, coordinates, nearest, spatial_radius_um=5.0
    )
    assert torch.equal(mask.diag(), torch.ones(3, dtype=torch.bool))
    assert not mask[0, 1] and not mask[1, 0]
    assert mask[0, 2]


def test_neighbourhood_dataset_never_crosses_its_pool():
    cells = 8
    arrays = PilotArrays(
        image=np.arange(cells * 3, dtype=np.float32).reshape(cells, 3),
        terra=np.arange(cells * 2, dtype=np.float32).reshape(cells, 2),
        neighbourhood_coordinates_um=np.column_stack(
            (np.arange(cells), np.zeros(cells))
        ),
        split_coordinates_um=np.column_stack((np.arange(cells), np.zeros(cells))),
        cell_ids=np.asarray([f"c{i}" for i in range(cells)]),
        labels=np.asarray(["a"] * cells),
        train_rows=np.arange(5),
        test_rows=np.arange(5, 8),
    )
    dataset = CellNeighbourhoodDataset(
        arrays,
        pool_rows=np.arange(5),
        anchor_rows=np.asarray([1, 3]),
        neighbours=2,
        coordinate_scale_um=2.0,
    )
    assert np.array_equal(dataset.context_rows[:, 0], np.asarray([1, 3]))
    assert set(dataset.context_rows.ravel()) <= set(range(5))
    assert dataset[0]["condition"].shape == (3, 3)


def test_error_summary_reports_retrieval_and_separation():
    errors = np.asarray(
        [[0.1, 0.5, 0.6], [0.4, 0.2, 0.7], [0.8, 0.6, 0.3]],
        dtype=np.float32,
    )
    metrics = summarize_error_matrices([errors])
    assert metrics["retrieval_top1"] == 1.0
    assert metrics["matched_error_mean"] < metrics["mismatched_error_mean"]
    assert metrics["chance_top1"] == 1 / 3
