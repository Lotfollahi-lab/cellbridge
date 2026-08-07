import numpy as np
import torch

from cellbridge.flow.checkpoints import load_prompts, save_prompts
from cellbridge.flow.compatibility import (
    false_negative_mask,
    softrepa_logits,
    softrepa_loss,
    summarize_error_matrices,
)
from cellbridge.flow.data import CellNeighbourhoodDataset, PilotArrays
from cellbridge.flow.model import CellFlow, SoftPromptTokens
from cellbridge.flow.path import linear_path, sample_endpoint
from cellbridge.flow.run import _set_run_seed


def test_linear_path_uses_stflow_endpoint_interpolation():
    target = torch.arange(12, dtype=torch.float32).view(3, 4)
    state, time, noise = linear_path(target, generator=torch.Generator().manual_seed(4))
    expected = (1 - time[:, None]) * noise + time[:, None] * target
    torch.testing.assert_close(state, expected)


def test_endpoint_sampler_uses_zero_and_returns_integrated_state():
    class ConstantEndpoint:
        def __init__(self):
            self.times = []

        def __call__(self, state, time, condition, coordinates, *, prompts=None):
            self.times.append(float(time[0]))
            return torch.full_like(state, 2.0)

    model = ConstantEndpoint()
    condition = torch.zeros(2, 3, 4)
    coordinates = torch.zeros(2, 3, 2)
    generated = sample_endpoint(
        model,
        condition,
        coordinates,
        4,
        steps=4,
        generator=torch.Generator().manual_seed(5),
    )
    torch.testing.assert_close(generated, torch.full((2, 4), 2.0))
    np.testing.assert_allclose(model.times, [0.0, 0.25, 0.5, 0.75])


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


def test_prompt_checkpoint_is_self_describing(tmp_path):
    prompts = SoftPromptTokens(2, 3, 8, time_dependent=True, time_bins=7)
    path = tmp_path / "prompts.pt"
    save_prompts(path, prompts)
    restored = load_prompts(path, torch.device("cpu"))
    assert restored.depth == 2
    assert restored.token_count == 3
    assert restored.time_bins == 7
    for expected, actual in zip(
        prompts.state_dict().values(), restored.state_dict().values(), strict=True
    ):
        torch.testing.assert_close(actual, expected)


def test_softrepa_uses_exponential_error_logits_and_diagonal_anchor():
    errors = torch.tensor([[0.1, 0.3], [0.4, 0.2]])
    expected = 4.0 * torch.exp(-errors / 0.07)
    torch.testing.assert_close(
        softrepa_logits(errors, scale=4.0, temperature=0.07), expected
    )
    base = softrepa_loss(errors, scale=4.0, temperature=0.07)
    anchored = softrepa_loss(
        errors, scale=4.0, temperature=0.07, diagonal_weight=2.0
    )
    torch.testing.assert_close(anchored - base, 2.0 * errors.diag().mean())


def test_false_negative_mask_is_symmetric_for_directed_knn():
    positions = torch.tensor([0, 1, 2])
    coordinates = torch.tensor([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    nearest = torch.tensor([[1], [2], [0]])
    mask = false_negative_mask(
        positions, coordinates, nearest, spatial_radius_um=None
    )
    assert torch.equal(mask.diag(), torch.ones(3, dtype=torch.bool))
    assert not mask[0, 1] and not mask[1, 0]
    assert not mask[1, 2] and not mask[2, 1]


def test_neighbourhood_dataset_never_crosses_its_pool():
    cells = 8
    arrays = PilotArrays(
        image=np.arange(cells * 3, dtype=np.float32).reshape(cells, 3),
        terra=np.arange(cells * 2, dtype=np.float32).reshape(cells, 2),
        neighbourhood_coordinates_um=np.column_stack(
            (np.arange(cells), np.zeros(cells))
        ),
        split_coordinates=np.column_stack((np.arange(cells), np.zeros(cells))),
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


def test_error_summary_supports_masks_scores_and_time_strata():
    errors = np.asarray(
        [[0.1, 0.5, 0.6], [0.4, 0.2, 0.7], [0.8, 0.6, 0.3]],
        dtype=np.float32,
    )
    valid = np.ones_like(errors, dtype=bool)
    valid[0, 1] = valid[1, 0] = False
    metrics = summarize_error_matrices(
        [errors],
        valid_masks=[valid],
        times=[np.asarray([0.1, 0.4, 0.9])],
        scale=4.0,
        temperature=0.07,
    )
    assert metrics["retrieval_top1"] == 1.0
    assert metrics["retrieval_top1_successes"] == 3
    assert metrics["matched_error_mean"] < metrics["mismatched_error_mean"]
    assert metrics["retrieval_candidates_min"] == 2
    assert metrics["retrieval_candidates_max"] == 3
    assert 0 < metrics["matched_probability_mean"] <= 1
    assert set(metrics["time_strata"]) == {
        "t_0.00_0.25",
        "t_0.25_0.50",
        "t_0.75_1.00",
    }


def test_run_seed_reaches_data_sampler_and_all_stochastic_sections():
    cfg = {
        "data": {"sampling_seed": 0},
        **{
            section: {"seed": 0}
            for section in (
                "base_training",
                "diagnostic",
                "softrepa",
                "generation",
                "evaluation",
            )
        },
    }
    _set_run_seed(cfg, 7)
    assert cfg["data"]["sampling_seed"] == 7
    assert all(
        settings["seed"] == 7
        for name, settings in cfg.items()
        if name != "data"
    )
