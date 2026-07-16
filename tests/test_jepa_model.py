import numpy as np
import pytest
import torch

from terra_multi.jepa.model import (
    CHECKPOINT_FORMAT,
    SpatialJEPA,
    TargetNormalizer,
    jepa_loss,
    load_model_state,
)


def make_model():
    return SpatialJEPA(
        image_dim=12,
        target_dim=6,
        model_dim=16,
        num_heads=4,
        context_depth=2,
        predictor_depth=2,
        dropout=0.0,
    )


def test_target_normalizer_centers_scales_and_stops_gradient():
    targets = np.array(
        [
            [1.0, 4.0, 8.0],
            [2.0, 6.0, 8.0],
            [3.0, 8.0, 8.0],
        ],
        dtype=np.float32,
    )
    normalizer = TargetNormalizer.fit(targets)
    values = torch.tensor(targets, requires_grad=True)

    normalized = normalizer(values)

    torch.testing.assert_close(
        normalized.mean(dim=0), torch.zeros(3), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        normalized[:, :2].std(dim=0, correction=0), torch.ones(2)
    )
    torch.testing.assert_close(normalizer.denormalize(normalized), values.detach())
    assert not normalized.requires_grad
    assert normalizer.scale[-1] == pytest.approx(1e-6)


def test_spatial_jepa_shapes_and_padding_masks():
    torch.manual_seed(0)
    model = make_model().eval()
    context = torch.randn(2, 5, 12)
    context_coords = torch.randn(2, 5, 2)
    target_coords = torch.randn(2, 3, 2)
    context_mask = torch.tensor(
        [[True, True, True, False, False], [True, True, True, True, True]]
    )
    target_mask = torch.tensor([[True, True, False], [True, True, True]])

    encoded = model.encode_context(context, context_coords, context_mask)
    predicted = model(
        context,
        context_coords,
        target_coords,
        context_mask,
        target_mask,
    )

    assert encoded.shape == (2, 5, 16)
    assert predicted.shape == (2, 3, 6)
    assert torch.count_nonzero(encoded[0, 3:]) == 0
    assert torch.count_nonzero(predicted[0, 2]) == 0


def test_padded_context_values_do_not_change_predictions():
    torch.manual_seed(1)
    model = make_model().eval()
    context = torch.randn(1, 5, 12)
    context_coords = torch.randn(1, 5, 2)
    target_coords = torch.randn(1, 2, 2)
    context_mask = torch.tensor([[True, True, True, False, False]])
    changed = context.clone()
    changed[:, 3:] = 1_000

    with torch.inference_mode():
        original = model(context, context_coords, target_coords, context_mask)
        modified = model(changed, context_coords, target_coords, context_mask)

    torch.testing.assert_close(original, modified)


def test_target_queries_are_independent_for_extraction():
    torch.manual_seed(4)
    model = make_model().eval()
    context = torch.randn(2, 5, 12)
    context_coords = torch.randn(2, 5, 2)
    target_coords = torch.randn(2, 3, 2)

    with torch.inference_mode():
        together = model(context, context_coords, target_coords)
        separately = torch.cat(
            [
                model(context, context_coords, target_coords[:, index : index + 1])
                for index in range(target_coords.shape[1])
            ],
            dim=1,
        )

    torch.testing.assert_close(together, separately)


def test_jepa_loss_detaches_teacher_but_trains_model():
    torch.manual_seed(2)
    model = make_model()
    context = torch.randn(2, 4, 12, requires_grad=True)
    context_coords = torch.randn(2, 4, 2)
    target_coords = torch.randn(2, 3, 2)
    terra = torch.randn(2, 3, 6, requires_grad=True)
    normalizer = TargetNormalizer.fit(torch.randn(20, 6))

    predictions = model(context, context_coords, target_coords)
    loss = jepa_loss(predictions, terra, normalizer)
    loss.backward()

    assert torch.isfinite(loss)
    assert context.grad is not None
    assert terra.grad is None
    assert model.predictor.output[-1].weight.grad is not None


def test_jepa_loss_is_smooth_l1_in_normalized_target_space():
    normalizer = TargetNormalizer(
        torch.tensor([1.0, 2.0]),
        torch.tensor([2.0, 4.0]),
    )
    predictions = torch.tensor([[[0.0, 2.0]]])
    terra = torch.tensor([[[3.0, 6.0]]], requires_grad=True)

    loss = jepa_loss(predictions, terra, normalizer)

    assert loss == pytest.approx(0.5)
    assert not loss.requires_grad
    assert terra.grad is None


def test_minimal_checkpoint_round_trip():
    torch.manual_seed(3)
    model = make_model().eval()
    checkpoint = {
        "format_version": CHECKPOINT_FORMAT,
        "model_config": model.model_config,
        "model_state": model.state_dict(),
    }
    restored = load_model_state(checkpoint).eval()
    context = torch.randn(1, 4, 12)
    context_coords = torch.randn(1, 4, 2)
    target_coords = torch.randn(1, 2, 2)

    with torch.inference_mode():
        expected = model(context, context_coords, target_coords)
        actual = restored(context, context_coords, target_coords)

    torch.testing.assert_close(actual, expected)
