import numpy as np
import pandas as pd
import pytest
import torch

from terra_multi.jepa.extract import (
    FORMAT_VERSION,
    SamplingSettings,
    current_output,
    fit_spatial_index,
    load_checkpoint,
    load_section,
    make_context_batch,
    predict_section,
    sampling_settings,
    write_output,
)
from terra_multi.jepa.model import CHECKPOINT_FORMAT, SpatialJEPA, TargetNormalizer


def settings():
    return SamplingSettings(
        n_context=2,
        context_sampling="uniform_within_radius",
        context_radius_um=250.0,
        guard_radius_um=60.0,
        coordinate_normalization="fixed_radius",
        sampling_seed=0,
    )


def test_context_batch_excludes_target_and_guard_band():
    coordinates = np.array([[0, 0], [50, 0], [100, 0], [200, 0], [300, 0]])
    embeddings = np.arange(10, dtype=np.float32).reshape(5, 2)
    batch = make_context_batch(
        embeddings,
        coordinates,
        target_rows=[2],
        spatial_index=fit_spatial_index(coordinates),
        settings=settings(),
    )

    selected = batch.rows[0, batch.mask[0]]
    assert 2 not in selected
    distances = np.linalg.norm(coordinates[selected] - coordinates[2], axis=1)
    assert np.all(distances >= settings().guard_radius_um)
    assert np.all(distances <= settings().context_radius_um)
    assert np.linalg.norm(
        batch.coordinates[0, batch.mask[0]], axis=1
    ).max() == pytest.approx(distances.max() / settings().context_radius_um)
    np.testing.assert_array_equal(batch.target_coordinates, np.zeros((1, 1, 2)))

    repeated = make_context_batch(
        embeddings,
        coordinates,
        target_rows=[2],
        spatial_index=fit_spatial_index(coordinates),
        settings=settings(),
    )
    np.testing.assert_array_equal(repeated.rows, batch.rows)


class RecordingPredictor:
    def __init__(self):
        self.contexts = []

    def predict(
        self,
        context_embeddings,
        context_coordinates,
        target_coordinates,
        context_mask,
    ):
        self.contexts.append((context_embeddings.cpu(), context_mask.cpu()))
        weights = context_mask.unsqueeze(-1)
        mean = (context_embeddings * weights).sum(1) / weights.sum(1)
        return mean[:, None, :]


def test_prediction_is_target_specific_and_inverse_normalized():
    coordinates = np.column_stack([np.arange(6) * 100.0, np.zeros(6)])
    embeddings = np.column_stack(
        [np.arange(6, dtype=np.float32), np.ones(6, dtype=np.float32)]
    )
    model = RecordingPredictor()
    normalizer = TargetNormalizer(torch.tensor([10.0, 20.0]), torch.tensor([2.0, 3.0]))

    predicted, counts = predict_section(
        model,
        normalizer,
        embeddings,
        coordinates,
        settings(),
        batch_size=2,
        device="cpu",
    )

    assert predicted.shape == embeddings.shape
    assert np.all(counts > 0)
    assert np.all(predicted[:, 0] >= 10.0)
    for batch_number, (contexts, mask) in enumerate(model.contexts):
        for row in range(len(contexts)):
            target_row = 2 * batch_number + row
            visible_ids = contexts[row, mask[row], 0].numpy()
            assert target_row not in visible_ids


def test_prediction_skips_incomplete_contexts():
    coordinates = np.array([[0.0, 0.0], [1000.0, 0.0], [1100.0, 0.0]])
    embeddings = np.column_stack(
        [np.arange(3, dtype=np.float32), np.ones(3, dtype=np.float32)]
    )
    predicted, counts = predict_section(
        RecordingPredictor(),
        TargetNormalizer(torch.zeros(2), torch.ones(2)),
        embeddings,
        coordinates,
        settings(),
        batch_size=3,
        device="cpu",
    )

    np.testing.assert_array_equal(counts, [0, 1, 1])
    np.testing.assert_array_equal(predicted, np.zeros_like(predicted))


def test_checkpoint_restores_model_normalizer_and_sampling(tmp_path):
    model = SpatialJEPA(
        image_dim=4,
        target_dim=2,
        model_dim=8,
        num_heads=2,
        context_depth=1,
        predictor_depth=1,
    )
    normalizer = TargetNormalizer(torch.zeros(2), torch.ones(2))
    checkpoint = {
        "format_version": CHECKPOINT_FORMAT,
        "model_config": model.model_config,
        "model_state": model.state_dict(),
        "target_normalizer": normalizer.state_dict(),
        "config": {
            "data": {
                "context_cells": 2,
                "context_sampling": "uniform_within_radius",
                "context_radius_um": 250,
                "guard_radius_um": 60,
                "coordinate_normalization": "fixed_radius",
                "sampling_seed": 0,
            }
        },
    }
    path = tmp_path / "best.pt"
    torch.save(checkpoint, path)

    restored, restored_normalizer, loaded = load_checkpoint(path, "cpu")

    assert restored.model_config == model.model_config
    torch.testing.assert_close(restored_normalizer.mean, normalizer.mean)
    assert sampling_settings(loaded) == settings()


def test_sampling_must_be_fully_recorded_in_checkpoint():
    checkpoint = {
        "config": {
            "data": {
                "context_cells": 2,
                "context_sampling": "uniform_within_radius",
                "context_radius_um": 250,
                "guard_radius_um": 60,
                "sampling_seed": 0,
            }
        }
    }
    with pytest.raises(ValueError, match="coordinate normalization"):
        sampling_settings(checkpoint)


def test_section_coordinates_use_manifest_pixel_scale(tmp_path):
    sample_dir = tmp_path / "section"
    sample_dir.mkdir()
    np.savez(
        sample_dir / "S_uni2.npz",
        cell_ids=np.array(["b", "a"]),
        uni_emb=np.array([[2.0], [1.0]], dtype=np.float32),
    )
    np.savez(
        sample_dir / "S_terra.npz",
        cell_ids=np.array(["a", "b"]),
        he_x=np.array([10.0, 30.0]),
        he_y=np.array([20.0, 40.0]),
    )
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "id": ["S"],
            "output_dir": [str(sample_dir)],
            "pixel_size_um": [0.5],
        }
    ).to_csv(manifest, index=False)

    uni, coordinates, cell_ids, sources = load_section(
        {
            "data_root": str(tmp_path),
            "manifest_path": str(manifest),
            "uni_key": "uni_emb",
        },
        "S",
    )

    np.testing.assert_array_equal(cell_ids, ["b", "a"])
    np.testing.assert_array_equal(uni[:, 0], [2.0, 1.0])
    np.testing.assert_allclose(coordinates, [[15.0, 20.0], [5.0, 10.0]])
    assert sources[-1] == manifest


def test_output_schema_is_strict_and_cacheable(tmp_path):
    path = tmp_path / "TENX.npz"
    cell_ids = np.array(["a", "b", "c"])
    embeddings = np.arange(6, dtype=np.float32).reshape(3, 2)
    counts = np.array([2, 2, 1])

    write_output(path, "TENX", cell_ids, embeddings, counts, "signature")

    assert current_output(path, "signature") == 3
    assert current_output(path, "different") is None
    with np.load(path, allow_pickle=False) as data:
        assert set(data.files) == {
            "format_version",
            "signature",
            "sample_id",
            "cell_ids",
            "jepa_emb",
            "n_context",
        }
        assert data["format_version"].item() == FORMAT_VERSION
        assert data["sample_id"].item() == "TENX"
        assert data["jepa_emb"].dtype == np.float16
