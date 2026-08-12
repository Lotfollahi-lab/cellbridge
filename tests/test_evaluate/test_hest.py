import numpy as np
import pandas as pd
import pytest

from cellbridge.inference.evaluate import hest


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    cell_ids = np.asarray([f"c{i}" for i in range(6)])
    features = np.arange(24, dtype=np.float32).reshape(6, 4)
    labels_path = tmp_path / "labels.csv"
    pd.DataFrame(
        {"cell_id": cell_ids, "group": ["A", "B", "A", "B", "A", "B"]}
    ).to_csv(labels_path, index=False)
    coordinates = np.column_stack((np.zeros(6), np.arange(6)))
    monkeypatch.setattr(hest, "load_coords", lambda _: (coordinates, cell_ids))
    return features, cell_ids, labels_path


@pytest.mark.parametrize("arm", ["uni_original", "lora_adapted"])
def test_prepare_task_fits_pca_on_training_rows(inputs, monkeypatch, arm):
    features, cell_ids, labels_path = inputs
    fitted = {}

    class RecordingPCA:
        def __init__(self, n_components, svd_solver, random_state):
            fitted["settings"] = (n_components, svd_solver, random_state)
            self.n_components = n_components

        def fit(self, values):
            fitted["values"] = values.copy()
            return self

        def transform(self, values):
            return values[:, : self.n_components]

    monkeypatch.setattr(hest, "PCA", RecordingPCA)
    task = hest.prepare_task(
        features,
        cell_ids,
        labels_path,
        "coordinates.h5ad",
        min_class_count=1,
        test_fraction=1 / 3,
        arm=arm,
        pca_components=2,
        seed=7,
    )

    np.testing.assert_array_equal(fitted["values"], features[task.train])
    assert fitted["settings"] == (2, "randomized", 7)
    assert task.features.shape == (6, 2)
    assert task.features.dtype == np.float32


def test_prepare_task_keeps_projected_arm_unchanged(inputs, monkeypatch):
    features, cell_ids, labels_path = inputs
    monkeypatch.setattr(
        hest,
        "PCA",
        lambda **_: pytest.fail("PCA must not run for lora_projected"),
    )

    task = hest.prepare_task(
        features,
        cell_ids,
        labels_path,
        "coordinates.h5ad",
        min_class_count=1,
        test_fraction=1 / 3,
        arm="lora_projected",
        pca_components=2,
    )

    np.testing.assert_array_equal(task.features, features)
