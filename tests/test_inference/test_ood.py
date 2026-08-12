import numpy as np
import pandas as pd
import pytest

from cellbridge.inference.evaluate import ood


@pytest.fixture
def inputs():
    features = np.arange(32, dtype=np.float32).reshape(8, 4)
    metadata = pd.DataFrame(
        {
            "row_id": np.arange(8),
            "sample_id": [f"s{i}" for i in range(8)],
            "label": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "split": ["train", "train", "train", "train", "val", "val", "test", "test"],
        }
    )
    return features, metadata


@pytest.mark.parametrize("arm", ["uni_original", "lora_adapted"])
@pytest.mark.parametrize("evaluation_split", ["test", "val"])
def test_prepare_task_fits_pca_on_official_training_rows(
    inputs, monkeypatch, arm, evaluation_split
):
    features, metadata = inputs
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

    monkeypatch.setattr(ood, "PCA", RecordingPCA)
    task = ood.prepare_task(
        features,
        metadata,
        evaluation_split=evaluation_split,
        arm=arm,
        pca_components=2,
        seed=7,
    )

    np.testing.assert_array_equal(fitted["values"], features[task.train])
    assert task.train.tolist() == [0, 1, 2, 3]
    assert fitted["settings"] == (2, "randomized", 7)
    assert task.features.shape == (8, 2)
    assert task.features.dtype == np.float32


def test_prepare_task_keeps_projected_arm_unchanged(inputs, monkeypatch):
    features, metadata = inputs
    monkeypatch.setattr(
        ood,
        "PCA",
        lambda **_: pytest.fail("PCA must not run for lora_projected"),
    )

    task = ood.prepare_task(
        features,
        metadata,
        arm="lora_projected",
        pca_components=2,
    )

    np.testing.assert_array_equal(task.features, features)
