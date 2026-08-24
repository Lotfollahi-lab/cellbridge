from pathlib import Path

import numpy as np
import pandas as pd

from cellbridge.config import load_config
from cellbridge.helpers.splits import spatial_split
from cellbridge.inference.evaluate.hest import three_way_spatial_split
from cellbridge.inference.evaluate.hest_conformal import _ontology_settings
from cellbridge.inference.evaluate.hest_features import load


ROOT = Path(__file__).resolve().parents[1]


def test_three_way_split_preserves_historical_test_stripe():
    cell_ids = np.asarray([f"cell-{index}" for index in range(100)])
    coordinates = np.column_stack([np.zeros(100), np.arange(100, dtype=float)])
    historical_fit, historical_test = spatial_split(
        cell_ids, coordinates, 0.2, "y"
    )

    train, calibration, test = three_way_spatial_split(
        cell_ids, coordinates, 0.2, 0.2, "y"
    )

    np.testing.assert_array_equal(test, historical_test)
    assert len(train) == 64
    assert len(calibration) == 16
    assert not set(train) & set(calibration)
    assert not set(train) & set(test)
    assert not set(calibration) & set(test)
    assert set(np.concatenate([train, calibration])) == set(historical_fit)


def test_frozen_hest_features_are_checked_against_manifest(tmp_path):
    artifact = tmp_path / "sample.npz"
    np.savez(
        artifact,
        cell_ids=np.asarray(["a", "b"]),
        embeddings=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16),
        signature=np.asarray("checked"),
    )
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "sample_id": "sample",
                "path": Path("/old/location") / artifact.name,
                "signature": "checked",
            }
        ]
    ).to_csv(manifest, index=False)
    settings = {
        "method": {
            "manifest": manifest,
            "arms": {"arm": "embeddings"},
        }
    }

    cell_ids, features = load(settings, "method", "arm", "sample")

    np.testing.assert_array_equal(cell_ids, ["a", "b"])
    np.testing.assert_allclose(features, [[1.0, 2.0], [3.0, 4.0]])
    assert features.dtype == np.float32


def test_fast_cohort_has_valid_per_sample_ontology_mappings():
    config = load_config(ROOT / "configs/inference/hest_conformal.yaml")
    samples = ("TENX116", "TENX191", "TENX195", "TENX196", "TENX197")

    for sample_id in samples:
        mapping, ontology, alpha, roots = _ontology_settings(config, sample_id)
        assert mapping.dataset == "hest"
        assert alpha == 0.1
        assert roots
        mapping.validate_ontology(ontology)
