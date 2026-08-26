import numpy as np
import pandas as pd
import pytest

from cellbridge.evaluation import evaluate_predictions
from cellbridge.inference.evaluate import bootstrap


def _result():
    y_true = np.array(["a", "b"] * 4)
    y_pred = np.array(["a", "b", "b", "a", "a", "b", "a", "a"])
    probabilities = np.full((len(y_true), 2), 0.2, dtype=np.float32)
    probabilities[np.arange(len(y_true)), (y_pred == "b").astype(int)] = 0.8
    return evaluate_predictions(
        ids=np.arange(len(y_true)),
        y_true=y_true,
        probabilities=probabilities,
        classes=np.array(["a", "b"]),
    )


def test_group_bootstrap_is_deterministic_and_preserves_point_estimate():
    result = _result()
    groups = np.repeat(["s1", "s2", "s3", "s4"], 2)
    settings = {"n_resamples": 200, "seed": 17}

    summary, replicates = bootstrap.summarize(result, groups, settings)
    repeated_summary, repeated_replicates = bootstrap.summarize(
        result, groups, settings
    )

    assert summary["metric"].tolist() == list(bootstrap.METRICS)
    assert len(replicates) == 200
    assert summary["n_test_groups"].eq(4).all()
    assert summary["min_class_group_count"].eq(4).all()
    assert summary.set_index("metric").loc["macro_f1", "estimate"] == pytest.approx(
        result.metrics["macro_f1"]
    )
    assert summary["bootstrap_se"].gt(0).all()
    pd.testing.assert_frame_equal(summary, repeated_summary)
    pd.testing.assert_frame_equal(replicates, repeated_replicates)


def test_resolve_groups_aligns_ids_and_extracts_biological_replicates():
    metadata = pd.DataFrame(
        {
            "row_id": [10, 11, 12],
            "sample_id": ["reg001_A.hdf", "reg001_B.hdf", "reg002_A.hdf"],
        }
    )

    groups = bootstrap.resolve_groups(
        metadata,
        np.array([12, 10, 11]),
        {"group_column": "sample_id", "group_regex": r"^(reg\d+)"},
    )

    assert groups.tolist() == ["reg002", "reg001", "reg001"]


def test_group_bootstrap_rejects_cell_level_single_group():
    with pytest.raises(ValueError, match="at least two test groups"):
        bootstrap.summarize(_result(), np.repeat("one", 8), {"n_resamples": 10})
