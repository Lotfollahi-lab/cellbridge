import numpy as np
import pandas as pd
import pytest

from terra_multi.evaluation.hierarchy_metrics import (
    UNRESOLVED,
    load_hierarchy_mapping,
    score_level,
    score_prediction_file,
)


@pytest.fixture
def mapping(tmp_path):
    """Two tumour clusters share an identity; one label is partial, one excluded."""

    path = tmp_path / "source_label_mapping.csv"
    pd.DataFrame(
        {
            "raw_label": ["0_Tumor", "7_Tumor", "1_Tcell", "2_Immune", "9_Necrotic"],
            "level_1_compartment": ["epithelial", "epithelial", "haem", "haem", ""],
            "level_2_lineage": ["neoplastic", "neoplastic", "lymphoid", "", ""],
            "level_3_identity": ["neoplastic cell", "neoplastic cell", "T cell", "", ""],
            "hierarchy_depth": [3, 3, 3, 1, 0],
        }
    ).to_csv(path, index=False)
    return load_hierarchy_mapping(path)


def test_within_identity_confusion_is_forgiven_at_identity_level(mapping):
    y_true = np.array(["0_Tumor", "0_Tumor", "1_Tcell"])
    y_pred = np.array(["7_Tumor", "0_Tumor", "1_Tcell"])

    identity = score_level(y_true, y_pred, mapping, "identity")

    assert identity["macro_f1"] == pytest.approx(1.0)
    assert identity["macro_f1_source_labels"] < 1.0
    assert identity["n_classes"] == 2


def test_cells_above_the_level_are_excluded_from_truth(mapping):
    y_true = np.array(["0_Tumor", "2_Immune", "9_Necrotic"])
    y_pred = np.array(["0_Tumor", "2_Immune", "9_Necrotic"])

    identity = score_level(y_true, y_pred, mapping, "identity")
    compartment = score_level(y_true, y_pred, mapping, "compartment")

    assert identity["n_scored"] == 1
    assert identity["coverage"] == pytest.approx(1 / 3)
    # The partial label reaches compartment, the excluded one never does.
    assert compartment["n_scored"] == 2


def test_prediction_that_stops_short_counts_as_an_error(mapping):
    y_true = np.array(["0_Tumor", "1_Tcell"])
    y_pred = np.array(["2_Immune", "1_Tcell"])

    identity = score_level(y_true, y_pred, mapping, "identity")

    assert identity["n_scored"] == 2
    assert identity["macro_f1"] == pytest.approx(0.5)


def test_score_prediction_file_covers_every_level(tmp_path, mapping):
    path = tmp_path / "predictions.npz"
    np.savez(
        path,
        y_true=np.array(["0_Tumor", "1_Tcell"]),
        y_pred=np.array(["7_Tumor", "1_Tcell"]),
    )

    rows = score_prediction_file(path, mapping)

    assert [row["level"] for row in rows] == ["compartment", "lineage", "identity"]
    assert all(row["n_test"] == 2 for row in rows)
    assert rows[-1]["macro_f1"] == pytest.approx(1.0)


def test_unresolved_is_never_a_truth_class(mapping):
    y_true = np.array(["0_Tumor", "1_Tcell"])
    y_pred = np.array(["9_Necrotic", "1_Tcell"])

    identity = score_level(y_true, y_pred, mapping, "identity")

    assert identity["n_classes"] == 2
    assert UNRESOLVED not in {"neoplastic cell", "T cell"}
    assert identity["macro_f1"] == pytest.approx(0.5)
