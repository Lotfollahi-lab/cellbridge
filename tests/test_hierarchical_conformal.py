import numpy as np
import pytest

from terra_multi.hierarchical_conformal.conformal import (
    calibrate_graph,
    calibrate_split,
    graph_sets,
    split_sets,
)
from terra_multi.hierarchical_conformal.hierarchy import Hierarchy
from terra_multi.hierarchical_conformal.metrics import (
    coverage_indicator,
    per_class_coverage,
)


@pytest.fixture
def hierarchy():
    return Hierarchy.from_edges(
        "cell",
        [
            ("cell", "left"),
            ("cell", "right"),
            ("left", "A"),
            ("left", "B"),
            ("right", "C"),
            ("right", "D"),
        ],
    )


def test_hierarchy_validates_leaf_classes_and_cycles(hierarchy):
    hierarchy.validate_classes(["A", "B", "C", "D"])
    assert hierarchy.ancestors("A") == ("A", "cell", "left")
    assert hierarchy.descendant_leaves("left") == ("A", "B")
    assert hierarchy.distance("cell", "A") == 2

    with pytest.raises(ValueError, match="cycle"):
        Hierarchy.from_edges(
            "cell",
            [("cell", "A"), ("A", "B"), ("B", "A")],
        )
    with pytest.raises(ValueError, match="classifier classes"):
        hierarchy.validate_classes(["A", "B", "C"])


def test_split_conformal_uses_finite_sample_order_statistic():
    classes = np.array(["A", "B"])
    true_probability = np.arange(0.05, 0.55, 0.05)
    probabilities = np.column_stack([true_probability, 1 - true_probability])
    labels = np.repeat("A", len(probabilities))

    q_hat = calibrate_split(probabilities, labels, classes, alpha=0.2)
    scores = 1 - true_probability
    assert q_hat == pytest.approx(np.sort(scores)[8])
    sets = split_sets(probabilities, q_hat)
    assert sets.shape == probabilities.shape


def test_graph_construction_moves_to_valid_ancestors(hierarchy):
    classes = np.array(["A", "B", "C", "D"])
    probabilities = np.array([[0.4, 0.3, 0.2, 0.1]])

    at_leaf, nodes = graph_sets(probabilities, classes, hierarchy, 0.4)
    np.testing.assert_array_equal(at_leaf[0], [True, False, False, False])
    assert nodes.tolist() == ["A"]

    at_lineage, nodes = graph_sets(
        probabilities,
        classes,
        hierarchy,
        np.nextafter(0.4, np.inf),
    )
    np.testing.assert_array_equal(at_lineage[0], [True, True, False, False])
    assert nodes.tolist() == ["left"]

    at_root, nodes = graph_sets(
        probabilities,
        classes,
        hierarchy,
        np.nextafter(0.7, np.inf),
    )
    assert at_root.all()
    assert nodes.tolist() == ["cell"]


def test_graph_tie_breaking_prefers_closest_node(hierarchy):
    classes = np.array(["A", "B", "C", "D"])
    probabilities = np.array([[0.5, 0.0, 0.3, 0.2]])
    sets, nodes = graph_sets(probabilities, classes, hierarchy, 0.5)
    np.testing.assert_array_equal(sets[0], [True, False, False, False])
    assert nodes.tolist() == ["A"]


def test_graph_calibration_applies_risk_correction(hierarchy):
    classes = np.array(["A", "B", "C", "D"])
    probabilities = np.tile([0.6, 0.3, 0.06, 0.04], (20, 1))
    labels = np.array(["A"] * 16 + ["B"] * 4)

    calibration = calibrate_graph(
        probabilities,
        labels,
        classes,
        hierarchy,
        alpha=0.2,
    )
    assert calibration.lambda_hat > 0.6
    assert calibration.corrected_risk <= 0.2
    sets, _ = graph_sets(
        probabilities,
        classes,
        hierarchy,
        calibration.lambda_hat,
    )
    assert coverage_indicator(sets, labels, classes).all()


def test_per_class_coverage_reports_each_leaf():
    classes = np.array(["A", "B"])
    labels = np.array(["A", "A", "B"])
    sets = np.array([[1, 0], [0, 1], [0, 1]], dtype=bool)
    rows = per_class_coverage("UNI", "split", sets, labels, classes)
    assert rows == [
        {
            "representation": "UNI",
            "method": "split",
            "class": "A",
            "n_test": 2,
            "coverage": 0.5,
        },
        {
            "representation": "UNI",
            "method": "split",
            "class": "B",
            "n_test": 1,
            "coverage": 1.0,
        },
    ]
