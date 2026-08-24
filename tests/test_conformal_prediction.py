"""Focused tests for global and conditional conformal prediction."""

import numpy as np

from cellbridge.inference.conformal import (
    TaskGraph,
    assign_ontology_branches,
    calibrate_conditional,
    calibrate_graph,
    close_graph_sets,
    graph_inclusion_scores,
    predict_conditional,
    predict_graph,
)
from cellbridge.inference.evaluate.conformal import _conditional_groups
from cellbridge.ontology import Ontology, OntologyMapping


def _graph():
    return TaskGraph(
        root="root",
        classes=("a", "b"),
        class_nodes=("leaf-a", "leaf-b"),
        names={
            "root": "cell",
            "term-a": "term a",
            "term-b": "term b",
            "leaf-a": "a",
            "leaf-b": "b",
        },
        parents={
            "root": (),
            "term-a": ("root",),
            "term-b": ("root",),
            "leaf-a": ("term-a",),
            "leaf-b": ("term-b",),
        },
        children={
            "root": ("term-a", "term-b"),
            "term-a": ("leaf-a",),
            "term-b": ("leaf-b",),
            "leaf-a": (),
            "leaf-b": (),
        },
        descendant_columns={
            "root": (0, 1),
            "term-a": (0,),
            "term-b": (1,),
            "leaf-a": (0,),
            "leaf-b": (1,),
        },
        distances=(
            {"leaf-a": 0, "term-a": 1, "root": 2},
            {"leaf-b": 0, "term-b": 1, "root": 2},
        ),
    )


def test_graph_threshold_one_selects_root_with_exact_zero_probability():
    graph = _graph()
    prediction = predict_graph(np.asarray([[1.0, 0.0]]), graph, 1.0)

    assert prediction.anchor_nodes.tolist() == ["root"]
    assert prediction.prediction_sets.tolist() == [[True, True]]


def test_small_graph_calibration_uses_conservative_threshold():
    graph = _graph()
    calibration = calibrate_graph(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        np.asarray(["b"]),
        graph,
        alpha=0.1,
    )

    assert calibration.lambda_hat == 1.0
    assert calibration.empirical_risk == 0.0


def test_graph_inclusion_scores_cover_each_candidate_at_its_score():
    graph = _graph()
    probabilities = np.asarray([[0.8, 0.2]])
    scores = graph_inclusion_scores(probabilities, graph)

    assert scores.shape == probabilities.shape
    for column in range(2):
        prediction = predict_graph(probabilities, graph, scores[0, column])
        assert prediction.prediction_sets[0, column]


def test_candidate_specific_conditional_thresholds():
    classes = ("a", "b")
    scores = np.asarray([[0.1, 0.9], [0.2, 0.8], [0.7, 0.3]])
    labels = np.asarray(["a", "a", "b"])
    calibration = calibrate_conditional(
        scores,
        labels,
        classes,
        {"a": "class:a", "b": "class:b"},
        alpha=0.5,
    )

    assert calibration.thresholds == {"class:a": 0.2, "class:b": 0.3}
    predicted = predict_conditional(
        np.asarray([[0.15, 0.4], [0.4, 0.2]]),
        classes,
        calibration,
    )
    assert predicted.tolist() == [[True, False], [False, True]]


def test_graph_closure_preserves_candidates():
    graph = _graph()
    candidate_sets = np.asarray([[True, False], [True, True], [False, False]])
    prediction = close_graph_sets(candidate_sets, graph)

    assert np.all(prediction.prediction_sets >= candidate_sets)
    assert prediction.anchor_nodes[1] == "root"
    assert prediction.anchor_nodes[2] == ""


def test_ordered_ontology_branches_are_unique():
    ontology = Ontology(
        release="test",
        source="test",
        sha256="",
        root="root",
        names={
            "root": "cell",
            "immune": "immune cell",
            "myeloid": "myeloid cell",
            "lymphoid": "lymphocyte",
            "term-a": "a",
            "term-b": "b",
            "term-c": "c",
        },
        parents={
            "root": (),
            "immune": ("root",),
            "myeloid": ("immune",),
            "lymphoid": ("immune",),
            "term-a": ("myeloid", "immune"),
            "term-b": ("lymphoid",),
            "term-c": ("root",),
        },
        children={
            "root": ("immune", "term-c"),
            "immune": ("myeloid", "lymphoid", "term-a"),
            "myeloid": ("term-a",),
            "lymphoid": ("term-b",),
            "term-a": (),
            "term-b": (),
            "term-c": (),
        },
    )
    mapping = OntologyMapping(
        dataset="test",
        release="test",
        sha256="",
        source_to_class={"a": "a", "b": "b", "c": "c"},
        class_to_term={"a": "term-a", "b": "term-b", "c": "term-c"},
        class_to_term_name={"a": "a", "b": "b", "c": "c"},
        excluded_classes=frozenset(),
    )

    groups = assign_ontology_branches(
        mapping,
        ontology,
        ("a", "b", "c"),
        ("myeloid", "immune"),
    )

    assert groups == {"a": "myeloid", "b": "immune", "c": "other"}


def test_hybrid_groups_pool_an_entire_branch_for_a_rare_class():
    labels = np.asarray(["a"] * 10 + ["b"] * 6 + ["c"] * 10)
    groups = _conditional_groups(
        labels,
        ("a", "b", "c"),
        {"a": "branch-1", "b": "branch-1", "c": "branch-2"},
        alpha=0.1,
    )

    assert groups["hybrid"] == {
        "a": "branch:branch-1",
        "b": "branch:branch-1",
        "c": "class:c",
    }
