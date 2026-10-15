"""Keep the three manuscript constructions and their saved result contract."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from cellbridge.inference.conformal import bootstrap, conformal_ood
from cellbridge.inference.evaluate import Task
from cellbridge.ontology import Ontology, OntologyMapping

PAPER_METHODS = ("split", "graph", "branch_graph")


def example(seed, n_calibration=40):
    """A small ontology DAG and fixed probe probabilities, including ties."""
    parents = {
        "root": (),
        "left": ("root",),
        "right": ("root",),
        "inner": ("left",),
        "a": ("left",),
        "b": ("left", "inner"),
        "c": ("right",),
        "d": ("right",),
    }
    ontology = Ontology(
        release="test",
        source="test",
        sha256="",
        root="root",
        names={node: node for node in parents},
        parents=parents,
        children={
            node: tuple(k for k, v in parents.items() if node in v) for node in parents
        },
    )
    classes = np.array(list("abcd"))
    mapping = OntologyMapping(
        dataset="test",
        release="test",
        sha256="",
        source_to_class={c: c for c in classes},
        class_to_term={c: c for c in classes},
        class_to_term_name={c: c for c in classes},
        excluded_classes=frozenset(),
    )
    n_train, n_test = 12, 24
    n_cells = n_train + n_calibration + n_test
    probabilities = np.random.default_rng(seed).dirichlet(np.ones(4), n_cells)
    probabilities[-4:] = [
        [0.25, 0.25, 0.25, 0.25],
        [1, 0, 0, 0],
        [0.5, 0.5, 0, 0],
        [0, 0, 0.5, 0.5],
    ]
    task = Task(
        features=probabilities,
        labels=np.resize(classes, n_cells),
        ids=np.array([f"cell{i}" for i in range(n_cells)]),
        train=np.arange(n_train),
        validation=np.arange(n_train, n_train + n_calibration),
        test=np.arange(n_train + n_calibration, n_cells),
    )

    class FixedProbe:
        classes_ = classes

        def predict_proba(self, features):
            return features

    return task, mapping, ontology, FixedProbe()


@pytest.mark.parametrize("n_calibration", [4, 40])
@pytest.mark.parametrize("seed", [0, 19])
def test_paper_outputs_and_saved_artifacts(tmp_path, seed, n_calibration):
    task, mapping, ontology, model = example(seed, n_calibration)
    with patch.object(conformal_ood.probe_api, "fit", return_value=(model, {})) as fit:
        result = conformal_ood.evaluate(
            task,
            "linear",
            {},
            mapping,
            ontology,
            branch_roots=["left", "right"],
            alpha=0.1,
        )
    # Classifier fitting must not consume the conformal calibration split.
    assert fit.call_args.args[0].validation is None
    np.testing.assert_array_equal(fit.call_args.args[0].train, task.train)
    assert tuple(row["conformal_method"] for row in result.metrics) == PAPER_METHODS
    assert set(result.conditional_sets) == {"branch_graph"}
    assert set(result.conditional_nodes) == {"branch_graph"}
    assert {row["conformal_method"] for row in result.calibration} == {"branch_graph"}
    assert {row["group"] for row in result.calibration} == {
        "branch:left",
        "branch:right",
    }

    sets = bootstrap.prediction_sets(result)
    assert tuple(sets) == PAPER_METHODS
    groups = np.repeat(np.arange(6), 4)
    uncertainty, draws = bootstrap.summarize(
        sets,
        result.y_true,
        result.classes,
        groups,
        {"n_resamples": 20, "seed": 0},
        target_coverage=0.9,
    )
    assert set(uncertainty.conformal_method) == set(PAPER_METHODS)
    assert set(draws.conformal_method) == set(PAPER_METHODS)
    for metric in result.metrics:
        rows = uncertainty.loc[
            uncertainty.conformal_method == metric["conformal_method"]
        ]
        for row in rows.itertuples():
            assert row.estimate == pytest.approx(metric[row.metric])

    conformal_ood.save_result(tmp_path, result)
    assert (
        tuple(pd.read_csv(tmp_path / "metrics.csv").conformal_method) == PAPER_METHODS
    )
    with np.load(tmp_path / "predictions.npz", allow_pickle=False) as saved:
        assert set(saved.files) == {
            "ids",
            "y_true",
            "probabilities",
            "classes",
            "split_sets",
            "graph_sets",
            "graph_nodes",
            "branch_graph_sets",
            "branch_graph_nodes",
        }
        for method, masks in sets.items():
            np.testing.assert_array_equal(saved[f"{method}_sets"], masks)
