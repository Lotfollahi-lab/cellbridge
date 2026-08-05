"""Metrics for set-valued and ordinary leaf predictions."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .hierarchy import Hierarchy


def coverage_indicator(prediction_sets, labels, classes):
    classes = np.asarray(classes).astype(str)
    lookup = {label: column for column, label in enumerate(classes)}
    columns = np.asarray([lookup[str(label)] for label in labels], dtype=int)
    return np.asarray(prediction_sets, dtype=bool)[np.arange(len(columns)), columns]


def summarize(
    representation,
    method,
    prediction_sets,
    labels,
    top1,
    classes,
    hierarchy: Hierarchy,
    target_coverage,
    returned_nodes=None,
):
    prediction_sets = np.asarray(prediction_sets, dtype=bool)
    labels = np.asarray(labels).astype(str)
    sizes = prediction_sets.sum(axis=1)
    covered = coverage_indicator(prediction_sets, labels, classes)
    nodes = None if returned_nodes is None else np.asarray(returned_nodes).astype(str)
    return {
        "representation": representation,
        "method": method,
        "target_coverage": float(target_coverage),
        "marginal_coverage": float(covered.mean()),
        "mean_prediction_set_size": float(sizes.mean()),
        "singleton_rate": float(np.mean(sizes == 1)),
        "leaf_prediction_rate": (
            float(np.mean(np.isin(nodes, hierarchy.leaves)))
            if nodes is not None
            else float("nan")
        ),
        "mean_returned_ancestor_depth": (
            float(np.mean([hierarchy.depths[node] for node in nodes]))
            if nodes is not None
            else float("nan")
        ),
        "root_inconclusive_rate": (
            float(np.mean(nodes == hierarchy.root))
            if nodes is not None
            else float(np.mean(sizes == len(classes)))
        ),
        "leaf_accuracy": float(accuracy_score(labels, top1)),
        "leaf_macro_f1": float(f1_score(labels, top1, average="macro")),
        "n_test": len(labels),
        "n_classes": len(classes),
    }


def per_class_coverage(
    representation,
    method,
    prediction_sets,
    labels,
    classes,
):
    labels = np.asarray(labels).astype(str)
    covered = coverage_indicator(prediction_sets, labels, classes)
    rows = []
    for label in classes:
        selected = labels == label
        rows.append(
            {
                "representation": representation,
                "method": method,
                "class": str(label),
                "n_test": int(selected.sum()),
                "coverage": float(covered[selected].mean()),
            }
        )
    return rows
