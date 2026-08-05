"""Exact split and graph-constrained conformal prediction sets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hierarchy import Hierarchy


@dataclass(frozen=True)
class GraphCalibration:
    lambda_hat: float
    empirical_risk: float
    corrected_risk: float


def _validate(probabilities, classes):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    classes = np.asarray(classes).astype(str)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(classes):
        raise ValueError("probabilities must have shape [cells, classes]")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("probabilities must be finite and non-negative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("probability rows must sum to one")
    if len(set(classes)) != len(classes):
        raise ValueError("classifier classes must be unique")
    return probabilities, classes


def _true_columns(labels, classes, expected_rows=None):
    labels = np.asarray(labels).astype(str)
    if labels.ndim != 1:
        raise ValueError("true labels must be one-dimensional")
    if expected_rows is not None and len(labels) != expected_rows:
        raise ValueError(
            f"received {len(labels)} true labels for {expected_rows} probability rows"
        )
    if not len(labels):
        raise ValueError("calibration data must not be empty")
    lookup = {label: column for column, label in enumerate(classes)}
    try:
        return np.asarray([lookup[str(label)] for label in labels], dtype=int)
    except KeyError as error:
        raise ValueError(
            f"true label is absent from classifier classes: {error.args[0]}"
        ) from error


def calibrate_split(probabilities, labels, classes, alpha=0.1) -> float:
    """Return the finite-sample split-conformal nonconformity quantile."""
    probabilities, classes = _validate(probabilities, classes)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    columns = _true_columns(labels, classes, len(probabilities))
    scores = 1.0 - probabilities[np.arange(len(probabilities)), columns]
    rank = int(np.ceil((len(scores) + 1) * (1.0 - alpha)))
    if rank > len(scores):
        return 1.0
    return float(np.partition(scores, rank - 1)[rank - 1])


def split_sets(probabilities, q_hat):
    probabilities = np.asarray(probabilities)
    return probabilities >= 1.0 - float(q_hat)


def _graph_set(probability, classes, hierarchy, lambda_hat):
    predicted = str(classes[int(np.argmax(probability))])
    ancestors = hierarchy.ancestors(predicted)
    class_index = {label: index for index, label in enumerate(classes)}
    leaf_sets = {node: hierarchy.descendant_leaves(node) for node in ancestors}
    scores = {
        node: round(
            sum(probability[class_index[leaf]] for leaf in leaves),
            15,
        )
        for node, leaves in leaf_sets.items()
    }
    ordered = sorted(
        ancestors,
        key=lambda node: (scores[node], hierarchy.distance(node, predicted)),
    )
    selected = next((node for node in ordered if scores[node] >= lambda_hat), None)
    if selected is None:
        raise ValueError(f"lambda={lambda_hat} exceeds the root probability mass")

    leaves = set(leaf_sets[selected])
    for node in ancestors:
        if scores[node] < lambda_hat:
            leaves.update(leaf_sets[node])
    mask = np.asarray([label in leaves for label in classes], dtype=bool)
    return mask, selected, tuple(scores.values())


def _inclusion_threshold(probability, true_label, classes, hierarchy):
    _, _, scores = _graph_set(probability, classes, hierarchy, 0.0)
    candidates = {0.0, 1.0}
    candidates.update(
        float(np.nextafter(score, np.inf)) for score in scores if 0.0 <= score < 1.0
    )
    true_column = int(np.flatnonzero(classes == true_label)[0])
    for candidate in sorted(candidates):
        mask, _, _ = _graph_set(probability, classes, hierarchy, candidate)
        if mask[true_column]:
            return candidate
    raise RuntimeError("root prediction set did not cover the true label")


def calibrate_graph(
    probabilities,
    labels,
    classes,
    hierarchy: Hierarchy,
    alpha=0.1,
) -> GraphCalibration:
    """Calibrate the paper's full graph construction at every exact breakpoint."""
    probabilities, classes = _validate(probabilities, classes)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    hierarchy.validate_classes(classes)
    labels = np.asarray(labels).astype(str)
    _true_columns(labels, classes, len(probabilities))
    n = len(labels)
    thresholds = np.asarray(
        [
            _inclusion_threshold(row, label, classes, hierarchy)
            for row, label in zip(probabilities, labels)
        ]
    )
    allowed = alpha - (1.0 - alpha) / n
    if allowed < 0:
        raise ValueError("calibration set is too small for the requested alpha")
    required_covered = int(np.ceil(n * (1.0 - allowed)))
    lambda_hat = float(
        np.partition(thresholds, required_covered - 1)[required_covered - 1]
    )
    risk = float(np.mean(thresholds > lambda_hat))
    corrected = float((n * risk + 1.0) / (n + 1.0))
    return GraphCalibration(lambda_hat, risk, corrected)


def graph_sets(probabilities, classes, hierarchy: Hierarchy, lambda_hat):
    probabilities, classes = _validate(probabilities, classes)
    hierarchy.validate_classes(classes)
    results = [_graph_set(row, classes, hierarchy, lambda_hat) for row in probabilities]
    masks = np.stack([result[0] for result in results])
    nodes = np.asarray([result[1] for result in results], dtype=str)
    return masks, nodes
