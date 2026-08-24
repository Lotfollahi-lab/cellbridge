"""Split and graph-constrained conformal prediction sets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .graph import TaskGraph


@dataclass(frozen=True)
class GraphCalibration:
    lambda_hat: float
    empirical_risk: float
    corrected_risk: float


@dataclass(frozen=True)
class ConditionalCalibration:
    class_groups: dict[str, str]
    thresholds: dict[str, float]
    counts: dict[str, int]
    empirical_risks: dict[str, float]
    corrected_risks: dict[str, float]


@dataclass(frozen=True)
class GraphPrediction:
    prediction_sets: np.ndarray
    anchor_nodes: np.ndarray


def _probabilities(values, n_classes: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != n_classes:
        raise ValueError(f"probabilities must have shape [cells, {n_classes}]")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("probabilities must be finite and non-negative")
    sums = values.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("probability rows must sum to one")
    return values / sums[:, None]


def _score_matrix(values, n_classes: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != n_classes:
        raise ValueError(f"scores must have shape [cells, {n_classes}]")
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("scores must be finite and between zero and one")
    return values


def _label_columns(labels, classes, expected_rows: int) -> np.ndarray:
    labels = np.asarray(labels).astype(str)
    classes = tuple(map(str, classes))
    if labels.ndim != 1 or len(labels) != expected_rows:
        raise ValueError("labels must be one-dimensional and row-aligned")
    if not len(labels):
        raise ValueError("calibration data must not be empty")
    lookup = {label: column for column, label in enumerate(classes)}
    unknown = sorted(set(labels) - set(lookup))
    if unknown:
        raise ValueError(f"true labels are absent from classifier classes: {unknown}")
    return np.asarray([lookup[label] for label in labels], dtype=int)


def _alpha(value) -> float:
    value = float(value)
    if not 0 < value < 1:
        raise ValueError("alpha must be between zero and one")
    return value


def _quantile(scores, alpha: float) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    rank = int(np.ceil((len(scores) + 1) * (1.0 - alpha)))
    if not len(scores) or rank > len(scores):
        return 1.0
    return float(np.partition(scores, rank - 1)[rank - 1])


def calibrate_split(probabilities, labels, classes, alpha=0.1) -> float:
    """Return the finite-sample split-conformal nonconformity quantile."""
    classes = tuple(map(str, classes))
    if len(set(classes)) != len(classes):
        raise ValueError("classifier classes must be unique")
    probabilities = _probabilities(probabilities, len(classes))
    columns = _label_columns(labels, classes, len(probabilities))
    alpha = _alpha(alpha)
    scores = 1.0 - probabilities[np.arange(len(probabilities)), columns]
    return _quantile(scores, alpha)


def predict_split(probabilities, q_hat) -> np.ndarray:
    """Construct ordinary split-conformal class sets."""
    q_hat = float(q_hat)
    if not 0 <= q_hat <= 1:
        raise ValueError("q_hat must be between zero and one")
    values = np.asarray(probabilities)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("probabilities must have shape [cells, classes]")
    values = _probabilities(values, values.shape[1])
    return values >= 1.0 - q_hat


def calibrate_conditional(
    scores,
    labels,
    classes,
    class_groups,
    alpha=0.1,
) -> ConditionalCalibration:
    """Calibrate one finite-sample threshold per configured class group."""
    classes = tuple(map(str, classes))
    if len(set(classes)) != len(classes):
        raise ValueError("classifier classes must be unique")
    scores = _score_matrix(scores, len(classes))
    labels = np.asarray(labels).astype(str)
    columns = _label_columns(labels, classes, len(scores))
    alpha = _alpha(alpha)
    if set(class_groups) != set(classes):
        raise ValueError(
            "class_groups must contain every classifier class exactly once"
        )
    class_groups = {label: str(class_groups[label]) for label in classes}

    true_scores = scores[np.arange(len(scores)), columns]
    thresholds = {}
    counts = {}
    empirical_risks = {}
    corrected_risks = {}
    for group in sorted(set(class_groups.values())):
        selected = np.asarray([class_groups[label] == group for label in labels])
        group_scores = true_scores[selected]
        threshold = _quantile(group_scores, alpha)
        empirical_risk = (
            float(np.mean(group_scores > threshold)) if len(group_scores) else 0.0
        )
        thresholds[group] = threshold
        counts[group] = len(group_scores)
        empirical_risks[group] = empirical_risk
        corrected_risks[group] = (
            float(
                (len(group_scores) * empirical_risk + 1.0) / (len(group_scores) + 1.0)
            )
            if len(group_scores)
            else 0.0
        )
    return ConditionalCalibration(
        class_groups=class_groups,
        thresholds=thresholds,
        counts=counts,
        empirical_risks=empirical_risks,
        corrected_risks=corrected_risks,
    )


def predict_conditional(scores, classes, calibration: ConditionalCalibration):
    """Include each candidate class using the threshold for its group."""
    classes = tuple(map(str, classes))
    scores = _score_matrix(scores, len(classes))
    if set(calibration.class_groups) != set(classes):
        raise ValueError("calibration classes do not match classifier classes")
    thresholds = np.asarray(
        [
            calibration.thresholds[calibration.class_groups[class_label]]
            for class_label in classes
        ]
    )
    return scores <= thresholds[None, :]


def _graph_set(
    probability: np.ndarray,
    graph: TaskGraph,
    lambda_hat: float,
) -> tuple[np.ndarray, str, tuple[float, ...]]:
    predicted_column = int(np.argmax(probability))
    distances = graph.ancestors_for_column(predicted_column)
    scores = {
        node: round(float(probability[list(graph.descendant_columns[node])].sum()), 15)
        for node in distances
    }
    scores[graph.root] = 1.0
    ordered = sorted(
        distances,
        key=lambda node: (scores[node], distances[node], node),
    )
    selected = (
        graph.root
        if lambda_hat == 1.0
        else next((node for node in ordered if scores[node] >= lambda_hat), None)
    )
    if selected is None:
        raise RuntimeError(
            "task-graph root did not reach the requested probability mass"
        )

    columns = set(graph.descendant_columns[selected])
    for node in distances:
        if scores[node] < lambda_hat:
            columns.update(graph.descendant_columns[node])
    mask = np.zeros(len(graph.classes), dtype=bool)
    mask[list(columns)] = True
    return mask, selected, tuple(scores.values())


def predict_graph(probabilities, graph: TaskGraph, lambda_hat) -> GraphPrediction:
    """Construct the paper's nested graph-constrained prediction sets."""
    lambda_hat = float(lambda_hat)
    if not 0 <= lambda_hat <= 1:
        raise ValueError("lambda_hat must be between zero and one")
    probabilities = _probabilities(probabilities, len(graph.classes))
    prediction_sets = np.empty(probabilities.shape, dtype=bool)
    anchor_nodes = []
    for row, probability in enumerate(probabilities):
        prediction_sets[row], anchor, _ = _graph_set(
            probability,
            graph,
            lambda_hat,
        )
        anchor_nodes.append(anchor)
    return GraphPrediction(
        prediction_sets=prediction_sets,
        anchor_nodes=np.asarray(anchor_nodes, dtype=str),
    )


def _inclusion_scores(probability: np.ndarray, graph: TaskGraph) -> np.ndarray:
    _, _, breakpoints = _graph_set(probability, graph, 0.0)
    candidates = {0.0, 1.0}
    candidates.update(
        float(np.nextafter(score, np.inf))
        for score in breakpoints
        if 0.0 <= score < 1.0
    )
    thresholds = np.full(len(graph.classes), np.nan)
    for candidate in sorted(candidates):
        mask, _, _ = _graph_set(probability, graph, candidate)
        thresholds[mask & np.isnan(thresholds)] = candidate
        if np.isfinite(thresholds).all():
            return thresholds
    raise RuntimeError("task-graph root did not cover every class")


def _inclusion_threshold(
    probability: np.ndarray,
    true_column: int,
    graph: TaskGraph,
) -> float:
    return float(_inclusion_scores(probability, graph)[true_column])


def graph_inclusion_scores(probabilities, graph: TaskGraph) -> np.ndarray:
    """Return the minimum graph mass needed to include every candidate class."""
    probabilities = _probabilities(probabilities, len(graph.classes))
    return np.asarray([_inclusion_scores(row, graph) for row in probabilities])


def calibrate_graph(
    probabilities,
    labels,
    graph: TaskGraph,
    alpha=0.1,
) -> GraphCalibration:
    """Calibrate graph conformal risk control at every exact breakpoint."""
    probabilities = _probabilities(probabilities, len(graph.classes))
    columns = _label_columns(labels, graph.classes, len(probabilities))
    alpha = _alpha(alpha)
    thresholds = np.asarray(
        [
            _inclusion_threshold(row, column, graph)
            for row, column in zip(probabilities, columns)
        ]
    )
    lambda_hat = _quantile(thresholds, alpha)
    empirical_risk = float(np.mean(thresholds > lambda_hat))
    corrected_risk = float(
        (len(thresholds) * empirical_risk + 1.0) / (len(thresholds) + 1.0)
    )
    return GraphCalibration(lambda_hat, empirical_risk, corrected_risk)


def close_graph_sets(prediction_sets, graph: TaskGraph) -> GraphPrediction:
    """Close arbitrary candidate sets to their smallest task-graph superset."""
    prediction_sets = np.asarray(prediction_sets, dtype=bool)
    if prediction_sets.ndim != 2 or prediction_sets.shape[1] != len(graph.classes):
        raise ValueError(
            f"prediction_sets must have shape [cells, {len(graph.classes)}]"
        )

    closed = np.zeros_like(prediction_sets)
    anchors = []
    for row, prediction_set in enumerate(prediction_sets):
        columns = set(np.flatnonzero(prediction_set))
        if not columns:
            anchors.append("")
            continue
        candidates = [
            node
            for node, descendants in graph.descendant_columns.items()
            if columns.issubset(descendants)
        ]
        anchor = min(
            candidates,
            key=lambda node: (len(graph.descendant_columns[node]), node),
        )
        closed[row, list(graph.descendant_columns[anchor])] = True
        anchors.append(anchor)
    return GraphPrediction(closed, np.asarray(anchors, dtype=str))
