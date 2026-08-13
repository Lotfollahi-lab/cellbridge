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
    if not np.allclose(sums, 1.0, atol=1e-7, rtol=0.0):
        raise ValueError("probability rows must sum to one")
    return values / sums[:, None]


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


def calibrate_split(probabilities, labels, classes, alpha=0.1) -> float:
    """Return the finite-sample split-conformal nonconformity quantile."""
    classes = tuple(map(str, classes))
    if len(set(classes)) != len(classes):
        raise ValueError("classifier classes must be unique")
    probabilities = _probabilities(probabilities, len(classes))
    columns = _label_columns(labels, classes, len(probabilities))
    alpha = _alpha(alpha)
    scores = 1.0 - probabilities[np.arange(len(probabilities)), columns]
    rank = int(np.ceil((len(scores) + 1) * (1.0 - alpha)))
    if rank > len(scores):
        return 1.0
    return float(np.partition(scores, rank - 1)[rank - 1])


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
    ordered = sorted(
        distances,
        key=lambda node: (scores[node], distances[node], node),
    )
    selected = next((node for node in ordered if scores[node] >= lambda_hat), None)
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


def _inclusion_threshold(
    probability: np.ndarray,
    true_column: int,
    graph: TaskGraph,
) -> float:
    _, _, scores = _graph_set(probability, graph, 0.0)
    candidates = {0.0, 1.0}
    candidates.update(
        float(np.nextafter(score, np.inf)) for score in scores if 0.0 <= score < 1.0
    )
    for candidate in sorted(candidates):
        mask, _, _ = _graph_set(probability, graph, candidate)
        if mask[true_column]:
            return candidate
    raise RuntimeError("task-graph root did not cover the true class")


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
    n = len(probabilities)
    allowed_risk = alpha - (1.0 - alpha) / n
    if allowed_risk < 0:
        raise ValueError("calibration set is too small for the requested alpha")

    thresholds = np.asarray(
        [
            _inclusion_threshold(row, column, graph)
            for row, column in zip(probabilities, columns)
        ]
    )
    required_covered = int(np.ceil(n * (1.0 - allowed_risk)))
    lambda_hat = float(
        np.partition(thresholds, required_covered - 1)[required_covered - 1]
    )
    empirical_risk = float(np.mean(thresholds > lambda_hat))
    corrected_risk = float((n * empirical_risk + 1.0) / (n + 1.0))
    return GraphCalibration(lambda_hat, empirical_risk, corrected_risk)
