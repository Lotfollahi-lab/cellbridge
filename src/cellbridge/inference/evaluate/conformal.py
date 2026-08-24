"""Leakage-free conformal evaluation of a fitted cell-type probe."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from cellbridge.inference.conformal import (
    ConditionalCalibration,
    GraphCalibration,
    assign_ontology_branches,
    build_task_graph,
    calibrate_conditional,
    calibrate_graph,
    calibrate_split,
    close_graph_sets,
    graph_inclusion_scores,
    predict_conditional,
    predict_graph,
    predict_split,
)
from cellbridge.inference.evaluate import Task, ood
from cellbridge.inference.evaluate import probe as probe_api
from cellbridge.ontology import Ontology, OntologyMapping


@dataclass(frozen=True)
class ConformalResult:
    """Metrics and row-aligned test predictions from all set methods."""

    metrics: tuple[dict, ...]
    per_class_coverage: tuple[dict, ...]
    calibration: tuple[dict, ...]
    ids: np.ndarray
    y_true: np.ndarray
    probabilities: np.ndarray
    classes: np.ndarray
    split_sets: np.ndarray
    graph_sets: np.ndarray
    graph_nodes: np.ndarray
    conditional_sets: dict[str, np.ndarray]
    conditional_nodes: dict[str, np.ndarray]


def prepare_task(
    features,
    metadata,
    mapping: OntologyMapping,
    *,
    task: str,
    arm: str,
    pca_components: int,
    seed: int,
) -> Task:
    """Harmonize source labels, exclude unmapped concepts, and prepare splits."""
    source_labels = metadata["label"].astype(str).to_numpy()
    labels, included = mapping.map_labels(source_labels)
    if task == "immune":
        included &= metadata["label"].isin(ood.IMMUNE_CLASSES).to_numpy()
    elif task != "all":
        raise ValueError("task must be 'all' or 'immune'")
    if not included.any():
        raise ValueError(f"{task} has no ontology-mapped cells")

    selected_metadata = metadata.loc[included].reset_index(drop=True).copy()
    selected_metadata["label"] = labels[included]
    selected_features = np.asarray(features)[included]
    return ood.prepare_task(
        selected_features,
        selected_metadata,
        task="all",
        arm=arm,
        pca_components=pca_components,
        seed=seed,
    )


def _indices(task: Task) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = np.asarray(task.train, dtype=int)
    if task.validation is None:
        raise ValueError("conformal evaluation requires a calibration split")
    calibration = np.asarray(task.validation, dtype=int)
    test = np.asarray(task.test, dtype=int)
    splits = {"train": train, "calibration": calibration, "test": test}
    for name, indices in splits.items():
        if len(indices) == 0:
            raise ValueError(f"{name} indices are empty")
        if np.any(indices < 0) or np.any(indices >= len(task.features)):
            raise ValueError(f"{name} indices are out of bounds")
        if len(np.unique(indices)) != len(indices):
            raise ValueError(f"{name} indices contain duplicates")
    if any(
        np.intersect1d(left, right).size
        for left, right in ((train, calibration), (train, test), (calibration, test))
    ):
        raise ValueError("train, calibration, and test splits must be disjoint")
    return train, calibration, test


def _coverage(prediction_sets, labels, classes) -> np.ndarray:
    lookup = {label: column for column, label in enumerate(classes)}
    try:
        columns = np.asarray([lookup[label] for label in labels], dtype=int)
    except KeyError as error:
        raise ValueError(
            f"test label is absent from classifier classes: {error.args[0]!r}"
        ) from error
    return prediction_sets[np.arange(len(labels)), columns]


def _calibration_risks(calibration):
    if calibration is None:
        return float("nan"), float("nan"), 1
    if isinstance(calibration, GraphCalibration):
        return calibration.empirical_risk, calibration.corrected_risk, 1
    counts = np.asarray(list(calibration.counts.values()), dtype=float)
    empirical = np.asarray(list(calibration.empirical_risks.values()), dtype=float)
    corrected = np.asarray(list(calibration.corrected_risks.values()), dtype=float)
    return (
        float(np.average(empirical, weights=counts)),
        float(corrected.max()),
        len(counts),
    )


def _summary(
    method,
    prediction_sets,
    labels,
    classes,
    alpha,
    split_counts,
    fit_details,
    *,
    parameter,
    value,
    conditioning="global",
    calibration=None,
    graph_nodes=None,
    root=None,
) -> dict:
    sizes = prediction_sets.sum(axis=1)
    covered = _coverage(prediction_sets, labels, classes)
    empirical_risk, corrected_risk, n_groups = _calibration_risks(calibration)
    return {
        "conformal_method": method,
        "conditioning": conditioning,
        "target_coverage": 1.0 - alpha,
        "marginal_coverage": float(covered.mean()),
        "mean_prediction_set_size": float(sizes.mean()),
        "singleton_rate": float(np.mean(sizes == 1)),
        "root_inconclusive_rate": (
            float(np.mean(graph_nodes == root))
            if graph_nodes is not None
            else float(np.mean(sizes == len(classes)))
        ),
        "parameter": parameter,
        "value": value,
        "empirical_risk": empirical_risk,
        "corrected_risk": corrected_risk,
        "n_groups": n_groups,
        "n_classes": len(classes),
        **split_counts,
        **fit_details,
    }


def _per_class(method, prediction_sets, labels, classes) -> tuple[dict, ...]:
    covered = _coverage(prediction_sets, labels, classes)
    rows = []
    for class_label in classes:
        selected = labels == class_label
        rows.append(
            {
                "conformal_method": method,
                "class": class_label,
                "n_test": int(selected.sum()),
                "coverage": (
                    float(covered[selected].mean()) if selected.any() else float("nan")
                ),
            }
        )
    return tuple(rows)


def _calibration_rows(
    method: str,
    calibration: ConditionalCalibration,
) -> tuple[dict, ...]:
    rows = []
    for group, threshold in calibration.thresholds.items():
        members = sorted(
            label
            for label, class_group in calibration.class_groups.items()
            if class_group == group
        )
        rows.append(
            {
                "conformal_method": method,
                "group": group,
                "classes": ";".join(members),
                "n_calibration": calibration.counts[group],
                "threshold": threshold,
                "empirical_risk": calibration.empirical_risks[group],
                "corrected_risk": calibration.corrected_risks[group],
            }
        )
    return tuple(rows)


def _conditional_groups(
    labels,
    classes,
    branch_groups,
    alpha,
) -> dict[str, dict[str, str]]:
    class_groups = {label: f"class:{label}" for label in classes}
    ontology_groups = {
        label: f"branch:{branch_groups[label]}" for label in classes
    }
    minimum_count = int(np.ceil(1.0 / alpha) - 1)
    counts = {label: int(np.sum(labels == label)) for label in classes}
    hybrid_groups = {
        label: (
            class_groups[label]
            if counts[label] >= minimum_count
            else ontology_groups[label]
        )
        for label in classes
    }
    return {
        "class": class_groups,
        "branch": ontology_groups,
        "hybrid": hybrid_groups,
    }


def evaluate(
    task: Task,
    probe,
    settings,
    mapping: OntologyMapping,
    ontology: Ontology,
    *,
    branch_roots,
    alpha=0.1,
) -> ConformalResult:
    """Fit on train, calibrate on validation, and predict sets on test."""
    train, calibration_indices, test = _indices(task)
    fit_task = replace(task, validation=None)
    model, fit_details = probe_api.fit(fit_task, probe, settings)

    features = np.asarray(task.features, dtype=np.float32)
    labels = np.asarray(task.labels).astype(str)
    classes = np.asarray(model.classes_).astype(str)
    graph = build_task_graph(mapping, ontology, classes)
    branch_groups = assign_ontology_branches(
        mapping,
        ontology,
        classes,
        branch_roots,
    )

    calibration_probabilities = model.predict_proba(features[calibration_indices])
    test_probabilities = model.predict_proba(features[test])
    calibration_labels = labels[calibration_indices]
    test_labels = labels[test]

    q_hat = calibrate_split(
        calibration_probabilities,
        calibration_labels,
        classes,
        alpha,
    )
    split_sets = predict_split(test_probabilities, q_hat)
    graph_calibration = calibrate_graph(
        calibration_probabilities,
        calibration_labels,
        graph,
        alpha,
    )
    graph_prediction = predict_graph(
        test_probabilities,
        graph,
        graph_calibration.lambda_hat,
    )

    calibration_scores = {
        "split": 1.0 - calibration_probabilities,
        "graph": graph_inclusion_scores(calibration_probabilities, graph),
    }
    test_scores = {
        "split": 1.0 - test_probabilities,
        "graph": graph_inclusion_scores(test_probabilities, graph),
    }
    groups = _conditional_groups(calibration_labels, classes, branch_groups, alpha)
    conditional_sets = {}
    conditional_nodes = {}
    conditional_calibrations = {}
    for conditioning, class_groups in groups.items():
        for set_method in ("split", "graph"):
            method = f"{conditioning}_{set_method}"
            calibrated = calibrate_conditional(
                calibration_scores[set_method],
                calibration_labels,
                classes,
                class_groups,
                alpha,
            )
            prediction_sets = predict_conditional(
                test_scores[set_method],
                classes,
                calibrated,
            )
            conditional_calibrations[method] = calibrated
            if set_method == "graph":
                prediction = close_graph_sets(prediction_sets, graph)
                conditional_sets[method] = prediction.prediction_sets
                conditional_nodes[method] = prediction.anchor_nodes
            else:
                conditional_sets[method] = prediction_sets

    split_counts = {
        "n_train": len(train),
        "n_calibration": len(calibration_indices),
        "n_test": len(test),
    }
    metrics = [
        _summary(
            "split",
            split_sets,
            test_labels,
            classes,
            float(alpha),
            split_counts,
            fit_details,
            parameter="q_hat",
            value=q_hat,
        ),
        _summary(
            "graph",
            graph_prediction.prediction_sets,
            test_labels,
            classes,
            float(alpha),
            split_counts,
            fit_details,
            parameter="lambda_hat",
            value=graph_calibration.lambda_hat,
            calibration=graph_calibration,
            graph_nodes=graph_prediction.anchor_nodes,
            root=graph.root,
        ),
    ]
    per_class = list(_per_class("split", split_sets, test_labels, classes))
    per_class.extend(
        _per_class(
            "graph",
            graph_prediction.prediction_sets,
            test_labels,
            classes,
        )
    )
    calibration_rows = []
    for method, prediction_sets in conditional_sets.items():
        conditioning, set_method = method.rsplit("_", 1)
        calibrated = conditional_calibrations[method]
        metrics.append(
            _summary(
                method,
                prediction_sets,
                test_labels,
                classes,
                float(alpha),
                split_counts,
                fit_details,
                parameter="group_thresholds",
                value=float("nan"),
                conditioning=conditioning,
                calibration=calibrated,
                graph_nodes=conditional_nodes.get(method),
                root=graph.root if set_method == "graph" else None,
            )
        )
        per_class.extend(_per_class(method, prediction_sets, test_labels, classes))
        calibration_rows.extend(_calibration_rows(method, calibrated))

    return ConformalResult(
        metrics=tuple(metrics),
        per_class_coverage=tuple(per_class),
        calibration=tuple(calibration_rows),
        ids=np.asarray(task.ids)[test],
        y_true=test_labels,
        probabilities=np.asarray(test_probabilities, dtype=np.float32),
        classes=classes,
        split_sets=split_sets,
        graph_sets=graph_prediction.prediction_sets,
        graph_nodes=graph_prediction.anchor_nodes,
        conditional_sets=conditional_sets,
        conditional_nodes=conditional_nodes,
    )


def save_result(output_dir, result: ConformalResult) -> tuple[Path, ...]:
    """Write metrics, calibration details, and aligned prediction artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    coverage_path = output_dir / "per_class_coverage.csv"
    calibration_path = output_dir / "calibration.csv"
    predictions_path = output_dir / "predictions.npz"

    pd.DataFrame(result.metrics).to_csv(metrics_path, index=False)
    pd.DataFrame(result.per_class_coverage).to_csv(coverage_path, index=False)
    pd.DataFrame(result.calibration).to_csv(calibration_path, index=False)
    arrays = {
        "ids": np.asarray(result.ids).astype(str),
        "y_true": np.asarray(result.y_true).astype(str),
        "probabilities": result.probabilities,
        "classes": np.asarray(result.classes).astype(str),
        "split_sets": result.split_sets,
        "graph_sets": result.graph_sets,
        "graph_nodes": np.asarray(result.graph_nodes).astype(str),
    }
    arrays.update(
        {f"{method}_sets": values for method, values in result.conditional_sets.items()}
    )
    arrays.update(
        {
            f"{method}_nodes": np.asarray(values).astype(str)
            for method, values in result.conditional_nodes.items()
        }
    )
    np.savez(predictions_path, **arrays)
    return metrics_path, coverage_path, calibration_path, predictions_path
