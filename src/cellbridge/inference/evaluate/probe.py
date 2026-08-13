"""Shared scoring and output for linear and kNN probes."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

from cellbridge.inference.evaluate import Task, knn, linear

PROBES = ("linear", "knn")


@dataclass(frozen=True)
class ProbeResult:
    """Common metrics and test-set predictions from one fitted probe."""

    metrics: dict
    ids: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    probabilities: np.ndarray
    classes: np.ndarray


def fit(task: Task, probe, settings=None):
    """Fit one configured probe and return its estimator and fit details."""
    settings = settings or {}
    if probe == "linear":
        fitted = linear.fit(
            task,
            Cs=settings.get("Cs", (0.1, 1.0, 10.0)),
            max_iter=settings.get("max_iter", 2000),
            seed=settings.get("seed", 0),
        )
        return fitted.model, {
            "C": fitted.C,
            "selection_macro_f1": fitted.selection_macro_f1,
        }
    if probe == "knn":
        fitted = knn.fit(task, n_neighbors=settings.get("n_neighbors", 20))
        return fitted.model, {
            "n_neighbors": fitted.n_neighbors,
            "n_reference": fitted.n_reference,
        }
    raise ValueError(f"unknown probe {probe!r}; choose from {PROBES}")


def _macro_auroc(y_true, probabilities, classes) -> float:
    try:
        if len(classes) == 2:
            return float(roc_auc_score(y_true, probabilities[:, 1]))
        return float(
            roc_auc_score(
                y_true,
                probabilities,
                labels=classes,
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        return float("nan")


def evaluate(task: Task, probe, settings=None) -> ProbeResult:
    """Fit one probe and score its predictions on the fixed test split."""
    settings = settings or {}
    model, details = fit(task, probe, settings)

    features = np.asarray(task.features, dtype=np.float32)
    labels = np.asarray(task.labels)
    test = np.asarray(task.test, dtype=int)
    if len(test) == 0 or np.any(test < 0) or np.any(test >= len(features)):
        raise ValueError("test indices are empty or out of bounds")

    probabilities = model.predict_proba(features[test])
    classes = np.asarray(model.classes_)
    predicted = classes[np.argmax(probabilities, axis=1)]
    y_true = labels[test]
    validation = 0 if task.validation is None else len(task.validation)
    reference = len(task.train) + validation

    metrics = {
        "probe": probe,
        "n_features": features.shape[1],
        "n_train": len(task.train),
        "n_validation": validation,
        "n_reference": reference,
        "n_test": len(test),
        "n_classes": len(classes),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "macro_f1": float(
            f1_score(y_true, predicted, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, predicted, average="weighted", zero_division=0)
        ),
        "macro_auroc": _macro_auroc(y_true, probabilities, classes),
        **details,
    }
    return ProbeResult(
        metrics=metrics,
        ids=np.asarray(task.ids)[test],
        y_true=np.asarray(y_true),
        y_pred=np.asarray(predicted),
        probabilities=np.asarray(probabilities, dtype=np.float32),
        classes=classes,
    )


def save_result(output_dir, result: ProbeResult) -> tuple[Path, Path]:
    """Write one metrics row and its aligned test predictions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    predictions_path = output_dir / "predictions.npz"

    pd.DataFrame([result.metrics]).to_csv(metrics_path, index=False)
    np.savez(
        predictions_path,
        ids=np.asarray(result.ids).astype(str),
        y_true=np.asarray(result.y_true).astype(str),
        y_pred=np.asarray(result.y_pred).astype(str),
        probabilities=result.probabilities,
        classes=np.asarray(result.classes).astype(str),
    )
    return metrics_path, predictions_path
