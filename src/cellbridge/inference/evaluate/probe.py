"""Fit embedding probes and score them with the shared evaluator."""

import numpy as np

from cellbridge.evaluation import (
    ClassificationResult,
    evaluate_predictions,
    save_result as save_result,
)
from cellbridge.inference.evaluate import Task, knn, linear

PROBES = ("linear", "knn")
ProbeResult = ClassificationResult


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
    validation = 0 if task.validation is None else len(task.validation)
    return evaluate_predictions(
        ids=np.asarray(task.ids)[test],
        y_true=labels[test],
        probabilities=probabilities,
        classes=classes,
        metrics={
            "probe": probe,
            "n_features": features.shape[1],
            "n_train": len(task.train),
            "n_validation": validation,
            "n_reference": len(task.train) + validation,
            "n_test": len(test),
            "n_classes": len(classes),
            **details,
        },
        min_test_support=settings.get("min_test_support"),
    )
