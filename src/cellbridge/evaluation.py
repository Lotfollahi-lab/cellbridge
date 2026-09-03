"""Shared classification scoring and prediction artifacts."""

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


@dataclass(frozen=True)
class ClassificationResult:
    """Metrics and aligned predictions for one classification result."""

    metrics: dict
    ids: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    probabilities: np.ndarray
    classes: np.ndarray


def scored_classes(y_true, classes, min_test_support) -> np.ndarray:
    y_true = np.asarray(y_true).astype(str)
    if min_test_support is None:
        present = len(np.unique(y_true)) == len(classes)
        return np.full(len(classes), present, dtype=bool)
    counts = pd.Series(y_true).value_counts()
    support = np.array([int(counts.get(str(name), 0)) for name in classes])
    return support >= max(int(min_test_support), 1)


def macro_auroc(
    y_true, probabilities, classes, *, min_test_support=None, sample_weight=None
):
    scored = scored_classes(y_true, classes, min_test_support)
    y_true = np.asarray(y_true).astype(str)
    values = []
    for column in np.flatnonzero(scored):
        positive = y_true == str(classes[column])
        if sample_weight is None:
            if not positive.any() or positive.all():
                continue
        elif sample_weight[positive].sum() <= 0 or sample_weight[~positive].sum() <= 0:
            continue
        try:
            values.append(
                roc_auc_score(
                    positive, probabilities[:, column], sample_weight=sample_weight
                )
            )
        except ValueError:
            continue
    if not values:
        return float("nan"), 0
    return float(np.mean(values)), len(values)


def evaluate_predictions(
    *,
    ids,
    y_true,
    probabilities,
    classes,
    metrics=None,
    min_test_support=None,
) -> ClassificationResult:
    """Score class probabilities and retain their aligned predictions."""
    ids = np.asarray(ids)
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    classes = np.asarray(classes)
    if probabilities.shape != (len(y_true), len(classes)):
        raise ValueError(
            "probabilities must have shape "
            f"({len(y_true)}, {len(classes)}), got {probabilities.shape}"
        )
    if len(ids) != len(y_true):
        raise ValueError("ids, labels, and probabilities must be row-aligned")

    predicted = classes[np.argmax(probabilities, axis=1)]
    scores = {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "macro_f1": float(
            f1_score(y_true, predicted, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, predicted, average="weighted", zero_division=0)
        ),
        "macro_auroc": macro_auroc(
            y_true, probabilities, classes, min_test_support=min_test_support
        )[0],
    }
    return ClassificationResult(
        metrics={**(metrics or {}), **scores},
        ids=ids,
        y_true=y_true,
        y_pred=np.asarray(predicted),
        probabilities=probabilities,
        classes=classes,
    )


def save_result(output_dir, result: ClassificationResult) -> tuple[Path, Path]:
    """Write one metrics row and its aligned predictions."""
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
