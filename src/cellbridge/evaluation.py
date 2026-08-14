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


def _macro_auroc(y_true, probabilities, classes) -> float:
    if len(np.unique(y_true)) != len(classes):
        return float("nan")
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


def evaluate_predictions(
    *,
    ids,
    y_true,
    probabilities,
    classes,
    metrics=None,
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
        "macro_auroc": _macro_auroc(y_true, probabilities, classes),
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
