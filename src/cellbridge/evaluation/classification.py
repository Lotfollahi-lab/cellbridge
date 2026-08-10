"""Shared classification protocol for linear and kNN evaluation."""

import os
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

from cellbridge.evaluation.artifacts import scalar_string, write_table
from cellbridge.evaluation.linear_probe import (
    fit_linear_classifier,
    fit_validated_linear_classifier,
)
from cellbridge.evaluation.zero_shot import fit_knn_classifier

METHODS = ("linear", "knn")


@dataclass(frozen=True)
class ClassificationResult:
    metrics: dict
    predictions: dict


def resolve_methods(settings, requested=None):
    """Return validated methods from an override or a config mapping."""
    methods = requested if requested is not None else settings.get("methods", ["linear"])
    if isinstance(methods, str):
        methods = methods.split(",")
    methods = [method.strip() for method in methods]
    if not methods or any(not method for method in methods):
        raise ValueError("evaluation methods must not be empty")
    if len(set(methods)) != len(methods):
        raise ValueError(f"duplicate evaluation methods: {methods}")
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"unknown evaluation method(s): {unknown}")
    return methods


def method_directory(path, method):
    """Keep the established linear path and namespace additional methods."""
    path = Path(path)
    return path if method == "linear" else path / method


def cached_probe_metrics(metrics_path, signature):
    """Return a finished probe's metrics when it and its predictions match."""
    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        return None
    try:
        table = pd.read_csv(metrics_path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None
    if len(table) != 1 or "signature" not in table:
        return None
    if str(table["signature"].iloc[0]) != str(signature):
        return None
    predictions_path = metrics_path.with_name("predictions.npz")
    try:
        with np.load(predictions_path, allow_pickle=False) as predictions:
            if (
                "signature" not in predictions
                or scalar_string(predictions["signature"]) != str(signature)
            ):
                return None
    except (OSError, ValueError, KeyError):
        return None
    return table.iloc[0].to_dict()


def write_probe_result(result_dir, metrics, predictions, signature):
    """Atomically write one probe's metrics and predictions under its signature."""
    result_dir = Path(result_dir)
    metrics = {**metrics, "signature": str(signature)}
    write_table(result_dir / "metrics.csv", [metrics])
    predictions_path = result_dir / "predictions.npz"
    temporary = predictions_path.with_name(f".{predictions_path.name}.{uuid4().hex}.tmp.npz")
    try:
        np.savez(temporary, signature=np.asarray(str(signature)), **predictions)
        os.replace(temporary, predictions_path)
    finally:
        temporary.unlink(missing_ok=True)
    return metrics


def select_knn_reference(labels, reference, sampling="all", seed=0):
    """Return all references or a reproducible equal-count sample per class."""
    reference = np.asarray(reference, dtype=int)
    if sampling == "all":
        return reference
    if sampling != "balanced":
        raise ValueError(
            "knn_reference_sampling must be either 'all' or 'balanced'"
        )

    reference_labels = np.asarray(labels)[reference]
    classes, counts = np.unique(reference_labels, return_counts=True)
    if not len(classes):
        raise ValueError("kNN reference set is empty")
    per_class = int(counts.min())
    rng = np.random.default_rng(int(seed))
    selected = []
    for label in classes:
        candidates = reference[reference_labels == label]
        selected.extend(
            rng.choice(candidates, size=per_class, replace=False).tolist()
        )
    return np.asarray(sorted(selected), dtype=int)


def _fit(features, labels, train, val, method, settings):
    if method == "linear" and val is not None:
        class_weight = settings.get("class_weight", "balanced")
        model, best_C, val_score = fit_validated_linear_classifier(
            features,
            labels,
            train,
            val,
            settings.get("Cs", [0.1, 1, 10]),
            class_weight,
            settings.get("max_iter", 2000),
        )
        return model, {"best_C": best_C, "val_macro_f1": val_score}
    if method == "linear":
        class_weight = settings.get("class_weight", "balanced")
        model = fit_linear_classifier(
            features[train],
            labels[train],
            settings.get("Cs", [0.1, 1, 10]),
            class_weight,
            settings.get("max_iter", 2000),
        )
        return model, {}
    if method == "knn":
        reference = train if val is None else np.concatenate([train, val])
        reference_sampling = settings.get("knn_reference_sampling", "all")
        reference = select_knn_reference(
            labels,
            reference,
            reference_sampling,
            settings.get("seed", 0),
        )
        model = fit_knn_classifier(
            features[reference],
            labels[reference],
            settings.get("knn_neighbors", 20),
            settings.get("knn_metric", "cosine"),
            settings.get("knn_weights", "uniform"),
        )
        return model, {
            "n_neighbors": int(settings.get("knn_neighbors", 20)),
            "neighbor_metric": settings.get("knn_metric", "cosine"),
            "neighbor_weights": settings.get("knn_weights", "uniform"),
            "reference_sampling": reference_sampling,
            "n_reference": int(len(reference)),
        }
    raise ValueError(f"unknown evaluation method {method!r}; choose from {METHODS}")


def _macro_auroc(y_true, probabilities, classes):
    try:
        if len(classes) == 2:
            return float(roc_auc_score(y_true, probabilities[:, 1]))
        return float(
            roc_auc_score(
                y_true,
                probabilities,
                multi_class="ovr",
                average="macro",
                labels=classes,
            )
        )
    except ValueError:
        return float("nan")


def evaluate_classification(
    features,
    labels,
    train_indices,
    test_indices,
    method,
    settings,
    *,
    val_indices=None,
    row_ids=None,
):
    """Fit one method on fixed splits and return common metrics and predictions."""
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels)
    train = np.asarray(train_indices, dtype=int)
    test = np.asarray(test_indices, dtype=int)
    val = None if val_indices is None else np.asarray(val_indices, dtype=int)

    model, selection = _fit(features, labels, train, val, method, settings)
    classes = np.asarray(model.classes_)
    probabilities = model.predict_proba(features[test])
    if method == "knn":
        # KNeighborsClassifier.predict repeats the full neighbour search.
        predicted = classes[np.argmax(probabilities, axis=1)]
    else:
        predicted = model.predict(features[test])
    y_true = labels[test]

    metrics = {
        "method": method,
        "n_features": int(features.shape[1]),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_classes": int(len(classes)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "macro_f1": float(
            f1_score(y_true, predicted, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, predicted, average="weighted", zero_division=0)
        ),
        "macro_auroc": _macro_auroc(y_true, probabilities, classes),
        **selection,
    }
    if val is not None:
        metrics["n_val"] = int(len(val))

    predictions = {
        "y_true": y_true,
        "y_pred": predicted,
        "y_proba": probabilities,
        "classes": classes,
    }
    if row_ids is not None:
        predictions["row_ids_test"] = np.asarray(row_ids)[test]
    return ClassificationResult(metrics, predictions)
