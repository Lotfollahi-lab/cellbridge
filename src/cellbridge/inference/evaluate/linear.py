"""Standardized multinomial logistic-regression probe."""

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cellbridge.inference.evaluate import Task


@dataclass(frozen=True)
class LinearProbe:
    model: object
    C: float
    selection_macro_f1: float


def _model(C, max_iter):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=float(C), max_iter=int(max_iter)),
    )


def _inputs(task):
    features = np.asarray(task.features, dtype=np.float32)
    labels = np.asarray(task.labels)
    indices = np.asarray(task.train, dtype=int)
    if features.ndim != 2 or len(features) != len(labels):
        raise ValueError("features and labels must be row-aligned")
    if not np.isfinite(features).all():
        raise ValueError("features contain non-finite values")
    if len(indices) == 0 or np.any(indices < 0) or np.any(indices >= len(features)):
        raise ValueError("training indices are empty or out of bounds")
    if len(np.unique(labels[indices])) < 2:
        raise ValueError("linear probing requires at least two training classes")
    return features, labels, indices


def fit(
    task: Task,
    *,
    Cs=(0.1, 1.0, 10.0),
    max_iter: int = 2000,
    seed: int = 0,
) -> LinearProbe:
    """Select C without test data, then fit on all available reference rows."""
    features, labels, train = _inputs(task)
    choices = [float(value) for value in Cs]
    if not choices or any(value <= 0 for value in choices):
        raise ValueError("C values must be positive")

    best_value, best_score = None, -np.inf
    if task.validation is not None:
        validation = np.asarray(task.validation, dtype=int)
        if (
            len(validation) == 0
            or np.any(validation < 0)
            or np.any(validation >= len(features))
        ):
            raise ValueError("validation indices are empty or out of bounds")
        if len(np.intersect1d(train, validation)):
            raise ValueError("training and validation indices overlap")
        for value in choices:
            model = _model(value, max_iter).fit(features[train], labels[train])
            score = f1_score(
                labels[validation],
                model.predict(features[validation]),
                average="macro",
                zero_division=0,
            )
            if score > best_score:
                best_value, best_score = value, float(score)
        reference = np.concatenate((train, validation))
    else:
        class_counts = np.unique(labels[train], return_counts=True)[1]
        folds = min(3, int(class_counts.min()))
        if folds < 2:
            raise ValueError("cross-validation requires two examples per class")
        cv = StratifiedKFold(folds, shuffle=True, random_state=int(seed))
        for value in choices:
            scores = cross_val_score(
                _model(value, max_iter),
                features[train],
                labels[train],
                cv=cv,
                scoring="f1_macro",
                n_jobs=-1,
            )
            score = float(scores.mean())
            if score > best_score:
                best_value, best_score = value, score
        reference = train

    model = _model(best_value, max_iter).fit(features[reference], labels[reference])
    return LinearProbe(model, best_value, best_score)
