"""Uniform cosine k-nearest-neighbour probe."""

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from cellbridge.inference.evaluate import Task


@dataclass(frozen=True)
class KnnProbe:
    model: KNeighborsClassifier
    n_neighbors: int
    n_reference: int


def fit(task: Task, *, n_neighbors: int = 20) -> KnnProbe:
    """Index every train/validation embedding without class resampling."""
    features = np.asarray(task.features, dtype=np.float32)
    labels = np.asarray(task.labels)
    train = np.asarray(task.train, dtype=int)
    if features.ndim != 2 or len(features) != len(labels):
        raise ValueError("features and labels must be row-aligned")
    if not np.isfinite(features).all():
        raise ValueError("features contain non-finite values")
    if len(train) == 0:
        raise ValueError("training indices are empty")

    reference = train
    if task.validation is not None:
        validation = np.asarray(task.validation, dtype=int)
        if len(validation) == 0:
            raise ValueError("validation indices are empty")
        reference = np.concatenate((train, validation))
    if np.any(reference < 0) or np.any(reference >= len(features)):
        raise ValueError("reference indices are out of bounds")
    if len(np.unique(reference)) != len(reference):
        raise ValueError("reference indices contain duplicates or overlapping splits")
    if len(np.unique(labels[reference])) < 2:
        raise ValueError("kNN probing requires at least two reference classes")
    if not 1 <= n_neighbors <= len(reference):
        raise ValueError(f"n_neighbors must be between 1 and {len(reference)}")

    model = KNeighborsClassifier(
        n_neighbors=int(n_neighbors),
        metric="cosine",
        weights="uniform",
        algorithm="brute",
        n_jobs=-1,
    )
    model.fit(features[reference], labels[reference])
    return KnnProbe(model, int(n_neighbors), len(reference))
