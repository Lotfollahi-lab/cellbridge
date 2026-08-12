"""Downstream evaluation of extracted cell embeddings."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Task:
    """One row-aligned classification task with fixed data splits."""

    features: np.ndarray
    labels: np.ndarray
    ids: np.ndarray
    train: np.ndarray
    validation: np.ndarray | None
    test: np.ndarray
