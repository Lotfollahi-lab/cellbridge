"""Deterministic grouped spatial-block cross-validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GroupKFold


@dataclass(frozen=True)
class SpatialFold:
    number: int
    train: np.ndarray
    test: np.ndarray


def spatial_block_ids(coords_um, block_size_um):
    coords = np.asarray(coords_um, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2 or not np.isfinite(coords).all():
        raise ValueError("coordinates must be a finite (N, 2) array")
    if float(block_size_um) <= 0:
        raise ValueError("block_size_um must be positive")
    origin = coords.min(axis=0)
    blocks = np.floor((coords - origin) / float(block_size_um)).astype(np.int64)
    return np.asarray([f"{x}:{y}" for x, y in blocks])


def spatial_block_folds(coords_um, n_splits, block_size_um, labels=None):
    """Make folds whose spatial blocks occur in exactly one test partition."""
    groups = spatial_block_ids(coords_um, block_size_um)
    n_splits = int(n_splits)
    if len(np.unique(groups)) < n_splits:
        raise ValueError("fewer spatial blocks than requested CV folds")
    labels = None if labels is None else np.asarray(labels)
    global_classes = None if labels is None else set(labels.tolist())
    folds = []
    for number, (train, test) in enumerate(
        GroupKFold(n_splits=n_splits).split(coords_um, groups=groups)
    ):
        train, test = np.sort(train), np.sort(test)
        if set(groups[train]) & set(groups[test]):
            raise RuntimeError("a spatial block leaked across train and test")
        if labels is not None:
            missing = sorted(global_classes - set(labels[train].tolist()))
            if missing:
                raise ValueError(
                    f"fold {number} training set misses classes: {missing}"
                )
        folds.append(SpatialFold(number, train, test))
    covered = np.concatenate([fold.test for fold in folds])
    if not np.array_equal(np.sort(covered), np.arange(len(coords_um))):
        raise RuntimeError("spatial test folds do not partition all cells")
    return folds, groups
