"""Shared out-of-distribution split and feature preparation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from terra_multi.evaluation.classification import evaluate_classification
from terra_multi.evaluation.linear_probe import reduce_features

IMMUNE_CLASSES = (
    "B cells",
    "Macrophages/Monocytes",
    "Dendritic cells",
    "T cells",
    "Granulocytes",
    "NK cells",
    "Plasma cells",
)


def validate_official_split(table, split_path, *, require_full_coverage=True):
    """Require exact official split agreement, and full coverage unless a dataset
    carries label-free samples (e.g. background-only tiles) that yield no cells."""
    official = pd.read_csv(split_path).rename(
        columns={"sample_name": "sample_id", "train_test_val_split": "official_split"}
    )
    official["sample_id"] = official["sample_id"].astype(str)
    official["official_split"] = official["official_split"].replace({"valid": "val"})
    if official["sample_id"].duplicated().any():
        raise ValueError("official split contains duplicate samples")

    observed = table[["sample_id", "split"]].drop_duplicates().copy()
    observed["sample_id"] = observed["sample_id"].astype(str)
    observed["split"] = observed["split"].replace({"valid": "val"})
    if observed["sample_id"].duplicated().any():
        raise ValueError("a sample has multiple observed splits")
    observed_ids, official_ids = set(observed["sample_id"]), set(official["sample_id"])
    if observed_ids - official_ids:
        raise ValueError("features contain samples absent from the official split")
    if require_full_coverage and observed_ids != official_ids:
        raise ValueError("features do not cover every official sample")

    merged = official.merge(observed, on="sample_id", validate="one_to_one")
    if not (merged["official_split"] == merged["split"]).all():
        raise ValueError("metadata disagrees with the official split")


@dataclass(frozen=True)
class OodTask:
    features: np.ndarray
    labels: np.ndarray
    row_ids: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    pca_explained_variance: float | None


def prepare_ood_task(features, table, task, settings, *, pca=False, split=None):
    """Select one official task and prepare its features once for all evaluators."""
    if task not in {"all", "immune"}:
        raise ValueError(f"unknown task {task!r}")
    split = (
        table["split"].replace({"valid": "val"}).to_numpy()
        if split is None
        else np.asarray(split)
    )
    selected = np.ones(len(table), dtype=bool)
    if task == "immune":
        selected = table["label"].isin(IMMUNE_CLASSES).to_numpy()

    train = np.flatnonzero(selected & (split == "train"))
    val = np.flatnonzero(selected & (split == "val"))
    test = np.flatnonzero(selected & (split == "test"))
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError(f"{task}: an official split is empty")

    features = np.asarray(features, dtype=np.float32)
    explained = None
    if pca and settings.get("uni_pca_components", 256):
        features, explained = reduce_features(
            features,
            train,
            settings.get("uni_pca_components", 256),
            settings.get("seed", 0),
        )
    return OodTask(
        features=features,
        labels=table["label"].to_numpy(),
        row_ids=table["row_id"].to_numpy(),
        train_indices=train,
        val_indices=val,
        test_indices=test,
        pca_explained_variance=explained,
    )


def evaluate_ood_task(data, method, settings):
    """Evaluate one prepared task with a shared classifier."""
    return evaluate_classification(
        data.features,
        data.labels,
        data.train_indices,
        data.test_indices,
        method,
        settings,
        val_indices=data.val_indices,
        row_ids=data.row_ids,
    )
