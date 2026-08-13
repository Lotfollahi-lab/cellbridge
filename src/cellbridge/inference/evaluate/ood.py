"""Prepare official-split OOD cell-type tasks."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from cellbridge.inference.evaluate import Task

PCA_ARMS = frozenset({"uni", "lora_adapted"})

IMMUNE_CLASSES = {
    "B cells",
    "Macrophages/Monocytes",
    "Dendritic cells",
    "T cells",
    "Granulocytes",
    "NK cells",
    "Plasma cells",
}


def validate_split(metadata, split_path, *, require_all_samples: bool = True) -> None:
    """Check that feature metadata agrees with the dataset's official split."""
    official = pd.read_csv(split_path).rename(
        columns={"sample_name": "sample_id", "train_test_val_split": "split"}
    )
    required = {"sample_id", "split"}
    if not required <= set(official) or not required <= set(metadata):
        raise ValueError("official split and metadata require sample_id and split columns")

    official = official[list(required)].copy()
    official["sample_id"] = official["sample_id"].astype(str)
    official["split"] = official["split"].replace({"valid": "val"})
    if official["sample_id"].duplicated().any():
        raise ValueError("official split contains duplicate samples")

    observed = metadata[list(required)].drop_duplicates().copy()
    observed["sample_id"] = observed["sample_id"].astype(str)
    observed["split"] = observed["split"].replace({"valid": "val"})
    if observed["sample_id"].duplicated().any():
        raise ValueError("a sample has multiple observed splits")

    unknown = set(observed["sample_id"]) - set(official["sample_id"])
    if unknown:
        raise ValueError(f"metadata contains unknown samples: {sorted(unknown)[:5]}")
    if require_all_samples and set(observed["sample_id"]) != set(official["sample_id"]):
        raise ValueError("metadata does not cover every official sample")
    compared = observed.merge(official, on="sample_id", suffixes=("_seen", "_official"))
    if not (compared["split_seen"] == compared["split_official"]).all():
        raise ValueError("metadata disagrees with the official split")


def prepare_task(
    features,
    metadata,
    *,
    task: str = "all",
    evaluation_split: str = "test",
    arm: str | None = None,
    pca_components: int = 256,
    seed: int = 0,
) -> Task:
    """Select an OOD label task and return its official train/val/test rows."""
    required = {"row_id", "sample_id", "label", "split"}
    if not required <= set(metadata):
        raise ValueError(f"metadata is missing {sorted(required - set(metadata))}")
    if metadata[list(required)].isna().any().any():
        raise ValueError("metadata contains null task fields")
    if task not in {"all", "immune"}:
        raise ValueError("task must be 'all' or 'immune'")
    if evaluation_split not in {"val", "test"}:
        raise ValueError("evaluation_split must be 'val' or 'test'")

    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or len(features) != len(metadata):
        raise ValueError(f"expected ({len(metadata)}, D) features, got {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("features contain non-finite values")
    if metadata["row_id"].duplicated().any():
        raise ValueError("metadata contains duplicate row IDs")

    labels = metadata["label"].astype(str).to_numpy()
    splits = metadata["split"].replace({"valid": "val"}).to_numpy()
    unknown_splits = set(splits) - {"train", "val", "test"}
    if unknown_splits:
        raise ValueError(f"metadata contains unknown splits: {sorted(unknown_splits)}")
    selected = np.ones(len(metadata), dtype=bool)
    if task == "immune":
        selected = metadata["label"].isin(IMMUNE_CLASSES).to_numpy()

    train = np.flatnonzero(selected & (splits == "train"))
    validation = np.flatnonzero(selected & (splits == "val"))
    test = np.flatnonzero(selected & (splits == "test"))
    if evaluation_split == "val":
        validation, test = None, validation
    required_splits = (train, test) if validation is None else (train, validation, test)
    if any(len(indices) == 0 for indices in required_splits):
        raise ValueError(f"{task} has an empty required split")
    missing = set(labels[test]) - set(labels[train])
    if missing:
        raise ValueError(f"test classes absent from training: {sorted(missing)}")

    if arm in PCA_ARMS:
        maximum = min(len(train), features.shape[1])
        if not 1 <= pca_components <= maximum:
            raise ValueError(
                f"pca_components must be between 1 and {maximum}, got {pca_components}"
            )
        pca = PCA(
            n_components=int(pca_components),
            svd_solver="randomized",
            random_state=int(seed),
        )
        pca.fit(features[train])
        features = pca.transform(features).astype(np.float32)

    return Task(
        features,
        labels,
        metadata["row_id"].to_numpy(),
        train,
        validation,
        test,
    )
