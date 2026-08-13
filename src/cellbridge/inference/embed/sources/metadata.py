"""Canonical metadata loading for OOD dataset sources."""

from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = frozenset({"train", "val", "test"})


def _official_split(path) -> pd.DataFrame:
    path = Path(path)
    table = pd.read_csv(path)
    required = {"sample_name", "train_test_val_split"}
    if not required <= set(table):
        raise ValueError(f"{path} is missing {sorted(required - set(table))}")
    if table[list(required)].isna().any().any():
        raise ValueError(f"{path} contains null sample metadata")

    table = table[list(required)].copy()
    table["sample_name"] = table["sample_name"].astype(str)
    table["train_test_val_split"] = table["train_test_val_split"].replace(
        {"valid": "val"}
    )
    if table["sample_name"].duplicated().any():
        raise ValueError(f"{path} contains duplicate samples")
    unknown = set(table["train_test_val_split"]) - SPLITS
    if unknown:
        raise ValueError(f"{path} contains unknown splits: {sorted(unknown)}")
    return table


def load_metadata(
    metadata_path,
    split_path,
    *,
    exclude_labels=(),
    require_all_samples: bool = True,
) -> pd.DataFrame:
    """Load canonical cells and attach their explicit sample-level split."""
    metadata_path = Path(metadata_path)
    metadata = pd.read_csv(metadata_path)
    required = {"sample_id", "cell_uid", "label"}
    if not required <= set(metadata):
        raise ValueError(
            f"{metadata_path} is missing {sorted(required - set(metadata))}"
        )
    if metadata[list(required)].isna().any().any():
        raise ValueError(f"{metadata_path} contains null cell metadata")

    if "row_id" in metadata:
        if metadata["row_id"].isna().any() or metadata["row_id"].duplicated().any():
            raise ValueError(f"{metadata_path} contains invalid row IDs")
    else:
        metadata.insert(0, "row_id", np.arange(len(metadata), dtype=np.int64))
    if metadata["cell_uid"].astype(str).duplicated().any():
        raise ValueError(f"{metadata_path} contains duplicate cell UIDs")

    official = _official_split(split_path)
    split_by_sample = dict(
        official[["sample_name", "train_test_val_split"]].itertuples(
            index=False, name=None
        )
    )
    sample_ids = metadata["sample_id"].astype(str)
    observed = set(sample_ids)
    expected = set(split_by_sample)
    metadata_only = observed - expected
    split_only = expected - observed
    if metadata_only or (require_all_samples and split_only):
        raise ValueError(
            "metadata and split contain different samples: "
            f"metadata_only={sorted(metadata_only)[:5]}, "
            f"split_only={sorted(split_only)[:5]}"
        )

    resolved = sample_ids.map(split_by_sample)
    if "split" in metadata:
        recorded = metadata["split"].replace({"valid": "val"}).astype(str)
        if not recorded.equals(resolved):
            raise ValueError(f"{metadata_path} disagrees with the explicit split")
    metadata["split"] = resolved

    excluded = set(map(str, exclude_labels))
    if excluded:
        metadata = metadata.loc[~metadata["label"].astype(str).isin(excluded)].copy()
    if metadata.empty:
        raise ValueError(f"{metadata_path} has no evaluable cells")
    return metadata
