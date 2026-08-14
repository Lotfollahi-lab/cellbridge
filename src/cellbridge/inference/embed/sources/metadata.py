"""Canonical metadata loading for OOD dataset sources."""

from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = frozenset({"train", "val", "test"})


def cell_uids(metadata) -> np.ndarray:
    """Return validated globally unique cell identifiers."""
    if "cell_uid" in metadata:
        values = metadata["cell_uid"].astype(str).to_numpy()
    else:
        required = {"sample_id", "tile_id", "cell_id"}
        if not required <= set(metadata):
            raise ValueError(f"metadata requires cell_uid or {sorted(required)}")
        values = (
            metadata["sample_id"].astype(str)
            + "/"
            + metadata["tile_id"].astype(str)
            + "/"
            + metadata["cell_id"].astype(str)
        ).to_numpy()
    if len(np.unique(values)) != len(values):
        raise ValueError("metadata contains duplicate cell UIDs")
    return values


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


def _attach_split(
    metadata,
    metadata_path,
    split_path,
    require_all_samples,
    recorded_columns=("split",),
):
    official = _official_split(split_path)
    split_by_sample = dict(
        official[["sample_name", "train_test_val_split"]].itertuples(
            index=False, name=None
        )
    )
    sample_ids = metadata["sample_id"].astype(str)
    observed, expected = set(sample_ids), set(split_by_sample)
    metadata_only, split_only = observed - expected, expected - observed
    if metadata_only or (require_all_samples and split_only):
        raise ValueError(
            "metadata and split contain different samples: "
            f"metadata_only={sorted(metadata_only)[:5]}, "
            f"split_only={sorted(split_only)[:5]}"
        )

    resolved = sample_ids.map(split_by_sample)
    for column in recorded_columns:
        if column in metadata:
            recorded = metadata[column].replace({"valid": "val"}).astype(str)
            if not recorded.equals(resolved):
                raise ValueError(f"{metadata_path} disagrees with the explicit split")
    metadata["split"] = resolved
    return metadata


def _ensure_row_ids(metadata, path):
    if "row_id" in metadata:
        if metadata["row_id"].isna().any() or metadata["row_id"].duplicated().any():
            raise ValueError(f"{path} contains invalid row IDs")
    else:
        metadata.insert(0, "row_id", np.arange(len(metadata), dtype=np.int64))


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

    _ensure_row_ids(metadata, metadata_path)
    if metadata["cell_uid"].astype(str).duplicated().any():
        raise ValueError(f"{metadata_path} contains duplicate cell UIDs")

    _attach_split(metadata, metadata_path, split_path, require_all_samples)

    excluded = set(map(str, exclude_labels))
    if excluded:
        metadata = metadata.loc[~metadata["label"].astype(str).isin(excluded)].copy()
    if metadata.empty:
        raise ValueError(f"{metadata_path} has no evaluable cells")
    return metadata


def load_tiled_metadata(
    metadata_path,
    split_path,
    *,
    require_all_samples: bool = True,
) -> pd.DataFrame:
    """Load tile-local cells and attach their explicit sample-level split."""
    metadata_path = Path(metadata_path)
    metadata = pd.read_csv(metadata_path)
    required = {"sample_id", "tile_id", "cell_id", "label"}
    if not required <= set(metadata):
        raise ValueError(
            f"{metadata_path} is missing {sorted(required - set(metadata))}"
        )
    if metadata[list(required)].isna().any().any():
        raise ValueError(f"{metadata_path} contains null cell metadata")
    if metadata.duplicated(["sample_id", "tile_id", "cell_id"]).any():
        raise ValueError(f"{metadata_path} contains duplicate cell IDs")

    _ensure_row_ids(metadata, metadata_path)
    return _attach_split(
        metadata,
        metadata_path,
        split_path,
        require_all_samples,
        recorded_columns=("split", "split_id"),
    )
