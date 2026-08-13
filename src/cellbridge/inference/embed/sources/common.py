"""Shared metadata helpers for OOD dataset sources."""

import numpy as np


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
