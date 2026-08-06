"""JEPA contextual-feature contract on the shared artifact storage."""

from pathlib import Path

import numpy as np

from cellbridge.evaluation.artifacts import (
    artifact_row_count,
    file_identity,
    stable_signature,
    write_array_artifact,
)

FORMAT_VERSION = "cellbridge.jepa.contextual/1"


def checkpoint_digest(path):
    return file_identity(path, checksum=True)["sha256"]


def feature_signature(checkpoint_digest, sources, settings):
    return stable_signature(
        {
            "format": FORMAT_VERSION,
            "checkpoint": checkpoint_digest,
            "sources": [file_identity(path) for path in sources],
            "sampling": settings.__dict__,
        }
    )


def current_output(path, signature):
    """Return row count when a contextual feature artifact is complete."""
    count = artifact_row_count(
        path,
        format_version=FORMAT_VERSION,
        signature=signature,
        array_dims={"jepa_emb": None},
        required_metadata=("sample_id", "n_context"),
    )
    if count is None:
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            counts = data["n_context"]
            if counts.shape != (count,) or np.any(counts < 1):
                return None
    except (OSError, ValueError, KeyError):
        return None
    return count


def write_output(path, sample_id, cell_ids, embeddings, counts, signature):
    """Validate and atomically write a contextual feature artifact."""
    counts = np.asarray(counts)
    if counts.shape != (len(cell_ids),) or np.any(counts < 1):
        raise ValueError("n_context must be positive and row-aligned")
    write_array_artifact(
        Path(path),
        format_version=FORMAT_VERSION,
        signature=signature,
        row_ids=cell_ids,
        arrays={"jepa_emb": embeddings},
        metadata={
            "sample_id": np.asarray(str(sample_id)),
            "n_context": counts.astype(np.int16),
        },
    )
