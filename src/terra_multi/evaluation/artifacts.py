"""Baseline feature-artifact contract built on the shared artifact storage."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from terra_multi.evaluation.provenance import (
    file_identity,
    git_identity,
    path_identity,
    stable_json,
    stable_signature,
)
from terra_multi.evaluation.storage import (
    artifact_row_count,
    load_array_artifact,
    scalar_string,
    write_array_artifact,
    write_json,
    write_table,
)

FORMAT_VERSION = "terra_multi.baseline.features/1"
CORE_METADATA = (
    "baseline_name",
    "dataset_name",
    "artifact_sample_id",
    "model_signature",
    "provenance_json",
)


def feature_signature(dataset, model_identity, source_identity, settings) -> str:
    """Sign the complete contract that determines one embedding artifact."""
    return stable_signature(
        {
            "format_version": FORMAT_VERSION,
            "dataset": dataset,
            "model": model_identity,
            "source": source_identity,
            "settings": settings,
        }
    )


def write_feature_file(
    path,
    cell_ids,
    embeddings,
    signature,
    *,
    metadata,
    expected_dim=None,
):
    """Validate and atomically write one baseline embedding artifact."""
    missing = set(CORE_METADATA) - set(metadata)
    if missing:
        raise ValueError(f"metadata is missing core keys: {sorted(missing)}")
    embeddings = np.asanyarray(embeddings)
    if expected_dim is not None and (
        embeddings.ndim != 2 or embeddings.shape[1] != int(expected_dim)
    ):
        width = embeddings.shape[1] if embeddings.ndim == 2 else embeddings.shape
        raise ValueError(f"embeddings: expected {expected_dim} columns, got {width}")
    write_array_artifact(
        path,
        format_version=FORMAT_VERSION,
        signature=signature,
        row_ids=cell_ids,
        arrays={"embeddings": embeddings},
        metadata=metadata,
    )


def feature_file_is_current(
    path,
    signature,
    *,
    expected_cells=None,
    expected_dim=None,
    expected_cell_ids=None,
    required_metadata=(),
) -> bool:
    """Return true only for a complete artifact matching this signature."""
    required = set(CORE_METADATA) | set(required_metadata)
    count = artifact_row_count(
        path,
        format_version=FORMAT_VERSION,
        signature=signature,
        array_dims={"embeddings": expected_dim},
        expected_cells=expected_cells,
        expected_row_ids=expected_cell_ids,
        required_metadata=required,
    )
    if count is None:
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            if any(not scalar_string(data[name]) for name in CORE_METADATA[:-1]):
                return False
            json.loads(scalar_string(data["provenance_json"]))
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return True


@dataclass(frozen=True)
class ArtifactSummary:
    path: str
    signature: str
    sample_id: str
    dataset_name: str
    baseline_name: str
    model_signature: str
    n_cells: int
    n_features: int


def summarize_feature_file(path, *, expected_dim=None) -> ArtifactSummary:
    """Read and validate the identifying metadata for one artifact."""
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as data:
            signature = scalar_string(data["signature"])
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"{path}: invalid feature artifact") from error
    if not feature_file_is_current(path, signature, expected_dim=expected_dim):
        raise ValueError(f"{path}: invalid or unsupported feature artifact")
    with np.load(path, allow_pickle=False) as data:
        embeddings = data["embeddings"]
        return ArtifactSummary(
            path=str(path.resolve()),
            signature=signature,
            sample_id=scalar_string(data["artifact_sample_id"]),
            dataset_name=scalar_string(data["dataset_name"]),
            baseline_name=scalar_string(data["baseline_name"]),
            model_signature=scalar_string(data["model_signature"]),
            n_cells=len(data["cell_ids"]),
            n_features=int(embeddings.shape[1]),
        )


def load_feature_file(
    path,
    *,
    expected_signature=None,
    expected_dim=None,
    metadata_keys=(),
):
    """Load one artifact for probing and verify manifest agreement."""
    path = Path(path)
    if expected_signature is None:
        try:
            with np.load(path, allow_pickle=False) as data:
                expected_signature = scalar_string(data["signature"])
        except (OSError, ValueError, KeyError) as error:
            raise ValueError(f"{path}: invalid feature artifact") from error
    arrays, cell_ids, metadata = load_array_artifact(
        path,
        format_version=FORMAT_VERSION,
        signature=expected_signature,
        array_dims={"embeddings": expected_dim},
        metadata_keys=metadata_keys,
    )
    return arrays["embeddings"], cell_ids, metadata
