"""Validated, atomic storage for row-aligned feature artifacts and tables."""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from cellbridge.evaluation.provenance import json_default

_RESERVED = {"format_version", "signature"}


def scalar_string(value) -> str:
    value = np.asarray(value)
    if value.size != 1:
        raise ValueError("expected a scalar string")
    return str(value.reshape(()).item())


def _finite(values, chunk_size=65_536) -> bool:
    return all(
        np.isfinite(values[start : start + chunk_size]).all()
        for start in range(0, len(values), chunk_size)
    )


def _metadata_array(name, values, rows):
    values = np.asarray(values)
    if values.ndim and len(values) != rows:
        raise ValueError(f"{name}: {len(values)} rows for {rows} cells")
    return values.astype(str) if values.dtype.kind == "O" else values


def write_array_artifact(
    path,
    *,
    format_version,
    signature,
    row_ids,
    arrays,
    metadata=None,
    row_id_key="cell_ids",
    storage_dtype=np.float16,
):
    """Validate and atomically write aligned 2D feature arrays."""
    path = Path(path)
    row_ids = np.asarray(row_ids).astype(str)
    if row_ids.ndim != 1 or len(np.unique(row_ids)) != len(row_ids):
        raise ValueError(f"{path}: row IDs must be one-dimensional and unique")
    if not arrays:
        raise ValueError("at least one feature array is required")
    reserved = _RESERVED | {row_id_key}
    metadata = metadata or {}
    overlap = reserved & (set(arrays) | set(metadata))
    duplicate = set(arrays) & set(metadata)
    if overlap or duplicate:
        raise ValueError(f"reserved or duplicate artifact keys: {sorted(overlap | duplicate)}")

    payload = {
        "format_version": np.asarray(str(format_version)),
        "signature": np.asarray(str(signature)),
        row_id_key: row_ids,
    }
    for name, values in arrays.items():
        values = np.asanyarray(values)
        if values.ndim != 2 or len(values) != len(row_ids):
            raise ValueError(f"{name}: expected ({len(row_ids)}, D), got {values.shape}")
        if not _finite(values):
            raise ValueError(f"{name} contains non-finite values")
        payload[name] = values.astype(storage_dtype, copy=False)
    payload.update(
        {
            name: _metadata_array(name, values, len(row_ids))
            for name, values in metadata.items()
        }
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp.npz")
    try:
        np.savez(temporary, **payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def artifact_row_count(
    path,
    *,
    format_version,
    signature,
    array_dims,
    row_id_key="cell_ids",
    expected_cells=None,
    expected_row_ids=None,
    required_metadata=(),
):
    """Return row count only when an artifact satisfies its complete contract."""
    path = Path(path)
    if not path.exists():
        return None
    required = {"format_version", "signature", row_id_key, *array_dims, *required_metadata}
    try:
        with np.load(path, allow_pickle=False) as data:
            if not required.issubset(data.files):
                return None
            if scalar_string(data["format_version"]) != str(format_version):
                return None
            if scalar_string(data["signature"]) != str(signature):
                return None
            row_ids = data[row_id_key].astype(str)
            if row_ids.ndim != 1 or len(np.unique(row_ids)) != len(row_ids):
                return None
            if expected_cells is not None and len(row_ids) != int(expected_cells):
                return None
            if expected_row_ids is not None and not np.array_equal(
                row_ids, np.asarray(expected_row_ids).astype(str)
            ):
                return None
            for name in required_metadata:
                values = data[name]
                if values.ndim and len(values) != len(row_ids):
                    return None
            for name, dimension in array_dims.items():
                values = data[name]
                if values.ndim != 2 or len(values) != len(row_ids) or not _finite(values):
                    return None
                if dimension is not None and values.shape[1] != int(dimension):
                    return None
            return len(row_ids)
    except (OSError, ValueError, KeyError):
        return None


def load_array_artifact(
    path,
    *,
    format_version,
    signature,
    array_dims,
    row_id_key="cell_ids",
    metadata_keys=(),
):
    count = artifact_row_count(
        path,
        format_version=format_version,
        signature=signature,
        array_dims=array_dims,
        row_id_key=row_id_key,
        required_metadata=metadata_keys,
    )
    if count is None:
        raise ValueError(f"{path}: artifact does not match its expected contract")
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name].astype(np.float32) for name in array_dims}
        row_ids = data[row_id_key].astype(str)
        metadata = {name: np.asarray(data[name]) for name in metadata_keys}
    return arrays, row_ids, metadata


def write_table(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        pd.DataFrame(rows).to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, default=json_default) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
