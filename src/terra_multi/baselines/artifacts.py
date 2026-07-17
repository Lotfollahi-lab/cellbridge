"""Signed, model-agnostic embedding artifacts used by every baseline.

One artifact owns one sample and keeps cell identifiers row-aligned with a
single ``embeddings`` matrix.  Signatures cover the model, source data, and
preprocessing settings so an interrupted or stale extraction cannot be reused
silently.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

FORMAT_VERSION = "terra_multi.baseline.features/1"
CORE_METADATA = (
    "baseline_name",
    "dataset_name",
    "artifact_sample_id",
    "model_signature",
    "provenance_json",
)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def stable_json(payload) -> str:
    """Serialize JSON-compatible provenance deterministically."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def stable_signature(payload) -> str:
    """Return a deterministic SHA-256 signature for provenance metadata."""
    return hashlib.sha256(stable_json(payload).encode()).hexdigest()


def file_identity(path, *, checksum=False) -> dict:
    """Describe a source file, optionally including its content checksum."""
    unresolved = Path(path).expanduser()
    resolved = unresolved.resolve()
    stat = resolved.stat()
    identity = {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if unresolved != resolved:
        identity["requested_path"] = str(unresolved)
    if checksum:
        digest = hashlib.sha256()
        with open(resolved, "rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        identity["sha256"] = digest.hexdigest()
    return identity


def path_identity(path) -> list[dict] | dict:
    """Describe a file or the files immediately inside one directory."""
    path = Path(path)
    if path.is_file():
        return file_identity(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    return [file_identity(item) for item in sorted(path.iterdir()) if item.is_file()]


def git_identity(repository) -> dict:
    """Record the commit and a content hash for every uncommitted change."""
    repository = Path(repository).resolve()
    if not (repository / ".git").exists():
        raise ValueError(f"not a git checkout: {repository}")

    def run_bytes(*arguments):
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return completed.stdout

    def run(*arguments):
        return run_bytes(*arguments).decode().strip()

    status = run("status", "--porcelain")
    digest = hashlib.sha256()
    digest.update(run_bytes("diff", "--binary", "HEAD", "--"))
    untracked = run("ls-files", "--others", "--exclude-standard").splitlines()
    for relative in sorted(untracked):
        path = repository / relative
        digest.update(relative.encode())
        digest.update(b"\0")
        if path.is_file():
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
    return {
        "path": str(repository),
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status),
        "working_tree_signature": digest.hexdigest(),
    }


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


def scalar_string(value) -> str:
    value = np.asarray(value)
    if value.size != 1:
        raise ValueError("expected a scalar string")
    return str(value.reshape(()).item())


def _finite_in_chunks(values, chunk_size=65_536) -> bool:
    return all(
        np.isfinite(values[start : start + chunk_size]).all()
        for start in range(0, len(values), chunk_size)
    )


def _metadata_array(name, values, n_cells):
    values = np.asarray(values)
    if values.ndim and len(values) != n_cells:
        raise ValueError(f"{name}: {len(values)} rows for {n_cells} cells")
    if values.dtype.kind == "O":
        values = values.astype(str)
    return values


def write_feature_file(
    path,
    cell_ids,
    embeddings,
    signature,
    *,
    metadata,
    expected_dim=None,
):
    """Validate and atomically write one aligned embedding artifact."""
    path = Path(path)
    cell_ids = np.asarray(cell_ids).astype(str)
    embeddings = np.asanyarray(embeddings)
    if cell_ids.ndim != 1:
        raise ValueError(f"cell_ids must be one-dimensional, got {cell_ids.shape}")
    if len(np.unique(cell_ids)) != len(cell_ids):
        raise ValueError(f"{path}: duplicate cell_ids")
    if embeddings.ndim != 2 or len(embeddings) != len(cell_ids):
        raise ValueError(
            f"embeddings: expected ({len(cell_ids)}, D), got {embeddings.shape}"
        )
    if expected_dim is not None and embeddings.shape[1] != int(expected_dim):
        raise ValueError(
            f"embeddings: expected {expected_dim} columns, got {embeddings.shape[1]}"
        )
    if not _finite_in_chunks(embeddings):
        raise ValueError("embeddings contain non-finite values")

    overlap = {"format_version", "signature", "cell_ids", "embeddings"} & set(
        metadata
    )
    if overlap:
        raise ValueError(f"metadata uses reserved keys: {sorted(overlap)}")
    missing = set(CORE_METADATA) - set(metadata)
    if missing:
        raise ValueError(f"metadata is missing core keys: {sorted(missing)}")

    arrays = {
        "format_version": np.asarray(FORMAT_VERSION),
        "signature": np.asarray(str(signature)),
        "cell_ids": cell_ids,
        "embeddings": embeddings.astype(np.float16, copy=False),
    }
    arrays.update(
        {
            name: _metadata_array(name, values, len(cell_ids))
            for name, values in metadata.items()
        }
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp.npz")
    try:
        np.savez(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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
    path = Path(path)
    if not path.exists():
        return False
    required = set(CORE_METADATA) | set(required_metadata)
    try:
        with np.load(path, allow_pickle=False) as data:
            if not {
                "format_version",
                "signature",
                "cell_ids",
                "embeddings",
                *required,
            }.issubset(data.files):
                return False
            if scalar_string(data["format_version"]) != FORMAT_VERSION:
                return False
            if scalar_string(data["signature"]) != str(signature):
                return False
            cell_ids = data["cell_ids"].astype(str)
            embeddings = data["embeddings"]
            if embeddings.ndim != 2 or len(embeddings) != len(cell_ids):
                return False
            if expected_cells is not None and len(cell_ids) != int(expected_cells):
                return False
            if expected_dim is not None and embeddings.shape[1] != int(expected_dim):
                return False
            if not _finite_in_chunks(embeddings):
                return False
            if len(np.unique(cell_ids)) != len(cell_ids):
                return False
            if any(not scalar_string(data[name]) for name in CORE_METADATA[:-1]):
                return False
            json.loads(scalar_string(data["provenance_json"]))
            if expected_cell_ids is not None and not np.array_equal(
                cell_ids, np.asarray(expected_cell_ids).astype(str)
            ):
                return False
            return True
    except (OSError, ValueError, KeyError):
        return False


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
    with np.load(path, allow_pickle=False) as data:
        required = {
            "format_version",
            "signature",
            "cell_ids",
            "embeddings",
            *CORE_METADATA,
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"{path}: missing keys {sorted(missing)}")
        if scalar_string(data["format_version"]) != FORMAT_VERSION:
            raise ValueError(f"{path}: unsupported feature format")
        embeddings = data["embeddings"]
        cell_ids = data["cell_ids"].astype(str)
        if embeddings.ndim != 2 or len(embeddings) != len(cell_ids):
            raise ValueError(f"{path}: embeddings and cell_ids are misaligned")
        if expected_dim is not None and embeddings.shape[1] != int(expected_dim):
            raise ValueError(
                f"{path}: expected {expected_dim} features, got {embeddings.shape[1]}"
            )
        if len(np.unique(cell_ids)) != len(cell_ids):
            raise ValueError(f"{path}: duplicate cell_ids")
        if not _finite_in_chunks(embeddings):
            raise ValueError(f"{path}: embeddings contain non-finite values")
        json.loads(scalar_string(data["provenance_json"]))
        return ArtifactSummary(
            path=str(path.resolve()),
            signature=scalar_string(data["signature"]),
            sample_id=scalar_string(data["artifact_sample_id"]),
            dataset_name=scalar_string(data["dataset_name"]),
            baseline_name=scalar_string(data["baseline_name"]),
            model_signature=scalar_string(data["model_signature"]),
            n_cells=len(cell_ids),
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
    with np.load(path, allow_pickle=False) as data:
        signature = scalar_string(data["signature"])
        if expected_signature is not None and signature != str(expected_signature):
            raise ValueError(f"{path}: signature disagrees with manifest")
        features = data["embeddings"].astype(np.float32)
        cell_ids = data["cell_ids"].astype(str)
        if features.ndim != 2 or len(features) != len(cell_ids):
            raise ValueError(f"{path}: embeddings and cell_ids are misaligned")
        if expected_dim is not None and features.shape[1] != int(expected_dim):
            raise ValueError(
                f"{path}: expected {expected_dim} features, got {features.shape[1]}"
            )
        if not np.isfinite(features).all():
            raise ValueError(f"{path}: embeddings contain non-finite values")
        missing = set(metadata_keys) - set(data.files)
        if missing:
            raise ValueError(f"{path}: missing metadata {sorted(missing)}")
        metadata = {name: np.asarray(data[name]) for name in metadata_keys}
    return features, cell_ids, metadata


def write_table(path, rows):
    """Atomically write a CSV table."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        pd.DataFrame(rows).to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, payload):
    """Atomically write human-readable provenance JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
