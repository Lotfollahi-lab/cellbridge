"""Shared sample-split and fixed-cohort validation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd

SPLITS = ("distill_train", "distill_val", "label_test")


def _normalizer(function):
    return str if function is None else function


def load_split_table(
    path,
    *,
    allowed_splits: Iterable[str] | None = None,
    required_splits: Iterable[str] = (),
    normalize_id: Callable | None = None,
) -> pd.DataFrame:
    """Load unique sample assignments and enforce the split contract."""
    path = Path(path)
    table = pd.read_csv(path)
    required = {"sample_id", "split"}
    if not required <= set(table):
        raise ValueError(f"{path} must contain columns {sorted(required)}")
    table = table.loc[:, ["sample_id", "split"]].copy()
    if table.isna().any().any():
        raise ValueError(f"{path} contains null sample IDs or splits")
    table["sample_id"] = table["sample_id"].map(_normalizer(normalize_id))
    table["split"] = table["split"].astype(str)
    duplicates = table.loc[table["sample_id"].duplicated(keep=False), "sample_id"]
    if not duplicates.empty:
        raise ValueError(f"{path} has non-disjoint duplicate samples: {duplicates.unique()[:5].tolist()}")
    if allowed_splits is not None:
        unknown = sorted(set(table["split"]) - set(allowed_splits))
        if unknown:
            raise ValueError(f"{path} contains unknown splits: {unknown}")
    missing = [name for name in required_splits if not (table["split"] == name).any()]
    if missing:
        raise ValueError(f"{path} has empty required splits: {missing}")
    return table


def samples_in_split(path, split, *, normalize_id=None, required=True) -> list[str]:
    table = load_split_table(path, normalize_id=normalize_id)
    samples = table.loc[table["split"] == str(split), "sample_id"].tolist()
    if required and not samples:
        raise ValueError(f"{path}: split {split!r} is empty")
    return samples


def split_groups(path, names=SPLITS, *, normalize_id=None) -> dict[str, list[str]]:
    table = load_split_table(
        path,
        required_splits=names,
        normalize_id=normalize_id,
    )
    return {
        name: table.loc[table["split"] == name, "sample_id"].tolist()
        for name in names
    }


def load_cohort(path, *, normalize_id=None, expected_count=None) -> list[str]:
    """Load a unique, ordered ``sample_id`` cohort."""
    path = Path(path)
    table = pd.read_csv(path)
    if "sample_id" not in table:
        raise ValueError(f"{path} must contain a sample_id column")
    if table["sample_id"].isna().any():
        raise ValueError(f"{path} contains null sample IDs")
    samples = table["sample_id"].map(_normalizer(normalize_id)).tolist()
    if not samples:
        raise ValueError(f"{path} is empty")
    if len(samples) != len(set(samples)):
        raise ValueError(f"{path} contains duplicate samples")
    if expected_count is not None and len(samples) != int(expected_count):
        raise ValueError(f"{path} has {len(samples)} samples; expected {expected_count}")
    return samples


def cohort_in_split(
    cohort_path,
    splits_path,
    split,
    *,
    normalize_id=None,
    expected_count=None,
) -> list[str]:
    """Load a fixed cohort and prove every member has the requested assignment."""
    samples = load_cohort(
        cohort_path,
        normalize_id=normalize_id,
        expected_count=expected_count,
    )
    assignments = load_split_table(splits_path, normalize_id=normalize_id).set_index(
        "sample_id"
    )["split"]
    invalid = [sample for sample in samples if assignments.get(sample) != split]
    if invalid:
        raise ValueError(f"{cohort_path} contains samples outside {split}: {invalid[:5]}")
    return samples
