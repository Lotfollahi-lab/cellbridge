"""Lightweight split helpers for the isolated HistoPLUS environment."""

from pathlib import Path

import pandas as pd

from terra_multi.cohorts import cohort_in_split


def label_test_fast_samples(hest_cfg):
    return cohort_in_split(
        hest_cfg["label_test_fast_path"],
        hest_cfg["splits_path"],
        "label_test",
        expected_count=int(hest_cfg.get("label_test_fast_n", 5)),
    )


def hest_manifest_row(hest_cfg, sample_id):
    path = Path(hest_cfg["manifest_path"])
    manifest = pd.read_csv(path)
    if "id" not in manifest:
        raise ValueError(f"{path} requires an id column")
    manifest["id"] = manifest["id"].astype(str)
    selected = manifest.loc[manifest["id"] == str(sample_id)]
    if len(selected) != 1:
        raise KeyError(
            f"expected one manifest row for {sample_id}, found {len(selected)}"
        )
    return selected.iloc[0]


def validate_official_split(table, split_path, *, require_full_coverage=True):
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
    observed_ids = set(observed["sample_id"])
    official_ids = set(official["sample_id"])
    if observed_ids - official_ids:
        raise ValueError("metadata contains samples absent from the official split")
    if require_full_coverage and observed_ids != official_ids:
        raise ValueError("metadata does not cover every official sample")

    merged = official.merge(observed, on="sample_id", validate="one_to_one")
    if not (merged["official_split"] == merged["split"]).all():
        raise ValueError("metadata disagrees with the official split")
