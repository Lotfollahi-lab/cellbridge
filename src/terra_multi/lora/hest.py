"""HEST paths and held-out cohort helpers for LoRA evaluation."""

from pathlib import Path

import pandas as pd

from terra_multi.cohorts import cohort_in_split
from terra_multi.evaluation.artifacts import file_identity


def label_test_fast_samples(cfg):
    settings = cfg["eval"]
    return cohort_in_split(
        settings["label_test_fast_path"],
        cfg["splits_path"],
        "label_test",
        expected_count=int(settings.get("label_test_fast_n", 5)),
    )


def manifest_row(cfg, sample_id):
    root = Path(cfg["data_root"])
    path = Path(cfg.get("manifest_path", root / "manifest.csv"))
    manifest = pd.read_csv(path)
    if "id" not in manifest:
        raise ValueError(f"{path} requires an id column")
    manifest["id"] = manifest["id"].astype(str)
    matches = manifest.loc[manifest["id"] == str(sample_id)]
    if len(matches) != 1:
        raise KeyError(f"{sample_id} is absent or duplicated in {path}")
    return path, matches.iloc[0]


def sample_dir(cfg, sample_id, row=None):
    if row is None:
        _, row = manifest_row(cfg, sample_id)
    output_dir = row.get("output_dir")
    if output_dir is None or pd.isna(output_dir):
        return Path(cfg["data_root"]) / str(sample_id)
    return Path(output_dir)


def source_identity(cfg, sample_id):
    path, row = manifest_row(cfg, sample_id)
    directory = sample_dir(cfg, sample_id, row)
    return {
        "manifest": file_identity(path),
        "wsi": file_identity(row["wsi"]),
        "terra": file_identity(directory / f"{sample_id}_terra.npz"),
    }


def dataset(cfg, sample_id, transform):
    from terra_multi.lora.data import build_label_test_dataset

    return build_label_test_dataset(
        cfg,
        transform,
        max_cells_per_sample=None,
        sample_ids=[sample_id],
    )
