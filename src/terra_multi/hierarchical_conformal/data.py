"""Load aligned frozen PhenoCell representations and official split metadata."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from terra_multi.evaluation.ood import validate_official_split
from terra_multi.lora.probe_phenocell import load_phenocell_arm


def load_frozen_representation(
    manifest_path,
    arm,
    split_path,
    exclude_labels=(),
):
    manifest_path = Path(manifest_path)
    manifest = pd.read_csv(manifest_path)
    required = {"path", "signature"}
    if not required <= set(manifest):
        raise ValueError(
            f"{manifest_path}: missing columns {sorted(required - set(manifest))}"
        )

    features, table = load_phenocell_arm(manifest, arm)
    if features is None:
        raise KeyError(f"{manifest_path}: feature arm {arm!r} is absent")
    validate_official_split(table, split_path)

    excluded = set(map(str, exclude_labels))
    keep = ~table["label"].astype(str).isin(excluded)
    features = np.asarray(features)[keep.to_numpy()]
    table = table.loc[keep].reset_index(drop=True)
    if table["row_id"].duplicated().any():
        raise ValueError("PhenoCell row IDs must be unique")
    if not {"train", "val", "test"} <= set(table["split"]):
        raise ValueError("filtered data must retain train, val, and test cells")
    return features, table
