"""Read the frozen HEST feature artifacts used by the original probes."""

from pathlib import Path

import numpy as np
import pandas as pd


def load(settings, method, arm, sample_id):
    """Load one configured method/arm/sample with manifest validation."""
    try:
        profile = settings[method]
        array_name = profile["arms"][arm]
    except KeyError as error:
        raise ValueError(
            f"HEST features are not configured for {method}/{arm}"
        ) from error

    manifest_path = Path(profile["manifest"])
    manifest = pd.read_csv(
        manifest_path,
        dtype={"sample_id": str, "path": str, "signature": str},
    )
    required = {"sample_id", "path", "signature"}
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"{manifest_path} is missing {sorted(missing)}")
    selected = manifest.loc[manifest["sample_id"] == str(sample_id)]
    if len(selected) != 1:
        raise ValueError(
            f"{manifest_path} contains {len(selected)} rows for {sample_id!r}"
        )
    record = selected.iloc[0]
    artifact_path = Path(record["path"])
    if not artifact_path.exists():
        relocated = manifest_path.parent / artifact_path.name
        if not relocated.exists():
            raise FileNotFoundError(
                f"HEST feature artifact does not exist: {artifact_path}"
            )
        artifact_path = relocated

    with np.load(artifact_path, allow_pickle=False) as artifact:
        required_arrays = {"cell_ids", "signature", array_name}
        missing_arrays = required_arrays - set(artifact.files)
        if missing_arrays:
            raise ValueError(
                f"{artifact_path} is missing arrays {sorted(missing_arrays)}"
            )
        signature = str(np.asarray(artifact["signature"]).item())
        if signature != str(record["signature"]):
            raise ValueError(f"{artifact_path} signature does not match its manifest")
        cell_ids = artifact["cell_ids"].astype(str)
        features = np.asarray(artifact[array_name], dtype=np.float32)

    if (
        cell_ids.ndim != 1
        or features.ndim != 2
        or len(cell_ids) != len(features)
        or len(np.unique(cell_ids)) != len(cell_ids)
        or not np.isfinite(features).all()
    ):
        raise ValueError(f"{artifact_path} contains invalid or misaligned features")
    return cell_ids, features
