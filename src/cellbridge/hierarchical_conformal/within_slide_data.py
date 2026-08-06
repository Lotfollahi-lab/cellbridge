"""Within-slide HEST data preparation with a disjoint conformal stripe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cellbridge.evaluation.artifacts import load_array_artifact
from cellbridge.helpers.io import load_coords
from cellbridge.helpers.splits import spatial_split
from cellbridge.lora.features import FORMAT_VERSION

from .hest_data import _sample_labels


@dataclass(frozen=True)
class WithinSlideData:
    features: np.ndarray
    labels: np.ndarray
    cell_ids: np.ndarray
    train_indices: np.ndarray
    calibration_indices: np.ndarray
    test_indices: np.ndarray


def three_way_spatial_split(
    cell_ids,
    coordinates,
    test_fraction=0.2,
    calibration_fraction_within_fit=0.2,
    axis="y",
):
    """Preserve the historical test stripe and split its complement in two."""
    fit_calibration, test = spatial_split(
        cell_ids, coordinates, test_fraction, axis
    )
    train_local, calibration_local = spatial_split(
        np.asarray(cell_ids)[fit_calibration],
        np.asarray(coordinates)[fit_calibration],
        calibration_fraction_within_fit,
        axis,
    )
    train = fit_calibration[train_local]
    calibration = fit_calibration[calibration_local]
    return train, calibration, test


def load_within_slide_representation(
    manifest_path,
    labels_dir,
    data_root,
    sample_id,
    arm,
    *,
    min_class_count=25,
    test_fraction=0.2,
    calibration_fraction_within_fit=0.2,
    split_axis="y",
):
    manifest = pd.read_csv(manifest_path)
    required = {"sample_id", "path", "signature"}
    if not required <= set(manifest):
        raise ValueError(
            f"{manifest_path}: missing columns {sorted(required - set(manifest))}"
        )
    manifest["sample_id"] = manifest["sample_id"].astype(str)
    matched = manifest.loc[manifest["sample_id"].eq(str(sample_id))]
    if len(matched) != 1:
        raise ValueError(f"{manifest_path}: expected one row for {sample_id}")
    record = matched.iloc[0]
    arrays, cell_ids, _ = load_array_artifact(
        record["path"],
        format_version=FORMAT_VERSION,
        signature=str(record["signature"]),
        array_dims={arm: None},
    )
    features = np.asarray(arrays[arm])
    cell_ids = np.asarray(cell_ids).astype(str)

    labels = _sample_labels(labels_dir, sample_id)
    counts = labels.value_counts()
    keep_classes = set(counts.index[counts >= int(min_class_count)])
    coordinates, coordinate_ids = load_coords(
        Path(data_root) / sample_id / f"{sample_id}.h5ad"
    )
    coordinate_row = {
        str(cell_id): row for row, cell_id in enumerate(coordinate_ids)
    }
    aligned_labels = labels.reindex(cell_ids)
    keep = np.asarray(
        [
            cell_id in coordinate_row and label in keep_classes
            for cell_id, label in zip(cell_ids, aligned_labels)
        ],
        dtype=bool,
    )
    if keep.sum() < 3:
        raise ValueError(f"{sample_id}: fewer than three aligned full-label cells")

    selected_ids = cell_ids[keep]
    selected_features = features[keep]
    selected_labels = aligned_labels.to_numpy()[keep].astype(str)
    order = np.argsort(selected_ids)
    selected_ids = selected_ids[order]
    selected_features = selected_features[order]
    selected_labels = selected_labels[order]
    selected_coordinates = coordinates[
        [coordinate_row[cell_id] for cell_id in selected_ids]
    ]
    train, calibration, test = three_way_spatial_split(
        selected_ids,
        selected_coordinates,
        test_fraction,
        calibration_fraction_within_fit,
        split_axis,
    )
    train_classes = set(selected_labels[train])
    for name, indices in (("calibration", calibration), ("test", test)):
        unseen = sorted(set(selected_labels[indices]) - train_classes)
        if unseen:
            raise ValueError(
                f"{sample_id}: {name} identities absent from training: {unseen}"
            )
    return WithinSlideData(
        selected_features,
        selected_labels,
        selected_ids,
        train,
        calibration,
        test,
    )
