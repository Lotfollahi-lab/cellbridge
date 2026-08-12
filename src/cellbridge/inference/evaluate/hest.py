"""Prepare spatially separated HEST cell-type tasks."""

from pathlib import Path

import numpy as np
import pandas as pd

from cellbridge.helpers.io import load_coords
from cellbridge.helpers.splits import spatial_split
from cellbridge.inference.evaluate import Task


def _read_labels(path, id_column, label_column) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        table = pd.read_csv(path)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        table = pd.read_parquet(path)
    else:
        raise ValueError(f"unsupported label file: {path}")

    required = {id_column, label_column}
    if not required <= set(table):
        raise ValueError(f"{path} is missing {sorted(required - set(table))}")
    if table[list(required)].isna().any().any():
        raise ValueError(f"{path} contains null cell IDs or labels")
    table = table.copy()
    table[id_column] = table[id_column].astype(str)
    table[label_column] = table[label_column].astype(str)
    if table[id_column].duplicated().any():
        raise ValueError(f"{path} contains duplicate cell IDs")
    return table


def prepare_task(
    features,
    cell_ids,
    labels_path,
    coordinates_path,
    *,
    id_column: str = "cell_id",
    label_column: str = "group",
    min_class_count: int = 25,
    test_fraction: float = 0.2,
    split_axis: str = "y",
) -> Task:
    """Align one sample's features and labels, then make a spatial split."""
    features = np.asarray(features, dtype=np.float32)
    cell_ids = np.asarray(cell_ids).astype(str)
    if features.ndim != 2 or len(features) != len(cell_ids):
        raise ValueError(f"expected ({len(cell_ids)}, D) features, got {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("features contain non-finite values")
    if len(np.unique(cell_ids)) != len(cell_ids):
        raise ValueError("features contain duplicate cell IDs")
    if min_class_count < 1:
        raise ValueError("min_class_count must be positive")

    label_table = _read_labels(labels_path, id_column, label_column)
    counts = label_table[label_column].value_counts()
    classes = set(counts.index[counts >= min_class_count])
    if len(classes) < 2:
        raise ValueError("fewer than two classes satisfy min_class_count")
    labels = dict(
        label_table.loc[label_table[label_column].isin(classes), [id_column, label_column]]
        .itertuples(index=False, name=None)
    )

    coordinates, coordinate_ids = load_coords(coordinates_path)
    coordinate_ids = np.asarray(coordinate_ids).astype(str)
    if coordinates.shape != (len(coordinate_ids), 2) or not np.isfinite(coordinates).all():
        raise ValueError(f"{coordinates_path} contains invalid spatial coordinates")
    if len(np.unique(coordinate_ids)) != len(coordinate_ids):
        raise ValueError(f"{coordinates_path} contains duplicate coordinate IDs")
    coordinate_row = {cell_id: row for row, cell_id in enumerate(coordinate_ids)}

    keep = np.asarray(
        [cell_id in labels and cell_id in coordinate_row for cell_id in cell_ids]
    )
    selected_ids = cell_ids[keep]
    if len(selected_ids) < 2:
        raise ValueError("fewer than two cells have both labels and coordinates")
    selected_features = features[keep]
    order = np.argsort(selected_ids)
    selected_ids = selected_ids[order]
    selected_features = selected_features[order]
    selected_labels = np.asarray([labels[cell_id] for cell_id in selected_ids])
    selected_coordinates = coordinates[
        [coordinate_row[cell_id] for cell_id in selected_ids]
    ]
    train, test = spatial_split(
        selected_ids,
        selected_coordinates,
        test_fraction,
        split_axis,
    )
    missing = set(selected_labels[test]) - set(selected_labels[train])
    if missing:
        raise ValueError(f"test classes absent from training: {sorted(missing)}")
    return Task(selected_features, selected_labels, selected_ids, train, None, test)
