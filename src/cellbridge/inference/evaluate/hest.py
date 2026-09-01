"""Prepare spatially separated HEST cell-type tasks."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from cellbridge.helpers.io import load_coords
from cellbridge.helpers.splits import spatial_split
from cellbridge.inference.embed.sources.metadata import cell_uids
from cellbridge.inference.evaluate import Task
from cellbridge.ontology import load_mapping

PCA_ARMS = frozenset({"uni", "lora_adapted", "virchow2"})


def load_sample_mapping(mapping_dir, sample_id, dataset):
    """Load one reviewed HEST label mapping."""
    mapping = load_mapping(Path(mapping_dir) / f"{sample_id}.yaml")
    if mapping.dataset != str(dataset):
        raise ValueError(
            f"mapping dataset {mapping.dataset!r} does not match {dataset!r}"
        )
    return mapping


def align_metadata(source, data, loaded, sample_id):
    """Align one standard embedding artifact with its HEST cell metadata."""
    metadata = source.sample_metadata(data, sample_id)
    indexed = metadata.set_index("row_id", drop=False)
    missing = set(loaded.row_ids) - set(indexed.index)
    if missing:
        raise ValueError(
            f"{sample_id}: embedding artifacts contain unknown row IDs "
            f"{sorted(missing)[:5]}"
        )
    aligned = indexed.loc[loaded.row_ids].reset_index(drop=True)
    if not np.array_equal(cell_uids(aligned), loaded.cell_ids):
        raise ValueError(f"{sample_id}: embedding cell IDs do not match HEST metadata")
    if not np.all(loaded.sample_ids == str(sample_id)):
        raise ValueError(f"{sample_id}: embedding manifest contains another sample")
    return aligned


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


def _filter_split_support(
    features,
    labels,
    ids,
    train,
    calibration,
    test,
    minimum,
):
    minimum = int(minimum)
    if minimum < 1:
        raise ValueError("min_class_count_per_split must be positive")
    splits = [("train", train)]
    if calibration is not None:
        splits.append(("calibration", calibration))
    splits.append(("test", test))

    classes = set(labels)
    supported = set(classes)
    for _, indices in splits:
        supported &= {
            label
            for label in classes
            if int(np.sum(labels[indices] == label)) >= minimum
        }
    if len(supported) < 2:
        raise ValueError(
            "fewer than two classes have at least "
            f"{minimum} cells in every required split"
        )

    excluded = sorted(classes - supported)
    if excluded:
        print(
            f"Excluded classes with fewer than {minimum} cells per split: "
            f"{', '.join(excluded)}"
        )
    keep = np.isin(labels, tuple(supported))
    new_rows = np.full(len(labels), -1, dtype=int)
    new_rows[keep] = np.arange(int(keep.sum()))

    def remap(indices):
        mapped = new_rows[indices]
        return mapped[mapped >= 0]

    return (
        features[keep],
        labels[keep],
        ids[keep],
        remap(train),
        None if calibration is None else remap(calibration),
        remap(test),
    )


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
    arm: str | None = None,
    mapping=None,
    calibration_fraction_within_fit: float | None = None,
    min_class_count_per_split: int | None = None,
    pca_components: int = 256,
    seed: int = 0,
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
    if mapping is not None:
        mapped, included = mapping.map_labels(label_table[label_column])
        label_table = label_table.loc[included].copy()
        label_table[label_column] = mapped[included]

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
    if calibration_fraction_within_fit is None:
        train, test = spatial_split(
            selected_ids, selected_coordinates, test_fraction, split_axis
        )
        calibration = None
        held_out = (("test", test),)
    else:
        train, calibration, test = three_way_spatial_split(
            selected_ids,
            selected_coordinates,
            test_fraction,
            calibration_fraction_within_fit,
            split_axis,
        )
        held_out = (("calibration", calibration), ("test", test))
    if min_class_count_per_split is not None:
        (
            selected_features,
            selected_labels,
            selected_ids,
            train,
            calibration,
            test,
        ) = _filter_split_support(
            selected_features, selected_labels, selected_ids,
            train, calibration, test, min_class_count_per_split,
        )
        held_out = (("test", test),) if calibration is None else (
            ("calibration", calibration), ("test", test)
        )
    train_classes = set(selected_labels[train])
    for name, indices in held_out:
        missing = set(selected_labels[indices]) - train_classes
        if missing:
            raise ValueError(f"{name} classes absent from training: {sorted(missing)}")
    if arm in PCA_ARMS:
        maximum = min(len(train), selected_features.shape[1])
        if not 1 <= pca_components <= maximum:
            raise ValueError(
                f"pca_components must be between 1 and {maximum}, got {pca_components}"
            )
        pca = PCA(
            n_components=int(pca_components),
            svd_solver="randomized",
            random_state=int(seed),
        )
        pca.fit(selected_features[train])
        selected_features = pca.transform(selected_features).astype(np.float32)
    return Task(
        selected_features, selected_labels, selected_ids, train, calibration, test
    )


def three_way_spatial_split(
    cell_ids,
    coordinates,
    test_fraction,
    calibration_fraction_within_fit,
    axis,
):
    """Preserve the test stripe and split its complement into train/calibration."""
    fit, test = spatial_split(cell_ids, coordinates, test_fraction, axis)
    train_local, calibration_local = spatial_split(
        np.asarray(cell_ids)[fit],
        np.asarray(coordinates)[fit],
        calibration_fraction_within_fit,
        axis,
    )
    return fit[train_local], fit[calibration_local], test
