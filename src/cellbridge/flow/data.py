"""Leakage-safe cell neighbourhoods for the one-section flow pilot."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

from cellbridge.distill.data import select_indices
from cellbridge.evaluation.hest import load_labels
from cellbridge.helpers.io import load_coords
from cellbridge.helpers.splits import spatial_split
from cellbridge.jepa.data import load_section


@dataclass(frozen=True)
class PilotArrays:
    image: np.ndarray
    terra: np.ndarray
    neighbourhood_coordinates_um: np.ndarray
    split_coordinates_um: np.ndarray
    cell_ids: np.ndarray
    labels: np.ndarray
    train_rows: np.ndarray
    test_rows: np.ndarray


def load_pilot_arrays(cfg: dict) -> PilotArrays:
    """Load aligned labelled cells and reproduce CellBridge's spatial split."""

    section = load_section(
        cfg["data_root"],
        cfg["manifest_path"],
        cfg["sample_id"],
        image_key=cfg.get("uni_key", "uni_emb"),
        terra_key=cfg.get("terra_key", "cell_emb"),
    )
    settings = cfg["data"]
    label_table = pd.read_csv(cfg["labels_path"])
    label_column = settings.get("label_column", "group")
    minimum = int(settings.get("min_class_count", 25))
    counts = label_table[label_column].astype(str).value_counts()
    keep_classes = set(counts.index[counts >= minimum])
    label_map = load_labels(
        cfg["labels_path"],
        keep_classes,
        id_column=settings.get("label_id_column", "cell_id"),
        label_column=label_column,
    )
    split_coordinates, coordinate_ids = load_coords(cfg["adata_path"])
    coordinate_row = {
        str(cell_id): row for row, cell_id in enumerate(coordinate_ids.astype(str))
    }
    keep = np.asarray(
        [str(cell_id) in label_map and str(cell_id) in coordinate_row for cell_id in section.cell_ids]
    )
    rows = np.flatnonzero(keep)
    ids = section.cell_ids[rows].astype(str)
    order = np.argsort(ids)
    rows, ids = rows[order], ids[order]
    native_coordinates = split_coordinates[[coordinate_row[cell_id] for cell_id in ids]]
    train, test = spatial_split(
        ids,
        native_coordinates,
        float(settings.get("test_fraction", 0.2)),
        settings.get("split_axis", "y"),
    )
    return PilotArrays(
        image=np.asarray(section.image[rows]),
        terra=np.asarray(section.terra[rows]),
        neighbourhood_coordinates_um=np.asarray(section.coordinates_um[rows]),
        split_coordinates_um=np.asarray(native_coordinates),
        cell_ids=ids,
        labels=np.asarray([label_map[cell_id] for cell_id in ids]),
        train_rows=train,
        test_rows=test,
    )


def capped_rows(rows, cap: int | None, seed: int, name: str) -> np.ndarray:
    rows = np.asarray(rows, dtype=int)
    selected = select_indices(len(rows), cap, seed, name)
    return rows[selected]


class CellNeighbourhoodDataset(Dataset):
    """Center-cell targets with nearest H&E conditions from one split only."""

    def __init__(
        self,
        arrays: PilotArrays,
        pool_rows,
        anchor_rows,
        *,
        neighbours: int,
        coordinate_scale_um: float,
    ) -> None:
        pool_rows = np.asarray(pool_rows, dtype=int)
        anchor_rows = np.asarray(anchor_rows, dtype=int)
        if neighbours < 1 or coordinate_scale_um <= 0:
            raise ValueError("neighbours and coordinate_scale_um must be positive")
        if len(pool_rows) <= neighbours:
            raise ValueError("split pool must contain more cells than neighbours")
        if not set(anchor_rows).issubset(set(pool_rows)):
            raise ValueError("anchor rows must be contained in the split pool")
        self.arrays = arrays
        self.pool_rows = pool_rows
        self.anchor_rows = anchor_rows
        self.coordinate_scale_um = float(coordinate_scale_um)
        self.context_rows = self._nearest_rows(neighbours)

    def _nearest_rows(self, neighbours: int) -> np.ndarray:
        coordinates = self.arrays.neighbourhood_coordinates_um
        tree = cKDTree(coordinates[self.pool_rows])
        _, local_rows = tree.query(coordinates[self.anchor_rows], k=neighbours + 1)
        local_rows = np.atleast_2d(local_rows)
        pool_position = {row: position for position, row in enumerate(self.pool_rows)}
        contexts = []
        for anchor, candidates in zip(self.anchor_rows, local_rows, strict=True):
            anchor_local = pool_position[int(anchor)]
            others = [int(row) for row in np.atleast_1d(candidates) if row != anchor_local]
            if len(others) < neighbours:
                raise RuntimeError("could not construct a unique nearest-cell context")
            contexts.append([anchor, *self.pool_rows[others[:neighbours]]])
        return np.asarray(contexts, dtype=int)

    def __len__(self) -> int:
        return len(self.anchor_rows)

    def __getitem__(self, index: int) -> dict:
        anchor = self.anchor_rows[index]
        context = self.context_rows[index]
        coordinates = self.arrays.neighbourhood_coordinates_um
        relative = (coordinates[context] - coordinates[anchor]) / self.coordinate_scale_um
        return {
            "condition": torch.from_numpy(
                np.asarray(self.arrays.image[context], dtype=np.float32)
            ),
            "condition_coordinates": torch.from_numpy(relative.astype(np.float32)),
            "target": torch.from_numpy(
                np.asarray(self.arrays.terra[anchor], dtype=np.float32)
            ),
            "anchor_position": int(index),
            "center_coordinates": torch.from_numpy(
                coordinates[anchor].astype(np.float32)
            ),
        }

    @property
    def anchor_terra(self) -> np.ndarray:
        return np.asarray(self.arrays.terra[self.anchor_rows], dtype=np.float32)

    @property
    def anchor_image(self) -> np.ndarray:
        return np.asarray(self.arrays.image[self.anchor_rows], dtype=np.float32)

    @property
    def anchor_labels(self) -> np.ndarray:
        return self.arrays.labels[self.anchor_rows]

    @property
    def anchor_cell_ids(self) -> np.ndarray:
        return self.arrays.cell_ids[self.anchor_rows]

    @property
    def anchor_coordinates(self) -> np.ndarray:
        return self.arrays.neighbourhood_coordinates_um[self.anchor_rows]


def build_pilot_datasets(cfg: dict) -> tuple[PilotArrays, CellNeighbourhoodDataset, CellNeighbourhoodDataset]:
    arrays = load_pilot_arrays(cfg)
    settings = cfg["data"]
    seed = int(settings.get("sampling_seed", 0))
    train_anchors = capped_rows(
        arrays.train_rows, settings.get("max_train_cells"), seed, f"{cfg['sample_id']}:train"
    )
    test_anchors = capped_rows(
        arrays.test_rows, settings.get("max_test_cells"), seed, f"{cfg['sample_id']}:test"
    )
    common = {
        "neighbours": int(settings["neighbours"]),
        "coordinate_scale_um": float(settings["coordinate_scale_um"]),
    }
    return (
        arrays,
        CellNeighbourhoodDataset(arrays, arrays.train_rows, train_anchors, **common),
        CellNeighbourhoodDataset(arrays, arrays.test_rows, test_anchors, **common),
    )
