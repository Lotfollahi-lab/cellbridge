"""Image and label adapters for the selective-UNI supervised baseline."""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from torch.utils.data import Dataset, Sampler

from terra_multi.evaluation.hest import collapse_to_level, load_labels
from terra_multi.evaluation.ood import IMMUNE_CLASSES, validate_official_split
from terra_multi.evaluation.ood_data import (
    centered_crop,
    iter_lmdb_samples,
    locate_cells,
    reconstruct_image,
    to_uint8_rgb,
)
from terra_multi.helpers.io import load_coords
from terra_multi.helpers.splits import spatial_split
from terra_multi.lora.data import (
    canonical_id,
    crop_cell,
    open_wsi,
    split_samples,
)


def _canonical_ids(values) -> np.ndarray:
    return np.asarray([canonical_id(value) for value in values], dtype=str)


def _manifest_rows(path) -> dict[str, dict]:
    table = pd.read_csv(path)
    required = {"id", "wsi", "pixel_size_um"}
    if not required <= set(table):
        raise ValueError(f"{path} must contain {sorted(required)}")
    rows = {}
    for row in table.to_dict("records"):
        sample_id = canonical_id(row["id"])
        if sample_id in rows:
            raise ValueError(f"{path}: duplicate sample {sample_id!r}")
        rows[sample_id] = row
    return rows


def _sample_dir(data_root, sample_id, row) -> Path:
    output = row.get("output_dir")
    if output is not None and not pd.isna(output):
        return Path(output)
    return Path(data_root) / sample_id


def stratified_validation_split(indices, labels, fraction, seed):
    """Reserve a deterministic class-stratified subset without touching test."""
    indices = np.asarray(indices, dtype=int)
    labels = np.asarray(labels)
    if not 0 < fraction < 1:
        raise ValueError("validation fraction must be in (0, 1)")
    rng = np.random.default_rng(seed)
    train, val = [], []
    for label in np.unique(labels[indices]):
        rows = indices[labels[indices] == label]
        rows = rng.permutation(rows)
        if len(rows) == 1:
            train.extend(rows)
            continue
        n_val = min(max(round(len(rows) * fraction), 1), len(rows) - 1)
        val.extend(rows[:n_val])
        train.extend(rows[n_val:])
    if not train or not val:
        raise ValueError("could not construct nonempty stratified train/validation sets")
    return np.sort(np.asarray(train, dtype=int)), np.sort(np.asarray(val, dtype=int))


class HestClassificationDataset(Dataset):
    """One fast-HEST sample with the exact source labels and spatial split."""

    def __init__(self, cfg, sample_id, *, seed=0, transform=None):
        sample_id = canonical_id(sample_id)
        allowed = split_samples(
            cfg["splits_path"], "label_test_fast", cfg["label_test_fast_path"]
        )
        if sample_id not in allowed:
            raise ValueError(f"{sample_id!r} is not in label_test_fast")
        manifest = _manifest_rows(cfg["manifest_path"])
        if sample_id not in manifest:
            raise KeyError(f"manifest is missing sample {sample_id!r}")
        row = manifest[sample_id]
        sample_dir = _sample_dir(cfg["data_root"], sample_id, row)
        terra_path = sample_dir / f"{sample_id}_terra.npz"
        adata_path = sample_dir / f"{sample_id}.h5ad"
        labels_path = Path(
            cfg.get(
                "labels_pattern",
                "{data_root}/{sample_id}/{sample_id}_granular_cell_labels.csv",
            ).format(data_root=cfg["data_root"], sample_id=sample_id)
        )

        with np.load(terra_path, allow_pickle=True) as data:
            required = {"cell_ids", "he_x", "he_y"}
            if not required <= set(data.files):
                raise ValueError(f"{terra_path}: missing {sorted(required - set(data.files))}")
            crop_ids = _canonical_ids(data["cell_ids"])
            crop_centers = np.column_stack((data["he_x"], data["he_y"]))
        if len(np.unique(crop_ids)) != len(crop_ids):
            raise ValueError(f"{terra_path}: duplicate cell IDs")

        settings = cfg["evaluation"]
        label_column = settings.get("label_column", "group")
        labels_table = pd.read_csv(labels_path)
        labels_table[label_column] = collapse_to_level(
            labels_table[label_column], settings
        )
        labels_table = labels_table[labels_table[label_column].notna()].reset_index(
            drop=True
        )
        counts = labels_table[label_column].astype(str).value_counts()
        minimum = int(settings.get("min_class_count", 25))
        keep_classes = set(counts.index[counts >= minimum])
        label_map = load_labels(
            labels_path,
            keep_classes,
            id_column=settings.get("label_id_column", "cell_id"),
            label_column=label_column,
            table=labels_table,
        )

        spatial_coords, spatial_ids = load_coords(adata_path)
        spatial_ids = np.asarray(spatial_ids).astype(str)
        crop_row = {cell_id: index for index, cell_id in enumerate(crop_ids)}
        spatial_row = {cell_id: index for index, cell_id in enumerate(spatial_ids)}
        ids = np.asarray(
            sorted(set(crop_row) & set(spatial_row) & set(label_map)), dtype=str
        )
        if len(ids) < 2:
            raise ValueError(f"{sample_id}: fewer than two aligned labelled cells")
        self.cell_ids = ids
        self.labels = np.asarray([label_map[cell_id] for cell_id in ids], dtype=str)
        self.centers = np.asarray(
            [crop_centers[crop_row[cell_id]] for cell_id in ids], dtype=np.float64
        )
        aligned_spatial = np.asarray(
            [spatial_coords[spatial_row[cell_id]] for cell_id in ids], dtype=np.float64
        )
        self.train_indices, self.test_indices = spatial_split(
            ids,
            aligned_spatial,
            float(settings.get("test_frac", 0.2)),
            settings.get("split_axis", "y"),
        )
        self.selection_train_indices, self.val_indices = stratified_validation_split(
            self.train_indices,
            self.labels,
            float(settings.get("validation_frac", 0.1)),
            int(seed),
        )
        self.classes = np.unique(self.labels)
        class_to_index = {label: index for index, label in enumerate(self.classes)}
        self.targets = np.asarray([class_to_index[label] for label in self.labels])
        self.sample_id = sample_id
        self.wsi_path = str(row["wsi"])
        self.pixel_size_um = float(row["pixel_size_um"])
        self.transform = transform
        self.target_size = int(cfg["crop"]["target_size"])
        self.target_pixel_size_um = float(cfg["crop"]["target_pixel_size_um"])
        self._wsi = None

    def __len__(self):
        return len(self.cell_ids)

    def _get_wsi(self):
        if self._wsi is None:
            self._wsi = open_wsi(self.wsi_path)
        return self._wsi

    def __getitem__(self, index):
        cx, cy = self.centers[index]
        image = crop_cell(
            self._get_wsi(),
            cx,
            cy,
            self.pixel_size_um,
            self.target_size,
            self.target_pixel_size_um,
        )
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "target": int(self.targets[index]),
            "row_id": str(self.cell_ids[index]),
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_wsi"] = None
        return state


def _cache_signature(data_cfg, crop_cfg):
    return {
        "format_version": "terra_multi.ood_cell_crops/1",
        "data": data_cfg,
        "crop": {
            "source_size": int(crop_cfg["source_size"]),
            "target_size": int(crop_cfg["target_size"]),
        },
    }


def _safe_sample_stem(sample_name):
    digest = hashlib.sha1(str(sample_name).encode()).hexdigest()[:10]
    clean = Path(str(sample_name)).name.removesuffix(".hdf")
    clean = "".join(character if character.isalnum() or character in "-_" else "_" for character in clean)
    return f"{clean}_{digest}"


def _write_npz_atomic(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_ood_crop_cache(dataset_name, cfg, *, overwrite=False, limit_samples=None):
    """Materialize exact 96px benchmark crops once for repeated FT epochs."""
    data_cfg, crop_cfg = cfg["data"], cfg["crop"]
    output_dir = Path(cfg["cache_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    signature = _cache_signature(data_cfg, crop_cfg)
    signature_path = output_dir / "cache_config.yaml"
    if signature_path.exists() and not overwrite:
        with open(signature_path) as handle:
            if yaml.safe_load(handle) != signature:
                raise ValueError(f"{output_dir} contains a different crop-cache contract")

    metadata = pd.read_csv(data_cfg["metadata"])
    metadata["row_id"] = np.arange(len(metadata), dtype=np.int64)
    image_shape = tuple(data_cfg.get("image_shape", [1440, 1920]))
    tile_size = int(data_cfg.get("tile_size", 224))
    allow_unlabeled = bool(data_cfg.get("allow_unlabeled_samples", False))
    sample_rows, cell_tables = [], []
    completed = 0
    for sample_name, entries in iter_lmdb_samples(data_cfg["lmdb"]):
        if limit_samples is not None and completed >= limit_samples:
            break
        sample_metadata = metadata.loc[
            metadata["sample_id"].astype(str) == str(sample_name)
        ].copy()
        if sample_metadata.empty:
            if allow_unlabeled:
                continue
            raise ValueError(f"{sample_name}: no metadata rows")
        locations = locate_cells(
            sample_metadata,
            entries,
            image_shape,
            tile_size,
            skip_missing=allow_unlabeled,
        )
        if locations.empty:
            continue
        path = output_dir / "samples" / f"{_safe_sample_stem(sample_name)}.npz"
        if overwrite or not path.exists():
            tiles = {index: entry["image"] for index, entry in entries.items()}
            image = to_uint8_rgb(
                reconstruct_image(tiles, image_shape, tile_size),
                data_cfg.get("value_scale", 250),
            )
            crops = np.stack(
                [
                    centered_crop(image, row.x, row.y, crop_cfg["source_size"])
                    for row in locations.itertuples()
                ]
            )
            _write_npz_atomic(path, crops=crops)
        with np.load(path, allow_pickle=False) as cached:
            if len(cached["crops"]) != len(locations):
                raise ValueError(f"{path}: cached crop count does not match metadata")
        locations = locations.reset_index(drop=True)
        locations["sample_number"] = completed
        locations["local_index"] = np.arange(len(locations), dtype=np.int32)
        cell_tables.append(
            locations[
                [
                    "row_id",
                    "sample_id",
                    "tile_id",
                    "cell_id",
                    "label",
                    "class_id",
                    "split_id",
                    "sample_number",
                    "local_index",
                ]
            ].rename(columns={"split_id": "split"})
        )
        sample_rows.append(
            {
                "sample_number": completed,
                "sample_id": str(sample_name),
                "n_cells": len(locations),
                "path": str(path.resolve()),
            }
        )
        completed += 1
        print(f"[{dataset_name}/{sample_name}] cached {len(locations):,} crops")

    if not sample_rows:
        raise ValueError(f"{dataset_name}: no labelled samples were cached")
    samples = pd.DataFrame(sample_rows)
    cells = pd.concat(cell_tables, ignore_index=True)
    if limit_samples is None:
        validate_official_split(
            cells,
            data_cfg["split"],
            require_full_coverage=not allow_unlabeled,
        )
    samples.to_csv(output_dir / "manifest.csv", index=False)
    cells.to_csv(output_dir / "cells.csv", index=False)
    with open(signature_path, "w") as handle:
        yaml.safe_dump(signature, handle, sort_keys=False)
    print(f"cached {len(cells):,} {dataset_name} cells -> {output_dir}")


class CachedOodCropDataset(Dataset):
    """Random-access view over sample-sharded OOD cell crops."""

    def __init__(self, cfg, task="all", *, transform=None, max_open_samples=8):
        if task not in {"all", "immune"}:
            raise ValueError(f"unknown OOD task {task!r}")
        root = Path(cfg["cache_dir"])
        samples = pd.read_csv(root / "manifest.csv").sort_values("sample_number")
        cells = pd.read_csv(root / "cells.csv")
        if task == "immune":
            cells = cells[cells["label"].isin(IMMUNE_CLASSES)].copy()
        cells = cells.reset_index(drop=True)
        if cells.empty:
            raise ValueError(f"{task}: crop cache has no cells")
        cells["split"] = cells["split"].replace({"valid": "val"})
        self.records = cells
        self.sample_paths = dict(
            zip(samples["sample_number"].astype(int), samples["path"].astype(str))
        )
        self.classes = np.sort(cells["label"].astype(str).unique())
        class_to_index = {label: index for index, label in enumerate(self.classes)}
        self.targets = cells["label"].astype(str).map(class_to_index).to_numpy(int)
        split = cells["split"].astype(str).to_numpy()
        self.train_indices = np.flatnonzero(split == "train")
        self.val_indices = np.flatnonzero(split == "val")
        self.test_indices = np.flatnonzero(split == "test")
        if min(map(len, (self.train_indices, self.val_indices, self.test_indices))) == 0:
            raise ValueError(f"{task}: an official split is empty")
        self.row_ids = cells["row_id"].to_numpy(np.int64)
        self.transform = transform
        self.target_size = int(cfg["crop"]["target_size"])
        self.max_open_samples = int(max_open_samples)
        self._cache = OrderedDict()

    def __len__(self):
        return len(self.records)

    def _crops(self, sample_number):
        sample_number = int(sample_number)
        if sample_number in self._cache:
            return self._cache.pop(sample_number)
        with np.load(self.sample_paths[sample_number], allow_pickle=False) as data:
            crops = data["crops"]
        if len(self._cache) == self.max_open_samples:
            self._cache.popitem(last=False)
        return crops

    def __getitem__(self, index):
        row = self.records.iloc[index]
        sample_number = int(row.sample_number)
        crops = self._crops(sample_number)
        self._cache[sample_number] = crops
        image = Image.fromarray(crops[int(row.local_index)]).resize(
            (self.target_size, self.target_size), Image.Resampling.BILINEAR
        )
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "target": int(self.targets[index]),
            "row_id": int(self.row_ids[index]),
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        return state


class GroupedBatchSampler(Sampler[list[int]]):
    """Shuffle cells while keeping batches within one cached sample."""

    def __init__(self, dataset, indices, batch_size, *, seed=0, shuffle=True):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        indices = np.asarray(indices, dtype=int)
        sample_numbers = dataset.records.iloc[indices]["sample_number"].to_numpy(int)
        self.groups = tuple(indices[sample_numbers == value] for value in np.unique(sample_numbers))
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return sum((len(group) + self.batch_size - 1) // self.batch_size for group in self.groups)

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        groups = list(self.groups)
        if self.shuffle:
            groups = [groups[index] for index in rng.permutation(len(groups))]
        batches = []
        for group in groups:
            rows = rng.permutation(group) if self.shuffle else group
            batches.extend(
                rows[start : start + self.batch_size].tolist()
                for start in range(0, len(rows), self.batch_size)
            )
        yield from batches
