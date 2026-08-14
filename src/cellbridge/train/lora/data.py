"""H&E cell crops and TERRA targets for LoRA distillation."""

from __future__ import annotations

import bisect
import hashlib
import re
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from cellbridge.helpers.cohorts import samples_in_split

TARGET_PATCH_SIZE = 224
TARGET_PIXEL_SIZE_UM = 0.5
TERRA_DIM = 384
VALID_SPLITS = {"distill_train", "distill_val"}


def canonical_id(value) -> str:
    """Represent string, byte, and numeric identifiers consistently."""

    if isinstance(value, (bytes, np.bytes_)):
        value = bytes(value).decode("utf-8")
    if isinstance(value, (str, np.str_)):
        text = str(value)
        if re.fullmatch(r"[+-]?(?:0|[1-9]\d*)\.0+", text):
            return str(int(text.split(".", 1)[0]))
        return text
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        return str(int(value)) if float(value).is_integer() else str(value)
    return str(value)


def _canonical_ids(values) -> np.ndarray:
    return np.asarray([canonical_id(value) for value in values], dtype=str)


def select_indices(n, cap, seed, sample_id):
    """Choose at most ``cap`` rows deterministically for one sample."""

    if cap is None or n <= cap:
        return np.arange(n)
    sample_hash = int.from_bytes(
        hashlib.sha1(str(sample_id).encode()).digest()[:8], "big"
    )
    rng = np.random.default_rng([int(seed), sample_hash])
    return np.sort(rng.choice(n, cap, replace=False))


def split_samples(splits_path, split: str) -> list[str]:
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}, got {split!r}")
    return samples_in_split(splits_path, split, normalize_id=canonical_id)


def source_patch_size(
    pixel_size_um: float,
    target_size: int = TARGET_PATCH_SIZE,
    target_pixel_size_um: float = TARGET_PIXEL_SIZE_UM,
) -> int:
    """Return the level-0 crop side for a requested physical field of view."""

    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError(f"pixel_size_um must be positive, got {pixel_size_um}")
    if target_size < 1 or target_pixel_size_um <= 0:
        raise ValueError("target crop size and pixel size must be positive")
    return max(1, round(target_size * target_pixel_size_um / pixel_size_um))


def crop_cell(
    wsi,
    cx: float,
    cy: float,
    pixel_size_um: float,
    target_size: int = TARGET_PATCH_SIZE,
    target_pixel_size_um: float = TARGET_PIXEL_SIZE_UM,
) -> Image.Image:
    """Crop, white-pad, and resize one level-0 cell patch for UNI2-h."""

    patch_size = source_patch_size(
        pixel_size_um,
        target_size=target_size,
        target_pixel_size_um=target_pixel_size_um,
    )
    height, width = wsi.shape[:2]
    half = patch_size // 2
    x0, y0 = round(cx) - half, round(cy) - half
    x1, y1 = x0 + patch_size, y0 + patch_size
    valid_x0, valid_y0 = max(x0, 0), max(y0, 0)
    valid_x1, valid_y1 = min(x1, width), min(y1, height)
    canvas = np.full((patch_size, patch_size, 3), 255, dtype=np.uint8)
    if valid_x1 > valid_x0 and valid_y1 > valid_y0:
        region = np.asarray(wsi[valid_y0:valid_y1, valid_x0:valid_x1])
        if region.ndim == 2:
            region = np.repeat(region[..., None], 3, axis=-1)
        canvas[
            valid_y0 - y0 : valid_y1 - y0,
            valid_x0 - x0 : valid_x1 - x0,
        ] = region[..., :3]
    return Image.fromarray(canvas).resize(
        (target_size, target_size), Image.Resampling.BILINEAR
    )


def open_wsi(path):
    """Open a level-0 TIFF as a lazy zarr array."""

    import tifffile
    import zarr

    try:
        store = tifffile.imread(path, aszarr=True, level=0)
    except (IndexError, TypeError):
        store = tifffile.imread(path, aszarr=True, key=0)
    return zarr.open(store, mode="r")


def _manifest_rows(manifest_path) -> dict[str, dict]:
    table = pd.read_csv(manifest_path)
    required = {"id", "wsi", "pixel_size_um"}
    if not required <= set(table.columns):
        raise ValueError(f"{manifest_path} must contain columns {sorted(required)}")

    rows = {}
    for row in table.to_dict("records"):
        sample_id = canonical_id(row["id"])
        if sample_id in rows:
            raise ValueError(f"{manifest_path} contains duplicate sample {sample_id!r}")
        rows[sample_id] = row
    return rows


def _sample_output_dir(data_root, sample_id: str, manifest_row: dict) -> Path:
    output_dir = manifest_row.get("output_dir")
    if output_dir is not None and not pd.isna(output_dir):
        return Path(output_dir)
    return Path(data_root) / sample_id


def _load_sample(path, terra_key, cap, seed, sample_id):
    with np.load(path, allow_pickle=True) as data:
        required = {"cell_ids", terra_key, "he_x", "he_y"}
        if not required <= set(data.files):
            raise ValueError(
                f"{path} is missing keys {sorted(required - set(data.files))}"
            )
        cell_ids = _canonical_ids(data["cell_ids"])
        terra = np.asarray(data[terra_key], dtype=np.float32)
        centers = np.column_stack((data["he_x"], data["he_y"])).astype(np.float64)

    if terra.shape != (len(cell_ids), TERRA_DIM):
        raise ValueError(
            f"{path}: expected TERRA shape ({len(cell_ids)}, {TERRA_DIM}), "
            f"got {terra.shape}"
        )
    if centers.shape != (len(cell_ids), 2) or not np.isfinite(centers).all():
        raise ValueError(f"{path}: invalid H&E centers")
    if len(set(cell_ids.tolist())) != len(cell_ids):
        raise ValueError(f"{path}: duplicate cell IDs after canonicalization")

    rows = select_indices(len(cell_ids), cap, seed, sample_id)
    cell_ids, terra, centers = cell_ids[rows], terra[rows], centers[rows]
    norms = np.linalg.norm(terra, axis=1, keepdims=True)
    if not np.isfinite(terra).all() or np.any(norms <= 0):
        raise ValueError(f"{path}: TERRA embeddings must be finite and nonzero")
    return cell_ids, terra / norms, centers


class HESTCellDataset(Dataset):
    """Lazy H&E crops paired with normalized TERRA embeddings."""

    def __init__(
        self,
        data_root,
        manifest_path,
        splits_path,
        split: str,
        *,
        transform: Callable | None = None,
        max_cells_per_sample: int | None = None,
        seed: int = 0,
        terra_key: str = "cell_emb",
        wsi_opener: Callable = open_wsi,
        max_open_wsis: int = 64,
        target_size: int = TARGET_PATCH_SIZE,
        target_pixel_size_um: float = TARGET_PIXEL_SIZE_UM,
    ) -> None:
        if max_cells_per_sample is not None and max_cells_per_sample < 1:
            raise ValueError("max_cells_per_sample must be positive or None")
        if max_open_wsis < 1:
            raise ValueError("max_open_wsis must be positive")

        self.transform = transform
        self.wsi_opener = wsi_opener
        self.max_open_wsis = int(max_open_wsis)
        self.target_size = int(target_size)
        self.target_pixel_size_um = float(target_pixel_size_um)
        source_patch_size(1.0, self.target_size, self.target_pixel_size_um)
        self._wsi_cache = OrderedDict()

        sample_ids = split_samples(splits_path, split)
        manifest = _manifest_rows(manifest_path)
        missing = [sample_id for sample_id in sample_ids if sample_id not in manifest]
        if missing:
            raise KeyError(f"manifest is missing split samples {missing[:5]}")

        self._samples = []
        self._ends = []
        self.sample_indices = {}
        start = 0
        for sample_id in sample_ids:
            row = manifest[sample_id]
            sample_dir = _sample_output_dir(data_root, sample_id, row)
            cell_ids, terra, centers = _load_sample(
                sample_dir / f"{sample_id}_terra.npz",
                terra_key,
                max_cells_per_sample,
                seed,
                sample_id,
            )
            stop = start + len(cell_ids)
            self.sample_indices[sample_id] = np.arange(start, stop, dtype=int)
            self._samples.append(
                {
                    "sample_id": sample_id,
                    "wsi": str(row["wsi"]),
                    "pixel_size_um": float(row["pixel_size_um"]),
                    "terra": terra,
                    "centers": centers,
                }
            )
            self._ends.append(stop)
            start = stop
        self._length = start

    def __len__(self):
        return self._length

    def _get_wsi(self, sample):
        sample_id = sample["sample_id"]
        if sample_id in self._wsi_cache:
            return self._wsi_cache.pop(sample_id)
        wsi = self.wsi_opener(sample["wsi"])
        if len(self._wsi_cache) == self.max_open_wsis:
            _, old_wsi = self._wsi_cache.popitem(last=False)
            _close_wsi(old_wsi)
        return wsi

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        sample_number = bisect.bisect_right(self._ends, index)
        start = 0 if sample_number == 0 else self._ends[sample_number - 1]
        row = index - start
        sample = self._samples[sample_number]
        wsi = self._get_wsi(sample)
        self._wsi_cache[sample["sample_id"]] = wsi
        center_x, center_y = sample["centers"][row]
        image = crop_cell(
            wsi,
            center_x,
            center_y,
            sample["pixel_size_um"],
            self.target_size,
            self.target_pixel_size_um,
        )
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "terra": torch.from_numpy(sample["terra"][row]),
        }

    def close(self):
        for wsi in self._wsi_cache.values():
            _close_wsi(wsi)
        self._wsi_cache.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_wsi_cache"] = OrderedDict()
        return state

    def __del__(self):
        self.close()


def _close_wsi(wsi):
    close = getattr(getattr(wsi, "store", None), "close", None)
    if close is not None:
        close()


class WithinSampleBatchSampler(Sampler[list[int]]):
    """Yield full single-sample batches in locality-preserving visits."""

    def __init__(
        self,
        dataset: HESTCellDataset,
        batch_size: int,
        seed: int = 0,
        shuffle: bool = True,
        batches_per_sample_visit: int = 32,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if batches_per_sample_visit < 1:
            raise ValueError("batches_per_sample_visit must be positive")
        if not dataset.sample_indices:
            raise ValueError("dataset has no samples")

        self.sample_indices = tuple(dataset.sample_indices.values())
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.batches_per_sample_visit = int(batches_per_sample_visit)
        self.epoch = 0
        smallest_sample = min(map(len, self.sample_indices))
        if smallest_sample < self.batch_size:
            raise ValueError(
                "every sample must contain at least batch_size cells; "
                f"smallest sample has {smallest_sample}, batch_size={batch_size}"
            )
        self.n_batches = sum(
            len(indices) // self.batch_size for indices in self.sample_indices
        )

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        pools = [
            rng.permutation(indices) if self.shuffle else indices
            for indices in self.sample_indices
        ]
        visits = []
        for sample_number, indices in enumerate(pools):
            starts = range(0, len(indices) - self.batch_size + 1, self.batch_size)
            starts = list(starts)
            visit_size = self.batches_per_sample_visit
            visits.extend(
                [
                    (sample_number, start)
                    for start in starts[offset : offset + visit_size]
                ]
                for offset in range(0, len(starts), visit_size)
            )
        if self.shuffle:
            visits = [visits[index] for index in rng.permutation(len(visits))]

        for visit in visits:
            for sample_number, start in visit:
                batch = pools[sample_number][start : start + self.batch_size]
                yield batch.tolist()


def build_datasets(cfg, transform):
    data_root = cfg["data_root"]
    crop = cfg["crop"]
    common = {
        "data_root": data_root,
        "manifest_path": cfg.get(
            "manifest_path", str(Path(data_root) / "manifest.csv")
        ),
        "splits_path": cfg["splits_path"],
        "transform": transform,
        "max_cells_per_sample": cfg["max_cells_per_sample"],
        "seed": int(cfg["sampling_seed"]),
        "terra_key": cfg["terra_key"],
        "target_size": int(crop["target_size"]),
        "target_pixel_size_um": float(crop["target_pixel_size_um"]),
        "max_open_wsis": int(cfg["max_open_wsis"]),
    }
    return (
        HESTCellDataset(split="distill_train", **common),
        HESTCellDataset(split="distill_val", **common),
    )
