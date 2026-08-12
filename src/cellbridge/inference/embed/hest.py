"""HEST cells and image crops for embedding inference."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image

IMAGE_SIZE = 224
TARGET_MPP = 0.5


def _identifier(value) -> str:
    """Normalize identifiers that CSV readers may interpret as numbers."""
    if isinstance(value, (bytes, np.bytes_)):
        value = bytes(value).decode()
    text = str(value)
    if re.fullmatch(r"[+-]?(?:0|[1-9]\d*)\.0+", text):
        return str(int(text.split(".", 1)[0]))
    return text


def open_wsi(path):
    """Open the level-0 image as a lazy zarr array."""
    import tifffile
    import zarr

    try:
        store = tifffile.imread(path, aszarr=True, level=0)
    except Exception:
        store = tifffile.imread(path, aszarr=True, key=0)
    return zarr.open(store, mode="r")


def close_wsi(wsi) -> None:
    close = getattr(getattr(wsi, "store", None), "close", None)
    if close is not None:
        close()


@dataclass(frozen=True)
class HestSample:
    """Ordered cells and image metadata for one HEST sample."""

    sample_id: str
    wsi_path: Path
    pixel_size_um: float
    cell_ids: np.ndarray
    centers: np.ndarray


def load_sample(
    manifest_path,
    data_root,
    sample_id,
    *,
    max_cells: int | None = None,
) -> HestSample:
    """Load one sample's cell IDs and level-0 H&E centers in artifact order."""
    sample_id = _identifier(sample_id)
    manifest_path = Path(manifest_path)
    manifest = pd.read_csv(manifest_path)
    required = {"id", "wsi", "pixel_size_um"}
    if not required <= set(manifest):
        raise ValueError(f"{manifest_path} is missing {sorted(required - set(manifest))}")

    ids = manifest["id"].map(_identifier)
    matches = manifest.loc[ids == sample_id]
    if len(matches) != 1:
        raise ValueError(f"expected one manifest row for {sample_id!r}, found {len(matches)}")
    row = matches.iloc[0]

    directory = row.get("output_dir")
    directory = (
        Path(data_root) / sample_id
        if directory is None or pd.isna(directory)
        else Path(directory)
    )
    cells_path = directory / f"{sample_id}_terra.npz"
    with np.load(cells_path, allow_pickle=True) as data:
        required = {"cell_ids", "he_x", "he_y"}
        if not required <= set(data.files):
            raise ValueError(f"{cells_path} is missing {sorted(required - set(data.files))}")
        cell_ids = np.asarray([_identifier(value) for value in data["cell_ids"]])
        centers = np.column_stack((data["he_x"], data["he_y"])).astype(np.float64)

    if len(np.unique(cell_ids)) != len(cell_ids):
        raise ValueError(f"{cells_path} contains duplicate cell IDs")
    if centers.shape != (len(cell_ids), 2) or not np.isfinite(centers).all():
        raise ValueError(f"{cells_path} contains invalid H&E centers")
    if max_cells is not None:
        if max_cells < 1:
            raise ValueError("max_cells must be positive")
        cell_ids, centers = cell_ids[:max_cells], centers[:max_cells]

    pixel_size_um = float(row["pixel_size_um"])
    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError(f"{sample_id}: pixel_size_um must be positive")
    return HestSample(
        sample_id=sample_id,
        wsi_path=Path(row["wsi"]),
        pixel_size_um=pixel_size_um,
        cell_ids=cell_ids,
        centers=centers,
    )


def center_crop(
    wsi,
    sample: HestSample,
    index: int,
    *,
    image_size: int = IMAGE_SIZE,
    target_mpp: float = TARGET_MPP,
    fill: int = 255,
) -> Image.Image:
    """Return a centered, physically standardized RGB crop for one cell."""
    source_size = round(image_size * target_mpp / sample.pixel_size_um)
    if source_size < 1:
        raise ValueError("crop dimensions must be positive")
    x, y = sample.centers[index]
    x0, y0 = round(x) - source_size // 2, round(y) - source_size // 2
    x1, y1 = x0 + source_size, y0 + source_size
    height, width = wsi.shape[:2]
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x1, width), min(y1, height)

    crop = np.full((source_size, source_size, 3), fill, dtype=np.uint8)
    if sx1 > sx0 and sy1 > sy0:
        pixels = np.asarray(wsi[sy0:sy1, sx0:sx1])
        if pixels.ndim == 2:
            pixels = np.repeat(pixels[..., None], 3, axis=-1)
        crop[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = pixels[..., :3]
    return Image.fromarray(crop).resize((image_size, image_size), Image.Resampling.BILINEAR)


Cropper = Callable[[object, HestSample, int], Image.Image]


class HestCrops:
    """Lazy map-style dataset of image crops from one HEST sample."""

    def __init__(
        self,
        sample: HestSample,
        transform: Callable | None = None,
        cropper: Cropper | None = None,
        *,
        image_size: int = IMAGE_SIZE,
        target_mpp: float = TARGET_MPP,
    ):
        self.sample = sample
        self.transform = transform
        self.cropper = cropper or partial(
            center_crop, image_size=image_size, target_mpp=target_mpp
        )
        self._wsi = None

    def __len__(self) -> int:
        return len(self.sample.cell_ids)

    def __getitem__(self, index: int) -> dict:
        if not -len(self) <= index < len(self):
            raise IndexError(index)
        index %= len(self)
        if self._wsi is None:
            self._wsi = open_wsi(self.sample.wsi_path)
        image = self.cropper(self._wsi, self.sample, index)
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "sample_id": self.sample.sample_id,
            "cell_id": self.sample.cell_ids[index],
        }

    def close(self) -> None:
        if self._wsi is not None:
            close_wsi(self._wsi)
            self._wsi = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_wsi"] = None
        return state

    def __del__(self):
        self.close()
