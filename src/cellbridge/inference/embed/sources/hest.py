"""Adapter for the fixed, held-out HEST evaluation cohort."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image

from cellbridge.helpers.cohorts import cohort_in_split
from cellbridge.inference.embed.sources import SourceSample

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
    bounds: np.ndarray | None = None


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
        raise ValueError(
            f"{manifest_path} is missing {sorted(required - set(manifest))}"
        )

    ids = manifest["id"].map(_identifier)
    matches = manifest.loc[ids == sample_id]
    if len(matches) != 1:
        raise ValueError(
            f"expected one manifest row for {sample_id!r}, found {len(matches)}"
        )
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
            raise ValueError(
                f"{cells_path} is missing {sorted(required - set(data.files))}"
            )
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
    return Image.fromarray(crop).resize(
        (image_size, image_size),
        Image.Resampling.BILINEAR,
    )


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
            center_crop,
            image_size=image_size,
            target_mpp=target_mpp,
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


def sample_ids(data) -> list[str]:
    """Return the configured cohort after proving that it is held out."""
    cohort = cohort_in_split(
        data["cohort"],
        data["split"],
        data.get("cohort_split", "label_test"),
        expected_count=data.get("expected_samples"),
    )
    selected = data.get("sample_ids")
    if selected is None:
        return cohort
    if not isinstance(selected, list) or not selected:
        raise ValueError("data.sample_ids must be a non-empty list")
    selected = [_identifier(sample_id) for sample_id in selected]
    if len(selected) != len(set(selected)):
        raise ValueError("data.sample_ids contains duplicates")
    unknown = sorted(set(selected) - set(cohort))
    if unknown:
        raise ValueError(f"data.sample_ids contains non-cohort samples: {unknown}")
    return selected


def _manifest_rows(data) -> dict[str, dict]:
    path = Path(data["manifest"])
    table = pd.read_csv(path, dtype={"id": str})
    required = {"id", "wsi", "pixel_size_um", "xenium_cell_seg"}
    if not required <= set(table):
        raise ValueError(f"{path} is missing {sorted(required - set(table))}")
    if table["id"].isna().any() or table["id"].duplicated().any():
        raise ValueError(f"{path} contains invalid sample IDs")
    return {str(row["id"]): row for row in table.to_dict("records")}


def _sample_directory(data, sample_id, row) -> Path:
    directory = row.get("output_dir")
    if directory is None or pd.isna(directory):
        return Path(data["root"]) / sample_id
    return Path(directory)


def evaluation_paths(data, sample_id) -> tuple[Path, Path]:
    """Return the labels and spatial coordinates for one HEST sample."""
    sample_id = str(sample_id)
    rows = _manifest_rows(data)
    if sample_id not in rows:
        raise KeyError(f"manifest is missing HEST sample {sample_id!r}")
    directory = _sample_directory(data, sample_id, rows[sample_id])
    return (
        directory / f"{sample_id}_granular_cell_labels.csv",
        directory / f"{sample_id}.h5ad",
    )


def _cell_uids(sample_id, cell_ids) -> np.ndarray:
    return np.asarray([f"{sample_id}/{cell_id}" for cell_id in cell_ids], dtype=str)


def _metadata_frame(sample: HestSample) -> pd.DataFrame:
    cell_ids = sample.cell_ids.astype(str)
    uids = _cell_uids(sample.sample_id, cell_ids)
    return pd.DataFrame(
        {
            "row_id": uids,
            "cell_uid": uids,
            "sample_id": sample.sample_id,
            "cell_id": cell_ids,
        }
    )


def sample_metadata(data, sample_id) -> pd.DataFrame:
    """Load row alignment metadata for one HEST sample."""
    return _metadata_frame(load_sample(data["manifest"], data["root"], sample_id))


def load_metadata(data) -> pd.DataFrame:
    """Load globally unique, row-aligned metadata for the configured cohort."""
    return pd.concat(
        [sample_metadata(data, sample_id) for sample_id in sample_ids(data)],
        ignore_index=True,
    )


def check(data):
    """Validate HEST cohort inputs without opening images or geometries."""
    for key in ("root", "manifest", "split", "cohort"):
        if not Path(data[key]).exists():
            raise FileNotFoundError(f"data.{key} does not exist: {data[key]}")

    rows = _manifest_rows(data)
    metadata = []
    for sample_id in sample_ids(data):
        if sample_id not in rows:
            raise KeyError(f"manifest is missing HEST sample {sample_id!r}")
        row = rows[sample_id]
        sample = load_sample(data["manifest"], data["root"], sample_id)
        labels, coordinates = evaluation_paths(data, sample_id)
        paths = (sample.wsi_path, Path(row["xenium_cell_seg"]), labels, coordinates)
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{sample_id} is missing inputs: {missing}")
        metadata.append(_metadata_frame(sample))
    return pd.concat(metadata, ignore_index=True)


def _center_crop(wsi, sample, index, *, size, target_mpp, fill):
    target_mpp = sample.pixel_size_um if target_mpp is None else target_mpp
    return center_crop(
        wsi,
        sample,
        index,
        image_size=size,
        target_mpp=target_mpp,
        fill=fill,
    )


@dataclass(frozen=True)
class _InstanceCropper:
    scale_factor: float
    fill: int

    def __call__(self, wsi, sample, index):
        if sample.bounds is None:
            raise ValueError("HEST instance crops require cell bounds")
        x0, y0, x1, y1 = sample.bounds[index]
        side = max(1, int(np.ceil(max(x1 - x0, y1 - y0) * (1 + self.scale_factor))))
        left = round((x0 + x1 - side) / 2)
        top = round((y0 + y1 - side) / 2)
        right, bottom = left + side, top + side
        height, width = wsi.shape[:2]
        sx0, sy0 = max(left, 0), max(top, 0)
        sx1, sy1 = min(right, width), min(bottom, height)

        crop = np.full((side, side, 3), self.fill, dtype=np.uint8)
        if sx1 > sx0 and sy1 > sy0:
            pixels = np.asarray(wsi[sy0:sy1, sx0:sx1])
            if pixels.ndim == 2:
                pixels = np.repeat(pixels[..., None], 3, axis=-1)
            crop[sy0 - top : sy1 - top, sx0 - left : sx1 - left] = pixels[..., :3]
        return Image.fromarray(crop)


def make_cropper(profile, data):
    """Build the method-specific crop used for HEST."""
    settings = profile.get("crop", {})
    kind = settings.get("kind")
    fill = int(settings.get("fill", 255))
    if not 0 <= fill <= 255:
        raise ValueError("crop fill must be between 0 and 255")

    if kind == "center":
        size = int(settings.get("source_size", 0))
        target_mpp = settings.get("target_mpp")
        target_mpp = None if target_mpp is None else float(target_mpp)
        if size < 1 or (target_mpp is not None and target_mpp <= 0):
            raise ValueError("HEST center crop dimensions must be positive")
        return partial(
            _center_crop,
            size=size,
            target_mpp=target_mpp,
            fill=fill,
        )
    if kind == "instance_bbox":
        scale_factor = float(settings.get("scale_factor", 1.0))
        if scale_factor < 0:
            raise ValueError("scale_factor must be non-negative")
        return _InstanceCropper(scale_factor, fill)
    raise ValueError(f"unknown crop kind {kind!r}")


def _cell_bounds(path, cell_ids) -> np.ndarray:
    import geopandas as gpd

    cells = gpd.read_parquet(path)
    bounds = cells.geometry.bounds
    bounds.index = cells.index.astype(str)
    bounds = bounds.groupby(level=0, sort=False).agg(
        {"minx": "min", "miny": "min", "maxx": "max", "maxy": "max"}
    )
    positions = bounds.index.get_indexer(np.asarray(cell_ids).astype(str))
    if (positions < 0).any():
        missing = np.asarray(cell_ids)[positions < 0]
        raise ValueError(f"{path} is missing HEST cells {missing[:5].tolist()}")
    values = bounds.iloc[positions].to_numpy(dtype=np.float64)
    valid = (
        np.isfinite(values).all(axis=1)
        & (values[:, 2] > values[:, 0])
        & (values[:, 3] > values[:, 1])
    )
    if not valid.all():
        raise ValueError(f"{path} contains invalid HEST cell bounds")
    return values


def iter_samples(data, *, transform, cropper):
    """Yield one lazy crop dataset per configured HEST sample."""
    rows = _manifest_rows(data)
    for sample_id in sample_ids(data):
        sample = load_sample(data["manifest"], data["root"], sample_id)
        raw_cell_ids = sample.cell_ids
        if isinstance(cropper, _InstanceCropper):
            bounds = _cell_bounds(rows[sample_id]["xenium_cell_seg"], raw_cell_ids)
            sample = replace(sample, bounds=bounds)
        uids = _cell_uids(sample.sample_id, raw_cell_ids)
        sample = replace(sample, cell_ids=uids)
        yield SourceSample(
            sample_id=sample.sample_id,
            row_ids=uids,
            dataset=HestCrops(sample, transform=transform, cropper=cropper),
        )
