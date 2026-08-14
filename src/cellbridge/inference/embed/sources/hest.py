"""Adapter for the fixed, held-out HEST evaluation cohort."""

from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from cellbridge.helpers.cohorts import cohort_in_split
from cellbridge.inference.embed.hest import HestCrops, center_crop, load_sample
from cellbridge.inference.embed.sources import SourceSample


def sample_ids(data) -> list[str]:
    """Return the configured cohort after proving that it is held out."""
    return cohort_in_split(
        data["cohort"],
        data["split"],
        data.get("cohort_split", "label_test"),
        expected_count=data.get("expected_samples"),
    )


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


def sample_metadata(data, sample_id) -> pd.DataFrame:
    """Load row alignment metadata for one HEST sample."""
    sample = load_sample(data["manifest"], data["root"], sample_id)
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
        cell_ids = sample.cell_ids.astype(str)
        uids = _cell_uids(sample.sample_id, cell_ids)
        metadata.append(
            pd.DataFrame(
                {
                    "row_id": uids,
                    "cell_uid": uids,
                    "sample_id": sample.sample_id,
                    "cell_id": cell_ids,
                }
            )
        )
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
