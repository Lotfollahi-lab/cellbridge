"""Prepared Immunocto cell-crop source."""

import os
import struct
from functools import partial
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from cellbridge.inference.embed.methods.volta import patch2cell_crop
from cellbridge.inference.embed.sources import SourceSample
from cellbridge.inference.embed.sources.metadata import load_metadata as load_canonical

_BBOX = struct.Struct(">4I")


def _full_crop(image, _bounds):
    return image


def _instance_crop(image, bounds, *, scale_factor):
    return patch2cell_crop(image, bounds, scale_factor=scale_factor)


def make_cropper(profile, data):
    """Use each native crop whole, except for VOLTA's instance-bbox crop."""
    kind = profile.get("crop", {}).get("kind")
    if kind == "center":
        return _full_crop
    if kind == "instance_bbox":
        scale = float(profile.get("crop", {}).get("scale_factor", 1.0))
        if scale < 0:
            raise ValueError("scale_factor must be non-negative")
        return partial(_instance_crop, scale_factor=scale)
    raise ValueError(f"unknown crop kind {kind!r}")


def _validate_metadata(metadata, data):
    required = {"store_key", "x0", "y0", "x1", "y1"}
    if not required <= set(metadata):
        raise ValueError(f"metadata is missing {sorted(required - set(metadata))}")
    if metadata[list(required)].isna().any().any():
        raise ValueError("metadata contains null crop fields")
    if metadata.duplicated(["sample_id", "store_key"]).any():
        raise ValueError("metadata contains duplicate sample store keys")

    bounds = metadata[["x0", "y0", "x1", "y1"]].to_numpy(dtype=np.int64)
    shape = tuple(data.get("image_shape", (64, 64)))
    if len(shape) != 2:
        raise ValueError("data.image_shape must contain height and width")
    height, width = map(int, shape)
    valid = (
        (bounds[:, 0] >= 0)
        & (bounds[:, 1] >= 0)
        & (bounds[:, 2] > bounds[:, 0])
        & (bounds[:, 3] > bounds[:, 1])
        & (bounds[:, 2] <= width)
        & (bounds[:, 3] <= height)
    )
    if not valid.all():
        raise ValueError("metadata contains invalid crop bounds")
    return metadata


def load_metadata(data):
    """Load prepared, row-aligned Immunocto metadata."""
    metadata = load_canonical(data["metadata"], data["split"])
    return _validate_metadata(metadata, data)


def _shard_path(data, sample_id):
    return Path(data["shards"]) / f"{sample_id}.lmdb"


def check(data):
    """Validate prepared paths, metadata, and patient shard coverage."""
    for key in ("metadata", "split", "shards"):
        if not Path(data[key]).exists():
            raise FileNotFoundError(f"data.{key} does not exist: {data[key]}")
    metadata = load_metadata(data)
    missing = [
        str(_shard_path(data, sample_id))
        for sample_id in metadata["sample_id"].astype(str).unique()
        if not _shard_path(data, sample_id).exists()
    ]
    if missing:
        raise FileNotFoundError(f"missing Immunocto shards: {missing[:5]}")
    return metadata


class ImmunoctoCrops:
    """Lazily read one patient shard without sharing LMDB handles across workers."""

    def __init__(self, path, cells, *, transform, cropper):
        self.path = Path(path)
        self.cells = cells.reset_index(drop=True)
        self.transform = transform
        self.cropper = cropper
        self._env = None
        self._pid = None

    def __len__(self):
        return len(self.cells)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"] = None
        state["_pid"] = None
        return state

    def _environment(self):
        pid = os.getpid()
        if self._env is not None and self._pid == pid:
            return self._env
        if self._env is not None:
            self._env.close()
        try:
            import lmdb
        except ImportError as error:
            raise ImportError("reading Immunocto requires the lmdb package") from error
        self._env = lmdb.open(
            str(self.path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self._pid = pid
        return self._env

    def __getitem__(self, index):
        if not -len(self) <= index < len(self):
            raise IndexError(index)
        row = self.cells.iloc[index % len(self)]
        key = str(row["store_key"]).encode()
        with self._environment().begin(write=False) as transaction:
            png = transaction.get(b"I:" + key)
            packed_bounds = transaction.get(b"B:" + key)
        if png is None or packed_bounds is None:
            raise KeyError(
                f"{row['sample_id']}: missing prepared crop {row['store_key']}"
            )

        bounds = _BBOX.unpack(packed_bounds)
        recorded = tuple(int(row[name]) for name in ("x0", "y0", "x1", "y1"))
        if bounds != recorded:
            raise ValueError(
                f"{row['sample_id']}: stored bounds disagree with metadata"
            )
        image = Image.open(BytesIO(png)).convert("RGB")
        image = self.cropper(image, bounds)
        if not isinstance(image, Image.Image):
            raise TypeError("cropper must return a PIL image")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "sample_id": str(row["sample_id"]),
            "cell_id": str(row["cell_uid"]),
        }

    def __del__(self):
        if self._env is not None:
            self._env.close()


def iter_samples(data, *, transform, cropper):
    """Yield one lazy crop dataset per patient in metadata order."""
    metadata = load_metadata(data)
    for sample_id, cells in metadata.groupby("sample_id", sort=False):
        path = _shard_path(data, sample_id)
        if not path.exists():
            raise FileNotFoundError(f"missing Immunocto shard: {path}")
        yield SourceSample(
            sample_id=str(sample_id),
            row_ids=cells["row_id"].to_numpy(),
            dataset=ImmunoctoCrops(
                path,
                cells,
                transform=transform,
                cropper=cropper,
            ),
        )
