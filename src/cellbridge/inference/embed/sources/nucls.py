"""Adapter for prepared full-FOV NuCLS images and nucleus metadata."""

from pathlib import Path

import numpy as np
from PIL import Image

from cellbridge.inference.embed.crops import spatial_cropper
from cellbridge.inference.embed.sources import SourceSample
from cellbridge.inference.embed.sources.metadata import load_metadata as load_canonical

make_cropper = spatial_cropper
GEOMETRY_COLUMNS = ("x", "y", "x0", "y0", "x1", "y1")


def load_metadata(data):
    """Load prepared NuCLS cells, excluding configured non-target labels."""
    metadata = load_canonical(
        data["metadata"],
        data["split"],
        exclude_labels=data.get("exclude_labels", ()),
    )
    required = {"image_name", *GEOMETRY_COLUMNS}
    if not required <= set(metadata):
        raise ValueError(
            f"NuCLS metadata is missing {sorted(required - set(metadata))}"
        )
    geometry = metadata[list(GEOMETRY_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(geometry).all():
        raise ValueError("NuCLS metadata contains non-finite geometry")
    if ((metadata["x1"] <= metadata["x0"]) | (metadata["y1"] <= metadata["y0"])).any():
        raise ValueError("NuCLS metadata contains invalid bounding boxes")
    return metadata


def _validate_support(metadata) -> None:
    train = set(metadata.loc[metadata["split"] == "train", "label"].astype(str))
    for split in ("val", "test"):
        missing = (
            set(metadata.loc[metadata["split"] == split, "label"].astype(str)) - train
        )
        if missing:
            raise ValueError(
                f"NuCLS raw classes in {split} are absent from train: {sorted(missing)}"
            )
    if "slide_id" in metadata:
        leaked = metadata.groupby("slide_id")["split"].nunique()
        if (leaked > 1).any():
            raise ValueError("NuCLS slide leakage across splits")


def _image_path(images: Path, image_name: str) -> Path:
    name = Path(str(image_name))
    if name.name != str(image_name):
        raise ValueError(f"unsafe NuCLS image name: {image_name!r}")
    return images / name


def check(data):
    """Validate prepared paths, images, raw-label support, and split isolation."""
    for key in ("images", "metadata", "split"):
        if not Path(data[key]).exists():
            raise FileNotFoundError(f"data.{key} does not exist: {data[key]}")
    metadata = load_metadata(data)
    _validate_support(metadata)
    images = Path(data["images"])
    missing = {
        name
        for name in metadata["image_name"].astype(str).unique()
        if not _image_path(images, name).is_file()
    }
    if missing:
        raise FileNotFoundError(f"NuCLS images are missing: {sorted(missing)[:5]}")
    return metadata


class NuCLSCrops:
    """Method-specific crops from one in-memory NuCLS FOV."""

    def __init__(self, image, cells, transform, cropper):
        self.image = image
        self.cells = cells.reset_index(drop=True)
        self.transform = transform
        self.cropper = cropper

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, index):
        if not -len(self) <= index < len(self):
            raise IndexError(index)
        row = self.cells.iloc[index % len(self)]
        image = self.cropper(self.image, row)
        if not isinstance(image, Image.Image):
            raise TypeError("cropper must return a PIL image")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "sample_id": str(row["sample_id"]),
            "cell_id": str(row["cell_uid"]),
        }


def iter_samples(data, *, transform, cropper):
    """Yield one aligned crop dataset per prepared NuCLS FOV."""
    metadata = load_metadata(data)
    images = Path(data["images"])
    for sample_id, cells in metadata.groupby("sample_id", sort=False):
        names = cells["image_name"].astype(str).unique()
        if len(names) != 1:
            raise ValueError(f"{sample_id}: expected exactly one NuCLS image")
        path = _image_path(images, names[0])
        with Image.open(path) as source:
            image = np.asarray(source.convert("RGB"))
        yield SourceSample(
            sample_id=str(sample_id),
            row_ids=cells["row_id"].to_numpy(),
            dataset=NuCLSCrops(image, cells, transform, cropper),
        )
