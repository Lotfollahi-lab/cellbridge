"""Adapter for the existing tiled-LMDB OOD datasets."""

from pathlib import Path

from cellbridge.inference.embed.crops import spatial_cropper
from cellbridge.inference.embed.ood import OodCrops
from cellbridge.inference.embed.ood import iter_samples as iter_ood_samples
from cellbridge.inference.embed.ood import load_metadata as load_ood_metadata
from cellbridge.inference.embed.sources import SourceSample

make_cropper = spatial_cropper


def _arguments(data):
    return {
        "lmdb_path": data["lmdb"],
        "metadata_path": data["metadata"],
        "split_path": data["split"],
        "image_shape": data.get("image_shape"),
        "tile_size": int(data.get("tile_size", 224)),
        "value_scale": float(data.get("value_scale", 1)),
        "allow_unlabeled_samples": bool(data.get("allow_unlabeled_samples", False)),
        "skip_missing": bool(data.get("skip_missing_cells", False)),
    }


def load_metadata(data):
    """Load normalized metadata for a tiled-LMDB source."""
    return load_ood_metadata(
        data["metadata"],
        data["split"],
        require_all_samples=not data.get("allow_unlabeled_samples", False),
    )


def check(data):
    """Validate source paths and return normalized metadata."""
    for key in ("lmdb", "metadata", "split"):
        if not Path(data[key]).exists():
            raise FileNotFoundError(f"data.{key} does not exist: {data[key]}")
    return load_metadata(data)


def iter_samples(data, *, transform, cropper):
    """Yield aligned crop datasets in existing LMDB sample order."""
    for sample in iter_ood_samples(**_arguments(data)):
        yield SourceSample(
            sample_id=sample.sample_id,
            row_ids=sample.cells["row_id"].to_numpy(),
            dataset=OodCrops(sample, transform=transform, cropper=cropper),
        )
