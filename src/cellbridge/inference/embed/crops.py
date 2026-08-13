"""Resolve method-specific crops from located cells."""

from functools import partial

from PIL import Image

from cellbridge.inference.embed.methods.volta import patch2cell_crop
from cellbridge.inference.embed.ood import center_cell_crop


def _center_crop(image, cell, *, source_size, target_size, fill):
    crop = center_cell_crop(image, cell, size=source_size, fill=fill)
    if target_size is not None and crop.size != (target_size, target_size):
        crop = crop.resize((target_size, target_size), Image.Resampling.BILINEAR)
    return crop


def _instance_crop(image, cell, *, scale_factor):
    bounds = [cell[name] for name in ("x0", "y0", "x1", "y1")]
    return patch2cell_crop(image, bounds, scale_factor=scale_factor)


def spatial_cropper(profile, data):
    """Build a cropper for a source with cells located in larger images."""
    settings = profile.get("crop", {})
    kind = settings.get("kind")
    if kind == "center":
        has_pixels = "source_size" in settings
        has_fov = "fov_um" in settings
        if has_pixels == has_fov:
            raise ValueError(
                "center crop requires exactly one of source_size or fov_um"
            )
        if has_fov:
            source_mpp = float(data.get("source_mpp", 0))
            if source_mpp <= 0:
                raise ValueError("physical crops require positive data.source_mpp")
            source_size = round(float(settings["fov_um"]) / source_mpp)
        else:
            source_size = int(settings["source_size"])
        target_size = settings.get("target_size")
        target_size = None if target_size is None else int(target_size)
        fill = int(settings.get("fill", 255))
        if source_size < 1 or (target_size is not None and target_size < 1):
            raise ValueError("crop sizes must be positive")
        if not 0 <= fill <= 255:
            raise ValueError("crop fill must be between 0 and 255")
        return partial(
            _center_crop, source_size=source_size, target_size=target_size, fill=fill
        )
    if kind == "instance_bbox":
        scale_factor = float(settings.get("scale_factor", 1.0))
        if scale_factor < 0:
            raise ValueError("scale_factor must be non-negative")
        return partial(_instance_crop, scale_factor=scale_factor)
    raise ValueError(f"unknown crop kind {kind!r}")
