"""PAST's native-pixel, cell-centred image crop."""

from __future__ import annotations

import numpy as np
from PIL import Image


def native_center_crop(image, x, y, size=224, fill=0):
    """Crop native level-0 pixels and pad boundaries like PAST's dataset."""
    size = int(size)
    if size < 1:
        raise ValueError("crop size must be positive")
    height, width = image.shape[:2]
    x0, y0 = int(x) - size // 2, int(y) - size // 2
    x1, y1 = x0 + size, y0 + size
    source_x0, source_y0 = max(x0, 0), max(y0, 0)
    source_x1, source_y1 = min(x1, width), min(y1, height)
    crop = np.full((size, size, 3), fill, dtype=np.uint8)
    if source_x1 > source_x0 and source_y1 > source_y0:
        region = np.asarray(image[source_y0:source_y1, source_x0:source_x1])
        if region.ndim == 2:
            region = np.repeat(region[..., None], 3, axis=-1)
        crop[
            source_y0 - y0 : source_y1 - y0,
            source_x0 - x0 : source_x1 - x0,
        ] = region[..., :3]
    return Image.fromarray(crop)
