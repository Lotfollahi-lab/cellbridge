"""VOLTA ovarian momentum-encoder inference."""

import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image

from cellbridge.inference.embed.methods.base import TorchEmbedder, resolve_device

ARMS = ("volta",)
INPUT_SIZE = 32
EMBEDDING_DIM = 512
MEAN = np.asarray((0.7079, 0.6259, 0.7568), dtype=np.float32)
STD = np.asarray((0.1596, 0.1681, 0.1207), dtype=np.float32)


def patch2cell_crop(image, bounds, scale_factor=1.0) -> Image.Image:
    """Crop one instance using VOLTA's patch2cell converter geometry."""
    x0, y0, x1, y1 = map(float, bounds)
    if not np.isfinite((x0, y0, x1, y1)).all() or x1 <= x0 or y1 <= y0:
        raise ValueError(f"invalid cell bounds {bounds}")
    if scale_factor < 0:
        raise ValueError("scale_factor must be non-negative")

    if isinstance(image, Image.Image):
        width, height = image.size
    else:
        height, width = image.shape[:2]

    offset = float(scale_factor) * max(x1 - x0, y1 - y0) / 2
    left = int(max(x0 - offset, 0))
    top = int(max(y0 - offset, 0))
    right = int(min(x1 + offset, width))
    bottom = int(min(y1 + offset, height))

    if isinstance(image, Image.Image):
        return image.crop((left, top, right, bottom))
    crop = np.asarray(image[top:bottom, left:right])
    if crop.ndim == 2:
        crop = np.repeat(crop[..., None], 3, axis=-1)
    return Image.fromarray(crop[..., :3].astype(np.uint8))


def _transform(image):
    import cv2
    import torch

    image = Image.fromarray(np.asarray(image)).convert("RGB")
    values = cv2.resize(
        np.asarray(image),
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=cv2.INTER_CUBIC,
    ).astype(np.float32)
    values = (values / 255.0 - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(values.transpose(2, 0, 1)))


def _backbone(repository):
    path = Path(repository) / "backbones" / "resnet" / "model.py"
    spec = importlib.util.spec_from_file_location("volta_resnet", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import VOLTA backbone from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _teacher_state(checkpoint):
    weights = {}
    for name, value in checkpoint["state_dict"].items():
        name = str(name).removeprefix("module.")
        if name.startswith("encoder_k."):
            weights[name.removeprefix("encoder_k.")] = value
    if not weights:
        raise ValueError("VOLTA checkpoint contains no momentum-encoder weights")
    return weights


class VoltaEmbedder(TorchEmbedder):
    """Expose VOLTA's raw 512-dimensional momentum-encoder features."""

    def _embed(self, images) -> dict:
        features = self.model(images)
        if features.ndim != 2 or features.shape[1] != EMBEDDING_DIM:
            raise ValueError(f"expected (N, {EMBEDDING_DIM}) VOLTA features")
        return {"volta": features}


def load(checkpoint, repository, *, device=None) -> VoltaEmbedder:
    """Load the released ovarian momentum encoder."""
    import torch

    checkpoint = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if checkpoint.get("arch") != "preact_resnet18":
        raise ValueError(f"unexpected VOLTA architecture: {checkpoint.get('arch')!r}")

    model = _backbone(repository).preact_resnet18(num_classes=64)
    model.fc = torch.nn.Identity()
    model.load_state_dict(_teacher_state(checkpoint), strict=True)
    model.eval_transforms = _transform
    return VoltaEmbedder(model, resolve_device(device))
