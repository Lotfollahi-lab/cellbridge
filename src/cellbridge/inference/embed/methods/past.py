"""PAST image-encoder inference."""

import argparse
from pathlib import Path

from cellbridge.inference.embed.methods.base import TorchEmbedder, resolve_device

ARMS = ("past",)
IMAGE_PREFIX = "image_encoder.model."


class PastEmbedder(TorchEmbedder):
    """Expose PAST's raw 1024-dimensional image-encoder features."""

    def _embed(self, images) -> dict:
        return {"past": self.model(images)}


def load(checkpoint, *, device=None) -> PastEmbedder:
    """Load the image tower from the released PAST checkpoint."""
    import timm
    import torch
    from torchvision import transforms

    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(
            Path(checkpoint),
            map_location="cpu",
            weights_only=True,
        )

    state = checkpoint["model"]
    image_state = {
        key.split(IMAGE_PREFIX, 1)[1]: value
        for key, value in state.items()
        if IMAGE_PREFIX in key
    }
    if not image_state:
        raise ValueError("PAST checkpoint contains no image-encoder weights")

    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True,
    )
    model.load_state_dict(image_state, strict=True)
    model.eval_transforms = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return PastEmbedder(model, resolve_device(device))
