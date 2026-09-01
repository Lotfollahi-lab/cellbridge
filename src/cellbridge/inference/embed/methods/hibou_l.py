"""Frozen Hibou-L embedding inference."""

from cellbridge.inference.embed.methods.base import TorchEmbedder, resolve_device

ARMS = ("hibou_l",)


class _ImageProcessorTransform:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, image):
        return self.processor(images=image, return_tensors="pt")["pixel_values"][0]


class HibouLEmbedder(TorchEmbedder):
    """Expose Hibou-L's official pooled representation."""

    def _embed(self, images) -> dict:
        output = self.model(pixel_values=images)
        return {"hibou_l": output.pooler_output}


def load(
    hf_repo="histai/hibou-L",
    revision=None,
    *,
    device=None,
) -> HibouLEmbedder:
    """Load Hibou-L and its image processor from Hugging Face."""
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(
        hf_repo,
        revision=revision,
        trust_remote_code=True,
        use_fast=False,
    )
    model = AutoModel.from_pretrained(
        hf_repo, revision=revision, trust_remote_code=True
    )
    model.eval_transforms = _ImageProcessorTransform(processor)
    return HibouLEmbedder(model, resolve_device(device))
