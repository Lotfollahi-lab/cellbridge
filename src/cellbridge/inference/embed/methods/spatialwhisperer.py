"""SpatialWhisperer image inference."""

from cellbridge.inference.embed.methods.base import TorchEmbedder, resolve_device

ARMS = ("spatialwhisperer",)


class SpatialWhispererEmbedder(TorchEmbedder):
    """Expose normalized image embeddings in SpatialWhisperer's shared space."""

    def _embed(self, images) -> dict:
        import torch

        _, embeddings = self.model.get_image_features(
            patches_ctx=images,
            normalize_embeds=True,
        )
        if embeddings.ndim != 2 or not torch.isfinite(embeddings).all():
            raise ValueError("SpatialWhisperer returned invalid image embeddings")
        return {"spatialwhisperer": embeddings}


def load(
    hf_repo="Good-Lab/spatialwhisperer",
    *,
    device=None,
) -> SpatialWhispererEmbedder:
    """Load the released context-only SpatialWhisperer model."""
    import torch
    from spatialwhisperer import load_spatialwhisperer_model

    device = resolve_device(device)
    pl_model, _, _, image_processor = load_spatialwhisperer_model(
        hf_repo=hf_repo,
        device=device,
    )
    model = pl_model.model
    image_config = model.image_model.config
    if not image_config.context_model or image_config.cell_level_model:
        raise ValueError("expected the released context-only image model")

    model.eval_transforms = image_processor.transform
    model.precision = torch.float32
    return SpatialWhispererEmbedder(model, device)
