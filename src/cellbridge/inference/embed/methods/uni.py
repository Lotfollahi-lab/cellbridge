"""Frozen UNI embedding inference."""

from cellbridge.inference.embed.methods.base import TorchEmbedder, resolve_device

ARMS = ("uni",)


class UniEmbedder(TorchEmbedder):
    """Small adapter around a frozen Trident patch encoder."""

    def _embed(self, images) -> dict:
        return {"uni": self.model(images)}


def load(encoder="uni_v2", *, device=None) -> UniEmbedder:
    """Load a frozen Trident encoder for embedding inference."""
    from trident.patch_encoder_models import encoder_factory

    return UniEmbedder(encoder_factory(encoder), resolve_device(device))
