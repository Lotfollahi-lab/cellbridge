"""LoRA-adapted UNI embedding inference."""

from pathlib import Path

from cellbridge.inference.embed.methods.base import TorchEmbedder, resolve_device

ARMS = ("uni_original", "lora_adapted", "lora_projected")


class LoraEmbedder(TorchEmbedder):
    """Expose the image-derived representations from a LoRA checkpoint."""

    def _embed(self, images) -> dict:
        adapted = self.model.encode_images(images, use_lora=True)
        projected = self.model.apply_heads(adapted)["projected"]
        return {
            "uni_original": self.model.encode_base(images),
            "lora_adapted": adapted,
            "lora_projected": self.model.probe_projection(projected),
        }


def load(checkpoint, *, device=None, encoder_weights=None) -> LoraEmbedder:
    """Load one trained CellBridge checkpoint for embedding inference."""
    from cellbridge.lora.model import load_checkpoint

    device = resolve_device(device)
    loaded = load_checkpoint(
        Path(checkpoint),
        device=device,
        weights_path=encoder_weights,
    )
    model = loaded[0] if isinstance(loaded, tuple) else loaded
    return LoraEmbedder(model, device)
