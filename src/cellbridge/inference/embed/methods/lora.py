"""LoRA-adapted UNI embedding inference."""

from pathlib import Path

from cellbridge.inference.embed.methods.base import TorchEmbedder, resolve_device

ARMS = ("lora_adapted", "lora_projected")


class LoraEmbedder(TorchEmbedder):
    """Expose the image-derived representations from a LoRA checkpoint."""

    def _embed(self, images) -> dict:
        adapted = self.model.encode_images(images, use_lora=True)
        projected = self.model.apply_heads(adapted)["projected"]
        return {
            "lora_adapted": adapted,
            "lora_projected": projected,
        }


def resolve_checkpoint(
    checkpoint=None, *, repo_id=None, filename="cellbridge.pt", revision=None
) -> Path:
    """Prefer an explicit local adapter; otherwise download into the Hub cache."""
    if checkpoint is not None:
        path = Path(checkpoint).expanduser()
    elif repo_id:
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(repo_id, filename, revision=revision))
    else:
        raise ValueError(
            "CellBridge requires a local checkpoint or Hugging Face repo_id"
        )
    if not path.is_file():
        raise FileNotFoundError(f"CellBridge checkpoint does not exist: {path}")
    return path


def load(
    checkpoint=None,
    *,
    repo_id=None,
    filename="cellbridge.pt",
    revision=None,
    device=None,
    encoder_weights=None,
) -> LoraEmbedder:
    """Load a local or Hugging Face CellBridge adapter for embedding inference."""
    from cellbridge.train.lora.model import load_checkpoint

    device = resolve_device(device)
    loaded = load_checkpoint(
        resolve_checkpoint(
            checkpoint, repo_id=repo_id, filename=filename, revision=revision
        ),
        device=device,
        weights_path=encoder_weights,
    )
    model = loaded[0] if isinstance(loaded, tuple) else loaded
    return LoraEmbedder(model, device)
