"""Shared lifecycle for torch embedding models."""


def resolve_device(device=None):
    import torch

    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


class TorchEmbedder:
    """Handle device placement and inference context for a patch encoder."""

    def __init__(self, model, device=None):
        import torch

        self.device = resolve_device(device)
        self.model = model.to(self.device).eval()
        self.transform = self.model.eval_transforms
        self.precision = getattr(self.model, "precision", torch.float32)
        if self.transform is None:
            raise ValueError("the image encoder does not expose an evaluation transform")

    def embed(self, images) -> dict:
        import torch

        images = images.to(self.device, non_blocking=self.device.type == "cuda")
        use_autocast = self.device.type == "cuda" and self.precision in {
            torch.float16,
            torch.bfloat16,
        }
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=self.precision,
            enabled=use_autocast,
        ):
            return self._embed(images)

    def _embed(self, images) -> dict:
        raise NotImplementedError
