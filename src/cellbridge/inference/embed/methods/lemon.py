"""LEMON single-cell image-encoder inference."""

from functools import partial

from cellbridge.inference.embed.methods.base import TorchEmbedder, resolve_device

ARMS = ("lemon",)
ARCHITECTURES = {"vits8": 384, "vitb8": 768}
INPUT_SIZE = 40
MEAN = (0.5626310706138611, 0.3856309652328491, 0.5453572869300842)
STD = (0.18103212118148804, 0.1668265014886856, 0.13694556057453156)


def _backbone(arch):
    from timm.models.vision_transformer import VisionTransformer
    from torch import nn

    if arch not in ARCHITECTURES:
        raise ValueError(
            f"unknown LEMON architecture {arch!r}; choose from {tuple(ARCHITECTURES)}"
        )
    return VisionTransformer(
        img_size=INPUT_SIZE,
        patch_size=8,
        embed_dim=ARCHITECTURES[arch],
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=0,
    )


def _backbone_state(checkpoint):
    if "state_dict" not in checkpoint:
        raise KeyError("LEMON checkpoint does not contain a state_dict")
    state = {}
    for raw_name, value in checkpoint["state_dict"].items():
        name = str(raw_name).removeprefix("module.")
        if name.startswith("base_encoder.") and not name.startswith(
            ("base_encoder.head.", "base_encoder.fc.")
        ):
            state[name.removeprefix("base_encoder.")] = value
    if not state:
        raise ValueError("LEMON checkpoint contains no base-encoder weights")
    return state


class LemonEmbedder(TorchEmbedder):
    """Expose the released LEMON nuclear-morphology representation."""

    def __init__(self, model, arch, device=None):
        super().__init__(model, device)
        self.embedding_dim = ARCHITECTURES[arch]

    def _embed(self, images) -> dict:
        import torch

        features = self.model(images)
        if (
            features.ndim != 2
            or features.shape[1] != self.embedding_dim
            or not torch.isfinite(features).all()
        ):
            raise ValueError(
                f"expected finite (N, {self.embedding_dim}) LEMON features"
            )
        return {"lemon": features}


def load(
    hf_repo="aliceblondel/LEMON",
    arch="vitb8",
    revision=None,
    *,
    device=None,
) -> LemonEmbedder:
    """Load a released LEMON MoCo backbone from Hugging Face."""
    import torch
    from huggingface_hub import hf_hub_download
    from timm.layers import to_2tuple
    from torchvision.transforms import ToTensor
    from torchvision.transforms.v2 import Compose, Normalize, Resize

    checkpoint_path = hf_hub_download(
        repo_id=hf_repo,
        filename=f"lemon_{arch}.pth.tar",
        revision=revision,
    )
    model = _backbone(arch)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(_backbone_state(checkpoint), strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(f"missing LEMON backbone keys: {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"unexpected LEMON backbone keys: {incompatible.unexpected_keys}"
        )
    model.eval_transforms = Compose(
        [
            ToTensor(),
            Resize(size=to_2tuple(INPUT_SIZE)),
            Normalize(mean=MEAN, std=STD),
        ]
    )
    model.precision = torch.float16
    return LemonEmbedder(model, arch, resolve_device(device))
