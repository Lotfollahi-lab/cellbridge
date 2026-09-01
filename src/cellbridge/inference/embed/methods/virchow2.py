"""Frozen Virchow2 embedding inference."""

from cellbridge.inference.embed.methods.base import TorchEmbedder, resolve_device

ARMS = ("virchow2",)
REGISTER_TOKENS = 4


class Virchow2Embedder(TorchEmbedder):
    """Expose Virchow2's official class-plus-mean-patch representation."""

    def _embed(self, images) -> dict:
        import torch

        tokens = self.model(images)
        patch_tokens = tokens[:, 1 + REGISTER_TOKENS :]
        embedding = torch.cat((tokens[:, 0], patch_tokens.mean(dim=1)), dim=-1)
        return {"virchow2": embedding}


def load(
    hf_repo="paige-ai/Virchow2",
    revision=None,
    *,
    device=None,
) -> Virchow2Embedder:
    """Load Virchow2 from its gated Hugging Face repository."""
    import timm
    import torch
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from timm.layers import SwiGLUPacked

    model_id = hf_repo if revision is None else f"{hf_repo}@{revision}"
    model = timm.create_model(
        f"hf-hub:{model_id}",
        pretrained=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )
    model.eval_transforms = create_transform(
        **resolve_data_config(model.pretrained_cfg, model=model)
    )
    model.precision = torch.float16
    return Virchow2Embedder(model, resolve_device(device))
