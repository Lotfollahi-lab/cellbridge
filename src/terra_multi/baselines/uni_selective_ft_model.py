"""Selective supervised fine-tuning of the original UNI2-h encoder."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import torch
from torch import nn

CHECKPOINT_FORMAT = "terra_multi.uni_selective_ft/1"


def _backbone(encoder: nn.Module) -> nn.Module:
    return getattr(encoder, "model", encoder)


def transformer_blocks(encoder: nn.Module) -> Sequence[nn.Module]:
    """Return the ViT blocks exposed by Trident's UNI wrapper."""
    blocks = getattr(_backbone(encoder), "blocks", None)
    if not isinstance(blocks, (nn.Sequential, nn.ModuleList, list, tuple)):
        raise TypeError("UNI encoder must expose transformer blocks as model.blocks")
    return blocks


def final_normalization(encoder: nn.Module) -> nn.Module:
    """Return UNI's final token normalization module."""
    backbone = _backbone(encoder)
    for name in ("norm", "fc_norm"):
        module = getattr(backbone, name, None)
        if isinstance(module, nn.Module):
            return module
    raise ValueError("UNI encoder must expose a final model.norm or model.fc_norm")


class SelectiveUNIClassifier(nn.Module):
    """UNI2-h with only its last N blocks, final norm, and head trainable."""

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        *,
        image_dim: int = 1536,
        trainable_blocks: int = 1,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least two")
        if image_dim < 1:
            raise ValueError("image_dim must be positive")
        if trainable_blocks < 1:
            raise ValueError("trainable_blocks must be positive")

        blocks = transformer_blocks(encoder)
        if trainable_blocks > len(blocks):
            raise ValueError(
                f"requested {trainable_blocks} blocks, but UNI exposes {len(blocks)}"
            )

        encoder.requires_grad_(False)
        for block in blocks[-trainable_blocks:]:
            block.requires_grad_(True)
        final_normalization(encoder).requires_grad_(True)

        self.encoder = encoder
        self.head = nn.Linear(image_dim, num_classes)
        self.image_dim = int(image_dim)
        self.num_classes = int(num_classes)
        self.trainable_blocks = int(trainable_blocks)

    @property
    def eval_transforms(self) -> Any:
        return getattr(self.encoder, "eval_transforms", None)

    @property
    def precision(self) -> torch.dtype:
        return getattr(self.encoder, "precision", torch.bfloat16)

    @property
    def model_config(self) -> dict[str, int]:
        return {
            "image_dim": self.image_dim,
            "num_classes": self.num_classes,
            "trainable_blocks": self.trainable_blocks,
        }

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        if features.ndim != 2 or features.shape[1] != self.image_dim:
            raise ValueError(
                f"UNI returned {tuple(features.shape)}; expected (N, {self.image_dim})"
            )
        return self.head(features)

    def backbone_parameters(self) -> Iterator[nn.Parameter]:
        for parameter in self.encoder.parameters():
            if parameter.requires_grad:
                yield parameter

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        trainable = {name for name, value in self.named_parameters() if value.requires_grad}
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if name in trainable
        }

    def load_trainable_state_dict(
        self, state: Mapping[str, torch.Tensor], *, strict: bool = True
    ) -> None:
        expected = {
            name for name, value in self.named_parameters() if value.requires_grad
        }
        observed = set(state)
        if strict and observed != expected:
            raise ValueError(
                "trainable checkpoint mismatch: "
                f"missing={sorted(expected - observed)}, "
                f"unexpected={sorted(observed - expected)}"
            )
        current = self.state_dict()
        current.update(state)
        self.load_state_dict(current, strict=True)


def build_classifier(
    num_classes: int,
    *,
    weights_path: str | None = None,
    encoder: nn.Module | None = None,
    **model_config: Any,
) -> SelectiveUNIClassifier:
    """Build from the original UNI2-h weights, never a Cellbridge checkpoint."""
    if encoder is None:
        from trident.patch_encoder_models import encoder_factory

        kwargs = {"weights_path": weights_path} if weights_path else {}
        encoder = encoder_factory("uni_v2", **kwargs)
    return SelectiveUNIClassifier(
        encoder,
        num_classes,
        **model_config,
    )


def make_checkpoint(
    model: SelectiveUNIClassifier,
    *,
    classes: Sequence[str],
    best_epoch: int,
    seed: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Store only trainable tensors; frozen original UNI weights stay external."""
    return {
        "format_version": CHECKPOINT_FORMAT,
        "encoder": "uni_v2",
        "initialization": "original_uni2_h",
        "model_config": model.model_config,
        "classes": list(map(str, classes)),
        "best_epoch": int(best_epoch),
        "seed": int(seed),
        "trainable_state": model.trainable_state_dict(),
        "config": dict(config),
    }
