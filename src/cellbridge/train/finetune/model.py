"""UNI2-h classifier with a fixed selective-unfreezing policy."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

import torch
from torch import nn

CHECKPOINT_FORMAT = "cellbridge.uni_selective_finetune/1"
FEATURE_DIM = 1536
UNFREEZING_POLICY = "last_block+final_norm+head"


def _backbone(encoder: nn.Module) -> nn.Module:
    return getattr(encoder, "model", encoder)


def _last_block(encoder: nn.Module) -> nn.Module:
    blocks = getattr(_backbone(encoder), "blocks", None)
    if not isinstance(blocks, (nn.ModuleList, nn.Sequential, list, tuple)):
        raise TypeError("UNI must expose transformer blocks as model.blocks")
    if not blocks:
        raise ValueError("UNI exposes no transformer blocks")
    return blocks[-1]


def _final_norm(encoder: nn.Module) -> nn.Module:
    backbone = _backbone(encoder)
    for name in ("norm", "fc_norm"):
        module = getattr(backbone, name, None)
        if isinstance(module, nn.Module):
            return module
    raise ValueError("UNI must expose a final model.norm or model.fc_norm")


class SelectiveUNIClassifier(nn.Module):
    """Train UNI2-h's final block, final norm, and a classification head."""

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        *,
        feature_dim: int = FEATURE_DIM,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least two")
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")

        encoder.requires_grad_(False)
        _last_block(encoder).requires_grad_(True)
        _final_norm(encoder).requires_grad_(True)

        self.encoder = encoder
        self.head = nn.Linear(feature_dim, num_classes)
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)

    @property
    def transform(self):
        return getattr(self.encoder, "eval_transforms", None)

    @property
    def precision(self) -> torch.dtype:
        return getattr(self.encoder, "precision", torch.float32)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        expected = (len(images), self.feature_dim)
        if tuple(features.shape) != expected:
            raise ValueError(
                f"UNI returned {tuple(features.shape)}; expected {expected}"
            )
        return self.head(features)

    def backbone_parameters(self) -> Iterator[nn.Parameter]:
        return (
            parameter
            for parameter in self.encoder.parameters()
            if parameter.requires_grad
        )

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        names = {
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if name in names
        }


def build_classifier(
    num_classes: int,
    *,
    weights_path: str | None = None,
    encoder: nn.Module | None = None,
    feature_dim: int = FEATURE_DIM,
) -> SelectiveUNIClassifier:
    """Build a classifier from the original UNI2-h weights."""
    if encoder is None:
        from trident.patch_encoder_models import encoder_factory

        kwargs = {"weights_path": weights_path} if weights_path else {}
        encoder = encoder_factory("uni_v2", **kwargs)
    return SelectiveUNIClassifier(
        encoder,
        num_classes,
        feature_dim=feature_dim,
    )


def load_trainable_state_dict(
    model: SelectiveUNIClassifier,
    state: Mapping[str, torch.Tensor],
) -> None:
    """Load a compact checkpoint into a freshly initialized UNI classifier."""
    expected = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    observed = set(state)
    if observed != expected:
        raise ValueError(
            "trainable checkpoint mismatch: "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )
    current = model.state_dict()
    current.update(state)
    model.load_state_dict(current, strict=True)


def make_checkpoint(
    model: SelectiveUNIClassifier,
    *,
    classes: Sequence[str],
    epochs: int,
    seed: int,
) -> dict:
    """Return a checkpoint without duplicating frozen UNI weights."""
    classes = tuple(map(str, classes))
    if len(classes) != model.num_classes:
        raise ValueError("classes must match the classification head")
    return {
        "format_version": CHECKPOINT_FORMAT,
        "encoder": "uni_v2",
        "initialization": "original_uni2_h",
        "unfreezing": UNFREEZING_POLICY,
        "feature_dim": model.feature_dim,
        "classes": list(classes),
        "epochs": int(epochs),
        "seed": int(seed),
        "trainable_state": model.trainable_state_dict(),
    }
