"""LoRA-adapted UNI2-h model for image-to-transcriptomics distillation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

CHECKPOINT_FORMAT = "cellbridge.lora/2"


class LoRALinear(nn.Module):
    """A frozen linear layer with a trainable low-rank update."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1)")

        self.base_layer = base_layer
        self.rank = int(rank)
        self.scaling = float(alpha) / self.rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Linear(base_layer.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, base_layer.out_features, bias=False)
        self.enabled = True

        self.base_layer.requires_grad_(False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
        self.lora_A.to(base_layer.weight)
        self.lora_B.to(base_layer.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.base_layer(inputs)
        if self.enabled:
            output = output + self.scaling * self.lora_B(
                self.lora_A(self.dropout(inputs))
            )
        return output


def _parse_trainable_blocks(value: int | str) -> int:
    if isinstance(value, str):
        if value == "none":
            return 0
        if not value.startswith("last_") or not value[5:].isdigit():
            raise ValueError("trainable_blocks must be an integer, 'last_N', or 'none'")
        value = int(value[5:])
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("trainable_blocks must be a non-negative integer")
    return value


def _transformer_blocks(encoder: nn.Module) -> Sequence[nn.Module]:
    backbone = getattr(encoder, "model", encoder)
    blocks = getattr(backbone, "blocks", None)
    if not isinstance(blocks, (nn.Sequential, nn.ModuleList, list, tuple)):
        raise TypeError("UNI encoder must expose transformer blocks as model.blocks")
    return blocks


def _inject_lora(
    encoder: nn.Module,
    trainable_blocks: int,
    target_modules: Sequence[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> tuple[str, ...]:
    if trainable_blocks == 0:
        return ()

    blocks = _transformer_blocks(encoder)
    if trainable_blocks > len(blocks):
        raise ValueError(
            f"requested the last {trainable_blocks} blocks, but encoder has {len(blocks)}"
        )

    names = []
    for block_index in range(len(blocks) - trainable_blocks, len(blocks)):
        attention = getattr(blocks[block_index], "attn", None)
        if attention is None:
            raise ValueError(f"transformer block {block_index} has no attention module")
        for module_name in target_modules:
            layer = getattr(attention, module_name, None)
            if not isinstance(layer, nn.Linear):
                raise TypeError(
                    f"block {block_index}.attn.{module_name} is not a linear layer"
                )
            setattr(attention, module_name, LoRALinear(layer, rank, alpha, dropout))
            names.append(f"blocks.{block_index}.attn.{module_name}")
    return tuple(names)


class LoRAUNIDistiller(nn.Module):
    """UNI2-h with attention LoRA adapters and an alignment head."""

    def __init__(
        self,
        encoder: nn.Module,
        image_dim: int = 1536,
        terra_dim: int = 384,
        trainable_blocks: int | str = 4,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        target_modules: Sequence[str] = ("qkv", "proj"),
        alignment_dim: int | None = None,
        projection_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if image_dim < 1 or terra_dim < 1:
            raise ValueError("image_dim and terra_dim must be positive")
        if not target_modules:
            raise ValueError("target_modules cannot be empty")

        alignment_dim = terra_dim if alignment_dim is None else alignment_dim
        if alignment_dim < 1:
            raise ValueError("alignment_dim must be positive")
        if projection_hidden_dim is not None and projection_hidden_dim < 1:
            raise ValueError("projection_hidden_dim must be positive")

        trainable_blocks = _parse_trainable_blocks(trainable_blocks)
        target_modules = tuple(dict.fromkeys(target_modules))
        encoder.requires_grad_(False)
        self.encoder = encoder
        self.lora_module_names = _inject_lora(
            encoder,
            trainable_blocks,
            target_modules,
            lora_rank,
            lora_alpha,
            lora_dropout,
        )
        self.image_dim = int(image_dim)
        self.terra_dim = int(terra_dim)
        self.trainable_blocks = trainable_blocks
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.target_modules = target_modules
        self.alignment_dim = int(alignment_dim)
        self.projection_hidden_dim = (
            int(projection_hidden_dim) if projection_hidden_dim is not None else None
        )

        if self.projection_hidden_dim is None:
            self.projection_head = nn.Sequential(
                nn.LayerNorm(self.image_dim),
                nn.Linear(self.image_dim, self.alignment_dim),
            )
        else:
            self.projection_head = nn.Sequential(
                nn.LayerNorm(self.image_dim),
                nn.Linear(self.image_dim, self.projection_hidden_dim),
                nn.GELU(),
                nn.Linear(self.projection_hidden_dim, self.alignment_dim),
            )
        self.terra_projection_head = (
            nn.Identity()
            if self.alignment_dim == self.terra_dim
            else nn.Sequential(
                nn.LayerNorm(self.terra_dim),
                nn.Linear(self.terra_dim, self.alignment_dim),
            )
        )

    @property
    def eval_transforms(self) -> Any:
        return getattr(self.encoder, "eval_transforms", None)

    @property
    def model_config(self) -> dict[str, Any]:
        """Architecture settings needed to reconstruct this model."""

        return {
            "image_dim": self.image_dim,
            "terra_dim": self.terra_dim,
            "trainable_blocks": self.trainable_blocks,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": list(self.target_modules),
            "alignment_dim": self.alignment_dim,
            "projection_hidden_dim": self.projection_hidden_dim,
        }

    def lora_layers(self) -> Iterator[LoRALinear]:
        return (module for module in self.modules() if isinstance(module, LoRALinear))

    def lora_parameters(self) -> Iterator[nn.Parameter]:
        for layer in self.lora_layers():
            yield from layer.lora_A.parameters()
            yield from layer.lora_B.parameters()

    def head_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.projection_head.parameters()
        yield from self.terra_projection_head.parameters()

    @contextmanager
    def use_lora(self, enabled: bool) -> Iterator[None]:
        layers = list(self.lora_layers())
        previous = [layer.enabled for layer in layers]
        for layer in layers:
            layer.enabled = enabled
        try:
            yield
        finally:
            for layer, was_enabled in zip(layers, previous, strict=True):
                layer.enabled = was_enabled

    def encode_images(
        self, images: torch.Tensor, *, use_lora: bool | None = None
    ) -> torch.Tensor:
        if use_lora is None:
            features = self.encoder(images)
        else:
            with self.use_lora(use_lora):
                features = self.encoder(images)
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise ValueError("UNI encoder output must be a [batch, feature] tensor")
        if features.shape[1] != self.image_dim:
            raise ValueError(
                f"UNI feature dimension is {features.shape[1]}, expected {self.image_dim}"
            )
        return features

    def apply_heads(self, image_features: torch.Tensor) -> dict[str, torch.Tensor]:
        if image_features.ndim != 2 or image_features.shape[1] != self.image_dim:
            raise ValueError(
                f"image_features must have shape [batch, {self.image_dim}]"
            )
        return {"projected": F.normalize(self.projection_head(image_features), dim=-1)}

    def project_terra(self, terra_features: torch.Tensor) -> torch.Tensor:
        if terra_features.ndim != 2 or terra_features.shape[1] != self.terra_dim:
            raise ValueError(
                f"terra_features must have shape [batch, {self.terra_dim}]"
            )
        return F.normalize(self.terra_projection_head(terra_features), dim=-1)

    def forward(
        self, images: torch.Tensor, *, use_lora: bool | None = None
    ) -> dict[str, torch.Tensor]:
        return self.apply_heads(self.encode_images(images, use_lora=use_lora))

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().cpu()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }

    def load_trainable_state_dict(
        self, state: Mapping[str, torch.Tensor], *, strict: bool = True
    ) -> None:
        parameters = {
            name: parameter
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        missing = sorted(parameters.keys() - state.keys())
        unexpected = sorted(state.keys() - parameters.keys())
        if strict and (missing or unexpected):
            details = []
            if missing:
                details.append(f"missing keys: {missing}")
            if unexpected:
                details.append(f"unexpected keys: {unexpected}")
            raise RuntimeError(
                "invalid LoRA trainable state (" + "; ".join(details) + ")"
            )

        with torch.no_grad():
            for name in parameters.keys() & state.keys():
                source = state[name]
                destination = parameters[name]
                if source.shape != destination.shape:
                    raise RuntimeError(
                        f"shape mismatch for {name}: checkpoint {tuple(source.shape)}, "
                        f"model {tuple(destination.shape)}"
                    )
                destination.copy_(source)


def contrastive_loss(
    image_features: torch.Tensor,
    terra_features: torch.Tensor,
    temperature: float = 0.07,
    *,
    bidirectional: bool = True,
) -> dict[str, torch.Tensor]:
    """Exact-pair InfoNCE and in-batch retrieval accuracy."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if (
        image_features.ndim != 2
        or terra_features.ndim != 2
        or image_features.shape != terra_features.shape
        or len(image_features) == 0
    ):
        raise ValueError("features must have the same non-empty [batch, feature] shape")

    image_features = F.normalize(image_features, dim=-1)
    terra_features = F.normalize(terra_features, dim=-1)
    logits = image_features @ terra_features.T / temperature
    labels = torch.arange(len(logits), device=logits.device)
    loss = F.cross_entropy(logits, labels)
    if bidirectional:
        loss = 0.5 * (loss + F.cross_entropy(logits.T, labels))

    top_k = min(5, len(logits))
    image_top = logits.topk(top_k, dim=1).indices
    terra_top = logits.T.topk(top_k, dim=1).indices
    top1 = 0.5 * (
        (image_top[:, 0] == labels).float().mean()
        + (terra_top[:, 0] == labels).float().mean()
    )
    top5 = 0.5 * (
        (image_top == labels[:, None]).any(dim=1).float().mean()
        + (terra_top == labels[:, None]).any(dim=1).float().mean()
    )
    return {
        "loss_contrastive": loss,
        "loss_total": loss,
        "contrastive_top1": top1,
        "contrastive_top5": top5,
    }


def build_uni2_distiller(
    *,
    encoder: nn.Module | None = None,
    weights_path: str | None = None,
    **model_config: Any,
) -> LoRAUNIDistiller:
    if encoder is None:
        from trident.patch_encoder_models import encoder_factory

        encoder_kwargs = {"weights_path": weights_path} if weights_path else {}
        encoder = encoder_factory("uni_v2", **encoder_kwargs)
    return LoRAUNIDistiller(encoder=encoder, **model_config)


def make_checkpoint(
    model: LoRAUNIDistiller,
    *,
    step: int | None = None,
    val_loss: float | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT,
        "model_config": model.model_config,
        "trainable_state": model.trainable_state_dict(),
    }
    if step is not None:
        checkpoint["step"] = int(step)
    if val_loss is not None:
        checkpoint["val_loss"] = float(val_loss)
    if config is not None:
        checkpoint["config"] = dict(config)
    return checkpoint


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
    *,
    encoder: nn.Module | None = None,
    weights_path: str | None = None,
    strict: bool = True,
) -> tuple[LoRAUNIDistiller, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("format_version") != CHECKPOINT_FORMAT
    ):
        raise ValueError(f"{path} is not a supported LoRA checkpoint")
    if not isinstance(checkpoint.get("model_config"), dict):
        raise TypeError("LoRA checkpoint is missing model_config")
    if not isinstance(checkpoint.get("trainable_state"), dict):
        raise TypeError("LoRA checkpoint is missing trainable_state")

    model = build_uni2_distiller(
        encoder=encoder,
        weights_path=weights_path,
        **checkpoint["model_config"],
    ).to(device)
    model.load_trainable_state_dict(checkpoint["trainable_state"], strict=strict)
    return model, checkpoint
