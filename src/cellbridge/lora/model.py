"""LoRA-adapted UNI2-h model and objectives for image-to-TERRA distillation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

CHECKPOINT_FORMAT = "cellbridge.lora/1"
OBJECTIVE_MODES = {"contrastive", "contrastive_gex"}
TARGET_TYPES = {"exact", "terra_similarity"}


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
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")

        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
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
            update = self.lora_B(self.lora_A(self.dropout(inputs)))
            output = output + self.scaling * update
        return output


def _parse_trainable_blocks(value: int | str) -> int:
    if isinstance(value, str):
        if value == "none":
            return 0
        if not value.startswith("last_"):
            raise ValueError("trainable_blocks must be an integer, 'last_N', or 'none'")
        value = value.removeprefix("last_")
        if not value.isdigit():
            raise ValueError("trainable_blocks must be an integer, 'last_N', or 'none'")
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("trainable_blocks must be a non-negative integer")
    return value


def _transformer_blocks(encoder: nn.Module) -> Sequence[nn.Module]:
    backbone = getattr(encoder, "model", encoder)
    blocks = getattr(backbone, "blocks", None)
    if not isinstance(blocks, (nn.Sequential, nn.ModuleList, list, tuple)):
        raise ValueError("UNI encoder must expose transformer blocks as model.blocks")
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
    first_block = len(blocks) - trainable_blocks
    for block_index in range(first_block, len(blocks)):
        attention = getattr(blocks[block_index], "attn", None)
        if attention is None:
            raise ValueError(f"transformer block {block_index} has no attention module")
        for module_name in target_modules:
            layer = getattr(attention, module_name, None)
            if not isinstance(layer, nn.Linear):
                raise ValueError(
                    f"block {block_index}.attn.{module_name} is not a linear layer"
                )
            setattr(attention, module_name, LoRALinear(layer, rank, alpha, dropout))
            names.append(f"blocks.{block_index}.attn.{module_name}")
    return tuple(names)


class LoRAUNIDistiller(nn.Module):
    """UNI2-h with attention LoRA adapters and TERRA/GEX prediction heads."""

    def __init__(
        self,
        encoder: nn.Module,
        image_dim: int = 1536,
        terra_dim: int = 384,
        mode: str = "contrastive",
        num_genes: int | None = None,
        trainable_blocks: int | str = 2,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        gex_hidden_dim: int = 1024,
        gex_dropout: float = 0.1,
        target_modules: Sequence[str] = ("qkv", "proj"),
        geometry_mode: str = "euclidean",
        alignment_dim: int | None = None,
        curvature: float = 1.0,
        learnable_curvature: bool = True,
        min_curvature: float = 1e-3,
        max_curvature: float = 10.0,
        max_tangent_norm: float = 12.0,
        ontology_depths: Sequence[int] | None = None,
        ontology_parents: Sequence[int] | None = None,
        ontology_init_radius: float = 0.5,
    ) -> None:
        super().__init__()
        if mode not in OBJECTIVE_MODES:
            raise ValueError(f"mode must be one of {sorted(OBJECTIVE_MODES)}")
        if mode == "contrastive_gex" and (num_genes is None or num_genes < 1):
            raise ValueError("contrastive_gex mode requires a positive num_genes")
        if image_dim < 1 or terra_dim < 1:
            raise ValueError("image_dim and terra_dim must be positive")
        if not target_modules:
            raise ValueError("target_modules cannot be empty")

        if geometry_mode not in {"euclidean", "hyperbolic"}:
            raise ValueError("geometry_mode must be 'euclidean' or 'hyperbolic'")
        alignment_dim = int(terra_dim if alignment_dim is None else alignment_dim)
        if alignment_dim < 1:
            raise ValueError("alignment_dim must be positive")

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
        self.mode = mode
        self.num_genes = int(num_genes) if num_genes is not None else None
        self.trainable_blocks = trainable_blocks
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.gex_hidden_dim = int(gex_hidden_dim)
        self.gex_dropout = float(gex_dropout)
        self.target_modules = target_modules
        self.geometry_mode = geometry_mode
        self.alignment_dim = alignment_dim
        self.initial_curvature = float(curvature)
        self.learnable_curvature = bool(learnable_curvature)
        self.min_curvature = float(min_curvature)
        self.max_curvature = float(max_curvature)
        self.max_tangent_norm = float(max_tangent_norm)
        self.ontology_depths = (
            tuple(map(int, ontology_depths)) if ontology_depths is not None else None
        )
        self.ontology_parents = (
            tuple(map(int, ontology_parents)) if ontology_parents is not None else None
        )
        self.ontology_init_radius = float(ontology_init_radius)

        self.projection_head = nn.Sequential(
            nn.LayerNorm(self.image_dim),
            nn.Linear(self.image_dim, self.alignment_dim),
        )
        needs_terra_head = self.geometry_mode == "hyperbolic" or (
            self.alignment_dim != self.terra_dim
        )
        self.terra_projection_head = (
            nn.Sequential(
                nn.LayerNorm(self.terra_dim),
                nn.Linear(self.terra_dim, self.alignment_dim),
            )
            if needs_terra_head
            else nn.Identity()
        )
        self.lorentz = self.ontology = None
        if self.geometry_mode == "hyperbolic":
            from cellbridge.hyperbolic.manifold import LorentzOps, OntologyEmbedding

            self.lorentz = LorentzOps(
                self.alignment_dim,
                self.initial_curvature,
                learnable_curvature=self.learnable_curvature,
                min_curvature=self.min_curvature,
                max_curvature=self.max_curvature,
                max_tangent_norm=self.max_tangent_norm,
            )
            if self.ontology_depths is not None:
                if self.ontology_parents is None:
                    raise ValueError("ontology_depths requires ontology_parents")
                self.ontology = OntologyEmbedding(
                    self.lorentz,
                    self.ontology_depths,
                    self.ontology_parents,
                    init_radius=self.ontology_init_radius,
                )
        elif self.ontology_depths is not None or self.ontology_parents is not None:
            raise ValueError("ontology points require hyperbolic geometry")
        if self.mode == "contrastive_gex":
            self.gex_norm = nn.LayerNorm(self.image_dim)
            self.gex_hidden = nn.Linear(self.image_dim, self.gex_hidden_dim)
            self.gex_activation = nn.GELU()
            self.gex_hidden_dropout = nn.Dropout(self.gex_dropout)
            self.gex_output = nn.Linear(self.gex_hidden_dim, self.num_genes)

    @property
    def eval_transforms(self) -> Any:
        """The preprocessing transform provided by the trident encoder."""

        return getattr(self.encoder, "eval_transforms", None)

    @property
    def precision(self) -> torch.dtype:
        """The preferred autocast dtype provided by the trident encoder."""

        return getattr(self.encoder, "precision", torch.float32)

    @property
    def model_config(self) -> dict[str, Any]:
        """Architecture settings needed to reconstruct this model."""

        return {
            "image_dim": self.image_dim,
            "terra_dim": self.terra_dim,
            "mode": self.mode,
            "num_genes": self.num_genes,
            "trainable_blocks": self.trainable_blocks,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "gex_hidden_dim": self.gex_hidden_dim,
            "gex_dropout": self.gex_dropout,
            "target_modules": list(self.target_modules),
            "geometry_mode": self.geometry_mode,
            "alignment_dim": self.alignment_dim,
            "curvature": self.current_curvature,
            "learnable_curvature": self.learnable_curvature,
            "min_curvature": self.min_curvature,
            "max_curvature": self.max_curvature,
            "max_tangent_norm": self.max_tangent_norm,
            "ontology_depths": self.ontology_depths,
            "ontology_parents": self.ontology_parents,
            "ontology_init_radius": self.ontology_init_radius,
        }

    @property
    def current_curvature(self) -> float:
        if self.lorentz is None:
            return self.initial_curvature
        return float(self.lorentz.curvature.detach())

    def lora_layers(self) -> Iterator[LoRALinear]:
        for module in self.modules():
            if isinstance(module, LoRALinear):
                yield module

    def lora_parameters(self) -> Iterator[nn.Parameter]:
        """Yield adapter parameters for a dedicated optimizer group."""

        for layer in self.lora_layers():
            yield from layer.lora_A.parameters()
            yield from layer.lora_B.parameters()

    def head_parameters(self) -> Iterator[nn.Parameter]:
        """Yield projection and GEX parameters for a dedicated optimizer group."""

        yield from self.projection_head.parameters()
        yield from self.terra_projection_head.parameters()
        if self.mode == "contrastive_gex":
            yield from self.gex_norm.parameters()
            yield from self.gex_hidden.parameters()
            yield from self.gex_output.parameters()

    def curvature_parameters(self) -> Iterator[nn.Parameter]:
        if self.lorentz is not None:
            parameter = self.lorentz.curvature_parameter
            if parameter is not None:
                yield parameter

    def manifold_parameters(self) -> Iterator[nn.Parameter]:
        if self.ontology is not None:
            yield self.ontology.points

    @property
    def ontology_points(self) -> torch.Tensor | None:
        return None if self.ontology is None else self.ontology.points

    @torch.no_grad()
    def clamp_geometry_(self) -> None:
        if self.lorentz is not None:
            points = () if self.ontology is None else (self.ontology.points,)
            self.lorentz.clamp_(*points)

    @contextmanager
    def use_lora(self, enabled: bool) -> Iterator[None]:
        """Temporarily enable or disable all adapters, preserving prior state."""

        layers = list(self.lora_layers())
        previous = [layer.enabled for layer in layers]
        for layer in layers:
            layer.enabled = enabled
        try:
            yield
        finally:
            for layer, was_enabled in zip(layers, previous, strict=True):
                layer.enabled = was_enabled

    @contextmanager
    def adapters_disabled(self) -> Iterator[None]:
        """Temporarily return original frozen-UNI features."""

        with self.use_lora(False):
            yield

    def encode_images(
        self, images: torch.Tensor, *, use_lora: bool | None = None
    ) -> torch.Tensor:
        """Encode preprocessed crops, optionally bypassing all LoRA adapters."""

        if use_lora is None:
            image_feature = self.encoder(images)
        else:
            with self.use_lora(use_lora):
                image_feature = self.encoder(images)
        if not isinstance(image_feature, torch.Tensor) or image_feature.ndim != 2:
            raise ValueError("UNI encoder output must be a [batch, feature] tensor")
        if image_feature.shape[1] != self.image_dim:
            raise ValueError(
                f"UNI feature dimension is {image_feature.shape[1]}, expected {self.image_dim}"
            )
        return image_feature

    def encode_base(self, images: torch.Tensor) -> torch.Tensor:
        """Return original frozen-UNI features with all adapters disabled."""

        return self.encode_images(images, use_lora=False)

    def _map_alignment(self, tangent: torch.Tensor) -> torch.Tensor:
        if self.lorentz is None:
            return F.normalize(tangent, dim=-1)
        if tangent.dtype in {torch.float16, torch.bfloat16}:
            tangent = tangent.float()
        with torch.autocast(device_type=tangent.device.type, enabled=False):
            return self.lorentz.expmap0(tangent)

    def apply_heads(self, image_feature: torch.Tensor) -> dict[str, torch.Tensor]:
        """Apply the projection and optional GEX heads to cached UNI features."""

        if image_feature.ndim != 2 or image_feature.shape[1] != self.image_dim:
            raise ValueError(f"image_feature must have shape [batch, {self.image_dim}]")

        outputs = {
            "image_feature": image_feature,
            "projected": self._map_alignment(self.projection_head(image_feature)),
        }
        if self.mode == "contrastive_gex":
            hidden = self.gex_activation(self.gex_hidden(self.gex_norm(image_feature)))
            hidden = self.gex_hidden_dropout(hidden)
            outputs["gex_penultimate"] = hidden
            outputs["gex_prediction"] = self.gex_output(hidden)
        return outputs

    def project_terra(self, terra_feature: torch.Tensor) -> torch.Tensor:
        if terra_feature.ndim != 2 or terra_feature.shape[1] != self.terra_dim:
            raise ValueError(f"terra_feature must have shape [batch, {self.terra_dim}]")
        return self._map_alignment(self.terra_projection_head(terra_feature))

    def probe_projection(self, point: torch.Tensor) -> torch.Tensor:
        """Return origin-tangent coordinates consumed by the existing probes."""

        if self.lorentz is None:
            return point
        if point.dtype in {torch.float16, torch.bfloat16}:
            point = point.float()
        with torch.autocast(device_type=point.device.type, enabled=False):
            return self.lorentz.logmap0(point)

    def embedding_radius(self, point: torch.Tensor) -> torch.Tensor:
        if self.lorentz is None:
            return point.norm(dim=-1)
        if point.dtype in {torch.float16, torch.bfloat16}:
            point = point.float()
        with torch.autocast(device_type=point.device.type, enabled=False):
            return self.lorentz.radius(point)

    def forward(
        self, images: torch.Tensor, *, use_lora: bool | None = None
    ) -> dict[str, torch.Tensor]:
        return self.apply_heads(self.encode_images(images, use_lora=use_lora))

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Return only LoRA and head parameters, detached on CPU."""

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


def _similarity_logits(
    image_z: torch.Tensor,
    terra_z: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if image_z.ndim != 2 or terra_z.ndim != 2 or image_z.shape != terra_z.shape:
        raise ValueError(
            "image_z and terra_z must have the same [batch, feature] shape"
        )
    if len(image_z) == 0:
        raise ValueError("contrastive loss requires a non-empty batch")

    image_z = F.normalize(image_z, dim=-1)
    terra_z = F.normalize(terra_z, dim=-1)
    return image_z @ terra_z.T / temperature


def _retrieval_metrics(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    labels = torch.arange(len(logits), device=logits.device)
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
        "contrastive_top1": top1,
        "contrastive_top5": top5,
    }


def symmetric_contrastive_loss(
    image_z: torch.Tensor,
    terra_z: torch.Tensor,
    temperature: float = 0.07,
    *,
    bidirectional: bool = True,
) -> dict[str, torch.Tensor]:
    """CLIP-style exact-pair InfoNCE and in-batch retrieval accuracy."""

    logits = _similarity_logits(image_z, terra_z, temperature)
    labels = torch.arange(len(logits), device=logits.device)
    loss = F.cross_entropy(logits, labels)
    if bidirectional:
        loss = 0.5 * (loss + F.cross_entropy(logits.T, labels))
    return {"loss_contrastive": loss, **_retrieval_metrics(logits)}


def terra_similarity_targets(
    terra_z: torch.Tensor,
    teacher_temperature: float = 0.1,
    exact_match_boost: float = 0.05,
) -> torch.Tensor:
    """Create soft in-batch targets from pairwise TERRA cosine similarity."""

    if terra_z.ndim != 2 or len(terra_z) == 0:
        raise ValueError("terra_z must be a non-empty [batch, feature] tensor")
    if teacher_temperature <= 0:
        raise ValueError("teacher_temperature must be positive")
    if exact_match_boost <= 0:
        raise ValueError("exact_match_boost must be positive")

    with torch.autocast(device_type=terra_z.device.type, enabled=False):
        normalized = F.normalize(terra_z.detach().float(), dim=-1)
        teacher_logits = normalized @ normalized.T
        teacher_logits.diagonal().add_(exact_match_boost)
        targets = F.softmax(teacher_logits / teacher_temperature, dim=-1)
    return targets.detach()


def _distribution_kl(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.kl_div(F.log_softmax(logits, dim=-1), targets, reduction="batchmean")


def soft_target_contrastive_loss(
    image_z: torch.Tensor,
    terra_z: torch.Tensor,
    temperature: float = 0.07,
    *,
    teacher_temperature: float = 0.1,
    exact_match_boost: float = 0.05,
    bidirectional: bool = True,
) -> dict[str, torch.Tensor]:
    """Match image-to-TERRA similarities to a soft TERRA teacher distribution."""

    logits = _similarity_logits(image_z, terra_z, temperature)
    targets = terra_similarity_targets(
        terra_z,
        teacher_temperature=teacher_temperature,
        exact_match_boost=exact_match_boost,
    )
    loss = _distribution_kl(logits, targets)
    if bidirectional:
        loss = 0.5 * (loss + _distribution_kl(logits.T, targets))

    return {"loss_contrastive": loss, **_retrieval_metrics(logits)}


def compute_loss(
    outputs: Mapping[str, torch.Tensor],
    terra_target: torch.Tensor,
    gex_target: torch.Tensor | None = None,
    gex_mask: torch.Tensor | None = None,
    *,
    mode: str = "contrastive",
    lambda_contrastive: float = 1.0,
    lambda_gex: float = 1.0,
    temperature: float = 0.07,
    gex_loss: str = "smooth_l1",
    targets: str = "exact",
    teacher_temperature: float = 0.1,
    exact_match_boost: float = 0.05,
    bidirectional: bool = True,
    manifold=None,
) -> dict[str, torch.Tensor]:
    """Compute the configured LoRA objective and retrieval metrics."""

    if mode not in OBJECTIVE_MODES:
        raise ValueError(f"mode must be one of {sorted(OBJECTIVE_MODES)}")
    if targets not in TARGET_TYPES:
        raise ValueError(f"targets must be one of {sorted(TARGET_TYPES)}")
    if "projected" not in outputs:
        raise ValueError("model outputs are missing 'projected'")

    terra_z = outputs.get("terra_projected", terra_target)
    if targets == "exact":
        if manifold is None:
            metrics = symmetric_contrastive_loss(
                outputs["projected"],
                terra_z,
                temperature,
                bidirectional=bidirectional,
            )
        else:
            from cellbridge.hyperbolic.losses import (
                hyperbolic_contrastive_loss,
            )

            metrics = hyperbolic_contrastive_loss(
                outputs["projected"],
                terra_z,
                manifold,
                temperature,
                bidirectional=bidirectional,
            )
    else:
        if manifold is not None:
            raise ValueError("hyperbolic alignment does not support soft targets")
        metrics = soft_target_contrastive_loss(
            outputs["projected"],
            terra_z,
            temperature,
            teacher_temperature=teacher_temperature,
            exact_match_boost=exact_match_boost,
            bidirectional=bidirectional,
        )
    total = lambda_contrastive * metrics["loss_contrastive"]
    if mode == "contrastive_gex":
        if gex_loss != "smooth_l1":
            raise ValueError("gex_loss must be 'smooth_l1'")
        if gex_target is None or "gex_prediction" not in outputs:
            raise ValueError("contrastive_gex requires predictions and GEX targets")
        if outputs["gex_prediction"].shape != gex_target.shape:
            raise ValueError("GEX predictions and targets must have the same shape")
        elementwise = F.smooth_l1_loss(
            outputs["gex_prediction"], gex_target, reduction="none"
        )
        if gex_mask is None:
            loss_gex = elementwise.mean()
        else:
            if gex_mask.shape != gex_target.shape:
                raise ValueError("GEX mask and targets must have the same shape")
            weights = gex_mask.to(dtype=elementwise.dtype)
            denominator = weights.sum()
            loss_gex = (elementwise * weights).sum() / denominator.clamp_min(1)
        metrics["loss_gex"] = loss_gex
        total = total + lambda_gex * loss_gex
    metrics["loss_total"] = total
    return metrics


def build_uni2_distiller(
    *,
    encoder: nn.Module | None = None,
    weights_path: str | None = None,
    **model_config: Any,
) -> LoRAUNIDistiller:
    """Build a distiller around trident's UNI2-h encoder or a supplied encoder."""

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
    """Create a compact checkpoint without duplicating frozen UNI weights."""

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
    """Rebuild UNI2-h, load adapters/heads, and return the checkpoint metadata."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("format_version") != CHECKPOINT_FORMAT
    ):
        raise ValueError(f"{path} is not a supported LoRA distillation checkpoint")
    if not isinstance(checkpoint.get("model_config"), dict):
        raise ValueError("LoRA checkpoint is missing model_config")
    if not isinstance(checkpoint.get("trainable_state"), dict):
        raise ValueError("LoRA checkpoint is missing trainable_state")

    model = build_uni2_distiller(
        encoder=encoder,
        weights_path=weights_path,
        **checkpoint["model_config"],
    ).to(device)
    model.load_trainable_state_dict(checkpoint["trainable_state"], strict=strict)
    return model, checkpoint
