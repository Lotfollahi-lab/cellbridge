"""Spatial cross-modal JEPA model components.

The context encoder receives image embeddings for visible cells. The predictor
uses their encoded representations and target-cell coordinates to predict
TERRA latents for cells that were hidden from the context.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CHECKPOINT_FORMAT = "cellbridge.jepa/1"


def _as_float_tensor(values: np.ndarray | torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(values)
    if not tensor.is_floating_point():
        tensor = tensor.float()
    return tensor.to(dtype=torch.float32)


class TargetNormalizer(nn.Module):
    """Apply fixed train-set centering and scaling to TERRA targets."""

    def __init__(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        super().__init__()
        mean = _as_float_tensor(mean).detach().clone()
        scale = _as_float_tensor(scale).detach().clone()
        if mean.ndim != 1 or scale.shape != mean.shape:
            raise ValueError("mean and scale must be matching one-dimensional tensors")
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
            raise ValueError("target statistics must be finite")
        if torch.any(scale <= 0):
            raise ValueError("target scales must be positive")
        self.register_buffer("mean", mean)
        self.register_buffer("scale", scale)

    @classmethod
    def fit(
        cls,
        targets: np.ndarray | torch.Tensor,
        *,
        min_scale: float = 1e-6,
    ) -> TargetNormalizer:
        """Fit per-feature statistics from training targets only."""

        if min_scale <= 0:
            raise ValueError("min_scale must be positive")
        targets = _as_float_tensor(targets)
        if targets.ndim != 2 or len(targets) == 0:
            raise ValueError("targets must be a non-empty [cells, features] array")
        if not torch.isfinite(targets).all():
            raise ValueError("targets must be finite")
        mean = targets.mean(dim=0)
        scale = targets.std(dim=0, correction=0).clamp_min(min_scale)
        return cls(mean, scale)

    @property
    def target_dim(self) -> int:
        return len(self.mean)

    def normalize(self, targets: torch.Tensor) -> torch.Tensor:
        """Normalize targets after detaching them from the gradient graph."""

        self._check_last_dimension(targets)
        return (targets.detach() - self.mean) / self.scale

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        """Map normalized predictions back to the original TERRA space."""

        self._check_last_dimension(values)
        return values * self.scale + self.mean

    def inverse_transform(self, values: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`denormalize` used by embedding extraction."""

        return self.denormalize(values)

    def forward(self, targets: torch.Tensor) -> torch.Tensor:
        return self.normalize(targets)

    def _check_last_dimension(self, values: torch.Tensor) -> None:
        if values.ndim < 1 or values.shape[-1] != self.target_dim:
            raise ValueError(
                f"expected last dimension {self.target_dim}, got {tuple(values.shape)}"
            )


class CoordinateEmbedding(nn.Module):
    """Embed relative two-dimensional cell coordinates."""

    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        if coordinates.ndim != 3 or coordinates.shape[-1] != 2:
            raise ValueError("coordinates must have shape [batch, cells, 2]")
        return self.layers(coordinates)


class SpatialContextEncoder(nn.Module):
    """Contextualize visible-cell image embeddings and relative positions."""

    def __init__(
        self,
        image_dim: int,
        model_dim: int,
        num_heads: int,
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        _validate_transformer_settings(model_dim, num_heads, depth, dropout)
        if image_dim < 1:
            raise ValueError("image_dim must be positive")

        self.image_dim = int(image_dim)
        self.model_dim = int(model_dim)
        self.image_projection = nn.Sequential(
            nn.LayerNorm(self.image_dim),
            nn.Linear(self.image_dim, self.model_dim),
        )
        self.coordinate_embedding = CoordinateEmbedding(self.model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=num_heads,
            dim_feedforward=4 * self.model_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=depth,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(self.model_dim)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        coordinates: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _check_cell_inputs(image_embeddings, coordinates, self.image_dim, "context")
        mask = _validate_mask(mask, image_embeddings.shape[:2], image_embeddings.device)
        tokens = self.image_projection(image_embeddings)
        tokens = tokens + self.coordinate_embedding(coordinates)
        encoded = self.transformer(tokens, src_key_padding_mask=_padding_mask(mask))
        encoded = self.output_norm(encoded)
        return _zero_padding(encoded, mask)


class CrossAttentionBlock(nn.Module):
    """Update independent target queries from a shared context."""

    def __init__(self, model_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(model_dim)
        self.context_norm = nn.LayerNorm(model_dim)
        self.attention = nn.MultiheadAttention(
            model_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.feed_forward_norm = nn.LayerNorm(model_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, 4 * model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * model_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        context: torch.Tensor,
        context_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        context = self.context_norm(context)
        update, _ = self.attention(
            self.query_norm(queries),
            context,
            context,
            key_padding_mask=context_padding_mask,
            need_weights=False,
        )
        queries = queries + self.attention_dropout(update)
        return queries + self.feed_forward(self.feed_forward_norm(queries))


class TargetPredictor(nn.Module):
    """Predict target-cell latents by cross-attending to encoded context."""

    def __init__(
        self,
        target_dim: int,
        model_dim: int,
        num_heads: int,
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        _validate_transformer_settings(model_dim, num_heads, depth, dropout)
        if target_dim < 1:
            raise ValueError("target_dim must be positive")

        self.model_dim = int(model_dim)
        self.target_dim = int(target_dim)
        self.target_token = nn.Parameter(torch.empty(1, 1, self.model_dim))
        nn.init.normal_(self.target_token, std=0.02)
        self.coordinate_embedding = CoordinateEmbedding(self.model_dim)
        self.blocks = nn.ModuleList(
            CrossAttentionBlock(self.model_dim, num_heads, dropout)
            for _ in range(depth)
        )
        self.output = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, self.target_dim),
        )

    def forward(
        self,
        context: torch.Tensor,
        target_coordinates: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context.ndim != 3 or context.shape[-1] != self.model_dim:
            raise ValueError(
                f"context must have shape [batch, cells, {self.model_dim}]"
            )
        if target_coordinates.ndim != 3 or target_coordinates.shape[-1] != 2:
            raise ValueError("target coordinates must have shape [batch, targets, 2]")
        if context.shape[0] != target_coordinates.shape[0]:
            raise ValueError("context and target coordinates must share a batch size")

        context_mask = _validate_mask(context_mask, context.shape[:2], context.device)
        target_mask = _validate_mask(
            target_mask, target_coordinates.shape[:2], target_coordinates.device
        )
        queries = self.target_token.expand(
            target_coordinates.shape[0], target_coordinates.shape[1], -1
        )
        queries = queries + self.coordinate_embedding(target_coordinates)
        context_padding_mask = _padding_mask(context_mask)
        for block in self.blocks:
            queries = block(queries, context, context_padding_mask)
        return _zero_padding(self.output(queries), target_mask)


class SpatialJEPA(nn.Module):
    """Predict held-out TERRA latents from spatial image context.

    Shapes:
        context_embeddings: ``[batch, context_cells, image_dim]``
        context_coordinates: ``[batch, context_cells, 2]``
        target_coordinates: ``[batch, target_cells, 2]``
        output: ``[batch, target_cells, target_dim]``

    Coordinate normalization is deliberately left to the data pipeline so the
    same spatial convention can be reused during training and extraction.
    """

    def __init__(
        self,
        image_dim: int,
        target_dim: int = 384,
        model_dim: int = 384,
        num_heads: int = 6,
        context_depth: int = 2,
        predictor_depth: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.image_dim = int(image_dim)
        self.target_dim = int(target_dim)
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.context_depth = int(context_depth)
        self.predictor_depth = int(predictor_depth)
        self.dropout = float(dropout)

        self.context_encoder = SpatialContextEncoder(
            self.image_dim,
            self.model_dim,
            self.num_heads,
            self.context_depth,
            self.dropout,
        )
        self.predictor = TargetPredictor(
            self.target_dim,
            self.model_dim,
            self.num_heads,
            self.predictor_depth,
            self.dropout,
        )

    @property
    def model_config(self) -> dict[str, int | float]:
        return {
            "image_dim": self.image_dim,
            "target_dim": self.target_dim,
            "model_dim": self.model_dim,
            "num_heads": self.num_heads,
            "context_depth": self.context_depth,
            "predictor_depth": self.predictor_depth,
            "dropout": self.dropout,
        }

    def encode_context(
        self,
        context_embeddings: torch.Tensor,
        context_coordinates: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return image-only contextual features with shape ``[B, C, D]``."""

        return self.context_encoder(
            context_embeddings,
            context_coordinates,
            context_mask,
        )

    def predict(
        self,
        context_embeddings: torch.Tensor,
        context_coordinates: torch.Tensor,
        target_coordinates: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict standardized TERRA targets with shape ``[B, T, target_dim]``."""

        context = self.encode_context(
            context_embeddings,
            context_coordinates,
            context_mask,
        )
        return self.predictor(
            context,
            target_coordinates,
            context_mask,
            target_mask,
        )

    def forward(
        self,
        context_embeddings: torch.Tensor,
        context_coordinates: torch.Tensor,
        target_coordinates: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.predict(
            context_embeddings,
            context_coordinates,
            target_coordinates,
            context_mask,
            target_mask,
        )


def jepa_loss(
    predictions: torch.Tensor,
    terra_targets: torch.Tensor,
    normalizer: TargetNormalizer,
    target_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Smooth-L1 loss against centered, scaled, stop-gradient TERRA targets."""

    if predictions.shape != terra_targets.shape:
        raise ValueError("predictions and TERRA targets must have the same shape")
    if predictions.ndim != 3:
        raise ValueError("predictions must have shape [batch, targets, features]")
    target_mask = _validate_mask(target_mask, predictions.shape[:2], predictions.device)
    targets = normalizer.normalize(terra_targets)
    distance = F.smooth_l1_loss(predictions, targets, reduction="none").mean(dim=-1)
    if target_mask is None:
        return distance.mean()
    return distance[target_mask].mean()


def load_model_state(
    checkpoint: Mapping[str, object],
    *,
    device: str | torch.device = "cpu",
) -> SpatialJEPA:
    """Reconstruct a model from the minimal shared JEPA checkpoint format."""

    if checkpoint.get("format_version") != CHECKPOINT_FORMAT:
        raise ValueError("unsupported JEPA checkpoint format")
    config = checkpoint.get("model_config")
    state = checkpoint.get("model_state")
    if not isinstance(config, Mapping) or not isinstance(state, Mapping):
        raise ValueError("checkpoint must contain model_config and model_state")
    model = SpatialJEPA(**dict(config)).to(device)
    model.load_state_dict(state)
    return model


def _validate_transformer_settings(
    model_dim: int,
    num_heads: int,
    depth: int,
    dropout: float,
) -> None:
    if model_dim < 1 or num_heads < 1 or depth < 1:
        raise ValueError("model_dim, num_heads, and depth must be positive")
    if model_dim % num_heads:
        raise ValueError("model_dim must be divisible by num_heads")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")


def _check_cell_inputs(
    embeddings: torch.Tensor,
    coordinates: torch.Tensor,
    embedding_dim: int,
    name: str,
) -> None:
    if embeddings.ndim != 3 or embeddings.shape[-1] != embedding_dim:
        raise ValueError(
            f"{name} embeddings must have shape [batch, cells, {embedding_dim}]"
        )
    if coordinates.shape != (*embeddings.shape[:2], 2):
        raise ValueError(f"{name} coordinates must have shape [batch, cells, 2]")
    if coordinates.device != embeddings.device:
        raise ValueError(f"{name} embeddings and coordinates must share a device")


def _validate_mask(
    mask: torch.Tensor | None,
    shape: torch.Size | tuple[int, int],
    device: torch.device,
) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.dtype != torch.bool or tuple(mask.shape) != tuple(shape):
        raise ValueError(f"mask must be boolean with shape {tuple(shape)}")
    if mask.device != device:
        raise ValueError("mask and inputs must share a device")
    if not mask.any(dim=1).all():
        raise ValueError("every batch item must contain at least one valid cell")
    return mask


def _padding_mask(valid_mask: torch.Tensor | None) -> torch.Tensor | None:
    return None if valid_mask is None else ~valid_mask


def _zero_padding(
    values: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> torch.Tensor:
    if valid_mask is None:
        return values
    return values.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
