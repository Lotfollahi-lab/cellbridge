"""A small cell-level adaptation of STFlow with SoftREPA prompt hooks."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from cellbridge.jepa.model import CrossAttentionBlock, SpatialContextEncoder


class TimestepEmbedder(nn.Module):
    """STFlow's sinusoidal timestep embedding."""

    def __init__(self, model_dim: int, frequency_dim: int = 256) -> None:
        super().__init__()
        self.frequency_dim = int(frequency_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.frequency_dim // 2
        frequencies = torch.exp(
            -math.log(10_000)
            * torch.arange(half, device=time.device, dtype=torch.float32)
            / half
        )
        angles = time.float()[:, None] * frequencies[None]
        embedding = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
        if self.frequency_dim % 2:
            embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), -1)
        return self.mlp(embedding)


class SoftPromptTokens(nn.Module):
    """Layer-specific SoftREPA conditioning tokens with optional time dependence."""

    def __init__(
        self,
        depth: int,
        token_count: int,
        model_dim: int,
        *,
        time_dependent: bool = False,
        time_bins: int = 100,
    ) -> None:
        super().__init__()
        if min(depth, token_count, model_dim, time_bins) < 1:
            raise ValueError("prompt dimensions and time_bins must be positive")
        self.depth = int(depth)
        self.token_count = int(token_count)
        self.model_dim = int(model_dim)
        self.time_bins = int(time_bins)
        self.tokens = nn.Parameter(torch.empty(depth, token_count, model_dim))
        nn.init.normal_(self.tokens, std=0.02)
        self.time_tokens = (
            nn.Embedding(time_bins, depth * model_dim) if time_dependent else None
        )
        if self.time_tokens is not None:
            nn.init.normal_(self.time_tokens.weight, std=0.02)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        values = self.tokens[None].expand(len(time), -1, -1, -1)
        if self.time_tokens is None:
            return values
        indices = torch.clamp((time * self.time_bins).long(), max=self.time_bins - 1)
        offsets = self.time_tokens(indices).view(
            len(time), self.depth, 1, self.model_dim
        )
        return values + offsets


class CellFlow(nn.Module):
    """Predict a center cell's clean TERRA endpoint from its H&E neighbourhood."""

    def __init__(
        self,
        image_dim: int = 1536,
        target_dim: int = 384,
        model_dim: int = 128,
        num_heads: int = 4,
        context_depth: int = 2,
        predictor_depth: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if min(image_dim, target_dim, model_dim, context_depth, predictor_depth) < 1:
            raise ValueError("model dimensions and depths must be positive")
        self.image_dim = int(image_dim)
        self.target_dim = int(target_dim)
        self.model_dim = int(model_dim)
        self.predictor_depth = int(predictor_depth)
        self.context_encoder = SpatialContextEncoder(
            image_dim, model_dim, num_heads, context_depth, dropout
        )
        self.state_projection = nn.Linear(target_dim, model_dim)
        self.time_embedding = TimestepEmbedder(model_dim)
        self.blocks = nn.ModuleList(
            CrossAttentionBlock(model_dim, num_heads, dropout)
            for _ in range(predictor_depth)
        )
        self.output = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, target_dim))

    def forward(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
        coordinates: torch.Tensor,
        *,
        prompts: SoftPromptTokens | None = None,
    ) -> torch.Tensor:
        if state.ndim != 2 or state.shape[-1] != self.target_dim:
            raise ValueError(f"state must have shape [batch, {self.target_dim}]")
        if time.shape != (len(state),):
            raise ValueError("time must have shape [batch]")
        if condition.ndim != 3 or condition.shape[-1] != self.image_dim:
            raise ValueError(
                f"condition must have shape [batch, cells, {self.image_dim}]"
            )
        if coordinates.shape != (*condition.shape[:2], 2):
            raise ValueError("coordinates must match condition with final dimension two")
        context = self.context_encoder(condition, coordinates)
        query = (self.state_projection(state) + self.time_embedding(time))[:, None]
        prompt_values = None if prompts is None else prompts(time)
        if prompt_values is not None and prompt_values.shape[1] != len(self.blocks):
            raise ValueError("prompt depth must equal predictor depth")
        for index, block in enumerate(self.blocks):
            layer_context = context
            if prompt_values is not None:
                layer_context = torch.cat((prompt_values[:, index], context), dim=1)
            query = block(query, layer_context, None)
        return self.output(query[:, 0])

    @property
    def config(self) -> dict[str, int]:
        layer = self.context_encoder.transformer.layers[0]
        return {
            "image_dim": self.image_dim,
            "target_dim": self.target_dim,
            "model_dim": self.model_dim,
            "num_heads": layer.self_attn.num_heads,
            "context_depth": len(self.context_encoder.transformer.layers),
            "predictor_depth": self.predictor_depth,
            "dropout": float(layer.dropout.p),
        }
