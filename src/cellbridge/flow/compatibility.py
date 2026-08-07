"""Native flow-error compatibility matrices and SoftREPA loss."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from cellbridge.flow.diagnostics import summarize_error_matrices


def endpoint_error_matrix(
    model,
    state: torch.Tensor,
    target: torch.Tensor,
    time: torch.Tensor,
    conditions: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    prompts=None,
    candidate_chunk: int = 8,
) -> torch.Tensor:
    """Evaluate every target state ``i`` under every H&E condition ``j``."""

    batch = len(state)
    if target.shape != state.shape or time.shape != (batch,):
        raise ValueError("state, target, and time have incompatible shapes")
    if len(conditions) != batch or len(coordinates) != batch:
        raise ValueError("conditions and coordinates must share the state batch")
    if candidate_chunk < 1:
        raise ValueError("candidate_chunk must be positive")
    columns = []
    for start in range(0, batch, candidate_chunk):
        stop = min(start + candidate_chunk, batch)
        width = stop - start
        expanded_state = state[:, None].expand(-1, width, -1).reshape(-1, state.shape[-1])
        expanded_target = target[:, None].expand(-1, width, -1)
        expanded_time = time[:, None].expand(-1, width).reshape(-1)
        condition = conditions[None, start:stop].expand(batch, -1, -1, -1)
        condition = condition.reshape(-1, *conditions.shape[1:])
        relative = coordinates[None, start:stop].expand(batch, -1, -1, -1)
        relative = relative.reshape(-1, *coordinates.shape[1:])
        prediction = model(
            expanded_state,
            expanded_time,
            condition,
            relative,
            prompts=prompts,
        ).view(batch, width, -1)
        columns.append((prediction - expanded_target).square().mean(dim=-1))
    return torch.cat(columns, dim=1)


def softrepa_logits(errors: torch.Tensor, *, scale: float, temperature: float) -> torch.Tensor:
    """Use SoftREPA's checked-in ``scale * exp(-error / temperature)`` logits."""

    if errors.ndim != 2 or errors.shape[0] != errors.shape[1]:
        raise ValueError("errors must be a square matrix")
    if scale <= 0 or temperature <= 0:
        raise ValueError("scale and temperature must be positive")
    return float(scale) * torch.exp(-errors / float(temperature))


def softrepa_loss(
    errors: torch.Tensor,
    *,
    scale: float,
    temperature: float,
    diagonal_weight: float = 0.0,
    valid_negatives: torch.Tensor | None = None,
) -> torch.Tensor:
    """Combine SoftREPA compatibility CE and its optional diagonal anchor."""

    if diagonal_weight < 0:
        raise ValueError("diagonal_weight must be non-negative")
    logits = softrepa_logits(errors, scale=scale, temperature=temperature)
    if valid_negatives is not None:
        if valid_negatives.shape != errors.shape or valid_negatives.dtype != torch.bool:
            raise ValueError("valid_negatives must be a boolean error-matrix mask")
        logits = logits.masked_fill(~valid_negatives, torch.finfo(logits.dtype).min)
    labels = torch.arange(len(errors), device=errors.device)
    contrastive = F.cross_entropy(logits, labels)
    return contrastive + float(diagonal_weight) * errors.diag().mean()


def false_negative_mask(
    positions: torch.Tensor,
    center_coordinates: torch.Tensor,
    nearest_terra: torch.Tensor | None,
    *,
    spatial_radius_um: float | None,
) -> torch.Tensor:
    """Keep the diagonal and exclude configured TERRA/spatial false negatives."""

    batch = len(positions)
    valid = torch.ones((batch, batch), dtype=torch.bool, device=positions.device)
    if nearest_terra is not None:
        neighbours = nearest_terra[positions]
        near = (positions[None, :, None] == neighbours[:, None, :]).any(dim=-1)
        valid &= ~(near | near.T)
    if spatial_radius_um is not None:
        if spatial_radius_um <= 0:
            raise ValueError("spatial_radius_um must be positive or None")
        distances = torch.cdist(center_coordinates.float(), center_coordinates.float())
        valid &= distances > float(spatial_radius_um)
    valid.fill_diagonal_(True)
    return valid


@torch.no_grad()
def nearest_terra_indices(
    targets: np.ndarray,
    neighbours: int,
    device: torch.device,
    *,
    chunk_size: int = 512,
) -> torch.Tensor | None:
    """Find exact cosine-nearest frozen TERRA cells in bounded GPU chunks."""

    if neighbours == 0:
        return None
    if neighbours < 0 or neighbours >= len(targets):
        raise ValueError("nearest TERRA neighbour count is invalid")
    values = F.normalize(torch.as_tensor(targets, dtype=torch.float32, device=device), dim=-1)
    rows = []
    for start in range(0, len(values), chunk_size):
        similarities = values[start : start + chunk_size] @ values.T
        local = torch.arange(
            start,
            min(start + chunk_size, len(values)),
            device=device,
        )
        similarities[torch.arange(len(local), device=device), local] = -torch.inf
        rows.append(similarities.topk(neighbours, dim=1).indices.cpu())
    return torch.cat(rows).to(device)
