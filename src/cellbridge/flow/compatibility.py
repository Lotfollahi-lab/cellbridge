"""Native flow-error compatibility matrices and SoftREPA loss."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


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
    valid_negatives: torch.Tensor | None = None,
) -> torch.Tensor:
    logits = softrepa_logits(errors, scale=scale, temperature=temperature)
    if valid_negatives is not None:
        if valid_negatives.shape != errors.shape or valid_negatives.dtype != torch.bool:
            raise ValueError("valid_negatives must be a boolean error-matrix mask")
        logits = logits.masked_fill(~valid_negatives, torch.finfo(logits.dtype).min)
    labels = torch.arange(len(errors), device=errors.device)
    return F.cross_entropy(logits, labels)


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
        valid &= ~(positions[None, :, None] == neighbours[:, None, :]).any(dim=-1)
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


def summarize_error_matrices(matrices: list[np.ndarray]) -> dict[str, float]:
    """Summarize matched/shuffled separation and within-batch retrieval."""

    if not matrices:
        raise ValueError("at least one error matrix is required")
    matched, mismatched, top1, top5, better = [], [], [], [], []
    for errors in matrices:
        if errors.ndim != 2 or errors.shape[0] != errors.shape[1]:
            raise ValueError("diagnostic matrices must be square")
        size = len(errors)
        diagonal = np.diag(errors)
        off_diagonal = errors[~np.eye(size, dtype=bool)]
        ranks = np.argsort(errors, axis=1)
        expected = np.arange(size)
        matched.append(diagonal)
        mismatched.append(off_diagonal)
        top1.append(ranks[:, 0] == expected)
        top5.append((ranks[:, : min(5, size)] == expected[:, None]).any(axis=1))
        row_mismatch = (errors.sum(axis=1) - diagonal) / (size - 1)
        better.append(diagonal < row_mismatch)
    match = np.concatenate(matched)
    mismatch = np.concatenate(mismatched)
    mismatch_mean = float(mismatch.mean())
    return {
        "matched_error_mean": float(match.mean()),
        "mismatched_error_mean": mismatch_mean,
        "relative_error_reduction": float((mismatch_mean - match.mean()) / mismatch_mean),
        "matched_better_fraction": float(np.concatenate(better).mean()),
        "retrieval_top1": float(np.concatenate(top1).mean()),
        "retrieval_top5": float(np.concatenate(top5).mean()),
        "retrieval_candidates": int(matrices[0].shape[0]),
        "chance_top1": float(1.0 / matrices[0].shape[0]),
        "n_states": int(sum(len(matrix) for matrix in matrices)),
    }
