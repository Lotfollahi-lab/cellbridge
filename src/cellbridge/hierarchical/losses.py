"""Phenotype-aware pair weighting and hierarchy-ranking objectives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F

RELATION_ORDER = ("match", "identity", "lineage", "compartment", "unrelated")


def relation_masks(
    levels: torch.Tensor,
    depth: torch.Tensor,
    usable: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Classify pairs by their deepest jointly valid hierarchy level."""

    if levels.ndim != 2 or levels.shape[1] != 4:
        raise ValueError("hierarchy levels must have shape [batch, 4]")
    if depth.shape != (len(levels),) or usable.shape != (len(levels),):
        raise ValueError("hierarchy depth and usable masks must have shape [batch]")
    if torch.any((depth < 0) | (depth > 3)):
        raise ValueError("hierarchy depth must be in [0, 3]")

    usable = usable.bool()
    pair = usable[:, None] & usable[None, :]
    diagonal = torch.eye(len(levels), dtype=torch.bool, device=levels.device)
    off_diagonal = pair & ~diagonal
    identity = (
        off_diagonal
        & (depth[:, None] >= 3)
        & (depth[None, :] >= 3)
        & levels[:, None, 3].eq(levels[None, :, 3])
    )
    lineage = (
        off_diagonal
        & (depth[:, None] >= 2)
        & (depth[None, :] >= 2)
        & levels[:, None, 2].eq(levels[None, :, 2])
        & ~identity
    )
    compartment = (
        off_diagonal
        & (depth[:, None] >= 1)
        & (depth[None, :] >= 1)
        & levels[:, None, 1].eq(levels[None, :, 1])
        & ~identity
        & ~lineage
    )
    unrelated = (
        off_diagonal
        & (depth[:, None] >= 1)
        & (depth[None, :] >= 1)
        & ~levels[:, None, 1].eq(levels[None, :, 1])
    )
    return {
        "match": diagonal & usable[:, None],
        "identity": identity,
        "lineage": lineage,
        "compartment": compartment,
        "unrelated": unrelated,
    }


def negative_weight_matrix(
    relations: Mapping[str, torch.Tensor],
    relation_weights: Mapping[str, float],
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build denominator weights while leaving exact matches untouched."""

    size = len(next(iter(relations.values())))
    device = next(iter(relations.values())).device
    weights = torch.ones((size, size), dtype=dtype, device=device)
    for name, value in relation_weights.items():
        if name not in relations or name == "match":
            raise ValueError(f"cannot downweight unknown relation {name!r}")
        value = float(value)
        if not 0 <= value <= 1:
            raise ValueError("negative relation weights must be in [0, 1]")
        weights.masked_fill_(relations[name], value)
    weights.fill_diagonal_(1)
    return weights


def weighted_symmetric_pair_loss(
    image_z: torch.Tensor,
    terra_z: torch.Tensor,
    weights: torch.Tensor,
    temperature: float,
    *,
    bidirectional: bool = True,
) -> torch.Tensor:
    """Exact-pair InfoNCE with configurable off-diagonal denominator weights."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if image_z.ndim != 2 or image_z.shape != terra_z.shape:
        raise ValueError("image_z and terra_z must share a [batch, feature] shape")
    if weights.shape != (len(image_z), len(image_z)):
        raise ValueError("negative weights must have shape [batch, batch]")
    if torch.any(weights < 0):
        raise ValueError("negative weights cannot be negative")
    logits = F.normalize(image_z, dim=-1) @ F.normalize(terra_z, dim=-1).T
    logits = logits / temperature
    log_weights = torch.where(weights > 0, weights.log(), -torch.inf)

    def directional(values: torch.Tensor, log_weight: torch.Tensor) -> torch.Tensor:
        numerator = values.diagonal()
        denominator = torch.logsumexp(values + log_weight, dim=1)
        return (denominator - numerator).mean()

    loss = directional(logits, log_weights)
    if bidirectional:
        loss = 0.5 * (loss + directional(logits.T, log_weights.T))
    return loss


def _transition_margin(
    margins: Mapping[str, float], closer: str, farther: str
) -> float:
    key = f"{closer}>{farther}"
    if key not in margins:
        raise ValueError(f"missing hierarchy margin {key!r}")
    margin = float(margins[key])
    if margin < 0:
        raise ValueError("hierarchy margins must be non-negative")
    return margin


def _directional_ranking(
    similarities: torch.Tensor,
    relations: Mapping[str, torch.Tensor],
    sample_ids: Sequence[str],
    levels: Sequence[str],
    margins: Mapping[str, float],
    cross_sample: str,
) -> tuple[list[torch.Tensor], int, int, int]:
    means, triplets, violations, cross_triplets = [], 0, 0, 0
    samples = list(map(str, sample_ids))
    for closer, farther in zip(levels, levels[1:]):
        margin = _transition_margin(margins, closer, farther)
        transition = []
        for anchor in range(len(similarities)):
            near = relations[closer][anchor].clone()
            far = relations[farther][anchor].clone()
            cross = torch.tensor(
                [sample != samples[anchor] for sample in samples],
                dtype=torch.bool,
                device=similarities.device,
            )
            near_cross = near if closer == "match" else near & cross
            far_cross = far & cross
            if cross_sample == "only":
                near, far = near_cross, far_cross
            elif cross_sample == "prefer" and near_cross.any() and far_cross.any():
                near, far = near_cross, far_cross
            if not near.any() or not far.any():
                continue
            differences = similarities[anchor, near, None] - similarities[anchor, None, far]
            penalties = F.relu(margin - differences)
            transition.append(penalties.mean())
            count = penalties.numel()
            triplets += count
            violations += int((differences <= margin).sum().item())
            near_is_cross = cross[near]
            if closer == "match":
                near_is_cross = torch.ones_like(near_is_cross)
            cross_triplets += int(
                (near_is_cross[:, None] & cross[far][None, :]).sum().item()
            )
        if transition:
            means.append(torch.stack(transition).mean())
    return means, triplets, violations, cross_triplets


def symmetric_hierarchy_ranking_loss(
    image_z: torch.Tensor,
    terra_z: torch.Tensor,
    relations: Mapping[str, torch.Tensor],
    sample_ids: Sequence[str],
    *,
    levels: Sequence[str],
    margins: Mapping[str, float],
    temperature: float = 1.0,
    cross_sample: str = "prefer",
    min_triplets: int = 1,
    bidirectional: bool = True,
) -> dict[str, torch.Tensor]:
    """Order exact, phenotype, and ancestry relations in both modalities."""

    if temperature <= 0:
        raise ValueError("hierarchy temperature must be positive")
    if cross_sample not in {"any", "prefer", "only"}:
        raise ValueError("cross_sample must be 'any', 'prefer', or 'only'")
    if min_triplets < 1:
        raise ValueError("min_triplets must be positive")
    unknown = [name for name in levels if name not in RELATION_ORDER]
    positions = [RELATION_ORDER.index(name) for name in levels if name in RELATION_ORDER]
    if unknown or len(levels) < 2 or positions != sorted(set(positions)):
        raise ValueError("levels must be a unique coarse-order subsequence")
    similarities = F.normalize(image_z, dim=-1) @ F.normalize(terra_z, dim=-1).T
    similarities = similarities / temperature
    directions = [similarities, similarities.T] if bidirectional else [similarities]
    means, triplets, violations, cross_triplets = [], 0, 0, 0
    for values in directions:
        result = _directional_ranking(
            values, relations, sample_ids, levels, margins, cross_sample
        )
        means.extend(result[0])
        triplets += result[1]
        violations += result[2]
        cross_triplets += result[3]
    sufficient = triplets >= min_triplets
    loss = torch.stack(means).mean() if means and sufficient else similarities.sum() * 0
    scalar = lambda value: similarities.new_tensor(float(value))
    return {
        "loss_hier": loss,
        "hierarchy_triplets": scalar(triplets),
        "hierarchy_active_transitions": scalar(len(means)),
        "hierarchy_violation_rate": scalar(violations / triplets if triplets else 0),
        "hierarchy_cross_sample_fraction": scalar(
            cross_triplets / triplets if triplets else 0
        ),
        "hierarchy_sufficient": scalar(sufficient),
    }
