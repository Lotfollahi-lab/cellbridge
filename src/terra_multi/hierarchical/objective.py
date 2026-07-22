"""Compose the existing paired objective with optional hierarchy supervision."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from terra_multi.lora.model import compute_loss

from .losses import (
    negative_weight_matrix,
    relation_masks,
    symmetric_hierarchy_ranking_loss,
    weighted_symmetric_pair_loss,
)


def _empty_metrics(reference: torch.Tensor) -> dict[str, torch.Tensor]:
    zero = reference.sum() * 0
    return {
        "loss_hier": zero,
        "hierarchy_triplets": zero.detach(),
        "hierarchy_active_transitions": zero.detach(),
        "hierarchy_violation_rate": zero.detach(),
        "hierarchy_cross_sample_fraction": zero.detach(),
        "hierarchy_sufficient": zero.detach(),
        "hierarchy_usable_fraction": zero.detach(),
        "negative_downweighted_fraction": zero.detach(),
    }


def compute_hierarchy_objective(
    outputs: Mapping[str, torch.Tensor],
    terra_target: torch.Tensor,
    batch: Mapping,
    objective: Mapping,
    hierarchy: Mapping,
    *,
    gex_target: torch.Tensor | None = None,
    gex_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Keep exact pairs primary and add only mapped auxiliary assumptions."""

    metrics = compute_loss(
        outputs,
        terra_target,
        gex_target,
        gex_mask,
        **objective,
    )
    projected = outputs["projected"]
    auxiliary = _empty_metrics(projected)
    required = {"hierarchy_levels", "hierarchy_depth", "hierarchy_usable"}
    if not required <= set(batch):
        metrics.update(auxiliary)
        metrics["loss_pair"] = metrics["loss_contrastive"]
        return metrics

    levels = batch["hierarchy_levels"].to(projected.device)
    depth = batch["hierarchy_depth"].to(projected.device)
    usable = batch["hierarchy_usable"].to(projected.device).bool()
    relations = relation_masks(levels, depth, usable)
    auxiliary["hierarchy_usable_fraction"] = usable.float().mean()

    downweight = hierarchy["negative_downweighting"]
    if downweight["enabled"]:
        weights = negative_weight_matrix(
            relations, downweight["weights"], dtype=projected.dtype
        )
        off_diagonal = ~torch.eye(
            len(weights), dtype=torch.bool, device=weights.device
        )
        softened = weights[off_diagonal] < 1
        auxiliary["negative_downweighted_fraction"] = softened.float().mean()
        if bool(softened.any()):
            weighted = weighted_symmetric_pair_loss(
                projected,
                terra_target,
                weights,
                float(objective["temperature"]),
                bidirectional=bool(objective["bidirectional"]),
            )
            old_pair = metrics["loss_contrastive"]
            metrics["loss_contrastive"] = weighted
            metrics["loss_total"] = metrics["loss_total"] + float(
                objective["lambda_contrastive"]
            ) * (weighted - old_pair)

    ranking = hierarchy["ranking"]
    if ranking["enabled"]:
        ranked = symmetric_hierarchy_ranking_loss(
            projected,
            terra_target,
            relations,
            batch["sample_id"],
            levels=ranking["levels"],
            margins=ranking["margins"],
            temperature=float(ranking["temperature"]),
            cross_sample=ranking["cross_sample"],
            min_triplets=int(ranking["min_triplets"]),
            bidirectional=bool(ranking["bidirectional"]),
        )
        auxiliary.update(ranked)
        metrics["loss_total"] = metrics["loss_total"] + float(
            ranking["lambda_hier"]
        ) * ranked["loss_hier"]

    metrics.update(auxiliary)
    metrics["loss_pair"] = metrics["loss_contrastive"]
    return metrics
