"""Geodesic contrastive and Lorentz entailment-cone objectives."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .manifold import LorentzOps


def _retrieval_metrics(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    labels = torch.arange(len(logits), device=logits.device)
    top_k = min(5, len(logits))
    image_top = logits.topk(top_k, dim=1).indices
    terra_top = logits.T.topk(top_k, dim=1).indices
    return {
        "contrastive_top1": 0.5
        * (
            (image_top[:, 0] == labels).float().mean()
            + (terra_top[:, 0] == labels).float().mean()
        ),
        "contrastive_top5": 0.5
        * (
            (image_top == labels[:, None]).any(dim=1).float().mean()
            + (terra_top == labels[:, None]).any(dim=1).float().mean()
        ),
    }


def hyperbolic_contrastive_loss(
    image_points: torch.Tensor,
    terra_points: torch.Tensor,
    manifold: LorentzOps,
    temperature: float = 0.07,
    *,
    bidirectional: bool = True,
) -> dict[str, torch.Tensor]:
    """CLIP-style InfoNCE with negative geodesic distance as similarity."""

    if float(temperature) <= 0:
        raise ValueError("temperature must be positive")
    if (
        image_points.ndim != 2
        or terra_points.ndim != 2
        or image_points.shape != terra_points.shape
        or len(image_points) == 0
    ):
        raise ValueError("Lorentz inputs must share a non-empty [batch, feature] shape")
    logits = -manifold.pairwise_distance(image_points, terra_points) / float(
        temperature
    )
    labels = torch.arange(len(logits), device=logits.device)
    loss = F.cross_entropy(logits, labels)
    if bidirectional:
        loss = 0.5 * (loss + F.cross_entropy(logits.T, labels))
    return {"loss_contrastive": loss, **_retrieval_metrics(logits)}


def cone_half_aperture(
    apex: torch.Tensor,
    manifold: LorentzOps,
    K: float,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Ganea/MERU half-apertures and an unsaturated-cone mask.

    For Poincare norm ``r``, the requested expression is
    ``asin(K * (1 - r**2) / r)``. Its exactly equivalent Lorentz expression is
    ``asin(2K / (sqrt(c) * ||x_space||))``; using it keeps all learned points in
    the numerically stable Lorentz model.
    """

    if float(K) <= 0:
        raise ValueError("K must be positive")
    manifold._check_point(apex)
    apex = manifold.project(apex)
    with torch.autocast(device_type=apex.device.type, enabled=False):
        spatial_norm = apex[..., 1:].norm(dim=-1)
        argument = (
            2.0
            * float(K)
            / (manifold.curvature.sqrt() * spatial_norm.clamp_min(float(eps)))
        )
        active = argument < 1.0 - float(eps)
        aperture = torch.asin(argument.clamp(min=-1.0 + eps, max=1.0 - eps))
    return aperture, active


def cone_exterior_angle(
    apex: torch.Tensor,
    descendant: torch.Tensor,
    manifold: LorentzOps,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Exterior angle at ``apex`` in the hyperbolic triangle O-apex-cell."""

    manifold._check_point(apex)
    manifold._check_point(descendant)
    if apex.shape != descendant.shape:
        raise ValueError("cone apex and descendant tensors must have equal shapes")
    apex = manifold.project(apex)
    descendant = manifold.project(descendant)
    with torch.autocast(device_type=apex.device.type, enabled=False):
        inner = -apex[..., 0] * descendant[..., 0] + (
            apex[..., 1:] * descendant[..., 1:]
        ).sum(dim=-1)
        scaled_inner = manifold.curvature * inner
        numerator = descendant[..., 0] + apex[..., 0] * scaled_inner
        denominator = (
            apex[..., 1:].norm(dim=-1)
            * (scaled_inner.square() - 1.0).clamp_min(float(eps)).sqrt()
        )
        cosine = numerator / denominator.clamp_min(float(eps))
        return torch.acos(cosine.clamp(min=-1.0 + eps, max=1.0 - eps))


def entailment_cone_loss(
    cell_points: torch.Tensor,
    ancestor_points: torch.Tensor,
    manifold: LorentzOps,
    K: float,
    *,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Push each selected cell point inside each selected ancestor cone."""

    if cell_points.shape != ancestor_points.shape:
        raise ValueError("cell and ancestor points must have identical shapes")
    if valid_mask is None:
        valid_mask = torch.ones(
            cell_points.shape[:-1], dtype=torch.bool, device=cell_points.device
        )
    if valid_mask.shape != cell_points.shape[:-1]:
        raise ValueError("cone valid_mask must match every non-feature dimension")
    valid_mask = valid_mask.bool()
    if not bool(valid_mask.any()):
        zero = cell_points.sum() * 0
        return _cone_metrics(zero, 0, 0, 0)

    cells = cell_points[valid_mask]
    ancestors = ancestor_points[valid_mask]
    aperture, active = cone_half_aperture(ancestors, manifold, K)
    angle = cone_exterior_angle(ancestors, cells, manifold)
    penalties = F.relu(angle - aperture)
    return _cone_metrics(
        penalties.mean(),
        len(penalties),
        int(active.sum().item()),
        int((angle > aperture).sum().item()),
    )


def multimodal_entailment_cone_loss(
    modalities: Sequence[torch.Tensor],
    ancestor_points: torch.Tensor,
    manifold: LorentzOps,
    K: float,
    *,
    valid_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Average the cone loss over aligned image and TERRA cell embeddings."""

    if not modalities:
        raise ValueError("at least one cell modality is required")
    results = [
        entailment_cone_loss(
            points, ancestor_points, manifold, K, valid_mask=valid_mask
        )
        for points in modalities
    ]
    loss = torch.stack([result["loss_cone"] for result in results]).mean()
    return _cone_metrics(
        loss,
        sum(int(result["cone_pairs"].item()) for result in results),
        sum(int(result["cone_active_pairs"].item()) for result in results),
        sum(int(result["cone_violations"].item()) for result in results),
    )


def _cone_metrics(
    loss: torch.Tensor, pairs: int, active: int, violations: int
) -> dict[str, torch.Tensor]:
    def scalar(value):
        return loss.new_tensor(float(value)).detach()

    return {
        "loss_cone": loss,
        "cone_pairs": scalar(pairs),
        "cone_active_pairs": scalar(active),
        "cone_violations": scalar(violations),
        "cone_activation_rate": scalar(active / pairs if pairs else 0.0),
        "cone_violation_rate": scalar(violations / pairs if pairs else 0.0),
    }
