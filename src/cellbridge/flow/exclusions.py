"""Config-driven false-negative exclusions for SoftREPA batches."""

from __future__ import annotations

from cellbridge.flow.compatibility import false_negative_mask, nearest_terra_indices


def build_exclusion_index(dataset, settings, device):
    if not settings or not settings.get("enabled", False):
        return None
    return nearest_terra_indices(
        dataset.anchor_terra,
        int(settings["nearest_terra"]),
        device,
        chunk_size=int(settings.get("chunk_size", 512)),
    )


def batch_negative_mask(batch, nearest, settings):
    if not settings or not settings.get("enabled", False):
        return None
    radius = settings.get("spatial_radius_um")
    return false_negative_mask(
        batch["anchor_position"],
        batch["center_coordinates"],
        nearest,
        spatial_radius_um=None if radius is None else float(radius),
    )
