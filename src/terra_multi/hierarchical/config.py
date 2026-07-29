"""Configuration defaults and validation for hierarchy-aware LoRA runs."""

from __future__ import annotations

from .losses import RELATION_ORDER


def configure_hierarchy(cfg: dict) -> dict:
    hierarchy = cfg.setdefault("hierarchy", {})
    hierarchy.setdefault("enabled", False)
    if not isinstance(hierarchy["enabled"], bool):
        raise ValueError("hierarchy.enabled must be true or false")
    if not hierarchy["enabled"]:
        return cfg
    for key in ("split_manifest", "labels_dir"):
        if not hierarchy.get(key):
            raise ValueError(f"hierarchy.{key} is required when enabled")

    downweight = hierarchy.setdefault("negative_downweighting", {})
    downweight.setdefault("enabled", False)
    downweight.setdefault("weights", {"identity": 0.25, "lineage": 0.5})
    if not isinstance(downweight["enabled"], bool):
        raise ValueError("hierarchy.negative_downweighting.enabled must be boolean")
    for relation, weight in downweight["weights"].items():
        if relation not in RELATION_ORDER[1:]:
            raise ValueError(f"unknown negative-downweight relation {relation!r}")
        if not 0 <= float(weight) <= 1:
            raise ValueError("negative-downweight weights must be in [0, 1]")
    if downweight["enabled"] and cfg["objective"]["targets"] != "exact":
        raise ValueError("negative downweighting requires objective.targets: exact")

    ranking = hierarchy.setdefault("ranking", {})
    ranking.setdefault("enabled", False)
    ranking.setdefault("lambda_hier", 0.1)
    ranking.setdefault("levels", ["match", "identity", "lineage", "unrelated"])
    ranking.setdefault(
        "margins",
        {"match>identity": 0.05, "identity>lineage": 0.05, "lineage>unrelated": 0.05},
    )
    ranking.setdefault("temperature", 1.0)
    ranking.setdefault("cross_sample", "prefer")
    ranking.setdefault("min_triplets", 1)
    ranking.setdefault("bidirectional", True)
    if not isinstance(ranking["enabled"], bool):
        raise ValueError("hierarchy.ranking.enabled must be boolean")
    if float(ranking["lambda_hier"]) < 0 or float(ranking["temperature"]) <= 0:
        raise ValueError("hierarchy ranking weight/temperature is invalid")
    levels = ranking["levels"]
    positions = [
        RELATION_ORDER.index(level) for level in levels if level in RELATION_ORDER
    ]
    if (
        len(levels) < 2
        or len(positions) != len(levels)
        or positions != sorted(set(positions))
    ):
        raise ValueError("hierarchy ranking levels must be unique and fine-to-coarse")
    for closer, farther in zip(levels, levels[1:]):
        key = f"{closer}>{farther}"
        if key not in ranking["margins"] or float(ranking["margins"][key]) < 0:
            raise ValueError(f"hierarchy ranking requires a non-negative margin {key}")
    if ranking["cross_sample"] not in {"any", "prefer", "only"}:
        raise ValueError("hierarchy.ranking.cross_sample is invalid")
    if int(ranking["min_triplets"]) < 1:
        raise ValueError("hierarchy.ranking.min_triplets must be positive")
    if not isinstance(ranking["bidirectional"], bool):
        raise ValueError("hierarchy.ranking.bidirectional must be boolean")

    sampling = hierarchy.setdefault("sampling", {})
    sampling.setdefault("strategy", "balanced")
    sampling.setdefault("validation_strategy", "balanced")
    sampling.setdefault("relations", ["identity", "lineage", "unrelated"])
    sampling.setdefault("cross_sample", "prefer")
    sampling.setdefault("annotated_batch_fraction", None)
    if sampling["strategy"] not in {"balanced", "hierarchy"}:
        raise ValueError("hierarchy.sampling.strategy must be balanced or hierarchy")
    if sampling["validation_strategy"] not in {"balanced", "hierarchy"}:
        raise ValueError(
            "hierarchy.sampling.validation_strategy must be balanced or hierarchy"
        )
    if any(name not in RELATION_ORDER[1:] for name in sampling["relations"]):
        raise ValueError("hierarchy.sampling.relations contains an unknown relation")
    if sampling["cross_sample"] not in {"any", "prefer", "only"}:
        raise ValueError("hierarchy.sampling.cross_sample is invalid")
    annotated_fraction = sampling["annotated_batch_fraction"]
    if annotated_fraction is not None and not 0 < float(annotated_fraction) < 1:
        raise ValueError(
            "hierarchy.sampling.annotated_batch_fraction must be between 0 and 1"
        )
    cfg["training"].setdefault("selection_metric", "breast_val/loss_total")
    return cfg
