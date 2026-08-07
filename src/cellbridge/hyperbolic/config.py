"""Minimal configuration defaults for Euclidean/Lorentz alignment."""

from __future__ import annotations


def configure_geometry(cfg: dict) -> dict:
    geometry = cfg.setdefault("geometry", {})
    geometry.setdefault("mode", "euclidean")
    if geometry["mode"] not in {"euclidean", "hyperbolic"}:
        raise ValueError("geometry.mode must be 'euclidean' or 'hyperbolic'")

    default_dim = 64 if geometry["mode"] == "hyperbolic" else cfg["model"]["terra_dim"]
    geometry.setdefault("dim", default_dim)
    geometry.setdefault("curvature", 1.0)
    geometry.setdefault("learnable_curvature", True)
    geometry.setdefault("min_curvature", 1e-3)
    geometry.setdefault("max_curvature", 10.0)
    geometry.setdefault("max_tangent_norm", 12.0)
    geometry.setdefault("ontology_init_radius", 0.5)
    if int(geometry["dim"]) < 1:
        raise ValueError("geometry.dim must be positive")
    if not isinstance(geometry["learnable_curvature"], bool):
        raise ValueError("geometry.learnable_curvature must be boolean")
    curvature = float(geometry["curvature"])
    minimum = float(geometry["min_curvature"])
    maximum = float(geometry["max_curvature"])
    if not 0 < minimum <= curvature <= maximum:
        raise ValueError("geometry.curvature must lie within its positive bounds")
    if (
        min(
            float(geometry["max_tangent_norm"]),
            float(geometry["ontology_init_radius"]),
        )
        <= 0
    ):
        raise ValueError("geometry norm and ontology initialization must be positive")

    hierarchy = cfg.setdefault("hierarchy", {"enabled": False})
    cones = hierarchy.setdefault("cones", {})
    cones.setdefault("enabled", False)
    cones.setdefault("K", 0.1)
    cones.setdefault("lambda_cone", 0.1)
    if not isinstance(cones["enabled"], bool):
        raise ValueError("hierarchy.cones.enabled must be boolean")
    if float(cones["K"]) <= 0:
        raise ValueError("hierarchy.cones.K must be positive")
    if float(cones["lambda_cone"]) < 0:
        raise ValueError("hierarchy.cones.lambda_cone must be non-negative")

    if geometry["mode"] == "hyperbolic" and cfg["objective"]["targets"] != "exact":
        raise ValueError("hyperbolic alignment requires objective.targets: exact")
    if geometry["mode"] == "hyperbolic" and hierarchy.get("enabled", False):
        if hierarchy.get("negative_downweighting", {}).get("enabled", False):
            raise ValueError(
                "hyperbolic alignment cannot use Euclidean negative weighting"
            )
        if hierarchy.get("ranking", {}).get("enabled", False):
            raise ValueError("hyperbolic alignment cannot use Euclidean ranking")

    if cones["enabled"]:
        if geometry["mode"] != "hyperbolic":
            raise ValueError("entailment cones require geometry.mode: hyperbolic")
        if not hierarchy.get("enabled", False):
            raise ValueError("entailment cones require hierarchy.enabled: true")
        if not hierarchy.get("nodes_path"):
            raise ValueError("entailment cones require hierarchy.nodes_path")
        if hierarchy["negative_downweighting"]["enabled"]:
            raise ValueError("entailment cones cannot use Euclidean negative weighting")
        if hierarchy["ranking"]["enabled"]:
            raise ValueError("entailment cones replace the Euclidean ranking loss")

    training = cfg["training"]
    training.setdefault("lr_manifold", training["lr_heads"])
    training.setdefault("lr_curvature", training["lr_heads"])
    if min(float(training["lr_manifold"]), float(training["lr_curvature"])) <= 0:
        raise ValueError("manifold and curvature learning rates must be positive")
    return cfg
