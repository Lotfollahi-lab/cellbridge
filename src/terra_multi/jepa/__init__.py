"""Spatial joint-embedding predictive alignment."""

from terra_multi.jepa.model import (
    CHECKPOINT_FORMAT,
    SpatialJEPA,
    TargetNormalizer,
    jepa_loss,
    load_model_state,
)

__all__ = [
    "CHECKPOINT_FORMAT",
    "SpatialJEPA",
    "TargetNormalizer",
    "jepa_loss",
    "load_model_state",
]
