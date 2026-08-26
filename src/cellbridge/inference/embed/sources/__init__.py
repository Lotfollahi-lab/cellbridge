"""Dataset sources for shared embedding inference."""

from dataclasses import dataclass
from importlib import import_module

import numpy as np

FORMATS = ("tiled_lmdb", "nucls", "hest")
_MODULES = {
    "tiled_lmdb": "lmdb",
    "nucls": "nucls",
    "hest": "hest",
}


@dataclass(frozen=True)
class SourceSample:
    """One independently stored group of aligned cell crops."""

    sample_id: str
    row_ids: np.ndarray
    dataset: object


def load(data):
    """Load the source module selected by ``data.format``."""
    name = data.get("format", "tiled_lmdb")
    try:
        module = _MODULES[name]
    except KeyError as error:
        raise ValueError(
            f"unknown data format {name!r}; choose from {FORMATS}"
        ) from error
    return import_module(f"{__name__}.{module}")
