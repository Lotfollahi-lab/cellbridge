"""Run an embedding method over an ordered dataset of image crops."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EmbeddingResult:
    cell_ids: np.ndarray
    embeddings: dict[str, np.ndarray]


def _numpy(values) -> np.ndarray:
    """Move a tensor-like batch to a finite float32 NumPy array."""
    if hasattr(values, "detach"):
        values = values.detach().float().cpu().numpy()
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(f"embeddings must be a finite 2D array, got {values.shape}")
    return values


def embed(
    dataset,
    embedder,
    *,
    batch_size: int = 128,
    num_workers: int = 0,
) -> EmbeddingResult:
    """Embed every crop once while preserving dataset row order."""
    import torch

    if len(dataset) == 0:
        raise ValueError("cannot embed an empty dataset")
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=embedder.device.type == "cuda",
    )
    cell_ids = []
    embeddings = None
    offset = 0
    for batch in loader:
        if "image" not in batch or "cell_id" not in batch:
            raise ValueError("embedding batches require image and cell_id")
        outputs = {
            name: _numpy(values)
            for name, values in embedder.embed(batch["image"]).items()
        }
        if not outputs:
            raise ValueError("the embedder returned no representations")

        rows = len(batch["cell_id"])
        if any(len(values) != rows for values in outputs.values()):
            raise ValueError("embedding rows do not match the input batch")
        if embeddings is None:
            embeddings = {
                name: np.empty((len(dataset), values.shape[1]), dtype=np.float32)
                for name, values in outputs.items()
            }
        elif outputs.keys() != embeddings.keys():
            raise ValueError("the embedder returned inconsistent representation names")
        elif any(outputs[name].shape[1] != embeddings[name].shape[1] for name in outputs):
            raise ValueError("the embedder returned inconsistent feature dimensions")

        for name, values in outputs.items():
            embeddings[name][offset : offset + rows] = values
        cell_ids.extend(map(str, batch["cell_id"]))
        offset += rows

    if offset != len(dataset):
        raise ValueError(f"dataset has {len(dataset)} rows but the loader returned {offset}")
    cell_ids = np.asarray(cell_ids, dtype=str)
    if len(np.unique(cell_ids)) != len(cell_ids):
        raise ValueError("the dataset contains duplicate cell IDs")
    return EmbeddingResult(cell_ids, embeddings)
