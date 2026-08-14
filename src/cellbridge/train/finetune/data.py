"""Reuse OOD inference sources for supervised UNI fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from cellbridge.inference.embed import sources
from cellbridge.inference.evaluate.ood import IMMUNE_CLASSES

SPLITS = frozenset({"train", "val", "test"})
TASKS = frozenset({"all", "immune"})


@dataclass(frozen=True)
class OODFineTuneTask:
    """One label task backed by an existing OOD crop source."""

    data: dict
    profile: dict
    metadata: pd.DataFrame
    classes: tuple[str, ...]
    source: object

    @property
    def class_to_index(self) -> dict[str, int]:
        return {label: index for index, label in enumerate(self.classes)}

    @property
    def counts(self) -> dict[str, int]:
        return {
            split: int((self.metadata["split"] == split).sum())
            for split in sorted(SPLITS)
        }


def prepare_task(
    data: dict,
    profile: dict,
    *,
    task: str = "all",
    metadata: pd.DataFrame | None = None,
    source=None,
) -> OODFineTuneTask:
    """Validate labels and preserve the official train/val/test split."""
    if task not in TASKS:
        raise ValueError(f"task must be one of {sorted(TASKS)}")
    if profile.get("method") != "uni":
        raise ValueError("fine-tuning requires the UNI crop profile")

    source = source or sources.load(data)
    metadata = source.load_metadata(data) if metadata is None else metadata.copy()
    required = {"row_id", "sample_id", "label", "split"}
    if not required <= set(metadata):
        raise ValueError(f"metadata is missing {sorted(required - set(metadata))}")
    if metadata[list(required)].isna().any().any():
        raise ValueError("metadata contains null task fields")
    if metadata["row_id"].duplicated().any():
        raise ValueError("metadata contains duplicate row IDs")

    metadata["label"] = metadata["label"].astype(str)
    metadata["split"] = metadata["split"].replace({"valid": "val"}).astype(str)
    unknown = set(metadata["split"]) - SPLITS
    if unknown:
        raise ValueError(f"metadata contains unknown splits: {sorted(unknown)}")
    if task == "immune":
        metadata = metadata.loc[metadata["label"].isin(IMMUNE_CLASSES)].copy()

    counts = metadata["split"].value_counts()
    empty = sorted(SPLITS - set(counts.index[counts > 0]))
    if empty:
        raise ValueError(f"{task} has empty required splits: {empty}")

    classes = tuple(
        sorted(metadata.loc[metadata["split"] == "train", "label"].unique())
    )
    if len(classes) < 2:
        raise ValueError("fine-tuning requires at least two training classes")
    supported = set(classes)
    for split in ("val", "test"):
        missing = set(metadata.loc[metadata["split"] == split, "label"]) - supported
        if missing:
            raise ValueError(f"{split} classes absent from training: {sorted(missing)}")

    return OODFineTuneTask(
        data=dict(data),
        profile=dict(profile),
        metadata=metadata.reset_index(drop=True),
        classes=classes,
        source=source,
    )


class LabeledSubset(Dataset):
    """Attach aligned class targets and row IDs to source crops."""

    def __init__(self, dataset, positions, row_ids, targets) -> None:
        self.dataset = dataset
        self.positions = np.asarray(positions, dtype=int)
        self.row_ids = np.asarray(row_ids)
        self.targets = np.asarray(targets, dtype=int)
        if not len(self.positions) == len(self.row_ids) == len(self.targets):
            raise ValueError("positions, row_ids, and targets must be aligned")
        if len(self.positions) and (
            self.positions.min() < 0 or self.positions.max() >= len(dataset)
        ):
            raise ValueError("source positions are out of bounds")

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, index: int) -> dict:
        item = dict(self.dataset[int(self.positions[index])])
        item["target"] = int(self.targets[index])
        item["row_id"] = str(self.row_ids[index])
        return item


def iter_datasets(
    task: OODFineTuneTask,
    splits,
    *,
    transform,
):
    """Yield sample-local labeled datasets for the requested official splits."""
    splits = frozenset(splits)
    unknown = splits - SPLITS
    if not splits or unknown:
        raise ValueError(f"invalid splits: {sorted(unknown)}")

    selected = task.metadata.loc[task.metadata["split"].isin(splits)]
    indexed = selected.set_index("row_id", drop=False)
    cropper = task.source.make_cropper(task.profile, task.data)
    seen = set()
    for sample in task.source.iter_samples(
        task.data,
        transform=transform,
        cropper=cropper,
    ):
        if len(sample.row_ids) != len(sample.dataset):
            raise ValueError(
                f"{sample.sample_id}: source rows and crops are misaligned"
            )
        row_ids = np.asarray(sample.row_ids)
        positions = np.flatnonzero(pd.Index(row_ids).isin(indexed.index))
        if not len(positions):
            continue
        aligned = indexed.loc[list(row_ids[positions])]
        targets = aligned["label"].map(task.class_to_index)
        if targets.isna().any():
            raise ValueError(f"{sample.sample_id}: source contains an unknown class")
        seen.update(aligned["row_id"].tolist())
        yield LabeledSubset(
            sample.dataset,
            positions,
            aligned["row_id"].to_numpy(),
            targets.to_numpy(dtype=int),
        )

    missing = set(indexed.index) - seen
    if missing:
        raise ValueError(f"source is missing task rows: {sorted(missing)[:5]}")
