"""Build hierarchy-aware loaders while reusing the established LoRA datasets."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import ConcatDataset, DataLoader

from terra_multi.lora.data import (
    HESTCellDataset,
    WithinSampleBatchSampler,
    build_datasets,
    dataset_kwargs,
)

from .dataset import HierarchyDataset
from .sampler import AnnotatedBatchSampler, InterleavedBatchSampler
from .splits import PhenotypeSplit, load_phenotype_split


@dataclass
class HierarchyData:
    base_train: HESTCellDataset
    breast_train: HierarchyDataset
    base_val: HESTCellDataset
    breast_val: HierarchyDataset
    split: PhenotypeSplit
    genes: list[str] | None

    @property
    def train(self):
        return ConcatDataset((self.base_train, self.breast_train))


def build_hierarchy_data(cfg, transform, selected_genes_path=None) -> HierarchyData:
    settings = cfg["hierarchy"]
    split = load_phenotype_split(
        settings["split_manifest"],
        cfg["splits_path"],
        expected_counts=settings.get("expected_counts"),
    )
    base_train, base_val, genes = build_datasets(
        cfg, transform, selected_genes_path
    )
    if tuple(base_train.sample_ids) != split.base_train:
        raise ValueError("base training dataset differs from the phenotype split")
    if tuple(base_val.sample_ids) != split.base_val:
        raise ValueError("base validation dataset differs from the phenotype split")
    common = dataset_kwargs(cfg, transform, genes)
    breast_train = HESTCellDataset(
        split="label_test", sample_ids=split.breast_train, **common
    )
    breast_val = HESTCellDataset(
        split="label_test", sample_ids=split.breast_val, **common
    )
    labels_dir = settings["labels_dir"]
    return HierarchyData(
        base_train,
        HierarchyDataset(breast_train, labels_dir),
        base_val,
        HierarchyDataset(breast_val, labels_dir),
        split,
        genes,
    )


def _loader(dataset, sampler, workers, seed):
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        generator=generator,
    )


def make_hierarchy_loaders(data: HierarchyData, cfg):
    training = cfg["training"]
    hierarchy = cfg["hierarchy"]
    batch_size = int(training["batch_size"])
    seed = int(training["seed"])
    workers = int(training["num_workers"])
    base_train = WithinSampleBatchSampler(
        data.base_train,
        batch_size,
        seed=seed,
        shuffle=True,
        batches_per_sample_visit=int(training["batches_per_sample_visit"]),
    )
    sampling = hierarchy["sampling"]
    annotated_train = AnnotatedBatchSampler(
        data.breast_train,
        batch_size,
        seed=seed,
        shuffle=True,
        strategy=sampling["strategy"],
        relations=sampling["relations"],
        cross_sample=sampling["cross_sample"],
    )
    combined = InterleavedBatchSampler(
        base_train, annotated_train, len(data.base_train), seed
    )
    train_loader = _loader(data.train, combined, workers, seed)

    base_val = WithinSampleBatchSampler(
        data.base_val, batch_size, seed=seed, shuffle=False
    )
    annotated_val = AnnotatedBatchSampler(
        data.breast_val,
        batch_size,
        seed=seed,
        shuffle=False,
        strategy=sampling["strategy"],
        relations=sampling["relations"],
        cross_sample=sampling["cross_sample"],
    )
    validation = {
        "base_val": _loader(data.base_val, base_val, workers, seed + 1),
        "breast_val": _loader(data.breast_val, annotated_val, workers, seed + 2),
    }
    return train_loader, validation
