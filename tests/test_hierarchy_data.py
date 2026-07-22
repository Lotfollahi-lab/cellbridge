import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from terra_multi.hierarchical.dataset import HierarchyDataset
from terra_multi.hierarchical.sampler import AnnotatedBatchSampler


class TinyCells(Dataset):
    def __init__(self):
        self.sample_ids = ["A", "B"]
        self.sample_indices = {
            "A": np.arange(0, 4),
            "B": np.arange(4, 8),
        }
        self.sample_cell_ids = {
            "A": np.asarray(["a0", "a1", "a2", "a3"]),
            "B": np.asarray(["b0", "b1", "b2", "b3"]),
        }

    def __len__(self):
        return 8

    def __getitem__(self, index):
        sample = "A" if index < 4 else "B"
        return {
            "image": torch.tensor([float(index)]),
            "terra": torch.tensor([float(index)]),
            "sample_id": sample,
            "cell_id": self.sample_cell_ids[sample][index % 4],
        }

    def close(self):
        pass


def _write_labels(root, sample):
    identities = ["macrophage", "macrophage", "", "T cell"]
    status = ["full", "full", "partial", "excluded"]
    depth = [3, 3, 2, 0]
    use = [True, True, True, False]
    table = pd.DataFrame(
        {
            "sample_id": sample,
            "cell_id": [f"{sample.lower()}{index}" for index in range(4)],
            "mapping_status": status,
            "hierarchy_depth": depth,
            "use_in_hierarchy": use,
            "level_0": ["cell", "cell", "cell", ""],
            "level_1_compartment": [
                "hematopoietic",
                "hematopoietic",
                "hematopoietic",
                "",
            ],
            "level_2_lineage": ["myeloid", "myeloid", "myeloid", ""],
            "level_3_identity": identities,
        }
    )
    table.to_parquet(root / f"{sample}.parquet", index=False)


def test_hierarchy_dataset_aligns_codes_and_preserves_safe_depths(tmp_path):
    _write_labels(tmp_path, "A")
    _write_labels(tmp_path, "B")
    dataset = HierarchyDataset(TinyCells(), tmp_path)

    assert dataset[0]["mapping_status"] == "full"
    assert dataset[2]["mapping_status"] == "partial"
    assert dataset[2]["hierarchy_depth"].item() == 2
    assert dataset[2]["hierarchy_levels"][3].item() == -1
    assert not dataset[3]["hierarchy_usable"]
    assert dataset.levels[0, 3] == dataset.levels[4, 3]


def test_balanced_sampler_is_deterministic_cross_sample_and_without_replacement(
    tmp_path,
):
    _write_labels(tmp_path, "A")
    _write_labels(tmp_path, "B")
    dataset = HierarchyDataset(TinyCells(), tmp_path)
    sampler = AnnotatedBatchSampler(
        dataset, batch_size=4, seed=7, strategy="balanced"
    )

    first = list(sampler)
    second = list(sampler)

    assert first == second
    assert sorted(index for batch in first for index in batch) == list(range(8))
    assert all(len(set(dataset.sample_ids_by_index[batch])) == 2 for batch in first)


def test_hierarchy_sampler_draws_cross_sample_relations_without_replacement(tmp_path):
    _write_labels(tmp_path, "A")
    _write_labels(tmp_path, "B")
    dataset = HierarchyDataset(TinyCells(), tmp_path)
    sampler = AnnotatedBatchSampler(
        dataset,
        batch_size=4,
        seed=3,
        strategy="hierarchy",
        relations=["identity", "lineage", "unrelated"],
    )

    batches = list(sampler)

    assert len(batches) == 2
    assert len({index for batch in batches for index in batch}) == 8
