import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from cellbridge.hierarchical.dataset import HierarchyDataset
from cellbridge.hierarchical.sampler import (
    AnnotatedBatchSampler,
    InterleavedBatchSampler,
)
from cellbridge.hierarchical.training_data import make_hierarchy_loaders


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
    sampler = AnnotatedBatchSampler(dataset, batch_size=4, seed=7, strategy="balanced")

    first = list(sampler)
    second = list(sampler)

    assert first == second
    assert sorted(index for batch in first for index in batch) == list(range(8))
    assert all(len(set(dataset.sample_ids_by_index[batch])) == 2 for batch in first)


def test_breast_only_loaders_use_annotated_train_and_validation(tmp_path):
    _write_labels(tmp_path, "A")
    _write_labels(tmp_path, "B")
    breast_train = HierarchyDataset(TinyCells(), tmp_path)
    breast_val = HierarchyDataset(TinyCells(), tmp_path)
    data = type(
        "HierarchyData",
        (),
        {
            "base_train": TinyCells(),
            "breast_train": breast_train,
            "base_val": TinyCells(),
            "breast_val": breast_val,
            "train": None,
        },
    )()
    cfg = {
        "training": {"batch_size": 4, "num_workers": 0, "seed": 7},
        "hierarchy": {
            "breast_only": True,
            "sampling": {
                "strategy": "balanced",
                "validation_strategy": "balanced",
                "relations": ["identity", "lineage", "unrelated"],
                "cross_sample": "prefer",
            },
        },
    }

    train, validation = make_hierarchy_loaders(data, cfg)
    batches = list(train)

    assert set(validation) == {"breast_val"}
    assert len(batches) == len(breast_train) // 4
    assert all("hierarchy_depth" in batch for batch in batches)


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


def test_hierarchy_sampler_never_overfills_a_nearly_complete_batch(tmp_path):
    _write_labels(tmp_path, "A")
    _write_labels(tmp_path, "B")
    dataset = HierarchyDataset(TinyCells(), tmp_path)

    for seed in range(20):
        sampler = AnnotatedBatchSampler(
            dataset,
            batch_size=5,
            seed=seed,
            strategy="hierarchy",
            relations=["identity", "lineage", "unrelated"],
        )
        batches = list(sampler)

        assert len(batches) == len(sampler) == 1
        assert len(batches[0]) == 5
        assert len(set(batches[0])) == 5


def test_interleaved_sampler_enforces_even_fifty_percent_annotated_batches():
    base = [[index] for index in range(8)]
    annotated = [[index] for index in range(2)]
    sampler = InterleavedBatchSampler(
        base,
        annotated,
        annotated_offset=100,
        seed=7,
        annotated_fraction=0.5,
    )

    first = list(sampler)
    second = list(sampler)
    sources = [batch[0] >= 100 for batch in first]

    assert first == second
    assert sampler.source_counts == (5, 5)
    assert sources == [False, True] * 5
    assert [batch[0] for batch in first if batch[0] >= 100] == [
        100,
        101,
        100,
        101,
        100,
    ]


def test_interleaved_sampler_default_uses_each_source_batch_once():
    base = [[0], [1], [2]]
    annotated = [[0]]
    sampler = InterleavedBatchSampler(base, annotated, annotated_offset=100, seed=4)

    batches = list(sampler)

    assert sampler.source_counts == (3, 1)
    assert sorted(batch[0] for batch in batches) == [0, 1, 2, 100]


def test_cells_absent_from_the_label_file_are_marked_unusable(tmp_path):
    """A larger cell cap reaches cells the granular clustering dropped.

    Those cells carry no identity at any level, so they must be unusable for
    hierarchy terms without blocking training: the paired loss is label-free.
    """
    import numpy as np
    import pandas as pd

    from cellbridge.hierarchical.dataset import _load_aligned

    path = tmp_path / "S1.parquet"
    pd.DataFrame(
        {
            "sample_id": ["S1", "S1"],
            "cell_id": ["a-1", "b-1"],
            "mapping_status": ["full", "partial"],
            "hierarchy_depth": [3, 1],
            "use_in_hierarchy": [True, True],
            "level_0": ["cell", "cell"],
            "level_1_compartment": ["epithelial", "stromal"],
            "level_2_lineage": ["neoplastic", ""],
            "level_3_identity": ["neoplastic cell", ""],
        }
    ).to_parquet(path)

    aligned = _load_aligned(path, np.array(["a-1", "zzz-1", "b-1"]), "S1")

    assert aligned["cell_id"].tolist() == ["a-1", "zzz-1", "b-1"]
    assert aligned["use_in_hierarchy"].tolist() == [True, False, True]
    assert aligned["hierarchy_depth"].tolist() == [3, 0, 1]
    assert aligned.loc[1, "mapping_status"] == "excluded"
    assert aligned.loc[1, "level_3_identity"] == ""
    # The labelled rows must be untouched.
    assert aligned.loc[0, "level_3_identity"] == "neoplastic cell"
