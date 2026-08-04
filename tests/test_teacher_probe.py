import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from terra_multi.teacher_probe.data import ProbeData, load_probe_data
from terra_multi.teacher_probe.extract_gene_tokens import (
    stable_cell_subset,
    token_presence,
)
from terra_multi.teacher_probe.features import (
    build_fold_token_features,
    gene_feature_slice,
)
from terra_multi.teacher_probe.neighborhood import neighborhood_expression
from terra_multi.teacher_probe.splits import spatial_block_folds
from terra_multi.teacher_probe.store import write_token_store


class FakeTokenizedDataset:
    column_names = ["cell_id", "gene_tokens"]

    def __init__(self, tokens):
        self.tokens = np.asarray(tokens)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return {"gene_tokens": self.tokens[item]}
        raise TypeError(item)


def test_stable_subset_and_presence_are_deterministic_and_use_cell_segment():
    ids = np.asarray([f"cell-{index}" for index in range(100)])
    first = stable_cell_subset(ids, 20, 7)
    second = stable_cell_subset(ids, 20, 7)
    np.testing.assert_array_equal(first, second)
    assert len(first) == 20
    assert np.all(np.diff(first) > 0)

    dataset = FakeTokenizedDataset([[5, 1, 0, 7, 0], [1, 7, 0, 5, 0]])
    observed = token_presence(dataset, [5, 7], seq_len_cell=3, batch_size=1)
    np.testing.assert_array_equal(observed, [[True, False], [False, True]])


def test_spatial_blocks_never_cross_folds_and_partition_cells():
    x, y = np.meshgrid(np.arange(10) * 100.0, np.arange(10) * 100.0)
    coords = np.column_stack((x.ravel(), y.ravel()))
    labels = np.asarray(["a", "b"] * 50)
    folds, groups = spatial_block_folds(coords, 5, 200.0, labels)
    test_rows = []
    for fold in folds:
        assert not (set(groups[fold.train]) & set(groups[fold.test]))
        assert set(labels[fold.train]) == {"a", "b"}
        test_rows.extend(fold.test)
    np.testing.assert_array_equal(np.sort(test_rows), np.arange(100))


def test_neighborhood_expression_excludes_focal_cell_and_reuses_cache(tmp_path):
    cell_ids = np.asarray([f"cell-{index}" for index in range(8)])
    coords = np.column_stack((np.arange(8, dtype=float), np.zeros(8)))
    expression = np.column_stack(
        (np.arange(8, dtype=np.float32) ** 2, np.arange(8, dtype=np.float32))
    )
    genes = np.asarray(["A", "B"])
    cache = tmp_path / "neighborhood.npz"

    observed, diagnostics = neighborhood_expression(
        cell_ids, coords, expression, genes, 2, cache
    )
    np.testing.assert_allclose(observed[0], expression[[1, 2]].mean(axis=0))
    np.testing.assert_allclose(observed[3], expression[[2, 4]].mean(axis=0))
    assert diagnostics["n_neighbors"] == 2
    assert not diagnostics["cache_reused"]

    cached, cached_diagnostics = neighborhood_expression(
        cell_ids, coords, expression, genes, 2, cache
    )
    np.testing.assert_array_equal(cached, observed)
    assert cached_diagnostics["cache_reused"]

    changed, changed_diagnostics = neighborhood_expression(
        cell_ids, coords, expression + 1, genes, 2, cache
    )
    np.testing.assert_allclose(changed, observed + 1)
    assert not changed_diagnostics["cache_reused"]


def _token_store(path, cell_ids, genes, presence, seed=0):
    rng = np.random.default_rng(seed)
    spatial = {}
    for column, gene in enumerate(genes):
        values = rng.normal(size=(len(cell_ids), 8)).astype(np.float32)
        values[~presence[:, column]] = 0
        spatial[gene] = values
    write_token_store(
        path,
        cell_ids=cell_ids,
        genes=genes,
        ensembl_ids=[f"ENSG{index:011d}" for index in range(len(genes))],
        presence=presence,
        spatial_embeddings=spatial,
        metadata={"test": True},
    )


def test_gene_pca_uses_present_training_rows_and_caches_features(tmp_path):
    cell_ids = np.asarray([f"cell-{index}" for index in range(40)])
    genes = np.asarray(["A", "B"])
    presence = np.ones((40, 2), dtype=bool)
    presence[::3, 0] = False
    presence[::4, 1] = False
    store_path = tmp_path / "tokens.h5"
    _token_store(store_path, cell_ids, genes, presence)
    data = ProbeData(
        cell_ids=cell_ids,
        labels=np.asarray(["x", "y"] * 20),
        coords_um=np.zeros((40, 2)),
        cell_emb=np.zeros((40, 4), dtype=np.float32),
        expression=np.arange(80, dtype=np.float32).reshape(40, 2),
        presence=presence,
        genes=genes,
        store_rows=np.arange(40),
        token_store_path=store_path,
        token_store_signature="test-signature",
        alignment_report={},
    )
    settings = {"gene_pca_components": 3, "seed": 3}
    features, stats = build_fold_token_features(
        data, np.arange(30), 0, settings, tmp_path
    )
    assert features.shape == (40, 8)
    assert len(stats) == 2
    for index in range(2):
        pc_slice, mask_column = gene_feature_slice(index, 3)
        assert np.all(features[~presence[:, index], pc_slice] == 0)
        np.testing.assert_array_equal(features[:, mask_column], presence[:, index])
    cached, cached_stats = build_fold_token_features(
        data, np.arange(30), 0, settings, tmp_path
    )
    np.testing.assert_array_equal(cached, features)
    assert cached_stats == stats


def test_strict_loader_aligns_all_inputs_by_cell_id(tmp_path):
    genes = np.asarray([f"GENE{index}" for index in range(20)])
    all_ids = np.asarray([f"cell-{index:02d}" for index in range(30)])
    token_ids = all_ids[:20]
    counts = np.ones((30, 20), dtype=np.float32)
    adata = ad.AnnData(sp.csr_matrix(counts))
    adata.obs_names = all_ids
    adata.var_names = genes
    adata.obs["cell_id"] = all_ids
    adata.obs["total_counts"] = counts.sum(axis=1)
    coords = np.column_stack((np.arange(30), np.arange(30) * 2.0))
    adata.obsm["spatial"] = coords
    adata_path = tmp_path / "sample.h5ad"
    adata.write_h5ad(adata_path)

    base_embeddings = np.repeat(np.arange(30)[:, None], 4, axis=1).astype(np.float32)
    reverse = np.arange(29, -1, -1)
    terra_path = tmp_path / "terra.npz"
    np.savez(
        terra_path,
        cell_ids=all_ids[reverse],
        cell_emb=base_embeddings[reverse],
        he_x=coords[reverse, 0],
        he_y=coords[reverse, 1],
    )
    labels_path = tmp_path / "labels.csv"
    pd.DataFrame({"cell_id": all_ids, "group": np.asarray(["a", "b"] * 15)}).to_csv(
        labels_path, index=False
    )
    store_path = tmp_path / "tokens.h5"
    _token_store(store_path, token_ids, genes, np.ones((20, 20), dtype=bool))

    cfg = {
        "sample_id": "sample",
        "genes": genes.tolist(),
        "active_gene_tokens_path": str(store_path),
        "active_max_cells": 20,
        "terra_embeddings": str(terra_path),
        "adata_path": str(adata_path),
        "labels_path": str(labels_path),
        "pixel_size_um": 0.5,
        "eval": {
            "label_id_column": "cell_id",
            "label_column": "group",
            "min_class_count": 1,
        },
    }
    data = load_probe_data(cfg)
    np.testing.assert_array_equal(data.cell_ids, token_ids)
    np.testing.assert_array_equal(data.cell_emb[:, 0], np.arange(20))
    # obsm["spatial"] is already in microns, so pixel_size_um must not be applied
    # to it a second time; doing so shrank the section by 1/pixel_size_um and
    # left too few spatial blocks to build the requested CV folds.
    np.testing.assert_allclose(data.coords_um, coords[:20])
    assert data.expression.shape == (20, 20)
    assert data.alignment_report["n_evaluation_cells"] == 20
