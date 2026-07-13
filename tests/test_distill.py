from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import numpy as np
import torch
from torch.utils.data import DataLoader

import terra_multi.distill.data as distill_data
from terra_multi.distill.data import (
    CellEmbeddingDataset,
    SplitCache,
    build_split_cache,
    select_indices,
)
from terra_multi.distill.inference import (
    cosine_and_nmse,
    embedding_variance,
    nmse_from_cosine_loss,
)
from terra_multi.distill.make_splits import (
    partition_samples,
    select_label_test_samples,
)
from terra_multi.distill.model import (
    TerraProjector,
    cosine_distill_loss,
    covariance_loss,
    variance_loss,
)
from terra_multi.distill.train import fit, make_loader, run_epoch

LOSS_CFG = {
    "lambda_cos": 1.0,
    "lambda_var": 0.1,
    "lambda_cov": 0.01,
    "var_gamma": 1.0,
}


def test_projector_output_is_unit_norm():
    model = TerraProjector(in_dim=1536, out_dim=384).eval()
    inputs = torch.randn(8, 1536)
    normed, raw = model(inputs, return_raw=True)
    out = model(inputs)
    assert out.shape == (8, 384)
    assert torch.allclose(out.norm(dim=-1), torch.ones(8), atol=1e-5)
    assert torch.allclose(out, normed)
    assert normed.shape == raw.shape == (8, 384)
    assert torch.allclose(normed, torch.nn.functional.normalize(raw, dim=-1))


def test_cosine_distill_loss_bounds():
    v = torch.nn.functional.normalize(torch.randn(16, 384), dim=-1)
    assert cosine_distill_loss(v, v).item() < 1e-6  # identical -> 0
    assert abs(cosine_distill_loss(v, -v).item() - 2.0) < 1e-6  # opposite -> 2
    # teacher scale is irrelevant (loss normalizes it)
    assert torch.allclose(
        cosine_distill_loss(v, v), cosine_distill_loss(v, 7.0 * v), atol=1e-6
    )


def test_variance_and_covariance_losses():
    collapsed = torch.ones(8, 3)
    assert variance_loss(collapsed).item() > 0.9

    decorrelated = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    correlated = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
    assert covariance_loss(decorrelated).item() == 0.0
    assert covariance_loss(correlated).item() > 0.0


def test_select_indices_deterministic_and_capped():
    a = select_indices(1000, 100, base_seed=0, sample_id="TENX116")
    b = select_indices(1000, 100, base_seed=0, sample_id="TENX116")
    assert a.shape == (100,) and np.array_equal(a, b)
    assert np.array_equal(a, np.sort(a)) and len(np.unique(a)) == 100
    # A different sample draws a different subset from the same seed.
    assert not np.array_equal(a, select_indices(1000, 100, 0, "TENX143"))
    # No cap needed when n <= cap.
    assert np.array_equal(select_indices(50, 100, 0, "X"), np.arange(50))


def test_dataset_item_contract():
    uni = np.random.rand(5, 1536).astype(np.float16)
    terra = np.random.rand(5, 384).astype(np.float16)
    uni_row, terra_row = CellEmbeddingDataset(uni, terra)[0]
    assert uni_row.shape == (1536,) and uni_row.dtype == torch.float32
    assert terra_row.shape == (384,) and terra_row.dtype == torch.float32


def test_cache_is_rebuilt_for_a_different_sample(tmp_path):
    for sample_id, value in (("A", 1), ("B", 2)):
        sample_dir = tmp_path / sample_id
        sample_dir.mkdir()
        cell_ids = np.array([f"{sample_id}1"])
        np.savez(
            sample_dir / f"{sample_id}_uni2.npz",
            cell_ids=cell_ids,
            uni_emb=np.full((1, 2), value),
        )
        np.savez(
            sample_dir / f"{sample_id}_terra.npz",
            cell_ids=cell_ids,
            cell_emb=np.full((1, 2), value),
        )

    path = tmp_path / "cache.npz"
    build_split_cache(tmp_path, ["A"], 10, 0, cache_path=path)
    cache = build_split_cache(tmp_path, ["B"], 10, 0, cache_path=path)
    assert cache.sample_ids.tolist() == ["B"]
    assert cache.cell_ids.tolist() == ["B1"]


def test_concurrent_cache_writes_use_unique_temp_files(tmp_path, monkeypatch):
    cache = SplitCache(
        uni=np.ones((2, 2), dtype=np.float16),
        terra=np.ones((2, 1), dtype=np.float16),
        sample_ids=np.array(["A", "A"]),
        cell_ids=np.array(["A1", "A2"]),
    )
    barrier = Barrier(2)
    paths = []
    original_savez = distill_data.np.savez

    def synchronized_save(path, *args, **kwargs):
        original_savez(path, *args, **kwargs)
        paths.append(Path(path))
        barrier.wait()

    monkeypatch.setattr(distill_data.np, "savez", synchronized_save)
    path = tmp_path / "cache.npz"
    with ThreadPoolExecutor(max_workers=2) as pool:
        writes = [
            pool.submit(distill_data.save_cache, path, cache, "signature")
            for _ in range(2)
        ]
        for write in writes:
            write.result(timeout=5)

    assert len(set(paths)) == 2
    assert distill_data.load_cache(path, "signature") is not None
    assert not list(tmp_path.glob(".cache.npz.*.tmp.npz"))


def test_run_epoch_with_embedding_dataset():
    model = TerraProjector(in_dim=4, out_dim=2, hidden_dim=4, dropout=0)
    dataset = CellEmbeddingDataset(
        np.random.rand(6, 4).astype(np.float32),
        np.random.rand(6, 2).astype(np.float32),
    )
    loader = DataLoader(dataset, batch_size=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    metrics = run_epoch(model, loader, "cpu", LOSS_CFG, optimizer)
    assert all(np.isfinite(value) for value in metrics.values())
    assert np.isclose(metrics["cosine_similarity"], 1 - metrics["loss_cos"])
    assert np.isclose(
        metrics["loss_total"],
        metrics["loss_cos"] + 0.1 * metrics["loss_var"] + 0.01 * metrics["loss_cov"],
    )


def test_training_loader_drops_only_the_partial_training_batch():
    cache = SplitCache(
        uni=np.ones((6, 4), dtype=np.float32),
        terra=np.ones((6, 2), dtype=np.float32),
        sample_ids=np.full(6, "A"),
        cell_ids=np.arange(6).astype(str),
    )
    cfg = {"batch_size": 4, "num_workers": 0}
    assert [len(batch[0]) for batch in make_loader(cache, cfg, True)] == [4]
    assert [len(batch[0]) for batch in make_loader(cache, cfg, False)] == [4, 2]


def test_fit_logs_requested_train_and_val_metrics(tmp_path):
    class Run:
        def __init__(self):
            self.rows = []

        def log(self, row):
            self.rows.append(row)

    terra = np.random.rand(6, 2).astype(np.float32)
    loader = DataLoader(
        CellEmbeddingDataset(np.random.rand(6, 4).astype(np.float32), terra),
        batch_size=3,
    )
    model = TerraProjector(in_dim=4, out_dim=2, hidden_dim=4, dropout=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    run = Run()
    variance = embedding_variance(terra)
    fit(
        model,
        loader,
        loader,
        optimizer,
        "cpu",
        {"train": variance, "val": variance},
        {"max_epochs": 1, "patience": 1, **LOSS_CFG},
        tmp_path / "best.pt",
        run,
    )

    expected = {
        "epoch",
        "lr",
        "train/loss_total",
        "train/loss_cos",
        "train/loss_var",
        "train/loss_cov",
        "train/cosine_similarity",
        "train/normalized_mse",
        "val/loss_total",
        "val/loss_cos",
        "val/loss_var",
        "val/loss_cov",
        "val/cosine_similarity",
        "val/normalized_mse",
    }
    assert run.rows[0].keys() == expected


def test_partition_holds_out_all_annotated():
    samples = [f"S{i:02d}" for i in range(80)]
    annotated = set(samples[:15])
    split = partition_samples(samples, annotated, val_frac=0.1, seed=0)

    assert all(split[s] == "label_test" for s in annotated)
    assert not any(split[s] == "label_test" for s in samples if s not in annotated)
    counts = {v: sum(x == v for x in split.values()) for v in set(split.values())}
    assert counts["label_test"] == 15
    assert counts["distill_val"] == round(0.1 * 65)
    assert counts["distill_train"] + counts["distill_val"] == 65
    # Deterministic across calls.
    assert split == partition_samples(samples, annotated, 0.1, 0)


def test_fast_label_test_selection_is_configured_and_deterministic(tmp_path):
    path = tmp_path / "fast.csv"
    path.write_text("sample_id\nS1\nS3\nS4\n")
    cfg = {
        "eval": {
            "label_test_mode": "fast",
            "label_test_fast_path": str(path),
            "label_test_fast_n": 2,
            "label_test_fast_seed": 0,
        }
    }
    samples = ["S1", "S2", "S3", "S4"]
    selected = select_label_test_samples(cfg, samples)
    assert len(selected) == 2
    assert set(selected) <= {"S1", "S3", "S4"}
    assert selected == select_label_test_samples(cfg, samples)
    assert (
        select_label_test_samples({"eval": {"label_test_mode": "full"}}, samples)
        == samples
    )


def test_nmse_from_loss_matches_direct_metric():
    rng = np.random.default_rng(1)
    teacher = rng.normal(size=(101, 12)).astype(np.float32)
    student = rng.normal(size=(101, 12)).astype(np.float32)
    student_unit = torch.nn.functional.normalize(torch.from_numpy(student), dim=-1)
    loss = cosine_distill_loss(student_unit, torch.from_numpy(teacher)).item()

    variance = embedding_variance(teacher, batch_size=17)
    derived = nmse_from_cosine_loss(loss, variance)
    direct = cosine_and_nmse(student, teacher)["normalized_mse"]
    assert np.isclose(derived, direct, rtol=1e-6)
