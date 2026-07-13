import sys

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

import terra_multi.distill.train_prototypes as prototype_train
from terra_multi.distill.evaluate_probes import (
    _arm_config,
    arms_for_config,
)
from terra_multi.distill.prototype import (
    PrototypeDataset,
    PrototypeStudent,
    assign_prototypes,
    fit_prototypes,
    inverse_sqrt_class_weights,
    l2_normalize,
    predict_student_features,
    sample_kmeans_rows,
)
from terra_multi.distill.train_prototypes import (
    export_label_features,
    fit_student,
    run_epoch,
    save_assignments,
    save_features,
)


def test_prototype_student_returns_logits_and_features():
    model = PrototypeStudent(4, 8, 3, 5, dropout=0).eval()
    inputs = torch.randn(6, 4)
    logits, features = model(inputs, return_features=True)
    assert logits.shape == (6, 5)
    assert features.shape == (6, 3)
    assert torch.allclose(model(inputs), logits)


def test_prototype_fit_and_assign_are_deterministic_and_normalized():
    rng = np.random.default_rng(0)
    terra = np.concatenate(
        [rng.normal(-2, 0.1, (20, 4)), rng.normal(2, 0.1, (20, 4))]
    ).astype(np.float32)
    rows = sample_kmeans_rows(len(terra), 24, seed=0)
    assert len(rows) == 24
    assert np.array_equal(rows, sample_kmeans_rows(len(terra), 24, seed=0))
    assert np.allclose(np.linalg.norm(l2_normalize(terra), axis=1), 1.0)

    model_a, rows_a = fit_prototypes(terra, 2, 24, 0, 8, 20, 2)
    model_b, rows_b = fit_prototypes(terra, 2, 24, 0, 8, 20, 2)
    assignments = assign_prototypes(model_a, terra, batch_size=7)
    assert np.array_equal(rows_a, rows_b)
    assert np.allclose(model_a.cluster_centers_, model_b.cluster_centers_)
    assert assignments.shape == (40,)
    assert assignments.dtype == np.int32
    assert set(assignments) == {0, 1}


def test_inverse_sqrt_weights_are_clipped_and_ignore_empty_prototypes():
    prototype_ids = np.array([0] * 100 + [1] * 4)
    weights = inverse_sqrt_class_weights(prototype_ids, 3, clip=(0.5, 2.0)).numpy()
    assert weights[1] > weights[0]
    assert 0.5 <= weights[0] <= 2.0
    assert 0.5 <= weights[1] <= 2.0
    assert weights[2] == 0.0


def test_prototype_epoch_metrics_include_validation_top5():
    model = PrototypeStudent(4, 8, 3, 3, dropout=0)
    loader = DataLoader(
        PrototypeDataset(
            np.random.default_rng(1).normal(size=(9, 4)).astype(np.float32),
            np.arange(9) % 3,
        ),
        batch_size=4,
    )
    metrics = run_epoch(model, loader, torch.nn.CrossEntropyLoss(), "cpu", top5=True)
    assert metrics.keys() == {"loss_ce", "top1_acc", "top5_acc"}
    assert metrics["top5_acc"] == 1.0


def test_fit_student_logs_requested_metrics_and_saves_best(tmp_path):
    class Run:
        def __init__(self):
            self.rows = []

        def log(self, row):
            self.rows.append(row)

    rng = np.random.default_rng(2)
    loader = DataLoader(
        PrototypeDataset(
            rng.normal(size=(12, 4)).astype(np.float32), np.arange(12) % 3
        ),
        batch_size=4,
    )
    model = PrototypeStudent(4, 8, 3, 3, dropout=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    run = Run()
    _, _, train_rows, val_rows = fit_student(
        model,
        loader,
        loader,
        criterion,
        criterion,
        optimizer,
        "cpu",
        {"max_epochs": 1, "patience": 1},
        tmp_path / "best.pt",
        run,
    )
    assert (tmp_path / "best.pt").exists()
    assert train_rows[0].keys() == {"epoch", "loss_ce", "top1_acc", "lr"}
    assert val_rows[0].keys() == {"epoch", "loss_ce", "top1_acc", "top5_acc"}
    assert run.rows[0].keys() == {
        "epoch",
        "lr",
        "train/loss_ce",
        "train/top1_acc",
        "val/loss_ce",
        "val/top1_acc",
        "val/top5_acc",
    }


def test_assignment_and_feature_files_have_exact_required_keys(tmp_path):
    sample_ids = np.array(["A", "A"])
    cell_ids = np.array(["1", "2"])
    save_assignments(tmp_path / "assignments.npz", sample_ids, cell_ids, [0, 1])
    outputs = {
        "features_penultimate": np.ones((2, 3), dtype=np.float32),
        "prototype_probs": np.full((2, 2), 0.5, dtype=np.float32),
        "prototype_logits": np.zeros((2, 2), dtype=np.float32),
    }
    save_features(tmp_path / "features.npz", sample_ids, cell_ids, outputs)

    with np.load(tmp_path / "assignments.npz", allow_pickle=False) as data:
        assert set(data.files) == {"sample_id", "cell_id", "prototype_id"}
    with np.load(tmp_path / "features.npz", allow_pickle=False) as data:
        assert set(data.files) == {
            "sample_id",
            "cell_id",
            "features_penultimate",
            "prototype_probs",
            "prototype_logits",
        }


def test_feature_export_uses_uni_only_and_writes_probe_sidecar(tmp_path):
    sample_dir = tmp_path / "data" / "A"
    sample_dir.mkdir(parents=True)
    np.savez(
        sample_dir / "A_uni2.npz",
        cell_ids=np.array(["1", "2", "3"]),
        uni_emb=np.ones((3, 4), dtype=np.float32),
    )
    model = PrototypeStudent(4, 8, 3, 2, dropout=0).eval()
    cfg = {
        "data_root": str(tmp_path / "data"),
        "uni_key": "uni_emb",
        "batch_size": 2,
    }
    export_label_features(model, cfg, ["A"], "cpu", tmp_path, "label_test_fast")

    with np.load(tmp_path / "features_label_test_fast.npz") as data:
        assert data["features_penultimate"].shape == (3, 3)
        assert data["prototype_probs"].shape == (3, 2)
        assert np.allclose(data["prototype_probs"].sum(axis=1), 1.0)
        assert np.allclose(
            data["prototype_probs"],
            torch.softmax(torch.from_numpy(data["prototype_logits"]), dim=1).numpy(),
        )
    with np.load(tmp_path / "label_test_fast_features" / "A_prototype.npz") as data:
        assert "cell_ids" in data.files


def test_v1_probe_arms_use_per_sample_prototype_features(tmp_path):
    sample_dir = tmp_path / "data" / "A"
    sample_dir.mkdir(parents=True)
    pd.DataFrame({"cell_id": ["1", "2"], "group": ["x", "y"]}).to_csv(
        sample_dir / "A_granular_cell_labels.csv", index=False
    )
    cfg = {
        "distill_version": "v1",
        "output_dir": str(tmp_path / "output"),
        "data_root": str(tmp_path / "data"),
        "terra_key": "cell_emb",
        "uni_key": "uni_emb",
        "eval": {
            "label_test_mode": "fast",
            "label_column": "group",
            "label_id_column": "cell_id",
            "min_class_count": 1,
            "test_frac": 0.2,
            "split_axis": "y",
            "pca": True,
            "uni_pca_components": 2,
            "Cs": [1],
            "class_weight": "balanced",
        },
    }
    assert arms_for_config(cfg) == (
        "prototype_penultimate_512",
        "prototype_probs_256",
    )
    arm_cfg = _arm_config(
        cfg,
        "A",
        "prototype_probs_256",
        tmp_path / "unused",
        tmp_path / "probe",
    )
    assert arm_cfg["terra_key"] == "prototype_probs"
    assert arm_cfg["terra_embeddings"].endswith(
        "label_test_fast_features/A_prototype.npz"
    )


def test_student_feature_prediction_shapes_and_probabilities():
    model = PrototypeStudent(4, 8, 3, 2, dropout=0).eval()
    outputs = predict_student_features(
        model, np.ones((5, 4), dtype=np.float32), "cpu", batch_size=2
    )
    assert outputs["features_penultimate"].shape == (5, 3)
    assert outputs["prototype_probs"].shape == (5, 2)
    assert outputs["prototype_logits"].shape == (5, 2)
    assert np.allclose(outputs["prototype_probs"].sum(axis=1), 1.0)


def test_tiny_prototype_pipeline_writes_required_artifacts(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    rng = np.random.default_rng(3)
    for sample_id in ("TRAIN", "VAL", "LABEL"):
        sample_dir = data_root / sample_id
        sample_dir.mkdir(parents=True)
        cell_ids = np.array([f"{sample_id}_{i}" for i in range(12)])
        np.savez(
            sample_dir / f"{sample_id}_uni2.npz",
            cell_ids=cell_ids,
            uni_emb=rng.normal(size=(12, 4)).astype(np.float32),
        )
        np.savez(
            sample_dir / f"{sample_id}_terra.npz",
            cell_ids=cell_ids,
            cell_emb=rng.normal(size=(12, 3)).astype(np.float32),
        )

    splits_path = tmp_path / "splits.csv"
    pd.DataFrame(
        {
            "sample_id": ["TRAIN", "VAL", "LABEL"],
            "split": ["distill_train", "distill_val", "label_test"],
        }
    ).to_csv(splits_path, index=False)
    fast_path = tmp_path / "fast.csv"
    pd.DataFrame({"sample_id": ["LABEL"]}).to_csv(fast_path, index=False)
    output_dir = tmp_path / "output"
    checkpoint_dir = tmp_path / "checkpoint"
    cfg = {
        "distill_version": "v1",
        "run_name": "test_v1",
        "data_root": str(data_root),
        "splits_path": str(splits_path),
        "max_cells_per_sample": 10,
        "sampling_seed": 0,
        "uni_key": "uni_emb",
        "terra_key": "cell_emb",
        "uni_dim": 4,
        "hidden_dim": 8,
        "feat_dim": 3,
        "num_prototypes": 2,
        "dropout": 0.0,
        "kmeans_seed": 0,
        "max_cells_for_kmeans": 8,
        "kmeans_batch_size": 4,
        "kmeans_max_iter": 10,
        "kmeans_n_init": 1,
        "assignment_batch_size": 4,
        "use_class_weights": False,
        "batch_size": 4,
        "inference_batch_size": 4,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "max_epochs": 1,
        "patience": 1,
        "num_workers": 0,
        "seed": 0,
        "output_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "cache_dir": str(tmp_path / "cache"),
        "wandb": {"enabled": False},
        "eval": {
            "label_test_mode": "fast",
            "label_test_fast_path": str(fast_path),
            "label_test_fast_n": 1,
            "label_test_fast_seed": 0,
        },
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as fh:
        yaml.safe_dump(cfg, fh)
    monkeypatch.setattr(sys, "argv", ["train_prototypes", "--config", str(config_path)])
    prototype_train.main()

    expected = {
        "kmeans_k2_seed0.joblib",
        "kmeans_assignments_train.npz",
        "kmeans_assignments_val.npz",
        "kmeans_assignments_label_test_fast.npz",
        "prototype_config.yaml",
        "train_metrics.csv",
        "val_metrics.csv",
        "features_distill_val.npz",
        "features_label_test_fast.npz",
        "resolved_config.yaml",
        "prototype_eval.json",
    }
    assert expected <= {path.name for path in output_dir.iterdir()}
    assert (checkpoint_dir / "best.pt").exists()
