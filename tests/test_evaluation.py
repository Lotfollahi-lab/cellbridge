from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import cellbridge.lora.export_phenocell as lora_export
import cellbridge.lora.probe_phenocell as lora_probe
from cellbridge.baselines.evaluation import _probe_protocol_identity
from cellbridge.evaluation.classification import (
    evaluate_classification,
    method_directory,
    resolve_methods,
    select_knn_reference,
)
from cellbridge.evaluation.hest import HestData, evaluate_hest_data
from cellbridge.evaluation.ood import prepare_ood_task, validate_official_split
from cellbridge.lora import evaluate as lora_evaluate
from cellbridge.lora.evaluate import evaluate_phenocell_arm
from cellbridge.lora.evaluate_ood import merge_shard_manifests
from cellbridge.lora.probe_phenocell import resolve_probe_protocols


def classification_data():
    rng = np.random.default_rng(0)
    labels = np.repeat(["B", "T", "myeloid"], 20)
    centers = np.eye(3, 6, dtype=np.float32)
    features = np.concatenate(
        [centers[index] + 0.05 * rng.normal(size=(20, 6)) for index in range(3)]
    ).astype(np.float32)
    train = np.concatenate([np.arange(start, start + 15) for start in (0, 20, 40)])
    test = np.setdiff1d(np.arange(60), train)
    return features, labels, train, test


def test_shared_linear_probe_matches_legacy_metrics_and_predictions():
    features, labels, train, test = classification_data()
    settings = {"Cs": [0.1, 1, 10], "class_weight": "balanced"}

    legacy = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegressionCV(
                    Cs=settings["Cs"],
                    cv=3,
                    scoring="balanced_accuracy",
                    class_weight="balanced",
                    max_iter=2000,
                    n_jobs=-1,
                ),
            ),
        ]
    ).fit(features[train], labels[train])
    expected = legacy.predict(features[test])
    expected_proba = legacy.predict_proba(features[test])

    result = evaluate_classification(
        features, labels, train, test, "linear", settings
    )

    np.testing.assert_array_equal(result.predictions["y_pred"], expected)
    np.testing.assert_allclose(result.predictions["y_proba"], expected_proba)
    assert result.metrics["accuracy"] == accuracy_score(labels[test], expected)
    assert result.metrics["balanced_accuracy"] == balanced_accuracy_score(
        labels[test], expected
    )
    assert result.metrics["macro_f1"] == f1_score(
        labels[test], expected, average="macro"
    )
    assert result.metrics["weighted_f1"] == f1_score(
        labels[test], expected, average="weighted"
    )
    assert result.metrics["macro_auroc"] == pytest.approx(
        roc_auc_score(
            labels[test],
            expected_proba,
            multi_class="ovr",
            average="macro",
            labels=legacy.classes_,
        )
    )


def test_knn_official_protocol_uses_validation_references():
    features = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.02],
            [1.0, 0.03],
            [0.02, 1.0],
        ],
        dtype=np.float32,
    )
    labels = np.array(["A", "B", "B", "B", "A"])
    result = evaluate_classification(
        features,
        labels,
        train_indices=[0, 1],
        val_indices=[2],
        test_indices=[3, 4],
        method="knn",
        settings={"knn_neighbors": 1},
    )

    assert result.predictions["y_pred"].tolist() == ["B", "B"]
    assert result.metrics["n_train"] == 2
    assert result.metrics["n_val"] == 1


def test_balanced_knn_reference_has_equal_reproducible_class_counts():
    labels = np.array(["A"] * 4 + ["B"] * 2 + ["C"] * 3)
    reference = np.arange(len(labels))
    selected = select_knn_reference(labels, reference, "balanced", seed=4)

    assert selected.tolist() == select_knn_reference(
        labels, reference, "balanced", seed=4
    ).tolist()
    classes, counts = np.unique(labels[selected], return_counts=True)
    assert classes.tolist() == ["A", "B", "C"]
    assert counts.tolist() == [2, 2, 2]


def test_named_probe_protocols_merge_only_classifier_prior_settings():
    settings = {
        "Cs": [1],
        "protocols": {
            "natural": {
                "class_weight": None,
                "knn_reference_sampling": "all",
            },
            "equal_class": {
                "class_weight": "balanced",
                "knn_reference_sampling": "balanced",
            },
        },
    }
    protocols = dict(resolve_probe_protocols(settings))

    assert protocols["natural"]["Cs"] == [1]
    assert protocols["natural"]["class_weight"] is None
    assert protocols["equal_class"]["knn_reference_sampling"] == "balanced"


def test_validation_only_development_task_does_not_require_test_rows():
    features = np.eye(4, dtype=np.float32)
    table = pd.DataFrame(
        {
            "row_id": np.arange(4),
            "label": ["A", "B", "A", "B"],
            "split": ["train", "train", "val", "val"],
        }
    )
    data = prepare_ood_task(
        features,
        table,
        "all",
        {"evaluation_split": "val", "uni_pca_components": 0},
    )

    assert data.val_indices is None
    assert data.train_indices.tolist() == [0, 1]
    assert data.test_indices.tolist() == [2, 3]


def test_method_resolution_and_legacy_paths():
    assert resolve_methods({}) == ["linear"]
    assert resolve_methods({"methods": ["linear", "knn"]}) == ["linear", "knn"]
    assert resolve_methods({}, "knn,linear") == ["knn", "linear"]
    with pytest.raises(ValueError, match="must not be empty"):
        resolve_methods({"methods": []})
    with pytest.raises(ValueError, match="duplicate"):
        resolve_methods({"methods": ["knn", "knn"]})
    with pytest.raises(ValueError, match="unknown evaluation"):
        resolve_methods({"methods": ["svm"]})

    base = Path("outputs/probes/sample/arm")
    assert method_directory(base, "linear") == base
    assert method_directory(base, "knn") == base / "knn"


def test_hest_adapter_keeps_legacy_linear_schema():
    features, labels, train, test = classification_data()
    data = HestData(
        features=features,
        labels=labels,
        cell_ids=np.array([f"cell-{index}" for index in range(len(labels))]),
        train_indices=train,
        test_indices=test,
        terra_key="cell_emb",
        pca_explained_variance=None,
    )
    settings = {
        "label_column": "group",
        "split_axis": "y",
        "test_frac": 0.25,
        "pca": False,
        "Cs": [0.1, 1, 10],
        "class_weight": "balanced",
    }
    metrics, predictions = evaluate_hest_data(
        data, "TENX", "terra", "linear", settings
    )

    assert set(metrics) == {
        "sample_id",
        "model",
        "terra_key",
        "label_column",
        "n_features",
        "n_train",
        "n_test",
        "n_classes",
        "classes",
        "split_axis",
        "test_frac",
        "uni_pca",
        "uni_pca_components",
        "uni_pca_train_explained_variance",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_auroc",
    }
    assert set(predictions) == {
        "cell_ids_test",
        "y_true",
        "y_pred",
        "y_proba",
        "classes",
    }


def test_phenocell_adapter_supports_knn():
    features, labels, train, _ = classification_data()
    val = np.concatenate([indices[-3:] for indices in np.split(train, 3)])
    train = np.setdiff1d(train, val)
    split = np.full(len(labels), "test", dtype=object)
    split[train] = "train"
    split[val] = "val"
    table = pd.DataFrame(
        {
            "row_id": np.arange(len(labels)),
            "label": labels,
            "split": split,
        }
    )
    settings = {
        "Cs": [0.1, 1, 10],
        "class_weight": "balanced",
        "knn_neighbors": 1,
    }

    lora_metrics, lora_predictions = evaluate_phenocell_arm(
        features, table, "all", "lora_projected", settings, "knn"
    )
    assert lora_metrics["n_neighbors"] == 1
    assert lora_metrics["test_macro_f1"] == 1.0
    assert set(lora_predictions) == {"row_id", "y_true", "y_pred", "classes"}


def test_projected_arm_keeps_its_learned_dimensions_under_wide_pca_settings():
    # A d64 alignment head is narrower than the configured 256 components, so
    # reducing it would raise; the projected arm must reach the probe intact.
    features, labels, train, _ = classification_data()
    split = np.full(len(labels), "test", dtype=object)
    split[train[: len(train) // 2]] = "train"
    split[train[len(train) // 2 :]] = "val"
    table = pd.DataFrame(
        {"row_id": np.arange(len(labels)), "label": labels, "split": split}
    )
    settings = {"Cs": [1], "knn_neighbors": 1, "uni_pca_components": 256}

    metrics, _ = evaluate_phenocell_arm(
        features, table, "all", "lora_projected", settings, "knn"
    )

    assert metrics["n_features"] == features.shape[1]
    assert metrics["pca_explained_variance"] is None


def test_lora_export_signature_resolves_physical_fov(monkeypatch):
    captured = {}
    monkeypatch.setattr(lora_export, "path_identity", lambda _: "lmdb")
    monkeypatch.setattr(lora_export, "file_identity", lambda _: "metadata")

    def fake_signature(dataset, checkpoint, source, settings):
        captured.update(settings)
        return "signature"

    monkeypatch.setattr(lora_export, "feature_signature", fake_signature)
    cfg = {
        "data": {
            "lmdb": "unused",
            "metadata": "unused.csv",
            "source_mpp": 0.377,
        },
        "embedding": {"fov_um": 112.0, "target_size": 224},
    }

    assert lora_export._signature(cfg, {}, "sample", "phenocell") == "signature"
    assert captured["crop"]["mode"] == "physical_fov"
    assert captured["crop"]["source_size_px"] == 297
    assert captured["crop"]["target_size_px"] == 224


def test_lora_validation_probe_uses_every_named_protocol(tmp_path, monkeypatch):
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    pd.DataFrame(
        [{"sample_id": "sample", "path": "unused.npz", "signature": "features"}]
    ).to_csv(feature_dir / "manifest.csv", index=False)
    table = pd.DataFrame(
        {
            "row_id": np.arange(4),
            "sample_id": ["train", "train", "val", "val"],
            "label": ["A", "B", "A", "B"],
            "split": ["train", "train", "val", "val"],
        }
    )
    data = SimpleNamespace(
        pca_explained_variance=None,
        val_indices=None,
        test_indices=np.array([2, 3]),
    )
    captured = []

    monkeypatch.setattr(lora_probe, "FEATURE_ARMS", ("lora_projected",))
    monkeypatch.setattr(
        lora_probe,
        "load_phenocell_arm",
        lambda *_: (np.eye(4, dtype=np.float32), table),
    )
    monkeypatch.setattr(lora_probe, "validate_official_split", lambda *a, **k: None)
    monkeypatch.setattr(lora_probe, "prepare_ood_task", lambda *a, **k: data)
    monkeypatch.setattr(lora_probe, "file_identity", lambda _: "split")
    def fake_evaluate(_data, method, settings):
        captured.append((method, settings["class_weight"], settings["knn_reference_sampling"]))
        return SimpleNamespace(
            metrics={
                "n_features": 4,
                "n_train": 2,
                "n_test": 2,
                "n_classes": 2,
                "accuracy": 1.0,
                "balanced_accuracy": 1.0,
                "macro_f1": 1.0,
                "weighted_f1": 1.0,
                "reference_sampling": settings["knn_reference_sampling"],
                "n_reference": 2,
            },
            predictions={
                "row_ids_test": np.array([2, 3]),
                "y_true": np.array(["A", "B"]),
                "y_pred": np.array(["A", "B"]),
                "classes": np.array(["A", "B"]),
            },
        )

    monkeypatch.setattr(lora_probe, "evaluate_ood_task", fake_evaluate)
    cfg = {
        "data": {"split": "unused.csv"},
        "embedding": {"development_subset": {"max_cells_per_split": 2}},
        "probe": {
            "tasks": ["all"],
            "methods": ["knn"],
            "evaluation_split": "val",
            "protocols": {
                "natural": {"class_weight": None, "knn_reference_sampling": "all"},
                "equal_class": {
                    "class_weight": "balanced",
                    "knn_reference_sampling": "balanced",
                },
            },
        },
    }

    result = lora_probe.probe_phenocell(cfg, tmp_path)

    assert captured == [("knn", None, "all"), ("knn", "balanced", "balanced")]
    assert result.loc[0, "n_val"] == 0
    assert result.loc[0, "evaluation_split"] == "val"
    assert (tmp_path / "probes" / "natural" / "probe_summary_knn.csv").exists()
    assert (tmp_path / "probes" / "equal_class" / "probe_summary_knn.csv").exists()


def test_lora_aggregate_reads_every_phenocell_protocol_and_fails_if_missing(tmp_path):
    phenocell_root = tmp_path / "phenocell" / "probes"
    protocols = {
        "natural": {"class_weight": None, "knn_reference_sampling": "all"},
        "equal_class": {
            "class_weight": "balanced",
            "knn_reference_sampling": "balanced",
        },
    }
    for index, protocol in enumerate(protocols):
        directory = phenocell_root / protocol
        directory.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "task": "all",
                    "arm": "lora_projected",
                    "test_macro_f1": 0.7 + index / 10,
                    "test_balanced_accuracy": 0.8,
                }
            ]
        ).to_csv(directory / "probe_summary_knn.csv", index=False)

    result = lora_evaluate._read_phenocell_method_summary(
        tmp_path, {"probe": {"protocols": protocols}}, "knn"
    )

    assert result["protocol"].tolist() == ["natural", "equal_class"]
    (phenocell_root / "equal_class" / "probe_summary_knn.csv").unlink()
    with pytest.raises(FileNotFoundError, match="equal_class"):
        lora_evaluate._read_phenocell_method_summary(
            tmp_path, {"probe": {"protocols": protocols}}, "knn"
        )


def test_lora_probe_only_does_not_require_checkpoint(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(f"output_dir: {tmp_path / 'run'}\neval: {{}}\n")
    called = {}

    def fake_probe(cfg, output_dir, samples=None):
        called["output_dir"] = output_dir
        called["samples"] = samples

    monkeypatch.setattr(lora_evaluate, "probe_hest", fake_probe)
    monkeypatch.setattr(
        lora_evaluate,
        "_checkpoint_path",
        lambda *_: pytest.fail("probe-only resolved a checkpoint"),
    )

    lora_evaluate.main(["--config", str(config), "--probe-only"])

    assert called == {
        "output_dir": tmp_path / "run" / "evaluation" / "hest_label_test_fast",
        "samples": None,
    }


def test_probe_protocol_identity_references_existing_files():
    identities = _probe_protocol_identity()

    assert {Path(item["path"]).name for item in identities} >= {
        "evaluation.py",
        "ood.py",
    }


def test_phenocell_split_requires_complete_official_cohort(tmp_path):
    split_path = tmp_path / "split.csv"
    pd.DataFrame(
        {
            "sample_name": ["A", "B"],
            "train_test_val_split": ["train", "test"],
        }
    ).to_csv(split_path, index=False)
    observed = pd.DataFrame({"sample_id": ["A"], "split": ["train"]})

    with pytest.raises(ValueError, match="every official sample"):
        validate_official_split(observed, split_path)


def test_shard_manifests_merge_into_one_probe_manifest(tmp_path):
    for shard, samples in enumerate([["fold1_2", "fold1_0"], ["fold1_1"]]):
        pd.DataFrame(
            {"sample_id": samples, "n_cells": [1] * len(samples), "path": samples}
        ).to_csv(tmp_path / f"manifest.shard{shard:03d}.csv", index=False)

    merged = merge_shard_manifests(tmp_path, 2)

    assert merged["sample_id"].tolist() == ["fold1_0", "fold1_1", "fold1_2"]
    assert pd.read_csv(tmp_path / "manifest.csv")["sample_id"].tolist() == (
        merged["sample_id"].tolist()
    )


def test_merging_rejects_a_missing_export_shard(tmp_path):
    pd.DataFrame({"sample_id": ["fold1_0"], "n_cells": [1], "path": ["p"]}).to_csv(
        tmp_path / "manifest.shard000.csv", index=False
    )

    with pytest.raises(ValueError, match="expected 2 shard manifests"):
        merge_shard_manifests(tmp_path, 2)
