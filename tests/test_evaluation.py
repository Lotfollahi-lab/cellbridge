from pathlib import Path

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

from terra_multi.evaluation.classification import (
    evaluate_classification,
    method_directory,
    resolve_methods,
)
from terra_multi.evaluation.hest import HestData, evaluate_hest_data
from terra_multi.evaluation.phenocell import validate_official_split
from terra_multi.evaluation.probe_phenocell import evaluate as evaluate_phenocell
from terra_multi.lora import evaluate as lora_evaluate
from terra_multi.lora.evaluate import evaluate_phenocell_arm


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


def test_phenocell_adapters_keep_linear_schema_and_support_knn():
    features, labels, train, test = classification_data()
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

    metrics, _, predictions = evaluate_phenocell(
        features, table, split, "all", "distilled", settings, "linear"
    )
    assert set(metrics) == {
        "task",
        "arm",
        "n_features",
        "n_train",
        "n_val",
        "n_test",
        "n_classes",
        "best_C",
        "val_macro_f1",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
        "uni_pca_explained_variance",
    }
    assert set(predictions) == {"row_id", "y_true", "y_pred", "classes"}

    lora_metrics, lora_predictions = evaluate_phenocell_arm(
        features, table, "all", "lora_projected", settings, "knn"
    )
    assert lora_metrics["n_neighbors"] == 1
    assert lora_metrics["test_macro_f1"] == 1.0
    assert set(lora_predictions) == {"row_id", "y_true", "y_pred", "classes"}


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
