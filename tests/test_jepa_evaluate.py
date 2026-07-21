from pathlib import Path

import pandas as pd

from terra_multi.evaluation.hest import result_directory
from terra_multi.jepa.evaluate import ARMS, aggregate, arm_config, fast_samples


def make_config(tmp_path):
    data_root = tmp_path / "data"
    sample_dir = data_root / "section_a"
    sample_dir.mkdir(parents=True)
    pd.DataFrame(
        {"cell_id": ["a", "b", "c"], "group": ["immune", "immune", "tumour"]}
    ).to_csv(sample_dir / "section_a_granular_cell_labels.csv", index=False)
    fast_path = tmp_path / "fast.csv"
    pd.DataFrame({"sample_id": ["section_a"]}).to_csv(fast_path, index=False)
    splits_path = tmp_path / "splits.csv"
    pd.DataFrame(
        {"sample_id": ["section_a"], "split": ["label_test"]}
    ).to_csv(splits_path, index=False)
    return {
        "data_root": str(data_root),
        "output_dir": str(tmp_path / "output"),
        "splits_path": str(splits_path),
        "uni_key": "uni_emb",
        "terra_key": "cell_emb",
        "eval": {
            "label_test_fast_path": str(fast_path),
            "label_id_column": "cell_id",
            "label_column": "group",
            "min_class_count": 1,
            "test_frac": 0.2,
            "split_axis": "y",
            "pca": True,
            "uni_pca_components": 16,
            "methods": ["linear", "knn"],
            "Cs": [1.0],
            "class_weight": "balanced",
            "knn_neighbors": 3,
            "knn_metric": "cosine",
            "knn_weights": "uniform",
        },
    }


def test_arm_config_uses_jepa_features_and_shared_baselines(tmp_path):
    cfg = make_config(tmp_path)

    jepa = arm_config(cfg, "section_a", "jepa")
    uni = arm_config(cfg, "section_a", "uni_baseline")
    terra = arm_config(cfg, "section_a", "terra_teacher")
    jepa_path = Path(cfg["output_dir"]) / "features" / "section_a.npz"

    assert Path(jepa["terra_embeddings"]) == jepa_path
    assert jepa["terra_key"] == "jepa_emb"
    assert Path(uni["terra_embeddings"]) == jepa_path
    assert uni["terra_key"] == "jepa_emb"
    assert Path(terra["uni_embeddings"]) == jepa_path
    assert terra["uni_key"] == "jepa_emb"
    assert terra["terra_key"] == "cell_emb"
    assert uni["keep_classes"] == ["immune", "tumour"]
    assert fast_samples(cfg) == ["section_a"]


def test_aggregate_writes_separate_linear_and_knn_summaries(tmp_path):
    cfg = make_config(tmp_path)
    root = Path(cfg["output_dir"]) / "evaluation" / "hest_label_test_fast"
    metrics = {
        "accuracy": 0.5,
        "balanced_accuracy": 0.4,
        "macro_f1": 0.3,
        "macro_auroc": 0.6,
        "n_classes": 2,
    }
    for method in ("linear", "knn"):
        for arm, model in ARMS.items():
            base = root / "probes" / "section_a" / arm
            destination = result_directory(base, model, method)
            destination.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([metrics]).to_csv(destination / "metrics.csv", index=False)

    aggregate(cfg, ["section_a"])

    assert (root / "probe_metrics.csv").exists()
    assert (root / "probe_summary.csv").exists()
    assert (root / "probe_metrics_knn.csv").exists()
    assert (root / "probe_summary_knn.csv").exists()
