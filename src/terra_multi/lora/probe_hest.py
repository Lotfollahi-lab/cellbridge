"""Probe exported LoRA arms on the fixed HEST evaluation cohort."""

from pathlib import Path

import numpy as np
import pandas as pd

from terra_multi.evaluation.artifacts import (
    file_identity,
    load_array_artifact,
    stable_signature,
)
from terra_multi.evaluation.classification import (
    cached_probe_metrics,
    evaluate_classification,
    method_directory,
    resolve_methods,
    write_probe_result,
)
from terra_multi.evaluation.hest import prepare_labeled_hest_features
from terra_multi.lora.features import FEATURE_ARMS, FORMAT_VERSION, PCA_ARMS
from terra_multi.lora.hest import label_test_fast_samples, manifest_row, sample_dir


def _load_feature(path, arm, expected_signature):
    with np.load(path, allow_pickle=False) as data:
        if arm not in data:
            return None, None
    arrays, cell_ids, _ = load_array_artifact(
        path,
        format_version=FORMAT_VERSION,
        signature=expected_signature,
        array_dims={arm: None},
    )
    return arrays[arm], cell_ids


def evaluate_hest_sample(features, cell_ids, sample_id, arm, cfg, method="linear"):
    settings = cfg["eval"]
    _, row = manifest_row(cfg, sample_id)
    directory = sample_dir(cfg, sample_id, row)
    data = prepare_labeled_hest_features(
        features,
        cell_ids,
        directory / f"{sample_id}_granular_cell_labels.csv",
        directory / f"{sample_id}.h5ad",
        settings,
        pca=arm in PCA_ARMS and settings.get("pca", True),
        pca_components=settings.get("uni_pca_components", 256),
    )
    result = evaluate_classification(
        data.features,
        data.labels,
        data.train_indices,
        data.test_indices,
        method,
        settings,
        row_ids=data.cell_ids,
    )
    common = result.metrics
    metrics = {
        "sample_id": sample_id,
        "arm": arm,
        **{key: common[key] for key in ("n_features", "n_train", "n_test", "n_classes")},
        **{
            key: common[key]
            for key in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
        },
        "pca_explained_variance": data.pca_explained_variance,
    }
    for key in ("n_neighbors", "neighbor_metric", "neighbor_weights"):
        if key in common:
            metrics[key] = common[key]
    predictions = {
        "cell_ids_test": result.predictions["row_ids_test"],
        "y_true": result.predictions["y_true"],
        "y_pred": result.predictions["y_pred"],
        "classes": result.predictions["classes"].astype(str),
    }
    return metrics, predictions


def probe_hest(cfg, output_dir, samples=None):
    output_dir = Path(output_dir)
    manifest = pd.read_csv(output_dir / "features" / "manifest.csv")
    expected = label_test_fast_samples(cfg)
    if manifest["sample_id"].astype(str).tolist() != expected:
        raise ValueError("HEST feature manifest is not the configured label_test_fast set")
    if samples is not None:
        requested = {str(sample_id) for sample_id in samples}
        unknown = sorted(requested - set(expected))
        if unknown:
            raise ValueError(f"not label_test_fast samples: {unknown}")
        manifest = manifest[manifest["sample_id"].astype(str).isin(requested)]
    methods = resolve_methods(cfg["eval"])
    rows = {method: [] for method in methods}
    for record in manifest.itertuples(index=False):
        directory = sample_dir(cfg, str(record.sample_id))
        sources = {
            "labels": file_identity(
                directory / f"{record.sample_id}_granular_cell_labels.csv"
            ),
            "adata": file_identity(directory / f"{record.sample_id}.h5ad"),
        }
        for arm in FEATURE_ARMS:
            features = cell_ids = None
            for method in methods:
                payload = {
                    "features": str(record.signature),
                    "sources": sources,
                    "arm": arm,
                    "settings": cfg["eval"],
                }
                if method != "linear":
                    payload["method"] = method
                signature = stable_signature(payload)
                result_dir = method_directory(
                    output_dir / "probes" / str(record.sample_id) / arm, method
                )
                cached = cached_probe_metrics(result_dir / "metrics.csv", signature)
                if cached is not None:
                    rows[method].append(cached)
                    print(f"[HEST/{record.sample_id}/{arm}/{method}] current")
                    continue
                if features is None:
                    features, cell_ids = _load_feature(record.path, arm, record.signature)
                if features is None:
                    break
                metrics, predictions = evaluate_hest_sample(
                    features, cell_ids, str(record.sample_id), arm, cfg, method
                )
                metrics = write_probe_result(result_dir, metrics, predictions, signature)
                rows[method].append(metrics)
                print(
                    f"[HEST/{record.sample_id}/{arm}/{method}] "
                    f"macro-F1={metrics['macro_f1']:.4f}, "
                    f"balanced accuracy={metrics['balanced_accuracy']:.4f}"
                )
    if samples is not None:
        primary = "linear" if "linear" in methods else methods[0]
        return pd.DataFrame(rows[primary]), None
    summaries = {}
    metrics = ["macro_f1", "balanced_accuracy", "accuracy", "weighted_f1"]
    for method in methods:
        result = pd.DataFrame(rows[method])
        suffix = "" if method == "linear" else f"_{method}"
        result.to_csv(output_dir / f"probe_metrics{suffix}.csv", index=False)
        summary = result.groupby("arm")[metrics].agg(["mean", "std"])
        summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
        summary = summary.reindex([arm for arm in FEATURE_ARMS if arm in summary.index])
        summary.to_csv(output_dir / f"probe_summary{suffix}.csv")
        summaries[method] = result, summary
    return summaries["linear" if "linear" in methods else methods[0]]
