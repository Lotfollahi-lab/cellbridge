"""Probe exported LoRA arms on the official PhenoCell split."""

from pathlib import Path

import numpy as np
import pandas as pd

from cellbridge.evaluation.artifacts import (
    file_identity,
    load_array_artifact,
    stable_signature,
)
from cellbridge.evaluation.classification import (
    method_directory,
    resolve_methods,
    write_probe_result,
)
from cellbridge.evaluation.ood import (
    evaluate_ood_task,
    prepare_ood_task,
    validate_official_split,
)
from cellbridge.lora.features import FEATURE_ARMS, FORMAT_VERSION, PCA_ARMS


def load_phenocell_arm(manifest, arm):
    features, records = [], []
    for row in manifest.itertuples(index=False):
        with np.load(row.path, allow_pickle=False) as data:
            if arm not in data:
                return None, None
        arrays, _, metadata = load_array_artifact(
            row.path,
            format_version=FORMAT_VERSION,
            signature=row.signature,
            array_dims={arm: None},
            metadata_keys=("row_id", "sample_id", "label", "split"),
        )
        features.append(arrays[arm])
        records.append(
            pd.DataFrame(
                {
                    "row_id": metadata["row_id"],
                    "sample_id": metadata["sample_id"].astype(str),
                    "label": metadata["label"].astype(str),
                    "split": metadata["split"].astype(str),
                }
            )
        )
    return np.concatenate(features), pd.concat(records, ignore_index=True)


def _format_result(data, result, task, arm):
    common = result.metrics
    metrics = {
        "task": task,
        "arm": arm,
        "n_features": common["n_features"],
        "n_train": common["n_train"],
        "n_val": common["n_val"],
        "n_test": common["n_test"],
        "n_classes": common["n_classes"],
        "test_accuracy": common["accuracy"],
        "test_balanced_accuracy": common["balanced_accuracy"],
        "test_macro_f1": common["macro_f1"],
        "test_weighted_f1": common["weighted_f1"],
        "pca_explained_variance": data.pca_explained_variance,
    }
    for key in (
        "best_C",
        "val_macro_f1",
        "n_neighbors",
        "neighbor_metric",
        "neighbor_weights",
    ):
        if key in common:
            metrics[key] = common[key]
    predictions = {
        "row_id": result.predictions["row_ids_test"],
        "y_true": result.predictions["y_true"],
        "y_pred": result.predictions["y_pred"],
        "classes": result.predictions["classes"].astype(str),
    }
    return metrics, predictions


def evaluate_phenocell_arm(features, table, task, arm, settings, method="linear"):
    """Evaluate one LoRA feature arm with the shared PhenoCell protocol."""
    data = prepare_ood_task(features, table, task, settings, pca=arm in PCA_ARMS)
    result = evaluate_ood_task(data, method, settings)
    return _format_result(data, result, task, arm)


def probe_phenocell(cfg, output_dir):
    output_dir = Path(output_dir)
    manifest = pd.read_csv(output_dir / "features" / "manifest.csv")
    settings = cfg.get("probe", {})
    methods = resolve_methods(settings)
    all_metrics = {method: [] for method in methods}
    for arm in FEATURE_ARMS:
        features, table = load_phenocell_arm(manifest, arm)
        if features is None:
            continue
        validate_official_split(
            table,
            cfg["data"]["split"],
            require_full_coverage=not cfg["data"].get("allow_unlabeled_samples", False),
        )
        for task in settings.get("tasks", ["all", "immune"]):
            data = prepare_ood_task(features, table, task, settings, pca=arm in PCA_ARMS)
            for method in methods:
                result = evaluate_ood_task(data, method, settings)
                metrics, predictions = _format_result(data, result, task, arm)
                payload = {
                    "features": manifest["signature"].astype(str).tolist(),
                    "official_split": file_identity(cfg["data"]["split"]),
                    "arm": arm,
                    "task": task,
                    "settings": settings,
                }
                if method != "linear":
                    payload["method"] = method
                signature = stable_signature(payload)
                result_dir = method_directory(output_dir / "probes" / task / arm, method)
                metrics = write_probe_result(result_dir, metrics, predictions, signature)
                all_metrics[method].append(metrics)
                print(
                    f"[PhenoCell/{task}/{arm}/{method}] "
                    f"macro-F1={metrics['test_macro_f1']:.4f}, "
                    f"balanced accuracy={metrics['test_balanced_accuracy']:.4f}"
                )
    results = {}
    for method in methods:
        result = pd.DataFrame(all_metrics[method])
        suffix = "" if method == "linear" else f"_{method}"
        result.to_csv(output_dir / f"probe_summary{suffix}.csv", index=False)
        results[method] = result
    return results["linear" if "linear" in methods else methods[0]]
