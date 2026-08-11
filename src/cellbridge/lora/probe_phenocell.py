"""Probe exported LoRA arms on the official PhenoCell split."""

from collections.abc import Mapping
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


def resolve_probe_protocols(settings):
    """Merge named classifier-prior variants with shared feature settings."""
    configured = settings.get("protocols")
    if configured is None:
        return [(None, dict(settings))]
    if not isinstance(configured, Mapping) or not configured:
        raise ValueError("probe.protocols must be a non-empty mapping")

    base = {key: value for key, value in settings.items() if key != "protocols"}
    allowed = {"class_weight", "knn_reference_sampling"}
    resolved = []
    for name, overrides in configured.items():
        if not isinstance(name, str) or not name or "/" in name:
            raise ValueError(f"invalid probe protocol name {name!r}")
        if not isinstance(overrides, Mapping):
            raise TypeError(f"probe protocol {name!r} must be a mapping")
        unknown = set(overrides) - allowed
        if unknown:
            raise ValueError(
                f"probe protocol {name!r} has unsupported overrides {sorted(unknown)}"
            )
        resolved.append((name, {**base, **overrides}))
    return resolved


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


def _format_result(data, result, task, arm, protocol=None):
    common = result.metrics
    metrics = {
        "task": task,
        "arm": arm,
        "n_features": common["n_features"],
        "n_train": common["n_train"],
        "n_val": common.get("n_val", 0),
        "n_test": common["n_test"],
        "n_classes": common["n_classes"],
        "test_accuracy": common["accuracy"],
        "test_balanced_accuracy": common["balanced_accuracy"],
        "test_macro_f1": common["macro_f1"],
        "test_weighted_f1": common["weighted_f1"],
        "pca_explained_variance": data.pca_explained_variance,
        "evaluation_split": "val" if data.val_indices is None else "test",
    }
    if protocol is not None:
        metrics["protocol"] = protocol
    for key in (
        "best_C",
        "val_macro_f1",
        "n_neighbors",
        "neighbor_metric",
        "neighbor_weights",
        "reference_sampling",
        "n_reference",
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
    protocols = resolve_probe_protocols(settings)
    all_metrics = {
        (protocol, method): [] for protocol, _ in protocols for method in methods
    }
    for arm in FEATURE_ARMS:
        features, table = load_phenocell_arm(manifest, arm)
        if features is None:
            continue
        validate_official_split(
            table,
            cfg["data"]["split"],
            require_full_coverage=(
                not cfg["data"].get("allow_unlabeled_samples", False)
                and not cfg.get("embedding", {}).get("development_subset")
            ),
        )
        for task in settings.get("tasks", ["all", "immune"]):
            data = prepare_ood_task(
                features, table, task, settings, pca=arm in PCA_ARMS
            )
            for protocol, protocol_settings in protocols:
                for method in methods:
                    result = evaluate_ood_task(data, method, protocol_settings)
                    metrics, predictions = _format_result(
                        data,
                        result,
                        task,
                        arm,
                        protocol,
                    )
                    payload = {
                        "features": manifest["signature"].astype(str).tolist(),
                        "official_split": file_identity(cfg["data"]["split"]),
                        "arm": arm,
                        "task": task,
                        "method": method,
                        "protocol": protocol,
                        "settings": protocol_settings,
                    }
                    signature = stable_signature(payload)
                    probe_root = output_dir / "probes"
                    if protocol is not None:
                        probe_root = probe_root / protocol
                    result_dir = method_directory(
                        probe_root / task / arm, method
                    )
                    metrics = write_probe_result(
                        result_dir, metrics, predictions, signature
                    )
                    all_metrics[(protocol, method)].append(metrics)
                    label = method if protocol is None else f"{protocol}/{method}"
                    print(
                        f"[PhenoCell/{task}/{arm}/{label}] "
                        f"{metrics['evaluation_split']} "
                        f"macro-F1={metrics['test_macro_f1']:.4f}, "
                        f"balanced accuracy={metrics['test_balanced_accuracy']:.4f}"
                    )
    results = {}
    for protocol, _ in protocols:
        summary_dir = output_dir if protocol is None else output_dir / "probes" / protocol
        summary_dir.mkdir(parents=True, exist_ok=True)
        for method in methods:
            result = pd.DataFrame(all_metrics[(protocol, method)])
            suffix = "" if method == "linear" else f"_{method}"
            result.to_csv(summary_dir / f"probe_summary{suffix}.csv", index=False)
            results[(protocol, method)] = result
    preferred_method = "linear" if "linear" in methods else methods[0]
    return results[(protocols[0][0], preferred_method)]
