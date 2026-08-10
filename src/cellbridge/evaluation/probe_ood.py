"""Compare frozen raw UNI and distilled features on out-of-distribution labels."""

import argparse
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from cellbridge.config import load_config
from sklearn.metrics import f1_score

from cellbridge.evaluation.classification import (
    method_directory,
    resolve_methods,
)
from cellbridge.evaluation.linear_probe import reduce_features
from cellbridge.evaluation.ood import (
    IMMUNE_CLASSES,
    evaluate_ood_task,
    prepare_ood_task,
    validate_official_split,
)

ARMS = ("uni", "distilled")


def resolve_probe_protocols(settings):
    """Merge named classifier-prior variants without repeating feature settings."""
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
            raise ValueError(f"probe protocol {name!r} must be a mapping")
        unknown = set(overrides) - allowed
        if unknown:
            raise ValueError(
                f"probe protocol {name!r} has unsupported overrides {sorted(unknown)}"
            )
        resolved.append((name, {**base, **overrides}))
    return resolved


__all__ = [
    "ARMS",
    "IMMUNE_CLASSES",
    "evaluate",
    "load_distilled",
    "load_uni",
    "validate_official_split",
]


def load_uni(manifest):
    features, records = [], []
    for path in manifest["path"]:
        with np.load(path, allow_pickle=False) as data:
            features.append(data["uni_emb"].astype(np.float32))
            records.append(
                pd.DataFrame(
                    {
                        "row_id": data["row_id"],
                        "sample_id": data["sample_id"].astype(str),
                        "tile_id": data["tile_id"].astype(str),
                        "cell_id": data["cell_id"],
                        "label": data["label"].astype(str),
                        "class_id": data["class_id"],
                        "split": data["split"].astype(str),
                    }
                )
            )
    table = pd.concat(records, ignore_index=True)
    split = table["split"].replace({"valid": "val"}).to_numpy()
    return np.concatenate(features), table, split


def load_distilled(manifest, expected_row_ids):
    features, row_ids = [], []
    for path in manifest["path"]:
        with np.load(path, allow_pickle=False) as data:
            features.append(data["distilled_emb"].astype(np.float32))
            row_ids.append(data["row_id"])
    row_ids = np.concatenate(row_ids)
    if not np.array_equal(row_ids, expected_row_ids):
        raise ValueError("distilled embeddings are not aligned with cached UNI rows")
    return np.concatenate(features)


def reduce_uni(features, train_indices, components, seed):
    """Compatibility wrapper around the shared train-only PCA."""
    return reduce_features(features, train_indices, components, seed)


def _format_result(data, result, task, arm, protocol=None):
    common = result.metrics
    labels = result.predictions["classes"]
    per_class = f1_score(
        result.predictions["y_true"],
        result.predictions["y_pred"],
        labels=labels,
        average=None,
        zero_division=0,
    )
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
        "uni_pca_explained_variance": data.pca_explained_variance,
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
    class_rows = [
        {
            "task": task,
            "arm": arm,
            **({} if protocol is None else {"protocol": protocol}),
            "class": label,
            "f1": score,
        }
        for label, score in zip(labels, per_class)
    ]
    predictions = {
        "row_id": result.predictions["row_ids_test"],
        "y_true": result.predictions["y_true"],
        "y_pred": result.predictions["y_pred"],
        "classes": labels,
    }
    return metrics, class_rows, predictions


def evaluate(features, table, split, task, arm, settings, method="linear"):
    """Compatibility helper for evaluating one PhenoCell arm and method."""
    data = prepare_ood_task(
        features,
        table,
        task,
        settings,
        pca=arm == "uni",
        split=split,
    )
    result = evaluate_ood_task(data, method, settings)
    return _format_result(data, result, task, arm)


def _method_path(output_dir, task, arm, method, protocol=None):
    root = Path(output_dir) / "probes"
    if protocol is not None:
        root = root / protocol
    return method_directory(root / task / arm, method)


def _summary_name(prefix, arm, method):
    method_prefix = "" if method == "linear" else f"{method}_"
    return f"{prefix}_{method_prefix}{arm or 'all'}.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--methods", help="comma-separated: linear,knn")
    args = parser.parse_args()
    cfg = load_config(args.config)

    output_dir = Path(cfg["output_dir"])
    uni_dir = Path(cfg["embedding"]["output_dir"])
    uni_manifest = pd.read_csv(uni_dir / "manifest.csv")
    uni, table, split = load_uni(uni_manifest)
    validate_official_split(
        table,
        cfg["data"]["split"],
        require_full_coverage=(
            not cfg["data"].get("allow_unlabeled_samples", False)
            and not cfg["embedding"].get("development_subset")
        ),
    )
    arms = [args.arm] if args.arm else list(ARMS)
    methods = resolve_methods(cfg["probe"], args.methods)
    protocols = resolve_probe_protocols(cfg["probe"])
    keys = [(protocol, method) for protocol, _ in protocols for method in methods]
    distilled = None
    metrics_by_key = {key: [] for key in keys}
    classes_by_key = {key: [] for key in keys}
    for arm in arms:
        if arm == "uni":
            features = uni
        else:
            if distilled is None:
                projection_manifest = pd.read_csv(
                    output_dir / "embeddings" / "distilled" / "manifest.csv"
                )
                distilled = load_distilled(
                    projection_manifest, table["row_id"].to_numpy()
                )
            features = distilled
        for task in cfg["probe"].get("tasks", ["all", "immune"]):
            data = prepare_ood_task(
                features,
                table,
                task,
                cfg["probe"],
                pca=arm == "uni",
                split=split,
            )
            for protocol, protocol_settings in protocols:
                for method in methods:
                    result = evaluate_ood_task(data, method, protocol_settings)
                    metrics, class_rows, predictions = _format_result(
                        data, result, task, arm, protocol
                    )
                    result_dir = _method_path(
                        output_dir, task, arm, method, protocol
                    )
                    result_dir.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame([metrics]).to_csv(
                        result_dir / "metrics.csv", index=False
                    )
                    pd.DataFrame(class_rows).to_csv(
                        result_dir / "per_class_f1.csv", index=False
                    )
                    np.savez(result_dir / "predictions.npz", **predictions)
                    key = (protocol, method)
                    metrics_by_key[key].append(metrics)
                    classes_by_key[key].extend(class_rows)
                    label = method if protocol is None else f"{protocol}/{method}"
                    print(
                        f"[{task}/{arm}/{label}] "
                        f"{metrics['evaluation_split']} "
                        f"macro-F1={metrics['test_macro_f1']:.4f}"
                    )

    for protocol, _ in protocols:
        summary_dir = output_dir / "probes"
        if protocol is not None:
            summary_dir = summary_dir / protocol
        for method in methods:
            key = (protocol, method)
            summary_path = summary_dir / _summary_name(
                "summary", args.arm, method
            )
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(metrics_by_key[key]).to_csv(summary_path, index=False)
            pd.DataFrame(classes_by_key[key]).to_csv(
                summary_dir / _summary_name("per_class", args.arm, method),
                index=False,
            )
            print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
