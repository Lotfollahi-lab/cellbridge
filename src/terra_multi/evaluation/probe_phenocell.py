"""Compare frozen raw UNI and distilled features on PhenoCell labels."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score

from terra_multi.evaluation.classification import (
    method_directory,
    resolve_methods,
)
from terra_multi.evaluation.linear_probe import reduce_features
from terra_multi.evaluation.phenocell import (
    IMMUNE_CLASSES,
    evaluate_phenocell_task,
    prepare_phenocell_task,
    validate_official_split,
)

ARMS = ("uni", "distilled")
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


def _format_result(data, result, task, arm):
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
        "n_val": common["n_val"],
        "n_test": common["n_test"],
        "n_classes": common["n_classes"],
        "test_accuracy": common["accuracy"],
        "test_balanced_accuracy": common["balanced_accuracy"],
        "test_macro_f1": common["macro_f1"],
        "test_weighted_f1": common["weighted_f1"],
        "uni_pca_explained_variance": data.pca_explained_variance,
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
    class_rows = [
        {"task": task, "arm": arm, "class": label, "f1": score}
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
    data = prepare_phenocell_task(
        features,
        table,
        task,
        settings,
        pca=arm == "uni",
        split=split,
    )
    result = evaluate_phenocell_task(data, method, settings)
    return _format_result(data, result, task, arm)


def _method_path(output_dir, task, arm, method):
    return method_directory(Path(output_dir) / "probes" / task / arm, method)


def _summary_name(prefix, arm, method):
    method_prefix = "" if method == "linear" else f"{method}_"
    return f"{prefix}_{method_prefix}{arm or 'all'}.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--methods", help="comma-separated: linear,knn")
    args = parser.parse_args()
    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)

    output_dir = Path(cfg["output_dir"])
    uni_dir = Path(cfg["embedding"]["output_dir"])
    uni_manifest = pd.read_csv(uni_dir / "manifest.csv")
    uni, table, split = load_uni(uni_manifest)
    validate_official_split(table, cfg["data"]["split"])
    arms = [args.arm] if args.arm else list(ARMS)
    methods = resolve_methods(cfg["probe"], args.methods)
    distilled = None
    metrics_by_method = {method: [] for method in methods}
    classes_by_method = {method: [] for method in methods}
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
            data = prepare_phenocell_task(
                features,
                table,
                task,
                cfg["probe"],
                pca=arm == "uni",
                split=split,
            )
            for method in methods:
                result = evaluate_phenocell_task(data, method, cfg["probe"])
                metrics, class_rows, predictions = _format_result(
                    data, result, task, arm
                )
                result_dir = _method_path(output_dir, task, arm, method)
                result_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame([metrics]).to_csv(result_dir / "metrics.csv", index=False)
                pd.DataFrame(class_rows).to_csv(
                    result_dir / "per_class_f1.csv", index=False
                )
                np.savez(result_dir / "predictions.npz", **predictions)
                metrics_by_method[method].append(metrics)
                classes_by_method[method].extend(class_rows)
                print(
                    f"[{task}/{arm}/{method}] "
                    f"test macro-F1={metrics['test_macro_f1']:.4f}"
                )

    for method in methods:
        summary_path = output_dir / "probes" / _summary_name(
            "summary", args.arm, method
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(metrics_by_method[method]).to_csv(summary_path, index=False)
        pd.DataFrame(classes_by_method[method]).to_csv(
            output_dir
            / "probes"
            / _summary_name("per_class", args.arm, method),
            index=False,
        )
        print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
