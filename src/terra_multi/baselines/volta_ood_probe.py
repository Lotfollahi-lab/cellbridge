"""Artifact validation and shared probes for VOLTA OOD features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from terra_multi.baselines.evaluation import (
    finalize_phenocell_manifest,
    phenocell_cell_ids,
    probe_phenocell,
)
from terra_multi.baselines.volta_ood_extract import ood_feature_path
from terra_multi.evaluation.artifacts import (
    file_identity,
    load_feature_file,
    stable_signature,
    summarize_feature_file,
    write_table,
)
from terra_multi.evaluation.classification import (
    cached_probe_metrics,
    resolve_methods,
    write_probe_result,
)
from terra_multi.evaluation.ood import (
    evaluate_ood_task,
    prepare_ood_task,
    validate_official_split,
)


def _manifest_row(summary):
    return {
        "sample_id": summary.sample_id,
        "n_cells": summary.n_cells,
        "n_features": summary.n_features,
        "path": summary.path,
        "signature": summary.signature,
        "model_signature": summary.model_signature,
    }


def finalize_ood_manifest(cfg, output_dir, dataset_name):
    if dataset_name == "phenocell":
        return finalize_phenocell_manifest(cfg, output_dir)
    dataset = cfg[dataset_name]
    metadata = pd.read_csv(dataset["data"]["metadata"])
    metadata["row_id"] = np.arange(len(metadata), dtype=np.int64)
    samples = metadata["sample_id"].astype(str).drop_duplicates().tolist()
    expected_dim = int(cfg["baseline"]["embedding_dim"])
    summaries = []
    all_rows = []
    for sample_id in samples:
        summary = summarize_feature_file(
            ood_feature_path(output_dir, sample_id), expected_dim=expected_dim
        )
        if summary.sample_id != sample_id:
            raise ValueError(f"{summary.path}: expected sample {sample_id}")
        if summary.dataset_name != dataset_name:
            raise ValueError(f"{summary.path}: wrong dataset")
        if summary.baseline_name != cfg["baseline"]["name"]:
            raise ValueError(f"{summary.path}: wrong baseline")
        _, cell_ids, arrays = load_feature_file(
            summary.path,
            expected_signature=summary.signature,
            expected_dim=expected_dim,
            metadata_keys=("row_id", "sample_id", "tile_id", "cell_id"),
        )
        observed_ids = phenocell_cell_ids(
            pd.DataFrame(
                {
                    "sample_id": arrays["sample_id"],
                    "tile_id": arrays["tile_id"],
                    "cell_id": arrays["cell_id"],
                }
            )
        )
        if not np.array_equal(cell_ids, observed_ids):
            raise ValueError(f"{sample_id}: global cell IDs are inconsistent")
        expected_rows = metadata.loc[
            metadata["sample_id"].astype(str) == sample_id, "row_id"
        ].to_numpy()
        if not np.isin(arrays["row_id"], expected_rows).all():
            raise ValueError(f"{sample_id}: artifact contains foreign rows")
        all_rows.append(arrays["row_id"].astype(np.int64))
        summaries.append(summary)
    if len({item.model_signature for item in summaries}) != 1:
        raise ValueError("feature artifacts use different model revisions")
    row_ids = np.concatenate(all_rows)
    if len(np.unique(row_ids)) != len(row_ids):
        raise ValueError("feature artifacts contain duplicate metadata rows")
    result = [_manifest_row(item) for item in summaries]
    write_table(Path(output_dir) / "features" / "manifest.csv", result)
    return pd.DataFrame(result)


def _load_features(manifest, expected_dim):
    features = []
    tables = []
    keys = ("row_id", "sample_id", "label", "split")
    for row in manifest.itertuples(index=False):
        values, _, arrays = load_feature_file(
            row.path,
            expected_signature=row.signature,
            expected_dim=expected_dim,
            metadata_keys=keys,
        )
        features.append(values)
        tables.append(pd.DataFrame({key: arrays[key] for key in keys}))
    values = np.concatenate(features)
    table = pd.concat(tables, ignore_index=True)
    order = np.argsort(table["row_id"].to_numpy(), kind="stable")
    return values[order], table.iloc[order].reset_index(drop=True)


def _protocol_identity():
    package = Path(__file__).resolve().parents[1]
    paths = (
        Path(__file__),
        package / "evaluation" / "classification.py",
        package / "evaluation" / "linear_probe.py",
        package / "evaluation" / "ood.py",
        package / "evaluation" / "zero_shot.py",
    )
    return [file_identity(path, checksum=True) for path in paths]


def _format_result(data, result, task, feature_name):
    score_names = {
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_auroc",
    }
    metrics = {
        "task": task,
        "feature_name": feature_name,
        **{
            (f"test_{key}" if key in score_names else key): value
            for key, value in result.metrics.items()
        },
        "pca_train_explained_variance": data.pca_explained_variance,
    }
    predictions = {
        "row_id": result.predictions["row_ids_test"],
        "y_true": result.predictions["y_true"],
        "y_pred": result.predictions["y_pred"],
        "y_proba": result.predictions["y_proba"],
        "classes": result.predictions["classes"].astype(str),
    }
    return metrics, predictions


def probe_ood(cfg, output_dir, dataset_name):
    if dataset_name == "phenocell":
        return probe_phenocell(cfg, output_dir)
    manifest = finalize_ood_manifest(cfg, output_dir, dataset_name)
    expected_dim = int(cfg["baseline"]["embedding_dim"])
    features, table = _load_features(manifest, expected_dim)
    dataset = cfg[dataset_name]
    settings = dataset["probe"]
    if settings.get("pca", False):
        raise ValueError("VOLTA embeddings must use pca: false")
    validate_official_split(
        table,
        dataset["data"]["split"],
        require_full_coverage=not dataset["data"].get("allow_unlabeled_samples", False),
    )
    methods = resolve_methods(settings)
    output = {method: [] for method in methods}
    for task in settings["tasks"]:
        data = prepare_ood_task(features, table, task, settings, pca=False)
        for method in methods:
            signature = stable_signature(
                {
                    "feature_signatures": manifest["signature"].astype(str).tolist(),
                    "official_split": file_identity(dataset["data"]["split"]),
                    "task": task,
                    "method": method,
                    "settings": settings,
                    "probe_protocol": _protocol_identity(),
                }
            )
            result_dir = Path(output_dir) / "probes" / task / method
            cached = cached_probe_metrics(result_dir / "metrics.csv", signature)
            if cached is None:
                result = evaluate_ood_task(data, method, settings)
                metrics, predictions = _format_result(
                    data, result, task, cfg["baseline"]["name"]
                )
                cached = write_probe_result(result_dir, metrics, predictions, signature)
                print(
                    f"[{dataset_name}/{task}/{method}] "
                    f"macro-F1={cached['test_macro_f1']:.4f}"
                )
            else:
                print(f"[{dataset_name}/{task}/{method}] current")
            output[method].append(cached)
    for method, rows in output.items():
        output[method] = pd.DataFrame(rows)
        write_table(Path(output_dir) / f"probe_metrics_{method}.csv", rows)
    return output
