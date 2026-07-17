"""Model-agnostic HEST and PhenoCell probes for signed baseline features."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from terra_multi.baselines.artifacts import (
    file_identity,
    load_feature_file,
    scalar_string,
    stable_signature,
    summarize_feature_file,
    write_table,
)
from terra_multi.evaluation.classification import (
    evaluate_classification,
    resolve_methods,
)
from terra_multi.evaluation.hest import load_labels
from terra_multi.evaluation.linear_probe import reduce_features
from terra_multi.evaluation.phenocell import (
    evaluate_phenocell_task,
    prepare_phenocell_task,
    validate_official_split,
)
from terra_multi.helpers.io import load_coords
from terra_multi.helpers.splits import spatial_split


def _as_strings(values):
    return np.asarray(values).astype(str)


def label_test_fast_samples(hest_cfg) -> list[str]:
    """Return the fixed fast subset after proving it is held out by TERRA."""
    splits_path = Path(hest_cfg["splits_path"])
    fast_path = Path(hest_cfg["label_test_fast_path"])
    splits = pd.read_csv(splits_path)
    if not {"sample_id", "split"} <= set(splits):
        raise ValueError(f"{splits_path} requires sample_id and split columns")
    splits["sample_id"] = splits["sample_id"].astype(str)
    if splits["sample_id"].duplicated().any():
        raise ValueError(f"{splits_path} contains duplicate samples")

    fast = pd.read_csv(fast_path)
    if "sample_id" not in fast:
        raise ValueError(f"{fast_path} requires a sample_id column")
    samples = fast["sample_id"].astype(str).tolist()
    if len(samples) != len(set(samples)):
        raise ValueError(f"{fast_path} contains duplicate samples")
    assigned = splits.set_index("sample_id")["split"].astype(str).to_dict()
    invalid = [sample for sample in samples if assigned.get(sample) != "label_test"]
    if invalid:
        raise ValueError(
            "label_test_fast contains samples outside label_test: "
            f"{invalid[:5]}"
        )
    expected = int(hest_cfg.get("label_test_fast_n", 5))
    if len(samples) != expected:
        raise ValueError(
            f"{fast_path} has {len(samples)} samples; expected exactly {expected}"
        )
    return samples


def hest_manifest_row(hest_cfg, sample_id):
    manifest_path = Path(hest_cfg["manifest_path"])
    manifest = pd.read_csv(manifest_path)
    if "id" not in manifest:
        raise ValueError(f"{manifest_path} requires an id column")
    manifest["id"] = manifest["id"].astype(str)
    selected = manifest.loc[manifest["id"] == str(sample_id)]
    if len(selected) != 1:
        raise KeyError(f"expected one manifest row for {sample_id}, found {len(selected)}")
    return selected.iloc[0]


def hest_sample_dir(hest_cfg, sample_id, row=None) -> Path:
    row = hest_manifest_row(hest_cfg, sample_id) if row is None else row
    output_dir = row.get("output_dir")
    if output_dir is None or pd.isna(output_dir):
        return Path(hest_cfg["data_root"]) / str(sample_id)
    return Path(output_dir)


def hest_terra_path(hest_cfg, sample_id, row=None) -> Path:
    return hest_sample_dir(hest_cfg, sample_id, row) / f"{sample_id}_terra.npz"


def expected_hest_cell_ids(hest_cfg, sample_id) -> np.ndarray:
    path = hest_terra_path(hest_cfg, sample_id)
    with np.load(path, allow_pickle=True) as data:
        if "cell_ids" not in data:
            raise ValueError(f"{path}: missing cell_ids")
        cell_ids = _as_strings(data["cell_ids"])
    if len(np.unique(cell_ids)) != len(cell_ids):
        raise ValueError(f"{path}: duplicate cell IDs")
    return cell_ids


def hest_source_identity(hest_cfg, sample_id) -> dict:
    row = hest_manifest_row(hest_cfg, sample_id)
    return {
        "manifest": file_identity(hest_cfg["manifest_path"]),
        "wsi": file_identity(row["wsi"]),
        "terra": file_identity(hest_terra_path(hest_cfg, sample_id, row)),
    }


def hest_feature_path(output_dir, sample_id) -> Path:
    return Path(output_dir) / "features" / f"{sample_id}.npz"


def phenocell_file_stem(sample_id) -> str:
    value = str(sample_id).removesuffix(".hdf")
    if not value or Path(value).name != value:
        raise ValueError(f"unsafe PhenoCell sample ID {sample_id!r}")
    return value


def phenocell_feature_path(output_dir, sample_id) -> Path:
    return Path(output_dir) / "features" / f"{phenocell_file_stem(sample_id)}.npz"


def phenocell_cell_ids(locations) -> np.ndarray:
    """Build global IDs despite PhenoCell's tile-local instance IDs."""
    return (
        locations["sample_id"].astype(str)
        + "/"
        + locations["tile_id"].astype(str)
        + "/"
        + locations["cell_id"].astype(str)
    ).to_numpy(dtype=str)


def _manifest_rows(summaries):
    return [
        {
            "sample_id": summary.sample_id,
            "n_cells": summary.n_cells,
            "n_features": summary.n_features,
            "path": summary.path,
            "signature": summary.signature,
            "model_signature": summary.model_signature,
        }
        for summary in summaries
    ]


def _validate_common_artifacts(summaries, baseline_name, dataset_name):
    wrong_baseline = [s.sample_id for s in summaries if s.baseline_name != baseline_name]
    wrong_dataset = [s.sample_id for s in summaries if s.dataset_name != dataset_name]
    if wrong_baseline:
        raise ValueError(f"artifacts belong to another baseline: {wrong_baseline[:5]}")
    if wrong_dataset:
        raise ValueError(f"artifacts belong to another dataset: {wrong_dataset[:5]}")
    model_signatures = {summary.model_signature for summary in summaries}
    if len(model_signatures) != 1:
        raise ValueError("feature artifacts were produced by different model revisions")


def finalize_hest_manifest(cfg, output_dir) -> pd.DataFrame:
    """Require all five full feature files and atomically rebuild their manifest."""
    hest_cfg = cfg["hest"]
    expected_dim = int(cfg["baseline"]["embedding_dim"])
    samples = label_test_fast_samples(hest_cfg)
    summaries = [
        summarize_feature_file(
            hest_feature_path(output_dir, sample_id), expected_dim=expected_dim
        )
        for sample_id in samples
    ]
    for sample_id, summary in zip(samples, summaries):
        if summary.sample_id != sample_id:
            raise ValueError(
                f"{summary.path}: sample is {summary.sample_id}, expected {sample_id}"
            )
        _, cell_ids, _ = load_feature_file(
            summary.path,
            expected_signature=summary.signature,
            expected_dim=expected_dim,
        )
        expected_ids = expected_hest_cell_ids(hest_cfg, sample_id)
        if not np.array_equal(cell_ids, expected_ids):
            raise ValueError(f"{sample_id}: feature rows do not exactly cover TERRA cells")
    _validate_common_artifacts(
        summaries, cfg["baseline"]["name"], "hest/label_test_fast"
    )
    rows = _manifest_rows(summaries)
    write_table(Path(output_dir) / "features" / "manifest.csv", rows)
    return pd.DataFrame(rows)


def finalize_phenocell_manifest(cfg, output_dir) -> pd.DataFrame:
    """Require exact metadata-row coverage and rebuild the PhenoCell manifest."""
    phenocell_cfg = cfg["phenocell"]
    metadata = pd.read_csv(phenocell_cfg["data"]["metadata"])
    metadata["row_id"] = np.arange(len(metadata), dtype=np.int64)
    samples = metadata["sample_id"].astype(str).drop_duplicates().tolist()
    expected_dim = int(cfg["baseline"]["embedding_dim"])
    summaries = []
    for sample_id in samples:
        summary = summarize_feature_file(
            phenocell_feature_path(output_dir, sample_id), expected_dim=expected_dim
        )
        if summary.sample_id != sample_id:
            raise ValueError(
                f"{summary.path}: sample is {summary.sample_id}, expected {sample_id}"
            )
        _, cell_ids, arrays = load_feature_file(
            summary.path,
            expected_signature=summary.signature,
            expected_dim=expected_dim,
            metadata_keys=("row_id", "sample_id", "tile_id", "cell_id"),
        )
        expected = metadata.loc[metadata["sample_id"].astype(str) == sample_id]
        if not np.array_equal(arrays["row_id"], expected["row_id"].to_numpy()):
            raise ValueError(f"{sample_id}: feature row_ids disagree with metadata")
        if not np.all(arrays["sample_id"].astype(str) == sample_id):
            raise ValueError(f"{sample_id}: artifact contains another sample")
        observed_ids = (
            arrays["sample_id"].astype(str)
            + "/"
            + arrays["tile_id"].astype(str)
            + "/"
            + arrays["cell_id"].astype(str)
        )
        if not np.array_equal(cell_ids, observed_ids):
            raise ValueError(f"{sample_id}: global cell IDs are inconsistent")
        summaries.append(summary)
    _validate_common_artifacts(summaries, cfg["baseline"]["name"], "phenocell")
    rows = _manifest_rows(summaries)
    if sum(row["n_cells"] for row in rows) != len(metadata):
        raise ValueError("PhenoCell artifacts do not cover every metadata row")
    write_table(Path(output_dir) / "features" / "manifest.csv", rows)
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class HestFeatureData:
    features: np.ndarray
    labels: np.ndarray
    cell_ids: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray
    pca_explained_variance: float | None


def prepare_hest_features(features, cell_ids, labels_path, adata_path, settings):
    """Align arbitrary embeddings to HEST labels/coordinates and spatially split."""
    features = np.asarray(features, dtype=np.float32)
    cell_ids = _as_strings(cell_ids)
    if features.ndim != 2 or len(features) != len(cell_ids):
        raise ValueError(
            f"features must have shape ({len(cell_ids)}, D), got {features.shape}"
        )
    if not np.isfinite(features).all():
        raise ValueError("HEST features contain non-finite values")
    if len(np.unique(cell_ids)) != len(cell_ids):
        raise ValueError("HEST features contain duplicate cell IDs")

    id_column = settings.get("label_id_column", "cell_id")
    label_column = settings.get("label_column", "group")
    labels_table = pd.read_csv(labels_path)
    if label_column not in labels_table:
        raise KeyError(f"{labels_path}: missing label column {label_column!r}")
    counts = labels_table[label_column].astype(str).value_counts()
    minimum = int(settings.get("min_class_count", 25))
    keep_classes = set(counts.index[counts >= minimum])
    if not keep_classes:
        raise ValueError(f"{labels_path}: no class has at least {minimum} cells")
    label_map = load_labels(
        labels_path,
        keep_classes,
        id_column=id_column,
        label_column=label_column,
    )

    coords, coord_ids = load_coords(adata_path)
    coord_ids = _as_strings(coord_ids)
    if len(np.unique(coord_ids)) != len(coord_ids):
        raise ValueError(f"{adata_path}: duplicate coordinate cell IDs")
    coord_map = {cell_id: row for row, cell_id in enumerate(coord_ids)}
    keep = np.asarray(
        [
            cell_id in coord_map and label_map.get(cell_id) in keep_classes
            for cell_id in cell_ids
        ],
        dtype=bool,
    )
    selected_ids = cell_ids[keep]
    if len(selected_ids) < 2:
        raise ValueError("fewer than two feature rows have labels and coordinates")
    selected_features = features[keep]
    order = np.argsort(selected_ids)
    selected_ids = selected_ids[order]
    selected_features = selected_features[order]
    selected_labels = np.asarray([label_map[cell_id] for cell_id in selected_ids])
    selected_coords = coords[[coord_map[cell_id] for cell_id in selected_ids]]
    train, test = spatial_split(
        selected_ids,
        selected_coords,
        float(settings.get("test_frac", 0.2)),
        settings.get("split_axis", "y"),
    )

    explained = None
    if settings.get("pca", False):
        selected_features, explained = reduce_features(
            selected_features,
            train,
            settings.get("pca_components", 256),
            settings.get("seed", 0),
        )
    return HestFeatureData(
        selected_features,
        selected_labels,
        selected_ids,
        train,
        test,
        explained,
    )


def evaluate_hest_features(data, sample_id, feature_name, method, settings):
    result = evaluate_classification(
        data.features,
        data.labels,
        data.train_indices,
        data.test_indices,
        method,
        settings,
        row_ids=data.cell_ids,
    )
    metrics = {
        "sample_id": str(sample_id),
        "feature_name": str(feature_name),
        **result.metrics,
        "classes": "|".join(result.predictions["classes"].astype(str)),
        "label_column": settings.get("label_column", "group"),
        "split_axis": settings.get("split_axis", "y"),
        "test_frac": float(settings.get("test_frac", 0.2)),
        "pca": bool(settings.get("pca", False)),
        "pca_components": (
            int(data.features.shape[1])
            if data.pca_explained_variance is not None
            else 0
        ),
        "pca_train_explained_variance": data.pca_explained_variance,
    }
    predictions = {
        "cell_ids_test": result.predictions["row_ids_test"],
        "y_true": result.predictions["y_true"],
        "y_pred": result.predictions["y_pred"],
        "y_proba": result.predictions["y_proba"],
        "classes": result.predictions["classes"].astype(str),
    }
    return metrics, predictions


def _cached_metrics(path, signature):
    path = Path(path)
    if not path.exists():
        return None
    try:
        table = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None
    if len(table) != 1 or "signature" not in table:
        return None
    if str(table["signature"].iloc[0]) != str(signature):
        return None
    predictions_path = path.with_name("predictions.npz")
    try:
        with np.load(predictions_path, allow_pickle=False) as predictions:
            if (
                "signature" not in predictions
                or scalar_string(predictions["signature"]) != str(signature)
            ):
                return None
    except (OSError, ValueError, KeyError):
        return None
    return table.iloc[0].to_dict()


def _probe_protocol_identity():
    """Hash evaluator code so metric caches cannot survive protocol changes."""
    package = Path(__file__).resolve().parents[1]
    paths = (
        Path(__file__),
        package / "evaluation" / "classification.py",
        package / "evaluation" / "linear_probe.py",
        package / "evaluation" / "phenocell.py",
        package / "evaluation" / "zero_shot.py",
        package / "helpers" / "splits.py",
    )
    return [file_identity(path, checksum=True) for path in paths]


def _write_predictions(path, signature, predictions):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp.npz")
    try:
        np.savez(temporary, signature=np.asarray(signature), **predictions)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def probe_hest(cfg, output_dir, samples=None):
    """Run linear and kNN probes on every complete fast-HEST artifact."""
    manifest = finalize_hest_manifest(cfg, output_dir)
    expected = manifest["sample_id"].astype(str).tolist()
    if samples is not None:
        requested = [str(sample) for sample in samples]
        unknown = sorted(set(requested) - set(expected))
        if unknown:
            raise ValueError(f"not label_test_fast samples: {unknown}")
        manifest = manifest[manifest["sample_id"].astype(str).isin(requested)]
    settings = cfg["hest"]["probe"]
    methods = resolve_methods(settings)
    results = {method: [] for method in methods}
    for row in manifest.itertuples(index=False):
        sample_id = str(row.sample_id)
        sample_dir = hest_sample_dir(cfg["hest"], sample_id)
        labels_path = sample_dir / f"{sample_id}_granular_cell_labels.csv"
        adata_path = sample_dir / f"{sample_id}.h5ad"
        features, cell_ids, _ = load_feature_file(
            row.path,
            expected_signature=row.signature,
            expected_dim=cfg["baseline"]["embedding_dim"],
        )
        data = prepare_hest_features(
            features, cell_ids, labels_path, adata_path, settings
        )
        for method in methods:
            signature = stable_signature(
                {
                    "feature_signature": str(row.signature),
                    "labels": file_identity(labels_path),
                    "coordinates": file_identity(adata_path),
                    "method": method,
                    "settings": settings,
                    "probe_protocol": _probe_protocol_identity(),
                }
            )
            result_dir = Path(output_dir) / "probes" / sample_id / method
            cached = _cached_metrics(result_dir / "metrics.csv", signature)
            if cached is not None:
                results[method].append(cached)
                print(f"[HEST/{sample_id}/{method}] current")
                continue
            metrics, predictions = evaluate_hest_features(
                data,
                sample_id,
                cfg["baseline"]["name"],
                method,
                settings,
            )
            metrics["signature"] = signature
            write_table(result_dir / "metrics.csv", [metrics])
            _write_predictions(result_dir / "predictions.npz", signature, predictions)
            results[method].append(metrics)
            print(
                f"[HEST/{sample_id}/{method}] macro-F1={metrics['macro_f1']:.4f}, "
                f"balanced accuracy={metrics['balanced_accuracy']:.4f}"
            )

    output = {}
    metric_names = ["macro_f1", "balanced_accuracy", "accuracy", "weighted_f1"]
    for method, rows in results.items():
        table = pd.DataFrame(rows)
        output[method] = table
        if samples is None:
            write_table(Path(output_dir) / f"probe_metrics_{method}.csv", rows)
            summary = table[metric_names].agg(["mean", "std"]).T.reset_index()
            summary = summary.rename(columns={"index": "metric"})
            write_table(Path(output_dir) / f"probe_summary_{method}.csv", summary)
    return output


def _load_phenocell_features(manifest, expected_dim):
    features, records = [], []
    metadata_keys = ("row_id", "sample_id", "label", "split")
    for row in manifest.itertuples(index=False):
        values, _, metadata = load_feature_file(
            row.path,
            expected_signature=row.signature,
            expected_dim=expected_dim,
            metadata_keys=metadata_keys,
        )
        features.append(values)
        records.append(
            pd.DataFrame(
                {
                    "row_id": metadata["row_id"].astype(np.int64),
                    "sample_id": metadata["sample_id"].astype(str),
                    "label": metadata["label"].astype(str),
                    "split": metadata["split"].astype(str),
                }
            )
        )
    features = np.concatenate(features)
    table = pd.concat(records, ignore_index=True)
    order = np.argsort(table["row_id"].to_numpy(), kind="stable")
    features = features[order]
    table = table.iloc[order].reset_index(drop=True)
    expected_rows = np.arange(len(table), dtype=np.int64)
    if not np.array_equal(table["row_id"].to_numpy(), expected_rows):
        raise ValueError("PhenoCell row_ids are not a complete zero-based sequence")
    return features, table


def _format_phenocell_result(data, result, task, feature_name):
    metrics = {
        "task": task,
        "feature_name": feature_name,
        **{f"test_{key}": value for key, value in result.metrics.items() if key in {
            "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "macro_auroc"
        }},
        **{key: value for key, value in result.metrics.items() if key not in {
            "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "macro_auroc"
        }},
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


def probe_phenocell(cfg, output_dir):
    """Probe full PhenoCell artifacts on the official patient-disjoint split."""
    manifest = finalize_phenocell_manifest(cfg, output_dir)
    expected_dim = int(cfg["baseline"]["embedding_dim"])
    features, table = _load_phenocell_features(manifest, expected_dim)
    settings = cfg["phenocell"]["probe"]
    if settings.get("pca", False):
        raise ValueError("SpatialWhisperer projected embeddings must use pca: false")
    validate_official_split(table, cfg["phenocell"]["data"]["split"])
    methods = resolve_methods(settings)
    all_metrics = {method: [] for method in methods}
    for task in settings.get("tasks", ["all", "immune"]):
        data = prepare_phenocell_task(
            features, table, task, settings, pca=False
        )
        for method in methods:
            signature = stable_signature(
                {
                    "feature_signatures": manifest["signature"].astype(str).tolist(),
                    "official_split": file_identity(
                        cfg["phenocell"]["data"]["split"]
                    ),
                    "task": task,
                    "method": method,
                    "settings": settings,
                    "probe_protocol": _probe_protocol_identity(),
                }
            )
            result_dir = Path(output_dir) / "probes" / task / method
            cached = _cached_metrics(result_dir / "metrics.csv", signature)
            if cached is not None:
                all_metrics[method].append(cached)
                print(f"[PhenoCell/{task}/{method}] current")
                continue
            result = evaluate_phenocell_task(data, method, settings)
            metrics, predictions = _format_phenocell_result(
                data, result, task, cfg["baseline"]["name"]
            )
            metrics["signature"] = signature
            write_table(result_dir / "metrics.csv", [metrics])
            _write_predictions(result_dir / "predictions.npz", signature, predictions)
            all_metrics[method].append(metrics)
            print(
                f"[PhenoCell/{task}/{method}] "
                f"macro-F1={metrics['test_macro_f1']:.4f}, "
                f"balanced accuracy={metrics['test_balanced_accuracy']:.4f}"
            )

    output = {}
    for method, rows in all_metrics.items():
        output[method] = pd.DataFrame(rows)
        write_table(Path(output_dir) / f"probe_metrics_{method}.csv", rows)
    return output


def write_combined_summaries(cfg, output_root, hest_results, phenocell_results):
    """Write one compact comparison table per classification method."""
    methods = sorted(set(hest_results) | set(phenocell_results))
    for method in methods:
        rows = []
        hest = hest_results.get(method, pd.DataFrame())
        if not hest.empty:
            rows.append(
                {
                    "baseline": cfg["baseline"]["name"],
                    "dataset": "HEST",
                    "task": "label_test_fast_5_sample_mean",
                    "method": method,
                    "macro_f1": hest["macro_f1"].mean(),
                    "balanced_accuracy": hest["balanced_accuracy"].mean(),
                }
            )
        phenocell = phenocell_results.get(method, pd.DataFrame())
        if not phenocell.empty:
            rows.extend(
                {
                    "baseline": cfg["baseline"]["name"],
                    "dataset": "PhenoCell",
                    "task": row.task,
                    "method": method,
                    "macro_f1": row.test_macro_f1,
                    "balanced_accuracy": row.test_balanced_accuracy,
                }
                for row in phenocell.itertuples(index=False)
            )
        if rows:
            write_table(Path(output_root) / f"summary_{method}.csv", rows)


def probe_all(cfg, output_root, dataset="all"):
    hest_results = phenocell_results = {}
    if dataset in {"all", "hest"}:
        hest_results = probe_hest(
            cfg, Path(output_root) / "hest_label_test_fast"
        )
    if dataset in {"all", "phenocell"}:
        phenocell_results = probe_phenocell(
            cfg, Path(output_root) / "phenocell"
        )
    if dataset == "all":
        manifest_paths = (
            Path(output_root)
            / "hest_label_test_fast"
            / "features"
            / "manifest.csv",
            Path(output_root) / "phenocell" / "features" / "manifest.csv",
        )
        model_signatures = {
            signature
            for path in manifest_paths
            for signature in pd.read_csv(path)["model_signature"].astype(str).unique()
        }
        if len(model_signatures) != 1:
            raise ValueError("HEST and PhenoCell use different model revisions")
        write_combined_summaries(
            cfg, output_root, hest_results, phenocell_results
        )
    return hest_results, phenocell_results
