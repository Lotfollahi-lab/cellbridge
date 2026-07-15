"""HEST cell-type evaluation adapter."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from terra_multi.evaluation.classification import (
    evaluate_classification,
    method_directory,
)
from terra_multi.evaluation.linear_probe import reduce_features
from terra_multi.helpers.io import align_by_cell_id, load_coords, load_embeddings
from terra_multi.helpers.splits import spatial_split

MODELS = ("terra", "terra_neighbor", "uni", "concat", "neighbor_concat")
UNI_MODELS = {"uni", "concat", "neighbor_concat"}
NEIGHBOR_MODELS = {"terra_neighbor", "neighbor_concat"}


@dataclass(frozen=True)
class HestData:
    features: np.ndarray
    labels: np.ndarray
    cell_ids: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray
    terra_key: str
    pca_explained_variance: float | None


def load_labels(path, keep_classes=None, id_column="cell_id", label_column="cell_type"):
    """Load unique cell labels, optionally restricted to selected classes."""
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        table = pd.read_csv(path)
    elif suffix in {".parquet", ".pq"}:
        table = pd.read_parquet(path)
    else:
        raise ValueError(f"unsupported labels file type: {suffix}")

    missing = [column for column in (id_column, label_column) if column not in table]
    if missing:
        raise KeyError(f"{path}: missing label column(s): {missing}")
    if table[[id_column, label_column]].isna().any().any():
        raise ValueError(f"{path}: null cell IDs or labels")

    cell_ids = table[id_column].astype(str)
    duplicate_ids = cell_ids[cell_ids.duplicated(keep=False)].unique()
    if len(duplicate_ids):
        raise ValueError(
            f"{path}: {len(duplicate_ids)} duplicate cell ID(s), "
            f"including {duplicate_ids[:5].tolist()}"
        )

    labels = table[label_column].astype(str)
    if keep_classes is not None:
        keep = labels.isin({str(value) for value in keep_classes})
        cell_ids, labels = cell_ids[keep], labels[keep]
    return dict(zip(cell_ids, labels))


def _aligned_inputs(cfg, model):
    terra_key = cfg["terra_neighbor_key" if model in NEIGHBOR_MODELS else "terra_key"]
    terra, terra_ids = load_embeddings(cfg["terra_embeddings"], key=terra_key)
    uni, uni_ids = load_embeddings(cfg["uni_embeddings"], key=cfg["uni_key"])
    coords, coord_ids = load_coords(cfg["adata_path"])
    aligned, cell_ids = align_by_cell_id(
        [(terra, terra_ids), (uni, uni_ids), (coords, coord_ids)]
    )
    return (*aligned, cell_ids, terra_key)


def _labelled_rows(cfg, terra, uni, coords, cell_ids):
    label_map = load_labels(
        cfg["labels_path"],
        cfg.get("keep_classes"),
        id_column=cfg.get("label_id_column", "cell_id"),
        label_column=cfg.get("label_column", "cell_type"),
    )
    keep = np.array([cell_id in label_map for cell_id in cell_ids])
    labelled_ids = cell_ids[keep]
    if len(labelled_ids) < 2:
        raise ValueError(f"only {len(labelled_ids)} embedding cell(s) have labels")
    labels = np.asarray([label_map[cell_id] for cell_id in labelled_ids])
    return terra[keep], uni[keep], coords[keep], labelled_ids, labels


def prepare_hest_data(cfg, model):
    """Load, align, split, and optionally PCA-reduce one HEST feature arm."""
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {MODELS}")
    terra, uni, coords, cell_ids, terra_key = _aligned_inputs(cfg, model)
    terra, uni, coords, cell_ids, labels = _labelled_rows(
        cfg, terra, uni, coords, cell_ids
    )
    train, test = spatial_split(
        cell_ids, coords, cfg["test_frac"], cfg["split_axis"]
    )

    explained = None
    use_pca = bool(cfg.get("pca", False)) and model in UNI_MODELS
    if use_pca:
        uni, explained = reduce_features(
            uni,
            train,
            cfg.get("uni_pca_components", 256),
            cfg.get("seed", 0),
        )

    if model in {"terra", "terra_neighbor"}:
        features = terra
    elif model == "uni":
        features = uni
    else:
        features = np.concatenate([terra, uni], axis=1)
    return HestData(
        features,
        labels,
        cell_ids,
        train,
        test,
        terra_key,
        explained,
    )


def evaluate_hest_data(data, sample_id, model, method, settings):
    """Evaluate one prepared HEST embedding arm."""
    result = evaluate_classification(
        data.features,
        data.labels,
        data.train_indices,
        data.test_indices,
        method,
        settings,
        row_ids=data.cell_ids,
    )
    common = dict(result.metrics)
    common.pop("method")
    metrics = {
        "sample_id": sample_id,
        "model": model,
        "terra_key": data.terra_key if model != "uni" else None,
        "label_column": settings.get("label_column", "cell_type"),
        **common,
        "classes": "|".join(result.predictions["classes"].astype(str)),
        "split_axis": settings["split_axis"],
        "test_frac": float(settings["test_frac"]),
        "uni_pca": bool(settings.get("pca", False) and model in UNI_MODELS),
        "uni_pca_components": (
            int(data.features.shape[1])
            if model == "uni" and data.pca_explained_variance is not None
            else int(settings.get("uni_pca_components", 256))
            if model in {"concat", "neighbor_concat"}
            and data.pca_explained_variance is not None
            else 0
        ),
        "uni_pca_train_explained_variance": data.pca_explained_variance,
    }
    predictions = dict(result.predictions)
    predictions["cell_ids_test"] = predictions.pop("row_ids_test")
    return metrics, predictions


def result_directory(output_dir, model, method):
    """Keep historical linear paths while placing kNN in a named namespace."""
    return method_directory(output_dir, method) / model


def _evaluate_and_write(data, cfg, model, method):
    metrics, predictions = evaluate_hest_data(
        data, cfg.get("sample_id"), model, method, cfg
    )
    output_dir = result_directory(cfg["output_dir"], model, method)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)
    np.savez(output_dir / "predictions.npz", **predictions)
    print(
        f"[{model}/{method}] macro-F1={metrics['macro_f1']:.4f}, "
        f"balanced accuracy={metrics['balanced_accuracy']:.4f}"
    )
    return metrics, predictions


def run_hest_evaluations(cfg, model, methods):
    """Prepare one feature arm once, then run every requested evaluator."""
    data = prepare_hest_data(cfg, model)
    return {
        method: _evaluate_and_write(data, cfg, model, method) for method in methods
    }


def run_hest_evaluation(cfg, model, method):
    """Compatibility helper for evaluating one HEST model/method pair."""
    return run_hest_evaluations(cfg, model, [method])[method]
