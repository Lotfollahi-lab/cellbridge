"""Export and probe LoRA-UNI features on held-out HEST and PhenoCell.

The default CLI performs the complete evaluation from raw cell crops::

    python -m terra_multi.lora.evaluate --config CONFIG

Feature files carry a content signature derived from the checkpoint and source
data.  A file is reused only when that signature and its schema both match.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from terra_multi.evaluation.phenocell_data import (
    centered_crop,
    iter_lmdb_samples,
    locate_cells,
    reconstruct_image,
    to_uint8_rgb,
)
from terra_multi.evaluation.probe_phenocell import IMMUNE_CLASSES

FEATURE_ARMS = (
    "uni_original",
    "frozen_head_projected",
    "lora_adapted",
    "lora_projected",
    "gex_penultimate",
)
PCA_ARMS = {"uni_original", "lora_adapted"}
FORMAT_VERSION = "terra_multi.lora.features/1"


def stable_signature(payload):
    """Return a deterministic SHA-256 signature for JSON-compatible metadata."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_identity(path, *, checksum=False):
    """Describe a source file; checkpoints additionally receive a content hash."""
    path = Path(path).resolve()
    stat = path.stat()
    identity = {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if checksum:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        identity["sha256"] = digest.hexdigest()
    return identity


def path_identity(path):
    """Describe either one file or the files immediately inside a directory."""
    path = Path(path)
    if path.is_file():
        return file_identity(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    return [file_identity(item) for item in sorted(path.iterdir()) if item.is_file()]


def feature_signature(dataset, checkpoint_identity, source_identity, settings):
    return stable_signature(
        {
            "format": FORMAT_VERSION,
            "dataset": dataset,
            "checkpoint": checkpoint_identity,
            "source": source_identity,
            "settings": settings,
        }
    )


def _scalar_string(value):
    value = np.asarray(value)
    if value.size != 1:
        raise ValueError("expected a scalar string")
    return str(value.reshape(()).item())


def feature_file_is_current(
    path,
    signature,
    required_keys=None,
    *,
    required_metadata=(),
    expected_cells=None,
    expected_dims=None,
):
    """Return true only for a complete, aligned artifact with this signature."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            if (
                "format_version" not in data
                or _scalar_string(data["format_version"]) != FORMAT_VERSION
            ):
                return False
            if (
                "signature" not in data
                or _scalar_string(data["signature"]) != signature
            ):
                return False
            if "cell_ids" not in data:
                return False
            n_cells = len(data["cell_ids"])
            if expected_cells is not None and n_cells != expected_cells:
                return False
            if not set(required_metadata).issubset(data.files):
                return False
            required_keys = required_keys or FEATURE_ARMS[:-1]
            present = [key for key in required_keys if key in data]
            if not present or any(data[key].ndim != 2 for key in present):
                return False
            if any(len(data[key]) != n_cells for key in present):
                return False
            if expected_dims and any(
                data[key].shape[1] != dimension
                for key, dimension in expected_dims.items()
                if key in data
            ):
                return False
            return set(required_keys).issubset(data.files)
    except (OSError, ValueError, KeyError):
        return False


def write_feature_file(path, cell_ids, features, signature, metadata=None):
    """Atomically write aligned feature matrices and identifying metadata."""
    path = Path(path)
    cell_ids = np.asarray(cell_ids).astype(str)
    if len(np.unique(cell_ids)) != len(cell_ids):
        raise ValueError(f"{path}: duplicate cell_ids")
    arrays = {
        "format_version": np.asarray(FORMAT_VERSION),
        "signature": np.asarray(signature),
        "cell_ids": cell_ids,
    }
    for name, values in features.items():
        values = np.asarray(values)
        if values.ndim != 2 or len(values) != len(cell_ids):
            raise ValueError(
                f"{name}: expected ({len(cell_ids)}, D), got {values.shape}"
            )
        arrays[name] = values.astype(np.float16, copy=False)
    for name, values in (metadata or {}).items():
        values = np.asarray(values)
        if values.ndim and len(values) != len(cell_ids):
            raise ValueError(f"{name}: {len(values)} rows for {len(cell_ids)} cells")
        arrays[name] = values

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp.npz")
    try:
        np.savez(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _select_outputs(outputs, *, adapted):
    prefix = "lora" if adapted else "frozen_head"
    selected = {
        "lora_adapted" if adapted else "uni_original": outputs["image_feature"],
        f"{prefix}_projected": outputs["projected"],
    }
    if adapted and "gex_penultimate" in outputs:
        selected["gex_penultimate"] = outputs["gex_penultimate"]
    return selected


def infer_tensor(model, images, device):
    """Infer adapted and adapter-disabled representations for one image batch."""
    import torch

    images = images.to(device, non_blocking=True)
    device_type = torch.device(device).type
    precision = getattr(model, "precision", torch.float32)
    autocast = torch.autocast(
        device_type=device_type,
        dtype=precision,
        enabled=device_type == "cuda" and precision in {torch.float16, torch.bfloat16},
    )
    with torch.inference_mode(), autocast:
        adapted = _select_outputs(model(images), adapted=True)
        with model.adapters_disabled():
            frozen = _select_outputs(model(images), adapted=False)
    return {
        **adapted,
        **frozen,
    }


def infer_loader(model, loader, device, storage_dir=None):
    """Collect LoRA features from batches following the HESTCellDataset contract."""
    n_cells = len(loader.dataset)
    feature_arrays, cell_parts, sample_parts = {}, [], []
    offset = 0
    for batch in loader:
        if not {"image", "cell_id"}.issubset(batch):
            raise KeyError("evaluation batches require image and cell_id")
        outputs = infer_tensor(model, batch["image"], device)
        for name, values in outputs.items():
            values = values.float().cpu().numpy()
            if name not in feature_arrays:
                shape = (n_cells, values.shape[1])
                if storage_dir is None:
                    feature_arrays[name] = np.empty(shape, dtype=np.float32)
                else:
                    feature_arrays[name] = np.lib.format.open_memmap(
                        Path(storage_dir) / f"{name}.npy",
                        mode="w+",
                        dtype=np.float16,
                        shape=shape,
                    )
            feature_arrays[name][offset : offset + len(values)] = values
        batch_cells = np.asarray(batch["cell_id"]).astype(str)
        cell_parts.append(batch_cells)
        if "sample_id" in batch:
            sample_parts.append(np.asarray(batch["sample_id"]).astype(str))
        offset += len(batch_cells)
    if not cell_parts:
        raise ValueError("cannot export an empty dataset")
    if offset != n_cells:
        raise ValueError(f"loader yielded {offset} rows for a {n_cells}-cell dataset")
    for values in feature_arrays.values():
        flush = getattr(values, "flush", None)
        if flush is not None:
            flush()
    sample_ids = np.concatenate(sample_parts) if sample_parts else None
    return (
        np.concatenate(cell_parts),
        feature_arrays,
        sample_ids,
    )


def _checkpoint_path(cfg, override=None):
    if override:
        return Path(override)
    if cfg.get("checkpoint"):
        return Path(cfg["checkpoint"])
    return Path(cfg["checkpoint_dir"]) / "best.pt"


def load_model(path, device, weights_path=None):
    """Load a LoRA encoder and its validated trainable checkpoint state."""
    from terra_multi.lora.model import load_checkpoint

    loaded = load_checkpoint(path, device=device, weights_path=weights_path)
    model = loaded[0] if isinstance(loaded, tuple) else loaded
    return model.to(device).eval()


def label_test_fast_samples(cfg):
    """Return only configured label_test_fast samples, never train/validation rows."""
    settings = cfg["eval"]
    splits = pd.read_csv(cfg["splits_path"])
    allowed = set(splits.loc[splits["split"] == "label_test", "sample_id"].astype(str))
    fast_path = Path(settings["label_test_fast_path"])
    fast = pd.read_csv(fast_path)
    if "sample_id" not in fast:
        raise ValueError(f"{fast_path} must contain sample_id")
    samples = fast["sample_id"].astype(str).tolist()
    outside = sorted(set(samples) - allowed)
    if outside:
        raise ValueError(f"label_test_fast contains non-label_test samples: {outside}")
    n = int(settings.get("label_test_fast_n", 5))
    if len(samples) < n:
        raise ValueError(f"label_test_fast has {len(samples)} samples, expected {n}")
    return samples[:n]


def _hest_manifest_row(cfg, sample_id):
    root = Path(cfg["data_root"])
    manifest_path = Path(cfg.get("manifest_path", root / "manifest.csv"))
    manifest = pd.read_csv(manifest_path).set_index("id")
    if sample_id not in manifest.index.astype(str):
        raise KeyError(f"{sample_id} is absent from {manifest_path}")
    return manifest_path, manifest.loc[sample_id]


def _hest_sample_dir(cfg, sample_id, row=None):
    if row is None:
        _, row = _hest_manifest_row(cfg, sample_id)
    output_dir = row.get("output_dir")
    if output_dir is None or pd.isna(output_dir):
        return Path(cfg["data_root"]) / sample_id
    return Path(output_dir)


def _hest_source_identity(cfg, sample_id):
    manifest_path, row = _hest_manifest_row(cfg, sample_id)
    sample_dir = _hest_sample_dir(cfg, sample_id, row)
    return {
        "manifest": file_identity(manifest_path),
        "wsi": file_identity(row["wsi"]),
        "terra": file_identity(sample_dir / f"{sample_id}_terra.npz"),
    }


def _hest_dataset(cfg, sample_id, transform):
    from terra_multi.lora.data import build_label_test_dataset

    return build_label_test_dataset(
        cfg,
        transform,
        max_cells_per_sample=None,
        sample_ids=[sample_id],
    )


def export_hest(cfg, model, device, output_dir, checkpoint_identity, overwrite=False):
    """Export all requested arms for the five label_test_fast HEST samples."""
    import torch

    settings = cfg["eval"]
    rows = []
    required_arms = (
        FEATURE_ARMS if model.mode == "contrastive_gex" else FEATURE_ARMS[:-1]
    )
    expected_dims = {
        "uni_original": model.image_dim,
        "frozen_head_projected": model.terra_dim,
        "lora_adapted": model.image_dim,
        "lora_projected": model.terra_dim,
        "gex_penultimate": model.gex_hidden_dim,
    }
    for sample_id in label_test_fast_samples(cfg):
        source = _hest_source_identity(cfg, sample_id)
        with np.load(source["terra"]["path"], allow_pickle=True) as terra:
            expected_cells = len(terra["cell_ids"])
        signature = feature_signature(
            "hest/label_test_fast",
            checkpoint_identity,
            source,
            {
                "sample_id": sample_id,
                "encoder": "uni_v2",
                "crop": cfg.get("crop", {}),
            },
        )
        destination = output_dir / "features" / f"{sample_id}.npz"
        if not overwrite and feature_file_is_current(
            destination,
            signature,
            required_arms,
            expected_cells=expected_cells,
            expected_dims=expected_dims,
        ):
            with np.load(destination, allow_pickle=False) as data:
                count = len(data["cell_ids"])
            print(f"[HEST/{sample_id}] current ({count:,} cells)")
        else:
            dataset = _hest_dataset(cfg, sample_id, model.eval_transforms)
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=int(settings.get("batch_size", 128)),
                shuffle=False,
                num_workers=int(settings.get("num_workers", 8)),
                pin_memory=torch.device(device).type == "cuda",
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                with TemporaryDirectory(dir=destination.parent) as temporary:
                    cell_ids, features, sample_ids = infer_loader(
                        model, loader, device, temporary
                    )
                    if sample_ids is not None and set(sample_ids) != {sample_id}:
                        raise ValueError(
                            f"{sample_id}: dataset returned another sample"
                        )
                    write_feature_file(destination, cell_ids, features, signature)
            finally:
                dataset.close()
            count = len(cell_ids)
            print(f"[HEST/{sample_id}] wrote {count:,} cells -> {destination}")
        rows.append(
            {
                "sample_id": sample_id,
                "n_cells": count,
                "path": str(destination.resolve()),
                "signature": signature,
            }
        )
    _write_manifest(output_dir / "features" / "manifest.csv", rows)
    return pd.DataFrame(rows)


def _write_manifest(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    os.replace(temporary, path)


def _phenocell_signature(cfg, checkpoint_identity, sample_name):
    data = cfg["data"]
    return feature_signature(
        "phenocell",
        checkpoint_identity,
        {
            "lmdb": path_identity(data["lmdb"]),
            "metadata": file_identity(data["metadata"]),
        },
        {
            "sample_id": sample_name,
            "image_shape": data.get("image_shape", [1440, 1920]),
            "tile_size": int(data.get("tile_size", 224)),
            "value_scale": float(data.get("value_scale", 250)),
            "crop_size": int(cfg.get("embedding", {}).get("crop_size", 96)),
            "encoder": "uni_v2",
        },
    )


def _phenocell_features(image, locations, model, device, batch_size, crop_size):
    import torch

    parts = {}
    for start in range(0, len(locations), batch_size):
        rows = locations.iloc[start : start + batch_size]
        images = torch.stack(
            [
                model.eval_transforms(
                    Image.fromarray(
                        centered_crop(image, row.x, row.y, crop_size)
                    ).resize((224, 224), Image.Resampling.BILINEAR)
                )
                for row in rows.itertuples()
            ]
        )
        for name, values in infer_tensor(model, images, device).items():
            parts.setdefault(name, []).append(values.float().cpu().numpy())
    return {name: np.concatenate(values) for name, values in parts.items()}


def phenocell_cell_ids(locations):
    """Build globally unique IDs despite tile-local PhenoCell instance IDs."""
    return (
        locations["sample_id"].astype(str)
        + "/"
        + locations["tile_id"].astype(str)
        + "/"
        + locations["cell_id"].astype(str)
    ).to_numpy(dtype=str)


def export_phenocell(
    cfg, model, device, output_dir, checkpoint_identity, overwrite=False
):
    """Export adapted features from raw PhenoCell crops, never cached UNI."""
    data = cfg["data"]
    metadata = pd.read_csv(data["metadata"])
    metadata["row_id"] = np.arange(len(metadata), dtype=np.int64)
    image_shape = tuple(data.get("image_shape", [1440, 1920]))
    tile_size = int(data.get("tile_size", 224))
    crop_size = int(cfg.get("embedding", {}).get("crop_size", 96))
    batch_size = int(cfg.get("embedding", {}).get("batch_size", 128))
    rows = []
    observed_samples = set()
    required_arms = (
        FEATURE_ARMS if model.mode == "contrastive_gex" else FEATURE_ARMS[:-1]
    )
    expected_dims = {
        "uni_original": model.image_dim,
        "frozen_head_projected": model.terra_dim,
        "lora_adapted": model.image_dim,
        "lora_projected": model.terra_dim,
        "gex_penultimate": model.gex_hidden_dim,
    }
    required_metadata = (
        "row_id",
        "sample_id",
        "tile_id",
        "cell_id",
        "label",
        "class_id",
        "split",
    )

    for sample_name, entries in iter_lmdb_samples(data["lmdb"]):
        observed_samples.add(sample_name)
        sample_metadata = metadata.loc[
            metadata["sample_id"].astype(str) == sample_name
        ].copy()
        if sample_metadata.empty:
            raise ValueError(f"{sample_name}: no metadata rows")
        signature = _phenocell_signature(cfg, checkpoint_identity, sample_name)
        destination = (
            output_dir / "features" / f"{sample_name.removesuffix('.hdf')}.npz"
        )
        if not overwrite and feature_file_is_current(
            destination,
            signature,
            required_arms,
            required_metadata=required_metadata,
            expected_cells=len(sample_metadata),
            expected_dims=expected_dims,
        ):
            with np.load(destination, allow_pickle=False) as cached:
                count = len(cached["cell_ids"])
            print(f"[PhenoCell/{sample_name}] current ({count:,} cells)")
        else:
            locations = locate_cells(sample_metadata, entries, image_shape, tile_size)
            image = to_uint8_rgb(
                reconstruct_image(
                    {index: entry["image"] for index, entry in entries.items()},
                    image_shape,
                    tile_size,
                ),
                data.get("value_scale", 250),
            )
            features = _phenocell_features(
                image, locations, model, device, batch_size, crop_size
            )
            write_feature_file(
                destination,
                phenocell_cell_ids(locations),
                features,
                signature,
                metadata={
                    "row_id": locations["row_id"].to_numpy(np.int64),
                    "sample_id": locations["sample_id"].to_numpy(dtype=str),
                    "tile_id": locations["tile_id"].to_numpy(dtype=str),
                    "cell_id": locations["cell_id"].to_numpy(np.int32),
                    "label": locations["label"].to_numpy(dtype=str),
                    "class_id": locations["class_id"].to_numpy(np.int16),
                    "split": locations["split_id"].to_numpy(dtype=str),
                },
            )
            count = len(locations)
            print(f"[PhenoCell/{sample_name}] wrote {count:,} cells -> {destination}")
        rows.append(
            {
                "sample_id": sample_name,
                "n_cells": count,
                "path": str(destination.resolve()),
                "signature": signature,
            }
        )

    expected = set(metadata["sample_id"].astype(str))
    if observed_samples != expected:
        missing = sorted(expected - observed_samples)
        extra = sorted(observed_samples - expected)
        raise ValueError(
            f"PhenoCell LMDB/metadata sample mismatch: missing={missing}, extra={extra}"
        )
    _write_manifest(output_dir / "features" / "manifest.csv", rows)
    return pd.DataFrame(rows)


def _reduce_features(features, train_indices, components, seed=0):
    if not components:
        return features, None
    components = int(components)
    maximum = min(len(train_indices), features.shape[1])
    if components > maximum:
        raise ValueError(
            f"cannot fit {components} PCA components from {maximum} available"
        )
    pca = PCA(n_components=components, svd_solver="randomized", random_state=seed)
    pca.fit(features[train_indices])
    return pca.transform(features).astype(np.float32), float(
        pca.explained_variance_ratio_.sum()
    )


def _classifier(C, class_weight="balanced", max_iter=2000):
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(C), class_weight=class_weight, max_iter=int(max_iter)
                ),
            ),
        ]
    )


def _load_hest_feature(path, arm, expected_signature):
    with np.load(path, allow_pickle=False) as data:
        if _scalar_string(data["signature"]) != str(expected_signature):
            raise ValueError(f"{path}: signature disagrees with manifest")
        if arm not in data:
            return None, None
        return data[arm].astype(np.float32), data["cell_ids"].astype(str)


def evaluate_hest_sample(features, cell_ids, sample_id, arm, cfg):
    """Run the existing spatial StandardScaler cell-type probe for one sample."""
    from terra_multi.exploration.run_celltype_probe import load_labels
    from terra_multi.helpers.io import load_coords
    from terra_multi.helpers.probe import fit_classifier
    from terra_multi.helpers.splits import spatial_split

    settings = cfg["eval"]
    _, manifest_row = _hest_manifest_row(cfg, sample_id)
    sample_dir = _hest_sample_dir(cfg, sample_id, manifest_row)
    labels_path = sample_dir / f"{sample_id}_granular_cell_labels.csv"
    labels = pd.read_csv(labels_path)
    id_column = settings.get("label_id_column", "cell_id")
    label_column = settings.get("label_column", "group")
    counts = labels[label_column].astype(str).value_counts()
    keep_classes = set(counts.index[counts >= int(settings.get("min_class_count", 25))])
    label_map = load_labels(
        labels_path,
        keep_classes,
        id_column=id_column,
        label_column=label_column,
    )
    coords, coord_ids = load_coords(sample_dir / f"{sample_id}.h5ad")
    if len(np.unique(coord_ids.astype(str))) != len(coord_ids):
        raise ValueError(f"{sample_id}: duplicate coordinate cell IDs")
    coord_map = {cell_id: row for row, cell_id in enumerate(coord_ids.astype(str))}
    keep = np.array(
        [
            cell_id in coord_map and label_map.get(cell_id) in keep_classes
            for cell_id in cell_ids
        ]
    )
    ids = np.asarray(cell_ids)[keep]
    if len(ids) < 2:
        raise ValueError(f"{sample_id}: fewer than two labelled feature rows")
    X = np.asarray(features, dtype=np.float32)[keep]
    order = np.argsort(ids)
    ids, X = ids[order], X[order]
    y = np.asarray([label_map[cell_id] for cell_id in ids])
    aligned_coords = coords[[coord_map[cell_id] for cell_id in ids]]
    train, test = spatial_split(
        ids,
        aligned_coords,
        float(settings.get("test_frac", 0.2)),
        settings.get("split_axis", "y"),
    )
    explained = None
    if arm in PCA_ARMS and settings.get("pca", True):
        X, explained = _reduce_features(
            X, train, settings.get("uni_pca_components", 256)
        )
    model = fit_classifier(
        X[train],
        y[train],
        settings.get("Cs", [0.1, 1, 10]),
        settings.get("class_weight", "balanced"),
    )
    predicted = model.predict(X[test])
    metrics = {
        "sample_id": sample_id,
        "arm": arm,
        "n_features": X.shape[1],
        "n_train": len(train),
        "n_test": len(test),
        "n_classes": len(model.classes_),
        "accuracy": accuracy_score(y[test], predicted),
        "balanced_accuracy": balanced_accuracy_score(y[test], predicted),
        "macro_f1": f1_score(y[test], predicted, average="macro", zero_division=0),
        "weighted_f1": f1_score(
            y[test], predicted, average="weighted", zero_division=0
        ),
        "pca_explained_variance": explained,
    }
    predictions = {
        "cell_ids_test": ids[test],
        "y_true": y[test],
        "y_pred": predicted,
        "classes": model.classes_.astype(str),
    }
    return metrics, predictions


def probe_hest(cfg, output_dir):
    manifest_path = output_dir / "features" / "manifest.csv"
    manifest = pd.read_csv(manifest_path)
    expected = label_test_fast_samples(cfg)
    if manifest["sample_id"].astype(str).tolist() != expected:
        raise ValueError(
            "HEST feature manifest is not the configured label_test_fast set"
        )
    result_rows = []
    for row in manifest.itertuples(index=False):
        sample_dir = _hest_sample_dir(cfg, str(row.sample_id))
        probe_sources = {
            "labels": file_identity(
                sample_dir / f"{row.sample_id}_granular_cell_labels.csv"
            ),
            "adata": file_identity(sample_dir / f"{row.sample_id}.h5ad"),
        }
        for arm in FEATURE_ARMS:
            features, cell_ids = _load_hest_feature(row.path, arm, row.signature)
            if features is None:
                continue
            metrics, predictions = evaluate_hest_sample(
                features, cell_ids, str(row.sample_id), arm, cfg
            )
            signature = stable_signature(
                {
                    "features": str(row.signature),
                    "sources": probe_sources,
                    "arm": arm,
                    "settings": cfg["eval"],
                }
            )
            metrics["signature"] = signature
            result_dir = output_dir / "probes" / str(row.sample_id) / arm
            result_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([metrics]).to_csv(result_dir / "metrics.csv", index=False)
            np.savez(result_dir / "predictions.npz", signature=signature, **predictions)
            result_rows.append(metrics)
            print(
                f"[HEST/{row.sample_id}/{arm}] macro-F1={metrics['macro_f1']:.4f}, "
                f"balanced accuracy={metrics['balanced_accuracy']:.4f}"
            )
    results = pd.DataFrame(result_rows)
    results.to_csv(output_dir / "probe_metrics.csv", index=False)
    metric_names = ["macro_f1", "balanced_accuracy", "accuracy", "weighted_f1"]
    summary = results.groupby("arm")[metric_names].agg(["mean", "std"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    summary.reindex([arm for arm in FEATURE_ARMS if arm in summary.index]).to_csv(
        output_dir / "probe_summary.csv"
    )
    return results, summary


def load_phenocell_arm(manifest, arm):
    features, records = [], []
    for row in manifest.itertuples(index=False):
        with np.load(row.path, allow_pickle=False) as data:
            if _scalar_string(data["signature"]) != str(row.signature):
                raise ValueError(f"{row.path}: signature disagrees with manifest")
            if arm not in data:
                return None, None
            features.append(data[arm].astype(np.float32))
            records.append(
                pd.DataFrame(
                    {
                        "row_id": data["row_id"],
                        "sample_id": data["sample_id"].astype(str),
                        "label": data["label"].astype(str),
                        "split": data["split"].astype(str),
                    }
                )
            )
    return np.concatenate(features), pd.concat(records, ignore_index=True)


def validate_official_split(table, split_path):
    """Require complete sample coverage and exact agreement with the official split."""
    official = pd.read_csv(split_path).rename(
        columns={"sample_name": "sample_id", "train_test_val_split": "official_split"}
    )
    official["sample_id"] = official["sample_id"].astype(str)
    official["official_split"] = official["official_split"].replace({"valid": "val"})
    if official["sample_id"].duplicated().any():
        raise ValueError("official PhenoCell split contains duplicate samples")
    observed = table[["sample_id", "split"]].drop_duplicates().copy()
    observed["split"] = observed["split"].replace({"valid": "val"})
    if observed["sample_id"].duplicated().any():
        raise ValueError("a PhenoCell sample has multiple observed splits")
    if set(observed["sample_id"]) != set(official["sample_id"]):
        raise ValueError(
            "PhenoCell features do not cover every official sample exactly"
        )
    merged = official.merge(observed, on="sample_id", validate="one_to_one")
    if not (merged["official_split"] == merged["split"]).all():
        raise ValueError("PhenoCell feature metadata disagrees with the official split")


def evaluate_phenocell_arm(features, table, task, arm, settings):
    selected = np.ones(len(table), dtype=bool)
    if task == "immune":
        selected = table["label"].isin(IMMUNE_CLASSES).to_numpy()
    split = table["split"].replace({"valid": "val"}).to_numpy()
    train = np.flatnonzero(selected & (split == "train"))
    val = np.flatnonzero(selected & (split == "val"))
    test = np.flatnonzero(selected & (split == "test"))
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError(f"{task}: an official PhenoCell split is empty")

    explained = None
    if arm in PCA_ARMS and settings.get("uni_pca_components", 256):
        features, explained = _reduce_features(
            features,
            train,
            settings.get("uni_pca_components", 256),
            settings.get("seed", 0),
        )
    y = table["label"].to_numpy()
    best_C, best_score = None, -np.inf
    for C in settings.get("Cs", [0.1, 1, 10]):
        model = _classifier(
            C, settings.get("class_weight", "balanced"), settings.get("max_iter", 2000)
        )
        model.fit(features[train], y[train])
        score = f1_score(
            y[val], model.predict(features[val]), average="macro", zero_division=0
        )
        if score > best_score:
            best_C, best_score = float(C), float(score)
    model = _classifier(
        best_C, settings.get("class_weight", "balanced"), settings.get("max_iter", 2000)
    )
    fit = np.concatenate([train, val])
    model.fit(features[fit], y[fit])
    predicted = model.predict(features[test])
    metrics = {
        "task": task,
        "arm": arm,
        "n_features": features.shape[1],
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "n_classes": len(model.classes_),
        "best_C": best_C,
        "val_macro_f1": best_score,
        "test_accuracy": accuracy_score(y[test], predicted),
        "test_balanced_accuracy": balanced_accuracy_score(y[test], predicted),
        "test_macro_f1": f1_score(y[test], predicted, average="macro", zero_division=0),
        "test_weighted_f1": f1_score(
            y[test], predicted, average="weighted", zero_division=0
        ),
        "pca_explained_variance": explained,
    }
    predictions = {
        "row_id": table.iloc[test]["row_id"].to_numpy(),
        "y_true": y[test],
        "y_pred": predicted,
        "classes": model.classes_.astype(str),
    }
    return metrics, predictions


def probe_phenocell(cfg, output_dir):
    manifest = pd.read_csv(output_dir / "features" / "manifest.csv")
    all_metrics = []
    for arm in FEATURE_ARMS:
        features, table = load_phenocell_arm(manifest, arm)
        if features is None:
            continue
        validate_official_split(table, cfg["data"]["split"])
        for task in cfg.get("probe", {}).get("tasks", ["all", "immune"]):
            metrics, predictions = evaluate_phenocell_arm(
                features, table, task, arm, cfg.get("probe", {})
            )
            signature = stable_signature(
                {
                    "features": manifest["signature"].astype(str).tolist(),
                    "official_split": file_identity(cfg["data"]["split"]),
                    "arm": arm,
                    "task": task,
                    "settings": cfg.get("probe", {}),
                }
            )
            metrics["signature"] = signature
            result_dir = output_dir / "probes" / task / arm
            result_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([metrics]).to_csv(result_dir / "metrics.csv", index=False)
            np.savez(result_dir / "predictions.npz", signature=signature, **predictions)
            all_metrics.append(metrics)
            print(
                f"[PhenoCell/{task}/{arm}] macro-F1={metrics['test_macro_f1']:.4f}, "
                f"balanced accuracy={metrics['test_balanced_accuracy']:.4f}"
            )
    result = pd.DataFrame(all_metrics)
    result.to_csv(output_dir / "probe_summary.csv", index=False)
    return result


def write_evaluation_summary(
    cfg, output_dir, hest_metrics=None, phenocell_metrics=None
):
    """Write the headline metrics used to compare LoRA runs and ablations."""
    run_name = cfg.get("run_name", Path(cfg["output_dir"]).name)
    rows = []
    if hest_metrics is not None and not hest_metrics.empty:
        means = hest_metrics.groupby("arm")[["macro_f1", "balanced_accuracy"]].mean()
        rows.extend(
            {
                "run_name": run_name,
                "dataset": "HEST",
                "task": "label_test_fast_5_sample_mean",
                "arm": arm,
                "macro_f1": values["macro_f1"],
                "balanced_accuracy": values["balanced_accuracy"],
            }
            for arm, values in means.iterrows()
        )
    if phenocell_metrics is not None and not phenocell_metrics.empty:
        rows.extend(
            {
                "run_name": run_name,
                "dataset": "PhenoCell",
                "task": row.task,
                "arm": row.arm,
                "macro_f1": row.test_macro_f1,
                "balanced_accuracy": row.test_balanced_accuracy,
            }
            for row in phenocell_metrics.itertuples(index=False)
        )
    if not rows:
        return None
    path = Path(output_dir) / "summary.csv"
    _write_manifest(path, rows)
    print(f"wrote evaluation summary -> {path}")
    return pd.DataFrame(rows)


def load_phenocell_config(cfg):
    path = cfg.get("phenocell_config")
    if not path:
        raise ValueError("LoRA config requires top-level phenocell_config")
    with open(path) as handle:
        return yaml.safe_load(handle)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-hest", action="store_true")
    parser.add_argument("--skip-phenocell", action="store_true")
    parser.add_argument("--skip-probes", action="store_true")
    args = parser.parse_args(argv)

    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)
    checkpoint_path = _checkpoint_path(cfg, args.checkpoint)
    weights_path = cfg.get("model", {}).get("weights_path")
    checkpoint_identity = {
        "checkpoint": file_identity(checkpoint_path, checksum=True),
        "encoder": "uni_v2",
        "encoder_weights": (
            file_identity(weights_path, checksum=True)
            if weights_path
            else "trident-default"
        ),
    }
    evaluation_root = Path(
        args.output_dir
        or cfg.get("eval", {}).get("output_dir", Path(cfg["output_dir"]) / "evaluation")
    )

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device, weights_path)
    hest_metrics = phenocell_metrics = None
    if not args.skip_hest:
        hest_dir = evaluation_root / "hest_label_test_fast"
        export_hest(cfg, model, device, hest_dir, checkpoint_identity, args.overwrite)
        if not args.skip_probes:
            hest_metrics, _ = probe_hest(cfg, hest_dir)
    if not args.skip_phenocell:
        phenocell_cfg = load_phenocell_config(cfg)
        phenocell_dir = evaluation_root / "phenocell"
        export_phenocell(
            phenocell_cfg,
            model,
            device,
            phenocell_dir,
            checkpoint_identity,
            args.overwrite,
        )
        if not args.skip_probes:
            phenocell_metrics = probe_phenocell(phenocell_cfg, phenocell_dir)
    if not args.skip_probes:
        write_evaluation_summary(cfg, evaluation_root, hest_metrics, phenocell_metrics)


if __name__ == "__main__":
    main()
