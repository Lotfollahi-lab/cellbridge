"""Run frozen HistoPLUS on Cellbridge's HEST and OOD evaluation sets.

HistoPLUS is an end-to-end detector, segmentor, and classifier. Unlike the
Cellbridge representation probes, this adapter never uses reference masks to
choose model inputs. Reference cells are used only after inference for geometric
matching and scoring.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from PIL import Image

from cellbridge.baselines.histoplus_data import (
    hest_manifest_row,
    label_test_fast_samples,
    validate_official_split,
)
from cellbridge.baselines.histoplus_metrics import aggregate_metrics
from cellbridge.config import load_config as load_yaml_config
from cellbridge.evaluation.artifacts import git_identity, write_json, write_table
from cellbridge.evaluation.ood_data import (
    iter_lmdb_samples,
    locate_cells,
    reconstruct_image,
    tile_geometries,
    tile_index,
    to_uint8_rgb,
)

BASELINE_NAME = "histoplus"
PREDICTION_COLUMNS = (
    "sample_id",
    "prediction_id",
    "x",
    "y",
    "cell_type",
    "confidence",
    "contour",
)

# HistoPLUS and the two fine-grained reference datasets share this reviewed
# evaluation vocabulary. Classes without a defensible correspondence are None.
HISTOPLUS_FINE = {
    "Cancer cell": "Neoplastic",
    "Lymphocytes": "Lymphocyte",
    "Fibroblasts": "Stroma",
    "Plasmocytes": "Plasma",
    "Eosinophils": "Granulocyte",
    "Neutrophils": "Granulocyte",
    "Macrophages": "Macrophage",
    "Muscle Cell": "Smooth muscle",
    "Endothelial Cell": "Vascular",
    "Red blood cell": None,
    "Epithelial": "Epithelial",
    "Apoptotic Body": None,
    "Mitotic Figures": None,
    "Minor Stromal Cell": "Stroma",
}

PHENOCELL_REFERENCE = {
    "Tumor cells": "Neoplastic",
    "T cells": "Lymphocyte",
    "B cells": "Lymphocyte",
    "NK cells": "Lymphocyte",
    "Plasma cells": "Plasma",
    "Granulocytes": "Granulocyte",
    "Macrophages/Monocytes": "Macrophage",
    "Stroma": "Stroma",
    "Smooth muscle": "Smooth muscle",
    "Vasculature/Lymphatics": "Vascular",
    "Adipocytes": None,
    "Dendritic cells": None,
    "Nerves": None,
    "Other cells": None,
}

PANNuke_REFERENCE = {
    "Neoplastic cells": "Neoplastic",
    "Inflammatory": "Inflammatory",
    "Connective/Soft tissue cells": "Connective/Soft tissue",
    "Dead Cells": "Dead",
    "Epithelial": "Epithelial",
}

HISTOPLUS_PANNUKE = {
    "Cancer cell": "Neoplastic",
    "Lymphocytes": "Inflammatory",
    "Fibroblasts": "Connective/Soft tissue",
    "Plasmocytes": "Inflammatory",
    "Eosinophils": "Inflammatory",
    "Neutrophils": "Inflammatory",
    "Macrophages": "Inflammatory",
    "Muscle Cell": "Connective/Soft tissue",
    "Endothelial Cell": "Connective/Soft tissue",
    "Red blood cell": None,
    "Epithelial": "Epithelial",
    "Apoptotic Body": "Dead",
    "Mitotic Figures": None,
    "Minor Stromal Cell": "Connective/Soft tissue",
}

HEST_IDENTITY = {
    "neoplastic cell": "Neoplastic",
    "B cell": "Lymphocyte",
    "memory B cell": "Lymphocyte",
    "naive B cell": "Lymphocyte",
    "T cell": "Lymphocyte",
    "CD4-positive, alpha-beta memory T cell": "Lymphocyte",
    "CD8-positive, alpha-beta memory T cell": "Lymphocyte",
    "effector CD4-positive, alpha-beta T cell": "Lymphocyte",
    "effector CD8-positive, alpha-beta T cell": "Lymphocyte",
    "naive thymus-derived CD4-positive, alpha-beta T cell": "Lymphocyte",
    "regulatory T cell": "Lymphocyte",
    "mature natural killer T cell": "Lymphocyte",
    "type I natural killer T cell": "Lymphocyte",
    "innate lymphoid cell": "Lymphocyte",
    "lymphocyte": "Lymphocyte",
    "plasma cell": "Plasma",
    "classical monocyte": "Macrophage",
    "intermediate monocyte": "Macrophage",
    "non-classical monocyte": "Macrophage",
    "macrophage": "Macrophage",
    "neutrophil": "Granulocyte",
    "fibroblast": "Stroma",
    "stromal cell": "Stroma",
    "smooth muscle cell": "Smooth muscle",
    "pericyte": "Smooth muscle",
    "perivascular cell": "Smooth muscle",
    "endothelial cell": "Vascular",
    "endothelial cell of lymphatic vessel": "Vascular",
    "ciliated epithelial cell": "Epithelial",
    "epithelial cell": "Epithelial",
    "fallopian tube epithelial cell": "Epithelial",
    "luminal epithelial cell of mammary gland": "Epithelial",
    "mammary gland epithelial cell": "Epithelial",
    "myoepithelial cell": "Epithelial",
    "pancreatic acinar cell": "Epithelial",
    "pancreatic ductal cell": "Epithelial",
    "pancreatic endocrine cell": "Epithelial",
}


def load_config(path) -> dict:
    cfg = load_yaml_config(path)
    validate_config(cfg)
    return cfg


def validate_config(cfg) -> None:
    required = {
        "baseline",
        "model",
        "output_dir",
        "evaluation",
        "hest",
        "phenocell",
        "pannuke",
    }
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"config is missing sections {sorted(missing)}")
    if cfg["baseline"].get("name") != BASELINE_NAME:
        raise ValueError(f"baseline.name must be {BASELINE_NAME!r}")
    size = int(cfg["baseline"].get("inference_image_size", 0))
    overlap = int(cfg["baseline"].get("inference_overlap", 0))
    if size != 784 or overlap != 64:
        raise ValueError(
            "the released HistoPLUS inference contract is size 784, overlap 64"
        )
    if size <= 2 * overlap:
        raise ValueError("inference overlap leaves no tile core")
    repository = Path(cfg["model"]["repository"])
    if not (repository / "pyproject.toml").is_file():
        raise FileNotFoundError(repository / "pyproject.toml")
    if float(cfg["hest"]["model_mpp"]) != 0.25:
        raise ValueError("HEST must use the released 40x (0.25 MPP) checkpoint")
    if float(cfg["phenocell"]["model_mpp"]) != 0.5:
        raise ValueError("PhenoCell must use the released 20x (0.5 MPP) checkpoint")
    if float(cfg["pannuke"]["model_mpp"]) != 0.25:
        raise ValueError("PanNuke must use the released 40x (0.25 MPP) checkpoint")
    if float(cfg["evaluation"].get("match_radius_um", 0)) <= 0:
        raise ValueError("evaluation.match_radius_um must be positive")


def output_root(cfg, override=None) -> Path:
    return Path(override or cfg["output_dir"])


def prediction_path(root, dataset, sample_id) -> Path:
    stem = str(sample_id).removesuffix(".hdf")
    if not stem or Path(stem).name != stem:
        raise ValueError(f"unsafe sample ID {sample_id!r}")
    return Path(root) / dataset / "predictions" / f"{stem}.csv.gz"


def _write_predictions(path: Path, table: pd.DataFrame) -> None:
    missing = set(PREDICTION_COLUMNS) - set(table)
    if missing:
        raise ValueError(f"prediction table is missing {sorted(missing)}")
    table = table.loc[:, PREDICTION_COLUMNS]
    if table["prediction_id"].duplicated().any():
        raise ValueError("prediction IDs must be unique within a sample")
    # A tile with no detections yields empty object-dtype columns, which
    # np.isfinite rejects outright. Coerce so a genuinely empty prediction table
    # is written as such -- those cells then count as missed at scoring time,
    # which is the honest treatment -- while NaN or inf still fails loudly.
    measurements = table[["x", "y", "confidence"]].to_numpy(dtype=float)
    if not np.isfinite(measurements).all():
        raise ValueError("prediction coordinates and confidences must be finite")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp.csv.gz")
    try:
        table.to_csv(temporary, index=False, compression="gzip")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_predictions(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    missing = set(PREDICTION_COLUMNS) - set(table)
    if missing:
        raise ValueError(f"{path}: missing prediction columns {sorted(missing)}")
    return table


def _contour_json(points) -> str:
    return json.dumps(
        np.asarray(points, dtype=float).round(3).tolist(), separators=(",", ":")
    )


def _load_segmentor(cfg, model_mpp):
    from histoplus.helpers.segmentor import CellViTSegmentor

    return CellViTSegmentor.from_histoplus(
        mixed_precision=bool(cfg["baseline"].get("mixed_precision", True)),
        mpp=float(model_mpp),
        inference_image_size=int(cfg["baseline"]["inference_image_size"]),
    )


def _batch_tile_predictions(segmentor, tiles, batch_size):
    import torch

    postprocess = segmentor.get_postprocess_fn()
    for start in range(0, len(tiles), int(batch_size)):
        batch = tiles[start : start + int(batch_size)]
        tensors = torch.stack([segmentor.transform(tile["image"]) for tile in batch])
        with torch.inference_mode():
            raw = segmentor.forward(tensors)
        arrays = {name: value.detach().cpu().numpy() for name, value in raw.items()}
        yield from zip(batch, postprocess(arrays), strict=True)


def _image_tiles(image: Image.Image, scale, inference_size, overlap):
    source_width, source_height = image.size
    model_size = (
        max(1, round(source_width * scale)),
        max(1, round(source_height * scale)),
    )
    if model_size != image.size:
        image = image.resize(model_size, Image.Resampling.BILINEAR)
    width, height = image.size
    core = inference_size - 2 * overlap
    tiles = []
    for core_y in range(0, height, core):
        for core_x in range(0, width, core):
            x0, y0 = core_x - overlap, core_y - overlap
            tile = Image.new("RGB", (inference_size, inference_size), (255, 255, 255))
            source_box = (
                max(0, x0),
                max(0, y0),
                min(width, x0 + inference_size),
                min(height, y0 + inference_size),
            )
            crop = image.crop(source_box)
            tile.paste(crop, (source_box[0] - x0, source_box[1] - y0))
            tiles.append(
                {
                    "image": tile,
                    "x0": x0,
                    "y0": y0,
                    "core_x": core_x,
                    "core_y": core_y,
                    "core_x1": min(core_x + core, width),
                    "core_y1": min(core_y + core, height),
                    "width": width,
                    "height": height,
                }
            )
    return tiles


def infer_image(
    image,
    segmentor,
    *,
    source_mpp,
    model_mpp,
    batch_size,
    inference_size=784,
    overlap=64,
):
    """Infer one reconstructed field and return source-pixel predictions."""
    scale = float(source_mpp) / float(model_mpp)
    tiles = _image_tiles(image, scale, int(inference_size), int(overlap))
    rows = []
    for tile, prediction in _batch_tile_predictions(segmentor, tiles, batch_size):
        offset = np.array([tile["x0"], tile["y0"]], dtype=float)
        for centroid, contour, cell_type, confidence in zip(
            prediction.centroids,
            prediction.contours,
            prediction.cell_types,
            prediction.cell_type_probabilities,
            strict=True,
        ):
            centroid_model = np.asarray(centroid, dtype=float) + offset
            if not (
                tile["core_x"] <= centroid_model[0] < tile["core_x1"]
                and tile["core_y"] <= centroid_model[1] < tile["core_y1"]
            ):
                continue
            centroid_source = centroid_model / scale
            contour_source = (np.asarray(contour, dtype=float) + offset) / scale
            rows.append(
                {
                    "x": centroid_source[0],
                    "y": centroid_source[1],
                    "cell_type": cell_type,
                    "confidence": confidence,
                    "contour": _contour_json(contour_source),
                }
            )
    return pd.DataFrame(rows, columns=("x", "y", "cell_type", "confidence", "contour"))


def _official_test_samples(split_path):
    split = pd.read_csv(split_path).rename(
        columns={"sample_name": "sample_id", "train_test_val_split": "split"}
    )
    missing = {"sample_id", "split"} - set(split)
    if missing:
        raise ValueError(f"{split_path}: missing columns {sorted(missing)}")
    split["sample_id"] = split["sample_id"].astype(str)
    return split.loc[
        split["split"].replace({"valid": "val"}) == "test", "sample_id"
    ].tolist()


def infer_ood(
    cfg, root, dataset, *, shard_index=0, num_shards=1, samples=None, overwrite=False
):
    section = cfg[dataset]
    data = section["data"]
    metadata = pd.read_csv(data["metadata"])
    observed_split = metadata[["sample_id", "split_id"]].rename(
        columns={"split_id": "split"}
    )
    validate_official_split(
        observed_split,
        data["split"],
        require_full_coverage=not bool(data.get("allow_unlabeled_samples", False)),
    )
    test_samples = _official_test_samples(data["split"])
    if samples is not None:
        unknown = sorted(set(samples) - set(test_samples))
        if unknown:
            raise ValueError(
                f"{dataset}: requested samples are not official test samples: {unknown}"
            )
        test_samples = [sample for sample in test_samples if sample in set(samples)]
    if int(num_shards) < 1 or not 0 <= int(shard_index) < int(num_shards):
        raise ValueError("shard index must satisfy 0 <= index < num_shards")
    wanted = set(test_samples[int(shard_index) :: int(num_shards)])
    segmentor = _load_segmentor(cfg, section["model_mpp"])
    image_shape = tuple(data["image_shape"])
    tile_size = int(data["tile_size"])
    processed = set()
    for sample_id, entries in iter_lmdb_samples(data["lmdb"]):
        if sample_id not in wanted:
            continue
        destination = prediction_path(root, dataset, sample_id)
        if destination.is_file() and not overwrite:
            print(f"[{dataset}/{sample_id}] current")
            processed.add(sample_id)
            continue
        array = reconstruct_image(
            {index: entry["image"] for index, entry in entries.items()},
            image_shape,
            tile_size,
        )
        image = Image.fromarray(to_uint8_rgb(array, data.get("value_scale", 1)))
        predictions = infer_image(
            image,
            segmentor,
            source_mpp=float(data["source_mpp"]),
            model_mpp=float(section["model_mpp"]),
            batch_size=int(section["batch_size"]),
            inference_size=int(cfg["baseline"]["inference_image_size"]),
            overlap=int(cfg["baseline"]["inference_overlap"]),
        )
        predictions.insert(
            0, "prediction_id", [f"{sample_id}/{i}" for i in range(len(predictions))]
        )
        predictions.insert(0, "sample_id", sample_id)
        _write_predictions(destination, predictions)
        processed.add(sample_id)
        print(
            f"[{dataset}/{sample_id}] {len(predictions):,} detections -> {destination}"
        )
        if processed == wanted:
            break
    missing = sorted(wanted - processed)
    if missing:
        raise ValueError(f"{dataset}: samples absent from LMDB: {missing[:10]}")


def infer_hest(cfg, root, *, samples=None, overwrite=False):
    from histoplus.extract import extract
    from histoplus.helpers.tissue_detection import detect_tissue_on_wsi
    from openslide import OpenSlide
    from openslide.deepzoom import DeepZoomGenerator

    hest = cfg["hest"]
    expected = label_test_fast_samples(hest)
    wanted = expected if samples is None else samples
    unknown = sorted(set(wanted) - set(expected))
    if unknown:
        raise ValueError(f"not in fast HEST: {unknown}")
    segmentor = _load_segmentor(cfg, hest["model_mpp"])
    for sample_id in wanted:
        destination = prediction_path(root, "hest", sample_id)
        if destination.is_file() and not overwrite:
            print(f"[hest/{sample_id}] current")
            continue
        row = hest_manifest_row(hest, sample_id)
        slide = OpenSlide(str(row["wsi"]))
        try:
            coords, level = detect_tissue_on_wsi(
                slide,
                matter_threshold=float(hest.get("matter_threshold", 0.2)),
                target_mpp=float(hest.get("tissue_mpp", 8.0)),
                base_mpp=float(hest["model_mpp"]),
                tile_size_at_base_mpp=int(hest.get("tile_size", 224)),
            )
            result = extract(
                slide,
                coords,
                level,
                segmentor,
                tile_size=int(hest.get("tile_size", 224)),
                n_workers=int(hest.get("num_workers", 8)),
                batch_size=int(hest.get("batch_size", 4)),
                buffer_batch_size=int(hest.get("buffer_batch_size", 10)),
            )
            deepzoom = DeepZoomGenerator(
                slide, tile_size=int(hest.get("tile_size", 224)), overlap=0
            )
            level_scale = 2 ** ((deepzoom.level_count - 1) - level)
            records = []
            for tile in result.cell_masks:
                offset = np.array([tile.x, tile.y], dtype=float) * int(
                    hest.get("tile_size", 224)
                )
                for mask in tile.masks:
                    centroid = (
                        np.asarray(mask.centroid, dtype=float) + offset
                    ) * level_scale
                    contour = (
                        np.asarray(mask.coordinates, dtype=float) + offset
                    ) * level_scale
                    records.append(
                        {
                            "x": centroid[0],
                            "y": centroid[1],
                            "cell_type": mask.cell_type,
                            "confidence": mask.confidence,
                            "contour": _contour_json(contour),
                        }
                    )
        finally:
            slide.close()
        predictions = pd.DataFrame(
            records, columns=("x", "y", "cell_type", "confidence", "contour")
        )
        predictions.insert(
            0, "prediction_id", [f"{sample_id}/{i}" for i in range(len(predictions))]
        )
        predictions.insert(0, "sample_id", sample_id)
        _write_predictions(destination, predictions)
        print(f"[hest/{sample_id}] {len(predictions):,} detections -> {destination}")


def radius_match(reference_xy, prediction_xy, radius):
    """Maximum-cardinality, minimum-distance radius matching using a sparse solver."""
    from scipy.sparse import coo_matrix, hstack
    from scipy.sparse.csgraph import min_weight_full_bipartite_matching
    from scipy.spatial import cKDTree

    reference_xy = np.asarray(reference_xy, dtype=float)
    prediction_xy = np.asarray(prediction_xy, dtype=float)
    if reference_xy.ndim != 2 or reference_xy.shape[1:] != (2,):
        raise ValueError("reference coordinates must have shape (N, 2)")
    if prediction_xy.ndim != 2 or prediction_xy.shape[1:] != (2,):
        raise ValueError("prediction coordinates must have shape (N, 2)")
    if float(radius) <= 0:
        raise ValueError("matching radius must be positive")
    if len(reference_xy) == 0 or len(prediction_xy) == 0:
        return np.empty((0, 3), dtype=float)

    distances = cKDTree(reference_xy).sparse_distance_matrix(
        cKDTree(prediction_xy), float(radius), output_type="coo_matrix"
    )
    candidate = coo_matrix(
        (1.0 + distances.data / float(radius), (distances.row, distances.col)),
        shape=(len(reference_xy), len(prediction_xy)),
    )
    # A unique dummy for every reference guarantees a full matching. Its cost is
    # larger than every possible aggregate distance change, so cardinality wins
    # before distance in the optimization objective.
    dummy_cost = float(2 * len(reference_xy) + 1)
    dummy = coo_matrix(
        (
            np.full(len(reference_xy), dummy_cost),
            (np.arange(len(reference_xy)), np.arange(len(reference_xy))),
        ),
        shape=(len(reference_xy), len(reference_xy)),
    )
    graph = hstack([candidate, dummy], format="csr")
    rows, columns = min_weight_full_bipartite_matching(graph)
    real = columns < len(prediction_xy)
    rows, columns = rows[real], columns[real]
    matched_distance = np.linalg.norm(
        reference_xy[rows] - prediction_xy[columns], axis=1
    )
    return np.column_stack([rows, columns, matched_distance])


def _classification_rows(y_true, y_pred, classes):
    rows = []
    for label in classes:
        true = y_true == label
        predicted = y_pred == label
        tp = int(np.sum(true & predicted))
        fp = int(np.sum(~true & predicted))
        fn = int(np.sum(true & ~predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        rows.append(
            {
                "class": label,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def score_cells(reference, predictions, *, reference_map, prediction_map, radius_px):
    """Match all instances, then score only references in the common taxonomy."""
    matches = radius_match(
        reference[["x", "y"]].to_numpy(),
        predictions[["x", "y"]].to_numpy(),
        radius_px,
    )
    ref_to_pred = {int(r): (int(p), float(d)) for r, p, d in matches}
    reference_target = reference["label"].map(reference_map)
    prediction_target = predictions["cell_type"].map(prediction_map)
    scorable = reference_target.notna().to_numpy()
    # dtype is explicit because an empty official test field yields an empty
    # comprehension, which numpy would otherwise type as float64 and refuse to
    # combine with the boolean scorable mask below.
    # dtype is explicit because an empty official test field yields an empty
    # comprehension, which numpy would otherwise type as float64 and refuse to
    # combine with the boolean scorable mask below.
    matched = np.array(
        [index in ref_to_pred for index in range(len(reference))], dtype=bool
    )
    matched_scorable = scorable & matched
    classes = sorted({value for value in reference_map.values() if value is not None})

    matched_true, matched_pred = [], []
    end_true, end_pred = [], []
    match_rows = []
    used_predictions = set()
    for reference_index, row in enumerate(reference.itertuples(index=False)):
        target = reference_target.iloc[reference_index]
        pair = ref_to_pred.get(reference_index)
        predicted_target = None
        prediction_id = None
        distance = None
        predicted_type = None
        if pair is not None:
            prediction_index, distance = pair
            used_predictions.add(prediction_index)
            prediction_id = predictions.iloc[prediction_index]["prediction_id"]
            predicted_type = predictions.iloc[prediction_index]["cell_type"]
            predicted_target = prediction_target.iloc[prediction_index]
        if pd.notna(target):
            value = (
                predicted_target
                if pd.notna(predicted_target)
                else "__missed_or_unmapped__"
            )
            end_true.append(target)
            end_pred.append(value)
            if pair is not None:
                matched_true.append(target)
                matched_pred.append(value)
        match_rows.append(
            {
                "reference_id": row.reference_id,
                "reference_label": row.label,
                "reference_target": target,
                "prediction_id": prediction_id,
                "prediction_type": predicted_type,
                "prediction_target": predicted_target,
                "distance_px": distance,
            }
        )

    # Mapped detections not assigned to any reference are false positives.
    for prediction_index, target in enumerate(prediction_target):
        if prediction_index not in used_predictions and pd.notna(target):
            end_true.append("__background__")
            end_pred.append(target)

    end_true = np.asarray(end_true, dtype=object)
    end_pred = np.asarray(end_pred, dtype=object)
    matched_true = np.asarray(matched_true, dtype=object)
    matched_pred = np.asarray(matched_pred, dtype=object)
    end_rows = _classification_rows(end_true, end_pred, classes)
    matched_rows = _classification_rows(matched_true, matched_pred, classes)
    matched_macro = (
        float(np.mean([row["f1"] for row in matched_rows])) if matched_rows else 0.0
    )
    end_macro = float(np.mean([row["f1"] for row in end_rows])) if end_rows else 0.0
    coverage = float(matched_scorable.sum() / scorable.sum()) if scorable.sum() else 0.0
    n_matches = len(matches)
    detection_precision = n_matches / len(predictions) if len(predictions) else 0.0
    detection_recall = n_matches / len(reference) if len(reference) else 0.0
    metrics = {
        "n_reference": len(reference),
        "n_scorable_reference": int(scorable.sum()),
        "n_predictions": len(predictions),
        "n_geometric_matches": int(n_matches),
        "n_scorable_matches": int(matched_scorable.sum()),
        "coverage": coverage,
        "detection_precision": detection_precision,
        "detection_recall": detection_recall,
        "detection_f1": 2
        * detection_precision
        * detection_recall
        / (detection_precision + detection_recall)
        if detection_precision + detection_recall
        else 0.0,
        "matched_macro_f1": matched_macro,
        "coverage_adjusted_macro_f1": matched_macro * coverage,
        "end_to_end_macro_f1": end_macro,
        "classes": classes,
        "per_class_matched": matched_rows,
        "per_class_end_to_end": end_rows,
    }
    return metrics, pd.DataFrame(match_rows), matches


def polygon_iou(first, second):
    """Rasterized IoU for two nucleus contours in one coordinate system."""
    import cv2

    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.ndim != 2 or second.ndim != 2 or min(len(first), len(second)) < 3:
        return 0.0
    low = np.floor(np.minimum(first.min(axis=0), second.min(axis=0))).astype(int) - 1
    high = np.ceil(np.maximum(first.max(axis=0), second.max(axis=0))).astype(int) + 1
    width, height = (high - low + 1).tolist()
    if width <= 0 or height <= 0:
        return 0.0
    a = np.zeros((height, width), dtype=np.uint8)
    b = np.zeros_like(a)
    cv2.fillPoly(a, [np.rint(first - low).astype(np.int32)], 1)
    cv2.fillPoly(b, [np.rint(second - low).astype(np.int32)], 1)
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(intersection / union) if union else 0.0


def panoptic_quality(reference, predictions, matches, iou_threshold=0.5):
    ious = []
    for reference_index, prediction_index, _ in matches:
        reference_contour = reference.iloc[int(reference_index)]["contour"]
        prediction_contour = json.loads(
            predictions.iloc[int(prediction_index)]["contour"]
        )
        ious.append(polygon_iou(reference_contour, prediction_contour))
    true_positive_ious = [value for value in ious if value > float(iou_threshold)]
    tp = len(true_positive_ious)
    fp = len(predictions) - tp
    fn = len(reference) - tp
    dq = tp / (tp + 0.5 * fp + 0.5 * fn) if tp + fp + fn else 0.0
    sq = float(np.mean(true_positive_ious)) if true_positive_ious else 0.0
    return {
        "panoptic_tp": tp,
        "panoptic_fp": fp,
        "panoptic_fn": fn,
        "panoptic_iou_sum": float(sum(true_positive_ious)),
        "dq": dq,
        "sq": sq,
        "pq": dq * sq,
    }


def _hest_reference(cfg, sample_id):
    hest = cfg["hest"]
    row = hest_manifest_row(hest, sample_id)
    sample_dir = Path(row["output_dir"])
    with np.load(sample_dir / f"{sample_id}_terra.npz", allow_pickle=False) as data:
        cells = pd.DataFrame(
            {
                "reference_id": data["cell_ids"].astype(str),
                "x": data["he_x"].astype(float),
                "y": data["he_y"].astype(float),
            }
        )
    reviewed_path = Path(hest["reviewed_cells_pattern"].format(sample_id=sample_id))
    reviewed = pd.read_parquet(reviewed_path)[
        ["cell_id", "status", "malignancy", "canonical_identity"]
    ]
    reviewed["cell_id"] = reviewed["cell_id"].astype(str)
    cells = cells.merge(
        reviewed, left_on="reference_id", right_on="cell_id", validate="one_to_one"
    )
    cells["label"] = cells["canonical_identity"].astype(str)
    cells.loc[cells["status"] == "excluded", "label"] = "__excluded__"
    cells.loc[cells["malignancy"] == "malignant", "label"] = "neoplastic cell"
    return cells, float(row["pixel_size_um"])


def _instance_contour(entry, cell_id, geometry):
    import cv2

    mask = (np.asarray(entry["instance_mask"]) == int(cell_id)).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
    contour[:, 0] += geometry.x0 - geometry.pad_left
    contour[:, 1] += geometry.y0 - geometry.pad_top
    return contour


def _ood_references(cfg, dataset):
    section = cfg[dataset]
    data = section["data"]
    metadata = pd.read_csv(data["metadata"])
    metadata = metadata.loc[metadata["split_id"] == "test"].copy()
    metadata["reference_id"] = (
        metadata["sample_id"].astype(str)
        + "/"
        + metadata["tile_id"].astype(str)
        + "/"
        + metadata["cell_id"].astype(str)
    )
    by_sample = {
        name: group.copy()
        for name, group in metadata.groupby(
            metadata["sample_id"].astype(str), sort=False
        )
    }
    wanted = set(_official_test_samples(data["split"]))
    seen = set()
    image_shape = tuple(data["image_shape"])
    tile_size = int(data["tile_size"])
    geometries = {item.index: item for item in tile_geometries(image_shape, tile_size)}
    for sample_id, entries in iter_lmdb_samples(data["lmdb"]):
        if sample_id not in wanted:
            continue
        seen.add(sample_id)
        if sample_id not in by_sample:
            yield (
                sample_id,
                pd.DataFrame(columns=("reference_id", "x", "y", "label", "contour")),
            )
            continue
        reference = locate_cells(
            by_sample[sample_id],
            entries,
            image_shape,
            tile_size,
            skip_missing=bool(data.get("allow_unlabeled_samples", False)),
        )
        if dataset == "pannuke":
            contours = []
            keep = []
            for index, row in reference.iterrows():
                entry_index = tile_index(row["tile_id"])
                contour = _instance_contour(
                    entries[entry_index], row["cell_id"], geometries[entry_index]
                )
                if contour is not None and len(contour) >= 3:
                    keep.append(index)
                    contours.append(contour)
            reference = reference.loc[keep].copy()
            reference["contour"] = contours
        yield sample_id, reference.reset_index(drop=True)
    missing = sorted(wanted - seen)
    if missing:
        raise ValueError(
            f"{dataset}: official samples absent from LMDB: {missing[:10]}"
        )


def score_dataset(cfg, root, dataset):
    if dataset == "hest":
        sources = (
            (sample, *_hest_reference(cfg, sample))
            for sample in label_test_fast_samples(cfg["hest"])
        )
        reference_map, prediction_map = HEST_IDENTITY, HISTOPLUS_FINE
    else:
        source_mpp = float(cfg[dataset]["data"]["source_mpp"])
        sources = (
            (sample, reference, source_mpp)
            for sample, reference in _ood_references(cfg, dataset)
        )
        if dataset == "phenocell":
            reference_map, prediction_map = PHENOCELL_REFERENCE, HISTOPLUS_FINE
        else:
            reference_map, prediction_map = PANNuke_REFERENCE, HISTOPLUS_PANNUKE

    summaries = []
    sample_metrics = []
    output = Path(root) / dataset / "scores"
    for sample_id, reference, source_mpp in sources:
        predictions = _read_predictions(prediction_path(root, dataset, sample_id))
        radius_px = float(cfg["evaluation"]["match_radius_um"]) / source_mpp
        metrics, matches, match_indices = score_cells(
            reference,
            predictions,
            reference_map=reference_map,
            prediction_map=prediction_map,
            radius_px=radius_px,
        )
        metrics.update(
            {
                "sample_id": sample_id,
                "dataset": dataset,
                "source_mpp": source_mpp,
                "match_radius_px": radius_px,
            }
        )
        if dataset == "pannuke":
            metrics.update(
                panoptic_quality(
                    reference,
                    predictions,
                    match_indices,
                    cfg["pannuke"].get("panoptic_iou_threshold", 0.5),
                )
            )
        write_json(output / f"{str(sample_id).removesuffix('.hdf')}.json", metrics)
        if cfg["evaluation"].get("save_matches", True):
            path = output / "matches" / f"{str(sample_id).removesuffix('.hdf')}.csv.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            matches.to_csv(path, index=False, compression="gzip")
        summaries.append(
            {
                key: value
                for key, value in metrics.items()
                if not isinstance(value, (list, dict))
            }
        )
        sample_metrics.append(metrics)
        print(
            f"[{dataset}/{sample_id}] coverage={metrics['coverage']:.3f} end-to-end macro-F1={metrics['end_to_end_macro_f1']:.3f}"
        )
    if not summaries:
        raise ValueError(f"{dataset}: no official test samples were scored")
    write_table(output / "summary.csv", summaries)
    aggregate = aggregate_metrics(sample_metrics, dataset)
    write_json(output / "aggregate.json", aggregate)
    write_table(output / "per_class.csv", aggregate["per_class_end_to_end"])
    print(
        f"[{dataset}] aggregate coverage={aggregate['coverage']:.3f} "
        f"end-to-end macro-F1={aggregate['end_to_end_macro_f1']:.3f}"
    )


def check_inputs(cfg):
    samples = label_test_fast_samples(cfg["hest"])
    for dataset in ("phenocell", "pannuke"):
        data = cfg[dataset]["data"]
        for key in ("lmdb", "metadata", "split"):
            if not Path(data[key]).exists():
                raise FileNotFoundError(data[key])
        metadata = pd.read_csv(data["metadata"])
        validate_official_split(
            metadata[["sample_id", "split_id"]].rename(columns={"split_id": "split"}),
            data["split"],
            require_full_coverage=not bool(data.get("allow_unlabeled_samples", False)),
        )
    for sample_id in samples:
        row = hest_manifest_row(cfg["hest"], sample_id)
        for path in (
            row["wsi"],
            cfg["hest"]["reviewed_cells_pattern"].format(sample_id=sample_id),
        ):
            if not Path(path).exists():
                raise FileNotFoundError(path)
    identity = git_identity(cfg["model"]["repository"])
    print(f"HistoPLUS repository: {identity['commit']}")
    print(f"Fast HEST: {len(samples)} samples ({', '.join(samples)})")
    for dataset in ("phenocell", "pannuke"):
        metadata = pd.read_csv(cfg[dataset]["data"]["metadata"])
        test = metadata[metadata["split_id"] == "test"]
        fields = _official_test_samples(cfg[dataset]["data"]["split"])
        print(f"{dataset}: {len(fields):,} test fields, {len(test):,} reference cells")
    print("Protocol: frozen full-image detection; reference masks are scoring-only")


def prepare_models(cfg):
    for mpp in (0.25, 0.5):
        segmentor = _load_segmentor(cfg, mpp)
        print(f"loaded {segmentor.segmentor_name} at {mpp} MPP")


def _parse_samples(value):
    if not value:
        return None
    samples = [part.strip() for part in value.split(",") if part.strip()]
    if not samples or len(samples) != len(set(samples)):
        raise ValueError("--samples must be a unique comma-separated list")
    return samples


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "prepare"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
    hest = commands.add_parser("infer-hest")
    hest.add_argument("--config", required=True)
    hest.add_argument("--output-dir")
    hest.add_argument("--samples")
    hest.add_argument("--overwrite", action="store_true")
    ood = commands.add_parser("infer-ood")
    ood.add_argument("--config", required=True)
    ood.add_argument("--output-dir")
    ood.add_argument("--dataset", choices=("phenocell", "pannuke"), required=True)
    ood.add_argument("--samples")
    ood.add_argument("--shard-index", type=int, default=0)
    ood.add_argument("--num-shards", type=int, default=1)
    ood.add_argument("--overwrite", action="store_true")
    score = commands.add_parser("score")
    score.add_argument("--config", required=True)
    score.add_argument("--output-dir")
    score.add_argument(
        "--dataset", choices=("all", "hest", "phenocell", "pannuke"), default="all"
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    if args.command == "check":
        check_inputs(cfg)
    elif args.command == "prepare":
        prepare_models(cfg)
    elif args.command == "infer-hest":
        infer_hest(
            cfg,
            output_root(cfg, args.output_dir),
            samples=_parse_samples(args.samples),
            overwrite=args.overwrite,
        )
    elif args.command == "infer-ood":
        infer_ood(
            cfg,
            output_root(cfg, args.output_dir),
            args.dataset,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            samples=_parse_samples(args.samples),
            overwrite=args.overwrite,
        )
    elif args.command == "score":
        datasets = (
            ("hest", "phenocell", "pannuke")
            if args.dataset == "all"
            else (args.dataset,)
        )
        for dataset in datasets:
            score_dataset(cfg, output_root(cfg, args.output_dir), dataset)


if __name__ == "__main__":
    main()
