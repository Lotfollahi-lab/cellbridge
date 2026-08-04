"""VOLTA extraction from the shared tiled OOD datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from terra_multi.baselines.evaluation import phenocell_cell_ids, phenocell_file_stem
from terra_multi.baselines.volta_crops import crop_box, locate_instance_boxes
from terra_multi.evaluation.artifacts import (
    feature_file_is_current,
    feature_signature,
    file_identity,
    path_identity,
    stable_json,
    stable_signature,
    summarize_feature_file,
    write_feature_file,
    write_table,
)
from terra_multi.evaluation.ood_data import (
    iter_lmdb_samples,
    reconstruct_image,
    to_uint8_rgb,
)


def ood_feature_path(output_dir, sample_id):
    return Path(output_dir) / "features" / f"{phenocell_file_stem(sample_id)}.npz"


def _source_identity(data):
    return {
        "lmdb": path_identity(data["lmdb"]),
        "metadata": file_identity(data["metadata"]),
        "extractor": file_identity(Path(__file__), checksum=True),
    }


def _embed(image, locations, encoder, batch_size, scale):
    import torch

    parts = []
    for start in range(0, len(locations), batch_size):
        rows = locations.iloc[start : start + batch_size]
        images = torch.stack(
            [
                encoder.transform(
                    crop_box(image, (row.x0, row.y0, row.x1, row.y1), scale)
                )
                for row in rows.itertuples()
            ]
        )
        parts.append(encoder.encode(images))
    if not parts:
        raise ValueError("cannot embed an empty sample")
    return np.concatenate(parts)


def _summary_rows(summaries):
    return [
        {
            "sample_id": item.sample_id,
            "n_cells": item.n_cells,
            "n_features": item.n_features,
            "path": item.path,
            "signature": item.signature,
            "model_signature": item.model_signature,
        }
        for item in summaries
    ]


def extract_ood(
    cfg,
    encoder,
    root,
    dataset_name,
    *,
    samples=None,
    max_cells_per_sample=None,
    overwrite=False,
):
    dataset = cfg[dataset_name]
    data = dataset["data"]
    metadata = pd.read_csv(data["metadata"])
    metadata["row_id"] = np.arange(len(metadata), dtype=np.int64)
    expected = metadata["sample_id"].astype(str).drop_duplicates().tolist()
    requested = expected if samples is None else [str(value) for value in samples]
    unknown = sorted(set(requested) - set(expected))
    if unknown:
        raise ValueError(f"unknown {dataset_name} samples: {unknown}")
    wanted = [sample for sample in expected if sample in set(requested)]
    if max_cells_per_sample is not None and max_cells_per_sample < 1:
        raise ValueError("max_cells_per_sample must be positive")

    output_dir = Path(root) / dataset_name
    image_shape = tuple(data["image_shape"])
    tile_size = int(data["tile_size"])
    source = _source_identity(data)
    wanted_set = set(wanted)
    processed = set()
    observed = set()
    summaries = []
    skip_missing = bool(data.get("allow_unlabeled_samples", False))
    for sample_id, entries in iter_lmdb_samples(data["lmdb"]):
        observed.add(sample_id)
        if sample_id not in wanted_set:
            continue
        rows = metadata.loc[metadata["sample_id"].astype(str) == sample_id].copy()
        if max_cells_per_sample is not None:
            rows = rows.iloc[:max_cells_per_sample].copy()
        locations = locate_instance_boxes(
            rows,
            entries,
            image_shape,
            tile_size,
            skip_missing=skip_missing,
        )
        cell_ids = phenocell_cell_ids(locations)
        settings = {
            "sample_id": sample_id,
            "image_shape": list(image_shape),
            "tile_size": tile_size,
            "value_scale": float(data["value_scale"]),
            "crop": cfg["crop"],
            "model_input_size": 32,
            "resize": "OpenCV INTER_CUBIC",
            "normalization": cfg["model"]["normalization"],
            "normalized_embedding": False,
            "embedding_dim": encoder.expected_dim,
            "max_cells_per_sample": max_cells_per_sample,
        }
        signature = feature_signature(dataset_name, encoder.identity, source, settings)
        destination = ood_feature_path(output_dir, sample_id)
        required = (
            "row_id",
            "sample_id",
            "tile_id",
            "cell_id",
            "label",
            "class_id",
            "split",
            "x0",
            "y0",
            "x1",
            "y1",
        )
        if not overwrite and feature_file_is_current(
            destination,
            signature,
            expected_cells=len(locations),
            expected_dim=encoder.expected_dim,
            expected_cell_ids=cell_ids,
            required_metadata=required,
        ):
            summary = summarize_feature_file(
                destination, expected_dim=encoder.expected_dim
            )
            print(f"[{dataset_name}/{sample_id}] current ({summary.n_cells:,} cells)")
        else:
            image = to_uint8_rgb(
                reconstruct_image(
                    {index: entry["image"] for index, entry in entries.items()},
                    image_shape,
                    tile_size,
                ),
                data["value_scale"],
            )
            embeddings = _embed(
                image,
                locations,
                encoder,
                int(dataset["extraction"]["batch_size"]),
                float(cfg["crop"]["scale"]),
            )
            provenance = stable_json(
                {"model": encoder.identity, "source": source, "settings": settings}
            )
            write_feature_file(
                destination,
                cell_ids,
                embeddings,
                signature,
                expected_dim=encoder.expected_dim,
                metadata={
                    "baseline_name": np.asarray(cfg["baseline"]["name"]),
                    "dataset_name": np.asarray(dataset_name),
                    "artifact_sample_id": np.asarray(sample_id),
                    "model_signature": np.asarray(encoder.model_signature),
                    "provenance_json": np.asarray(provenance),
                    "row_id": locations["row_id"].to_numpy(np.int64),
                    "sample_id": locations["sample_id"].to_numpy(dtype=str),
                    "tile_id": locations["tile_id"].to_numpy(dtype=str),
                    "cell_id": locations["cell_id"].to_numpy(np.int32),
                    "label": locations["label"].to_numpy(dtype=str),
                    "class_id": locations["class_id"].to_numpy(np.int16),
                    "split": locations["split_id"].to_numpy(dtype=str),
                    "x0": locations["x0"].to_numpy(np.float32),
                    "y0": locations["y0"].to_numpy(np.float32),
                    "x1": locations["x1"].to_numpy(np.float32),
                    "y1": locations["y1"].to_numpy(np.float32),
                },
            )
            summary = summarize_feature_file(
                destination, expected_dim=encoder.expected_dim
            )
            print(f"[{dataset_name}/{sample_id}] wrote {summary.n_cells:,} cells")
        summaries.append(summary)
        processed.add(sample_id)
        if samples is not None and processed == wanted_set:
            break

    missing = sorted(wanted_set - processed)
    if missing:
        raise ValueError(f"{dataset_name} samples absent from LMDB: {missing}")
    if samples is None and not skip_missing and observed != set(expected):
        raise ValueError(f"{dataset_name} LMDB/metadata sample sets disagree")
    if samples is None and max_cells_per_sample is None:
        from terra_multi.baselines.volta_ood_probe import finalize_ood_manifest

        return finalize_ood_manifest(cfg, output_dir, dataset_name)
    result = _summary_rows(summaries)
    names = [phenocell_file_stem(sample) for sample in wanted]
    suffix = "-".join(names) if len(names) <= 3 else stable_signature(names)[:12]
    write_table(output_dir / "features" / f"manifest.partial.{suffix}.csv", result)
    return pd.DataFrame(result)
