"""PAST extraction for the fixed fast-HEST cohort."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from cellbridge.baselines.evaluation import (
    expected_hest_cell_ids,
    finalize_hest_manifest,
    hest_feature_path,
    hest_manifest_row,
    hest_source_identity,
    hest_terra_path,
    label_test_fast_samples,
)
from cellbridge.baselines.past_crops import native_center_crop
from cellbridge.evaluation.artifacts import (
    feature_file_is_current,
    feature_signature,
    file_identity,
    stable_json,
    summarize_feature_file,
    write_feature_file,
    write_table,
)
from cellbridge.lora.data import open_wsi

DATASET_NAME = "hest/label_test_fast"


def _centers(path, expected_ids, limit=None):
    with np.load(path, allow_pickle=True) as data:
        required = {"cell_ids", "he_x", "he_y"}
        if not required <= set(data.files):
            raise ValueError(f"{path}: missing {sorted(required - set(data.files))}")
        cell_ids = data["cell_ids"].astype(str)
        centers = np.column_stack((data["he_x"], data["he_y"])).astype(np.float64)
    if not np.array_equal(cell_ids, expected_ids):
        raise ValueError(f"{path}: cell IDs disagree with the TERRA artifact contract")
    if centers.shape != (len(cell_ids), 2) or not np.isfinite(centers).all():
        raise ValueError(f"{path}: invalid H&E centers")
    stop = len(cell_ids) if limit is None else min(int(limit), len(cell_ids))
    return cell_ids[:stop], centers[:stop]


def _close_wsi(wsi):
    close = getattr(getattr(wsi, "store", None), "close", None)
    if close is not None:
        close()


def _embed(wsi, centers, encoder, batch_size, workers, crop_size, destination):
    import torch

    features = np.lib.format.open_memmap(
        destination,
        mode="w+",
        dtype=np.float16,
        shape=(len(centers), encoder.expected_dim),
    )

    def transform(center):
        return encoder.transform(
            native_center_crop(wsi, center[0], center[1], crop_size, fill=0)
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(centers), batch_size):
            images = torch.stack(
                list(pool.map(transform, centers[start : start + batch_size]))
            )
            features[start : start + len(images)] = encoder.encode(images)
    features.flush()
    return features


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


def extract_hest(
    cfg,
    encoder,
    root,
    *,
    samples=None,
    max_cells_per_sample=None,
    overwrite=False,
):
    hest = cfg["hest"]
    expected = label_test_fast_samples(hest)
    requested = expected if samples is None else [str(value) for value in samples]
    unknown = sorted(set(requested) - set(expected))
    if unknown:
        raise ValueError(f"not label_test_fast samples: {unknown}")
    wanted = [sample for sample in expected if sample in set(requested)]
    if max_cells_per_sample is not None and max_cells_per_sample < 1:
        raise ValueError("max_cells_per_sample must be positive")

    output_dir = Path(root) / "hest_label_test_fast"
    summaries = []
    for sample_id in wanted:
        row = hest_manifest_row(hest, sample_id)
        terra_path = hest_terra_path(hest, sample_id, row)
        cell_ids, centers = _centers(
            terra_path,
            expected_hest_cell_ids(hest, sample_id),
            max_cells_per_sample,
        )
        source = {
            **hest_source_identity(hest, sample_id),
            "extractor": file_identity(Path(__file__), checksum=True),
        }
        settings = {
            "sample_id": sample_id,
            "crop": cfg["crop"],
            "model_input_size": 224,
            "resize": "none for native 224-pixel crops",
            "normalization": "ImageNet mean/std",
            "normalized_embedding": False,
            "embedding_dim": encoder.expected_dim,
            "max_cells_per_sample": max_cells_per_sample,
        }
        signature = feature_signature(DATASET_NAME, encoder.identity, source, settings)
        destination = hest_feature_path(output_dir, sample_id)
        if not overwrite and feature_file_is_current(
            destination,
            signature,
            expected_cells=len(cell_ids),
            expected_dim=encoder.expected_dim,
            expected_cell_ids=cell_ids,
        ):
            summary = summarize_feature_file(
                destination, expected_dim=encoder.expected_dim
            )
            print(f"[HEST/{sample_id}] current ({summary.n_cells:,} cells)")
        else:
            wsi = open_wsi(row["wsi"])
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with TemporaryDirectory(dir=destination.parent) as temporary:
                    embeddings = _embed(
                        wsi,
                        centers,
                        encoder,
                        int(hest["extraction"]["batch_size"]),
                        int(hest["extraction"]["num_workers"]),
                        int(cfg["crop"]["source_size"]),
                        Path(temporary) / "embeddings.npy",
                    )
                    write_feature_file(
                        destination,
                        cell_ids,
                        embeddings,
                        signature,
                        expected_dim=encoder.expected_dim,
                        metadata={
                            "baseline_name": np.asarray(cfg["baseline"]["name"]),
                            "dataset_name": np.asarray(DATASET_NAME),
                            "artifact_sample_id": np.asarray(sample_id),
                            "model_signature": np.asarray(encoder.model_signature),
                            "provenance_json": np.asarray(
                                stable_json(
                                    {
                                        "model": encoder.identity,
                                        "source": source,
                                        "settings": settings,
                                    }
                                )
                            ),
                        },
                    )
            finally:
                _close_wsi(wsi)
            summary = summarize_feature_file(
                destination, expected_dim=encoder.expected_dim
            )
            print(
                f"[HEST/{sample_id}] wrote {summary.n_cells:,} cells -> {destination}"
            )
        summaries.append(summary)

    if samples is None and max_cells_per_sample is None:
        return finalize_hest_manifest(cfg, output_dir)
    rows = _summary_rows(summaries)
    write_table(
        output_dir / "features" / f"manifest.partial.{'-'.join(wanted)}.csv", rows
    )
    return pd.DataFrame(rows)
