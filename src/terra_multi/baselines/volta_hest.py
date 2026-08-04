"""VOLTA extraction for the fixed fast-HEST cohort."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from terra_multi.baselines.evaluation import (
    expected_hest_cell_ids,
    finalize_hest_manifest,
    hest_feature_path,
    hest_manifest_row,
    hest_source_identity,
    label_test_fast_samples,
)
from terra_multi.baselines.volta_crops import crop_box
from terra_multi.evaluation.artifacts import (
    feature_file_is_current,
    feature_signature,
    file_identity,
    stable_json,
    summarize_feature_file,
    write_feature_file,
    write_table,
)
from terra_multi.lora.data import open_wsi

DATASET_NAME = "hest/label_test_fast"


def _merge_bounds(bounds, cell_ids):
    bounds = bounds.copy()
    bounds.index = np.asarray(cell_ids).astype(str)
    reducers = {"minx": "min", "miny": "min", "maxx": "max", "maxy": "max"}
    return bounds.groupby(level=0, sort=False).agg(reducers)


def _cell_bounds(path, cell_ids):
    import geopandas as gpd

    cells = gpd.read_parquet(path)
    ids = cells.index.astype(str).to_numpy()
    bounds = _merge_bounds(cells.geometry.bounds, ids)
    positions = bounds.index.get_indexer(np.asarray(cell_ids).astype(str))
    missing = positions < 0
    if missing.any():
        raise ValueError(
            f"{path}: missing TERRA cell IDs {cell_ids[missing][:5].tolist()}"
        )
    return bounds.iloc[positions].to_numpy(np.float64)


def _close_wsi(wsi):
    store = getattr(wsi, "store", None)
    close = getattr(store, "close", None)
    if close is not None:
        close()


def _embed(wsi, bounds, encoder, batch_size, workers, scale, destination):
    import torch

    features = np.lib.format.open_memmap(
        destination,
        mode="w+",
        dtype=np.float16,
        shape=(len(bounds), encoder.expected_dim),
    )

    def transform(box):
        return encoder.transform(crop_box(wsi, box, scale))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(bounds), batch_size):
            images = torch.stack(
                list(pool.map(transform, bounds[start : start + batch_size]))
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
        segmentation = Path(row["xenium_cell_seg"])
        all_ids = expected_hest_cell_ids(hest, sample_id)
        cell_ids = all_ids[:max_cells_per_sample]
        bounds = _cell_bounds(segmentation, cell_ids)
        source = {
            **hest_source_identity(hest, sample_id),
            "xenium_cell_seg": file_identity(segmentation),
            "extractor": file_identity(Path(__file__), checksum=True),
        }
        settings = {
            "sample_id": sample_id,
            "segmentation": "xenium_cell_seg",
            "crop": cfg["crop"],
            "model_input_size": 32,
            "resize": "OpenCV INTER_CUBIC",
            "normalization": cfg["model"]["normalization"],
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
                        bounds,
                        encoder,
                        int(hest["extraction"]["batch_size"]),
                        int(hest["extraction"]["num_workers"]),
                        float(cfg["crop"]["scale"]),
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
    suffix = "-".join(wanted)
    write_table(output_dir / "features" / f"manifest.partial.{suffix}.csv", rows)
    return pd.DataFrame(rows)
