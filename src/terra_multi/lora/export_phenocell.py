"""Export LoRA feature arms directly from OOD image crops (PhenoCell, PanNuke)."""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from terra_multi.evaluation.artifacts import file_identity, path_identity, write_table
from terra_multi.evaluation.ood_data import (
    centered_crop,
    iter_lmdb_samples,
    locate_cells,
    reconstruct_image,
    to_uint8_rgb,
)
from terra_multi.lora.features import (
    FEATURE_ARMS,
    feature_file_is_current,
    feature_signature,
    infer_tensor,
    write_feature_file,
)

_METADATA = ("row_id", "sample_id", "tile_id", "cell_id", "label", "class_id", "split")


def _signature(cfg, checkpoint_identity, sample_name, dataset):
    data = cfg["data"]
    return feature_signature(
        dataset,
        checkpoint_identity,
        {"lmdb": path_identity(data["lmdb"]), "metadata": file_identity(data["metadata"])},
        {
            "sample_id": sample_name,
            "image_shape": data.get("image_shape", [1440, 1920]),
            "tile_size": int(data.get("tile_size", 224)),
            "value_scale": float(data.get("value_scale", 250)),
            "crop_size": int(cfg.get("embedding", {}).get("crop_size", 96)),
            "encoder": "uni_v2",
        },
    )


def _features(image, locations, model, device, batch_size, crop_size):
    import torch

    parts = {}
    for start in range(0, len(locations), batch_size):
        rows = locations.iloc[start : start + batch_size]
        images = torch.stack(
            [
                model.eval_transforms(
                    Image.fromarray(centered_crop(image, row.x, row.y, crop_size)).resize(
                        (224, 224), Image.Resampling.BILINEAR
                    )
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
    cfg,
    model,
    device,
    output_dir,
    checkpoint_identity,
    overwrite=False,
    dataset="phenocell",
    shard=None,
    num_shards=1,
):
    """Export adapted features from raw OOD crops, never cached UNI.

    With ``shard``/``num_shards`` set, samples are split round-robin across
    parallel jobs and each writes its own manifest slice.
    """
    output_dir = Path(output_dir)
    data = cfg["data"]
    metadata = pd.read_csv(data["metadata"])
    metadata["row_id"] = np.arange(len(metadata), dtype=np.int64)
    image_shape = tuple(data.get("image_shape", [1440, 1920]))
    tile_size = int(data.get("tile_size", 224))
    allow_unlabeled = bool(data.get("allow_unlabeled_samples", False))
    crop_size = int(cfg.get("embedding", {}).get("crop_size", 96))
    batch_size = int(cfg.get("embedding", {}).get("batch_size", 128))
    expected_dims = {
        "uni_original": model.image_dim,
        "lora_adapted": model.image_dim,
    }
    rows, observed_samples = [], set()
    for position, (sample_name, entries) in enumerate(iter_lmdb_samples(data["lmdb"])):
        if shard is not None and position % num_shards != shard:
            continue
        sample_metadata = metadata.loc[metadata["sample_id"].astype(str) == sample_name].copy()
        if sample_metadata.empty:
            if allow_unlabeled:
                continue
            raise ValueError(f"{sample_name}: no metadata rows")
        observed_samples.add(sample_name)
        signature = _signature(cfg, checkpoint_identity, sample_name, dataset)
        destination = output_dir / "features" / f"{sample_name.removesuffix('.hdf')}.npz"
        if not overwrite and feature_file_is_current(
            destination,
            signature,
            FEATURE_ARMS,
            required_metadata=_METADATA,
            expected_cells=len(sample_metadata),
            expected_dims=expected_dims,
        ):
            with np.load(destination, allow_pickle=False) as cached:
                count = len(cached["cell_ids"])
            print(f"[{dataset}/{sample_name}] current ({count:,} cells)")
        else:
            locations = locate_cells(
                sample_metadata, entries, image_shape, tile_size,
                skip_missing=allow_unlabeled,
            )
            if locations.empty:
                print(f"[{dataset}/{sample_name}] no locatable cells; skipped")
                continue
            image = to_uint8_rgb(
                reconstruct_image(
                    {index: entry["image"] for index, entry in entries.items()},
                    image_shape,
                    tile_size,
                ),
                data.get("value_scale", 250),
            )
            features = _features(image, locations, model, device, batch_size, crop_size)
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
            print(f"[{dataset}/{sample_name}] wrote {count:,} cells -> {destination}")
        rows.append(
            {
                "sample_id": sample_name,
                "n_cells": count,
                "path": str(destination.resolve()),
                "signature": signature,
            }
        )
    if shard is None:
        expected = set(metadata["sample_id"].astype(str))
        if observed_samples != expected:
            missing = sorted(expected - observed_samples)
            extra = sorted(observed_samples - expected)
            raise ValueError(
                f"{dataset} LMDB/metadata mismatch: missing={missing}, extra={extra}"
            )
        name = "manifest.csv"
    else:
        name = f"manifest.shard{shard:03d}.csv"
    write_table(output_dir / "features" / name, rows)
    return pd.DataFrame(rows)
