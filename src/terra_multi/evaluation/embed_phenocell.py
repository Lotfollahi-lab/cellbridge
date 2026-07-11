"""Extract raw UNI and distilled embeddings for PhenoCell cells."""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from terra_multi.distill.model import TerraProjector
from terra_multi.evaluation.phenocell_data import (
    centered_crop,
    iter_lmdb_samples,
    locate_cells,
    reconstruct_image,
    to_uint8_rgb,
)


def load_projector(path, device):
    import torch

    checkpoint = torch.load(path, map_location=device)
    cfg = checkpoint["config"]
    model = TerraProjector(
        cfg.get("uni_dim", 1536),
        cfg.get("terra_dim", 384),
        cfg.get("hidden_dim", 1024),
        cfg.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model.eval(), cfg


def embed_crops(image, locations, encoder, transform, projector, device, batch_size, crop_size):
    import torch

    raw, distilled = [], []
    with torch.inference_mode():
        for start in range(0, len(locations), batch_size):
            rows = locations.iloc[start : start + batch_size]
            batch = [
                transform(
                    Image.fromarray(centered_crop(image, row.x, row.y, crop_size)).resize(
                        (224, 224), Image.Resampling.BILINEAR
                    )
                )
                for row in rows.itertuples()
            ]
            inputs = torch.stack(batch).to(device)
            with torch.autocast(
                device_type="cuda",
                dtype=encoder.precision,
                enabled=device == "cuda",
            ):
                uni = encoder(inputs)
                terra = projector(uni.float())
            raw.append(uni.float().cpu().numpy())
            distilled.append(terra.float().cpu().numpy())
    return np.concatenate(raw), np.concatenate(distilled)


def output_path(output_dir, sample_name):
    return output_dir / "samples" / f"{sample_name.removesuffix('.hdf')}.npz"


def write_sample(path, locations, uni, distilled):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(
        temporary,
        row_id=locations["row_id"].to_numpy(np.int64),
        sample_id=locations["sample_id"].astype(str).to_numpy(),
        tile_id=locations["tile_id"].astype(str).to_numpy(),
        cell_id=locations["cell_id"].to_numpy(np.int32),
        label=locations["label"].astype(str).to_numpy(),
        class_id=locations["class_id"].to_numpy(np.int16),
        split=locations["split_id"].astype(str).to_numpy(),
        x=locations["x"].to_numpy(np.float32),
        y=locations["y"].to_numpy(np.float32),
        uni_emb=uni.astype(np.float16),
        distilled_emb=distilled.astype(np.float16),
    )
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)

    import torch
    from trident.patch_encoder_models import encoder_factory

    data = cfg["data"]
    embedding = cfg["embedding"]
    image_shape = tuple(data.get("image_shape", [1440, 1920]))
    tile_size = int(data.get("tile_size", 224))
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(data["metadata"])
    metadata["row_id"] = np.arange(len(metadata))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = encoder_factory(embedding.get("encoder", "uni_v2")).to(device).eval()
    projector, projector_cfg = load_projector(embedding["projector_checkpoint"], device)
    if projector_cfg.get("uni_dim", 1536) != 1536:
        raise ValueError("the projector does not expect 1536-dimensional UNI2 embeddings")

    rows, completed = [], 0
    for sample_name, entries in iter_lmdb_samples(data["lmdb"]):
        if args.limit_samples is not None and completed >= args.limit_samples:
            break
        sample_metadata = metadata.loc[metadata["sample_id"].astype(str) == sample_name].copy()
        if sample_metadata.empty:
            raise ValueError(f"{sample_name}: no rows in {data['metadata']}")
        destination = output_path(output_dir, sample_name)
        if destination.exists() and not args.overwrite:
            with np.load(destination, allow_pickle=False) as cached:
                count = len(cached["row_id"])
            print(f"[{sample_name}] already complete ({count:,} cells)")
        else:
            locations = locate_cells(sample_metadata, entries, image_shape, tile_size)
            tiles = {index: entry["image"] for index, entry in entries.items()}
            image = to_uint8_rgb(
                reconstruct_image(tiles, image_shape, tile_size),
                data.get("value_scale", 250),
            )
            uni, distilled = embed_crops(
                image,
                locations,
                encoder,
                encoder.eval_transforms,
                projector,
                device,
                int(embedding.get("batch_size", 128)),
                int(embedding.get("crop_size", 96)),
            )
            write_sample(destination, locations, uni, distilled)
            count = len(locations)
            print(f"[{sample_name}] wrote {count:,} cells -> {destination}")
        rows.append({"sample_id": sample_name, "n_cells": count, "path": str(destination)})
        completed += 1

    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    with open(output_dir / "resolved_config.yaml", "w") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    print(f"embedded {manifest['n_cells'].sum():,} cells from {len(manifest)} samples")


if __name__ == "__main__":
    main()
