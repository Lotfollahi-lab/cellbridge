"""Extract reusable UNI embeddings for out-of-distribution cells."""

import argparse
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from cellbridge.config import load_config
from cellbridge.evaluation.ood_data import (
    centered_crop,
    iter_lmdb_samples,
    locate_cells,
    reconstruct_image,
    resolve_tile_layout,
    to_uint8_rgb,
)


@dataclass(frozen=True)
class CropSpec:
    """Resolved pixel and physical geometry for one OOD crop cache."""

    mode: str
    source_size_px: int
    target_size_px: int
    source_mpp: float | None
    requested_fov_um: float | None
    actual_fov_um: float | None
    interpolation: str = "bilinear"
    padding: str = "constant_white"


def resolve_crop_spec(cfg):
    """Resolve a physical-FoV crop, retaining explicit legacy pixel support."""
    data = cfg["data"]
    embedding = cfg["embedding"]
    has_fov = embedding.get("fov_um") is not None
    has_pixels = embedding.get("crop_size") is not None
    if has_fov == has_pixels:
        raise ValueError("embedding must define exactly one of fov_um or crop_size")

    target_size = int(embedding.get("target_size", 224))
    if target_size < 1:
        raise ValueError("embedding.target_size must be positive")
    interpolation = str(embedding.get("interpolation", "bilinear")).lower()
    if interpolation != "bilinear":
        raise ValueError("only bilinear crop resizing is currently supported")

    source_mpp = data.get("source_mpp")
    source_mpp = None if source_mpp is None else float(source_mpp)
    if source_mpp is not None and source_mpp <= 0:
        raise ValueError("data.source_mpp must be positive")

    if has_fov:
        if source_mpp is None:
            raise ValueError("physical crops require data.source_mpp")
        requested_fov = float(embedding["fov_um"])
        if requested_fov <= 0:
            raise ValueError("embedding.fov_um must be positive")
        source_size = max(1, round(requested_fov / source_mpp))
        mode = "physical_fov"
    else:
        requested_fov = None
        source_size = int(embedding["crop_size"])
        if source_size < 1:
            raise ValueError("embedding.crop_size must be positive")
        mode = "legacy_pixels"

    actual_fov = None if source_mpp is None else source_size * source_mpp
    return CropSpec(
        mode=mode,
        source_size_px=source_size,
        target_size_px=target_size,
        source_mpp=source_mpp,
        requested_fov_um=requested_fov,
        actual_fov_um=actual_fov,
        interpolation=interpolation,
    )


def select_development_subset(metadata, settings):
    """Deterministically sample split-level or split-and-label strata."""
    if not settings:
        return metadata.copy()
    per_split = settings.get("max_cells_per_split")
    per_class = settings.get("max_cells_per_class_per_split")
    if (per_split is None) == (per_class is None):
        raise ValueError(
            "development_subset must define exactly one of max_cells_per_split "
            "or max_cells_per_class_per_split"
        )
    maximum = int(per_split if per_split is not None else per_class)
    seed = int(settings.get("seed", 0))
    if maximum < 1:
        raise ValueError("development_subset cap must be positive")
    required = {"label", "split_id", "row_id"}
    if not required <= set(metadata):
        raise ValueError(
            "development_subset requires metadata columns "
            f"{sorted(required - set(metadata))}"
        )

    requested_splits = settings.get("splits")
    working = metadata.copy()
    working["_subset_split"] = working["split_id"].replace({"valid": "val"})
    if requested_splits is not None:
        requested_splits = {
            str(value).replace("valid", "val") for value in requested_splits
        }
        unknown = requested_splits - set(working["_subset_split"].astype(str))
        if unknown:
            raise ValueError(
                f"development_subset requests absent splits {sorted(unknown)}"
            )
        working = working.loc[working["_subset_split"].isin(requested_splits)]

    rng = np.random.default_rng(seed)
    selected = []
    strata = ["_subset_split"]
    if per_class is not None:
        strata.append("label")
    groups = working.groupby(strata, sort=True, dropna=False)
    for _, rows in groups:
        indices = rows.index.to_numpy()
        if len(indices) > maximum:
            indices = rng.choice(indices, size=maximum, replace=False)
        selected.extend(indices.tolist())
    return metadata.loc[sorted(selected)].copy()


def crop_valid_fraction(image_shape, locations, crop_size):
    """Fraction of each requested crop covered by real rather than padded pixels."""
    height, width = map(int, image_shape[:2])
    size = int(crop_size)
    fractions = []
    for row in locations.itertuples():
        x0 = int(round(row.x)) - size // 2
        y0 = int(round(row.y)) - size // 2
        valid_width = max(0, min(x0 + size, width) - max(x0, 0))
        valid_height = max(0, min(y0 + size, height) - max(y0, 0))
        fractions.append(valid_width * valid_height / float(size * size))
    return np.asarray(fractions, dtype=np.float32)


def embed_crops(image, locations, encoder, transform, device, batch_size, crop_spec):
    import torch

    raw = []
    with torch.inference_mode():
        for start in range(0, len(locations), batch_size):
            rows = locations.iloc[start : start + batch_size]
            batch = [
                transform(
                    Image.fromarray(
                        centered_crop(image, row.x, row.y, crop_spec.source_size_px)
                    ).resize(
                        (crop_spec.target_size_px, crop_spec.target_size_px),
                        Image.Resampling.BILINEAR,
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
            raw.append(uni.float().cpu().numpy())
    return np.concatenate(raw)


def output_path(output_dir, sample_name):
    return output_dir / "samples" / f"{sample_name.removesuffix('.hdf')}.npz"


def write_sample(path, locations, uni, valid_fraction):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(
        temporary,
        row_id=locations["row_id"].to_numpy(np.int64),
        sample_id=locations["sample_id"].to_numpy(dtype=str),
        tile_id=locations["tile_id"].to_numpy(dtype=str),
        cell_id=locations["cell_id"].to_numpy(np.int32),
        label=locations["label"].to_numpy(dtype=str),
        class_id=locations["class_id"].to_numpy(np.int16),
        split=locations["split_id"].to_numpy(dtype=str),
        x=locations["x"].to_numpy(np.float32),
        y=locations["y"].to_numpy(np.float32),
        crop_valid_fraction=np.asarray(valid_fraction, dtype=np.float32),
        uni_emb=uni.astype(np.float16),
    )
    os.replace(temporary, path)


def complete_manifest(output_dir, metadata, allow_unlabeled=False):
    path = output_dir / "manifest.csv"
    if not path.exists():
        return None
    manifest = pd.read_csv(path)
    required = {"sample_id", "n_cells", "path"}
    if not required.issubset(manifest):
        return None
    expected = metadata.groupby("sample_id").size().astype(int).to_dict()
    observed = dict(
        zip(manifest["sample_id"].astype(str), manifest["n_cells"].astype(int))
    )
    files_ok = all(Path(path).exists() for path in manifest["path"])
    if allow_unlabeled:
        # Un-locatable cells are dropped, so a sample's embedded count may be
        # below its metadata count; require the same samples and no over-count.
        counts_ok = set(observed) == set(expected) and all(
            observed[sample] <= expected[sample] for sample in observed
        )
        return manifest if counts_ok and files_ok else None
    if observed != expected or not files_ok:
        return None
    return manifest


def embedding_signature(cfg):
    embedding = cfg["embedding"]
    return {
        "data": cfg["data"],
        "embedding": {
            "encoder": embedding.get("encoder", "uni_v2"),
            "crop": asdict(resolve_crop_spec(cfg)),
            "development_subset": embedding.get("development_subset"),
            "output_dir": embedding["output_dir"],
        },
    }


def validate_existing_signature(output_dir, cfg):
    path = output_dir / "embedding_config.yaml"
    if not path.exists():
        return
    with open(path) as handle:
        existing = yaml.safe_load(handle)
    if existing != embedding_signature(cfg):
        raise ValueError(
            f"{output_dir} was created with a different encoder, crop, or dataset; "
            "use a new embedding.output_dir"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)

    data = cfg["data"]
    embedding = cfg["embedding"]
    image_shape = data.get("image_shape")
    image_shape = None if image_shape is None else tuple(image_shape)
    tile_size = int(data.get("tile_size", 224))
    allow_unlabeled = bool(data.get("allow_unlabeled_samples", False))
    output_dir = Path(embedding["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(data["metadata"])
    metadata["row_id"] = np.arange(len(metadata))
    subset_settings = embedding.get("development_subset")
    metadata = select_development_subset(metadata, subset_settings)
    crop_spec = resolve_crop_spec(cfg)
    validate_existing_signature(output_dir, cfg)
    if not args.overwrite and args.limit_samples is None:
        manifest = complete_manifest(output_dir, metadata, allow_unlabeled)
        if manifest is not None:
            print(
                f"UNI cache already complete: {manifest['n_cells'].sum():,} cells "
                f"from {len(manifest)} samples -> {output_dir}"
            )
            return

    import torch
    from trident.patch_encoder_models import encoder_factory

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = encoder_factory(embedding.get("encoder", "uni_v2")).to(device).eval()

    rows, completed = [], 0
    for sample_name, entries in iter_lmdb_samples(data["lmdb"]):
        if args.limit_samples is not None and completed >= args.limit_samples:
            break
        sample_metadata = metadata.loc[
            metadata["sample_id"].astype(str) == sample_name
        ].copy()
        if sample_metadata.empty:
            if allow_unlabeled or subset_settings:
                continue
            raise ValueError(f"{sample_name}: no rows in {data['metadata']}")
        destination = output_path(output_dir, sample_name)
        layout = resolve_tile_layout(entries, image_shape, tile_size)
        if destination.exists() and not args.overwrite:
            with np.load(destination, allow_pickle=False) as cached:
                count = len(cached["row_id"])
            print(f"[{sample_name}] already complete ({count:,} cells)")
        else:
            locations = locate_cells(
                sample_metadata,
                entries,
                image_shape,
                tile_size,
                skip_missing=allow_unlabeled,
                layout=layout,
            )
            skipped = len(sample_metadata) - len(locations)
            if locations.empty:
                print(
                    f"[{sample_name}] no locatable cells ({skipped} skipped); skipped"
                )
                continue
            tiles = {index: entry["image"] for index, entry in entries.items()}
            image = to_uint8_rgb(
                reconstruct_image(tiles, image_shape, tile_size, layout=layout),
                data.get("value_scale", 250),
            )
            uni = embed_crops(
                image,
                locations,
                encoder,
                encoder.eval_transforms,
                device,
                int(embedding.get("batch_size", 128)),
                crop_spec,
            )
            valid_fraction = crop_valid_fraction(
                image.shape, locations, crop_spec.source_size_px
            )
            write_sample(destination, locations, uni, valid_fraction)
            count = len(locations)
            note = f" ({skipped} unlocatable skipped)" if skipped else ""
            print(f"[{sample_name}] wrote {count:,} cells -> {destination}{note}")
        with np.load(destination, allow_pickle=False) as cached:
            if "crop_valid_fraction" not in cached:
                raise ValueError(
                    f"{destination}: legacy cache lacks crop provenance; use --overwrite"
                )
            mean_valid = float(cached["crop_valid_fraction"].mean())
            padded = int(np.count_nonzero(cached["crop_valid_fraction"] < 1.0))
        rows.append(
            {
                "sample_id": sample_name,
                "n_cells": count,
                "image_height": layout.image_shape[0],
                "image_width": layout.image_shape[1],
                "grid_rows": len(layout.geometries) // layout.grid_width,
                "grid_columns": layout.grid_width,
                "layout_mode": layout.mode,
                "layout_seam_score": layout.seam_score,
                "n_padded_crops": padded,
                "mean_crop_valid_fraction": mean_valid,
                "path": str(destination),
            }
        )
        completed += 1

    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    if subset_settings:
        summary = (
            metadata.groupby(["split_id", "label"], dropna=False)
            .size()
            .rename("n_selected")
            .reset_index()
        )
        summary.to_csv(output_dir / "development_subset_summary.csv", index=False)
    with open(output_dir / "embedding_config.yaml", "w") as handle:
        yaml.safe_dump(embedding_signature(cfg), handle, sort_keys=False)
    print(f"embedded {manifest['n_cells'].sum():,} cells from {len(manifest)} samples")


if __name__ == "__main__":
    main()
