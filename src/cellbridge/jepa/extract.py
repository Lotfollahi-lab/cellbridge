"""Extract target-specific JEPA embeddings from visible spatial context."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

from cellbridge.cohorts import cohort_in_split, samples_in_split
from cellbridge.config import load_config
from cellbridge.jepa.artifacts import (
    checkpoint_digest as _checkpoint_digest,
    current_output,
    feature_signature,
    write_output,
)
from cellbridge.jepa.data import load_context_section, sample_context_rows
from cellbridge.jepa.model import TargetNormalizer, load_model_state


@dataclass(frozen=True)
class SamplingSettings:
    n_context: int
    context_sampling: str
    context_radius_um: float
    guard_radius_um: float
    coordinate_normalization: str
    sampling_seed: int


@dataclass(frozen=True)
class ContextBatch:
    embeddings: np.ndarray
    coordinates: np.ndarray
    target_coordinates: np.ndarray
    mask: np.ndarray
    rows: np.ndarray
    counts: np.ndarray


def sampling_settings(checkpoint):
    """Read the spatial convention recorded by the training checkpoint."""
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("JEPA checkpoint is missing its resolved config")
    values = config.get("sampling", config.get("data"))
    if not isinstance(values, dict):
        raise ValueError("JEPA checkpoint config is missing sampling settings")

    n_context = values.get("n_context", values.get("context_cells"))
    strategy = values.get("context_sampling")
    radius = values.get("context_radius_um")
    guard = values.get("guard_um", values.get("guard_radius_um"))
    normalization = values.get("coordinate_normalization")
    seed = values.get("sampling_seed")
    if None in (n_context, strategy, radius, guard, normalization, seed):
        raise ValueError(
            "JEPA checkpoint sampling settings must record context count, "
            "context radius, guard radius, coordinate normalization, and sampling seed"
        )
    if int(n_context) < 1:
        raise ValueError("context count must be positive")
    if strategy != "uniform_within_radius":
        raise ValueError(f"unsupported context sampling {strategy!r}")
    if float(radius) <= 0 or not 0 <= float(guard) < float(radius):
        raise ValueError("sampling radii must satisfy 0 <= guard < context radius")
    if int(seed) < 0:
        raise ValueError("sampling seed must be non-negative")
    if normalization != "fixed_radius":
        raise ValueError(f"unsupported coordinate normalization {normalization!r}")
    return SamplingSettings(
        int(n_context),
        str(strategy),
        float(radius),
        float(guard),
        str(normalization),
        int(seed),
    )


def load_checkpoint(path, device):
    """Load the JEPA predictor, target normalizer, and checkpoint metadata."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path}: expected a dictionary checkpoint")
    model = load_model_state(checkpoint, device=device).eval()
    state = checkpoint.get("target_normalizer")
    if not isinstance(state, dict) or not {"mean", "scale"}.issubset(state):
        raise ValueError(f"{path}: checkpoint is missing target_normalizer state")
    normalizer = TargetNormalizer(state["mean"], state["scale"]).to(device).eval()
    sampling_settings(checkpoint)
    return model, normalizer, checkpoint


def fit_spatial_index(coordinates):
    """Build the two-dimensional radius index used for local context lookup."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape [cells, 2]")
    if len(coordinates) < 2 or not np.isfinite(coordinates).all():
        raise ValueError("a section needs at least two finite cell coordinates")
    return NearestNeighbors(algorithm="kd_tree", n_jobs=-1).fit(coordinates)


def make_context_batch(
    uni_embeddings,
    coordinates,
    target_rows,
    spatial_index,
    settings,
):
    """Build local contexts while excluding every target and its guard band."""
    uni_embeddings = np.asarray(uni_embeddings)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    target_rows = np.asarray(target_rows, dtype=int)
    distances, rows = spatial_index.radius_neighbors(
        coordinates[target_rows],
        radius=settings.context_radius_um,
        return_distance=True,
        sort_results=True,
    )

    batch_size = len(target_rows)
    context = np.zeros(
        (batch_size, settings.n_context, uni_embeddings.shape[1]), dtype=np.float32
    )
    relative = np.zeros((batch_size, settings.n_context, 2), dtype=np.float32)
    selected_rows = np.full((batch_size, settings.n_context), -1, dtype=np.int64)
    mask = np.zeros((batch_size, settings.n_context), dtype=bool)
    counts = np.zeros(batch_size, dtype=np.int16)

    for batch_row, target_row in enumerate(target_rows):
        rng = np.random.default_rng([settings.sampling_seed, int(target_row)])
        chosen = sample_context_rows(
            rows[batch_row],
            distances[batch_row],
            target_row=target_row,
            n_context=settings.n_context,
            guard_radius_um=settings.guard_radius_um,
            rng=rng,
        )
        if not len(chosen):
            continue

        count = len(chosen)
        context[batch_row, :count] = uni_embeddings[chosen]
        relative[batch_row, :count] = (
            (coordinates[chosen] - coordinates[target_row]) / settings.context_radius_um
        ).astype(np.float32)
        selected_rows[batch_row, :count] = chosen
        mask[batch_row, :count] = True
        counts[batch_row] = count

    target_coordinates = np.zeros((batch_size, 1, 2), dtype=np.float32)
    return ContextBatch(
        context,
        relative,
        target_coordinates,
        mask,
        selected_rows,
        counts,
    )


def predict_section(
    model,
    normalizer,
    uni_embeddings,
    coordinates,
    settings,
    *,
    batch_size,
    device,
):
    """Predict rows with a complete context; incomplete rows remain zero."""
    uni_embeddings = np.asarray(uni_embeddings)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if (
        uni_embeddings.ndim != 2
        or len(uni_embeddings) != len(coordinates)
        or not np.issubdtype(uni_embeddings.dtype, np.floating)
        or not np.isfinite(uni_embeddings).all()
    ):
        raise ValueError("UNI embeddings and coordinates must be aligned 2D arrays")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")

    spatial_index = fit_spatial_index(coordinates)
    outputs, context_counts = [], []
    for start in range(0, len(uni_embeddings), int(batch_size)):
        target_rows = np.arange(start, min(start + int(batch_size), len(coordinates)))
        batch = make_context_batch(
            uni_embeddings,
            coordinates,
            target_rows,
            spatial_index,
            settings,
        )
        # Training always uses exactly ``n_context`` visible cells. Do not feed
        # shorter, padded neighborhoods to a model that never saw them.
        valid = batch.counts == settings.n_context
        batch_output = np.zeros(
            (len(target_rows), normalizer.target_dim), dtype=np.float32
        )
        if valid.any():
            with torch.inference_mode():
                predicted = model.predict(
                    torch.from_numpy(batch.embeddings[valid]).to(device),
                    torch.from_numpy(batch.coordinates[valid]).to(device),
                    torch.from_numpy(batch.target_coordinates[valid]).to(device),
                    torch.from_numpy(batch.mask[valid]).to(device),
                )
                if predicted.ndim != 3 or predicted.shape[1] != 1:
                    raise ValueError(
                        "JEPA predictor must return shape [batch, 1, target_dim]"
                    )
                predicted = normalizer.inverse_transform(predicted[:, 0])
                batch_output[valid] = predicted.float().cpu().numpy()
        outputs.append(batch_output)
        context_counts.append(batch.counts)
    return np.concatenate(outputs), np.concatenate(context_counts)


def sample_sources(cfg, sample_id):
    """Resolve HEST artifacts and the pixel-to-micron scale for one section."""
    sample_dir = Path(cfg["data_root"]) / sample_id
    manifest_path = cfg.get("manifest_path")
    if not manifest_path:
        raise ValueError("JEPA extraction requires manifest_path for pixel_size_um")
    manifest = pd.read_csv(manifest_path)
    required = {"id", "pixel_size_um"}
    if not required.issubset(manifest):
        raise ValueError(
            f"{manifest_path}: missing columns {sorted(required - set(manifest))}"
        )
    matches = manifest.loc[manifest["id"].astype(str) == str(sample_id)]
    if len(matches) != 1:
        raise ValueError(f"manifest must contain exactly one row for {sample_id}")
    row = matches.iloc[0]
    if "output_dir" in matches and pd.notna(row["output_dir"]):
        sample_dir = Path(row["output_dir"])
    pixel_size_um = float(row["pixel_size_um"])
    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError(f"{sample_id}: pixel_size_um must be positive and finite")
    return (
        sample_dir / f"{sample_id}_uni2.npz",
        sample_dir / f"{sample_id}_terra.npz",
        Path(manifest_path),
        pixel_size_um,
    )


def load_section(cfg, sample_id):
    """Load and cell-ID-align UNI embeddings with native-micron coordinates."""
    uni_path, terra_path, manifest_path, _ = sample_sources(cfg, sample_id)
    section = load_context_section(
        cfg["data_root"],
        cfg["manifest_path"],
        sample_id,
        image_key=cfg.get("uni_key", "uni_emb"),
    )
    return (
        section.image,
        section.coordinates_um,
        section.cell_ids,
        (uni_path, terra_path, manifest_path),
    )


def configured_samples(cfg, override=None):
    """Resolve CLI samples, the fast label set, or all label-test sections."""
    fast_path = cfg.get("eval", {}).get("label_test_fast_path")
    configured = (
        cohort_in_split(fast_path, cfg["splits_path"], "label_test")
        if fast_path
        else samples_in_split(cfg["splits_path"], "label_test")
    )
    if not override:
        return configured
    requested = [value.strip() for value in override.split(",") if value.strip()]
    invalid = sorted(set(requested) - set(configured))
    if not requested or len(requested) != len(set(requested)) or invalid:
        raise ValueError(f"invalid configured samples: {invalid or requested}")
    return requested


def extract_sample(
    cfg,
    sample_id,
    model,
    normalizer,
    settings,
    checkpoint_digest,
    output_dir,
    device,
    overwrite=False,
):
    """Extract or reuse one complete tissue section."""
    uni, coordinates, cell_ids, sources = load_section(cfg, sample_id)
    signature = feature_signature(checkpoint_digest, sources, settings)
    destination = Path(output_dir) / f"{sample_id}.npz"
    if not overwrite:
        count = current_output(destination, signature)
        if count is not None:
            print(f"[{sample_id}] current ({count:,} cells)")
            return count, destination, signature

    embeddings, counts = predict_section(
        model,
        normalizer,
        uni,
        coordinates,
        settings,
        batch_size=cfg.get("eval", {}).get("batch_size", 128),
        device=device,
    )
    valid = counts == settings.n_context
    if not valid.any():
        raise ValueError(f"{sample_id}: no cells have a complete spatial context")
    write_output(
        destination,
        sample_id,
        cell_ids[valid],
        embeddings[valid],
        counts[valid],
        signature,
    )
    skipped = int((~valid).sum())
    print(
        f"[{sample_id}] wrote {int(valid.sum()):,} contextual embeddings"
        f" (skipped {skipped:,} cells without a complete context)"
    )
    return int(valid.sum()), destination, signature


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--samples", help="comma-separated section IDs")
    parser.add_argument("--device")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    checkpoint_path = Path(args.checkpoint or Path(cfg["checkpoint_dir"]) / "best.pt")
    output_dir = Path(args.output_dir or Path(cfg["output_dir"]) / "features")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, normalizer, checkpoint = load_checkpoint(checkpoint_path, device)
    settings = sampling_settings(checkpoint)
    digest = _checkpoint_digest(checkpoint_path)

    rows = []
    for sample_id in configured_samples(cfg, args.samples):
        count, path, signature = extract_sample(
            cfg,
            sample_id,
            model,
            normalizer,
            settings,
            digest,
            output_dir,
            device,
            args.overwrite,
        )
        rows.append(
            {
                "sample_id": sample_id,
                "n_cells": count,
                "path": str(path.resolve()),
                "signature": signature,
            }
        )
    if args.samples is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_dir / "manifest.csv", index=False)
        print(f"wrote {output_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
