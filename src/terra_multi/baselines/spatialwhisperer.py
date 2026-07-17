"""Extract and probe SpatialWhisperer H&E embeddings.

The image model is imported lazily: ``check`` and ``probe`` run in the regular
TERRA environment, while extraction runs in SpatialWhisperer's own Pixi
environment.
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from terra_multi.baselines.artifacts import (
    feature_file_is_current,
    feature_signature,
    file_identity,
    git_identity,
    path_identity,
    stable_json,
    stable_signature,
    summarize_feature_file,
    write_feature_file,
    write_json,
    write_table,
)
from terra_multi.baselines.evaluation import (
    expected_hest_cell_ids,
    finalize_hest_manifest,
    finalize_phenocell_manifest,
    hest_feature_path,
    hest_source_identity,
    label_test_fast_samples,
    phenocell_cell_ids,
    phenocell_feature_path,
    probe_all,
)
from terra_multi.evaluation.phenocell import validate_official_split
from terra_multi.evaluation.phenocell_data import (
    centered_crop,
    iter_lmdb_samples,
    locate_cells,
    reconstruct_image,
    to_uint8_rgb,
)

BASELINE_NAME = "spatialwhisperer"
HEST_DATASET_NAME = "hest/label_test_fast"
PHENOCELL_DATASET_NAME = "phenocell"


def load_config(path) -> dict:
    with open(path) as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    required = {"baseline", "model", "output_dir", "hest", "phenocell"}
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"config is missing sections {sorted(missing)}")
    baseline = cfg["baseline"]
    if baseline.get("name") != BASELINE_NAME:
        raise ValueError(f"baseline.name must be {BASELINE_NAME!r}")
    if int(baseline.get("embedding_dim", 0)) != 512:
        raise ValueError("the released SpatialWhisperer projection is 512-D")
    if baseline.get("normalized") is not True:
        raise ValueError("SpatialWhisperer embeddings must be L2-normalized")
    if str(baseline.get("save_dtype", "float16")) != "float16":
        raise ValueError("the baseline artifact contract currently requires float16")

    repository = Path(cfg["model"]["repository"])
    if not repository.is_dir():
        raise FileNotFoundError(repository)
    if not (repository / "pixi.toml").is_file():
        raise FileNotFoundError(repository / "pixi.toml")
    if not cfg["model"].get("hf_repo"):
        raise ValueError("model.hf_repo is required")

    hest_crop = cfg["hest"].get("crop", {})
    if int(hest_crop.get("target_size", 0)) != 224:
        raise ValueError("SpatialWhisperer requires 224x224 context patches")
    if float(hest_crop.get("target_pixel_size_um", 0)) != 0.5:
        raise ValueError("the native HEST baseline requires 0.5 um/pixel")
    if cfg["hest"].get("probe", {}).get("pca", False):
        raise ValueError("projected SpatialWhisperer embeddings must use HEST pca: false")

    phenocell_crop = cfg["phenocell"].get("crop", {})
    if int(phenocell_crop.get("source_size", 0)) != 96:
        raise ValueError("the comparable PhenoCell protocol requires source_size: 96")
    if int(phenocell_crop.get("target_size", 0)) != 224:
        raise ValueError("SpatialWhisperer requires PhenoCell target_size: 224")
    if cfg["phenocell"].get("probe", {}).get("pca", False):
        raise ValueError(
            "projected SpatialWhisperer embeddings must use PhenoCell pca: false"
        )


def output_root(cfg, override=None) -> Path:
    return Path(override or cfg["output_dir"])


def _snapshot_revision(path, fallback):
    parts = Path(path).parts
    try:
        index = parts.index("snapshots")
    except ValueError:
        return str(fallback)
    return parts[index + 1] if index + 1 < len(parts) else str(fallback)


def _precision_dtype(torch, precision):
    choices = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if str(precision).lower() not in choices:
        raise ValueError(f"unsupported precision {precision!r}")
    return choices[str(precision).lower()]


class SpatialWhispererEncoder:
    """Small image-only adapter around the released trimodal checkpoint."""

    def __init__(
        self,
        pl_model,
        image_processor,
        device,
        *,
        expected_dim,
        precision,
        identity,
    ):
        import torch

        self.pl_model = pl_model
        self.model = pl_model.model
        self.transform = image_processor.transform_context
        self.device = torch.device(device)
        self.expected_dim = int(expected_dim)
        self.precision_name = str(precision)
        self.precision = _precision_dtype(torch, precision)
        self.identity = identity
        self.model_signature = stable_signature(identity)

        image_config = self.model.image_model.config
        if not bool(getattr(image_config, "context_model", True)):
            raise ValueError("the loaded checkpoint has no context image model")
        if bool(getattr(image_config, "cell_level_model", False)):
            raise ValueError(
                "this runner targets the released context-only checkpoint; "
                "a cell-level checkpoint needs an explicit 56x56 crop contract"
            )
        projection_dim = int(getattr(self.model, "projection_dim", -1))
        if projection_dim != self.expected_dim:
            raise ValueError(
                f"checkpoint projection_dim={projection_dim}, expected {self.expected_dim}"
            )

    def encode(self, images):
        """Return the normalized second output, never the raw UNI2 feature."""
        import torch

        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 224, 224):
            raise ValueError(
                f"SpatialWhisperer input must be (B,3,224,224), got {tuple(images.shape)}"
            )
        images = images.to(self.device, non_blocking=True)
        use_autocast = self.device.type == "cuda" and self.precision in {
            torch.float16,
            torch.bfloat16,
        }
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=self.precision,
            enabled=use_autocast,
        ):
            _, projected = self.model.get_image_features(
                patches_ctx=images,
                normalize_embeds=True,
            )
        projected = projected.float()
        if projected.ndim != 2 or projected.shape[1] != self.expected_dim:
            raise ValueError(
                f"expected (B,{self.expected_dim}) embeddings, got {tuple(projected.shape)}"
            )
        if not torch.isfinite(projected).all():
            raise ValueError("SpatialWhisperer returned non-finite embeddings")
        norms = torch.linalg.vector_norm(projected, dim=1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=2e-2, rtol=2e-2):
            raise ValueError("SpatialWhisperer projected embeddings are not normalized")
        return projected.cpu().numpy()


def load_encoder(cfg, device=None) -> SpatialWhispererEncoder:
    """Download pinned assets, load the model, and capture exact provenance."""
    import torch
    import spatialwhisperer
    from huggingface_hub import hf_hub_download
    from spatialwhisperer.utils.model_io import (
        ensure_geneformer_weights,
        ensure_uni2_weights,
        load_cellwhisperer_model,
    )

    repository = Path(cfg["model"]["repository"]).resolve()
    installed_repository = Path(spatialwhisperer.__file__).resolve().parents[2]
    if installed_repository != repository:
        raise ValueError(
            "the active environment imports SpatialWhisperer from "
            f"{installed_repository}, but config points to {repository}"
        )
    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    geneformer_dir = ensure_geneformer_weights()
    uni_dir = ensure_uni2_weights()
    uni_weights = Path(uni_dir) / "pytorch_model.bin"
    if not uni_weights.is_file():
        raise FileNotFoundError(
            f"{uni_weights} is required by SpatialWhisperer's UNI2 loader. "
            "Accept the UNI2 terms, set HF_TOKEN, and rerun prepare."
        )

    model_cfg = cfg["model"]
    hf_repo = model_cfg["hf_repo"]
    requested_revision = str(model_cfg.get("revision", "main"))
    checkpoint = hf_hub_download(
        repo_id=hf_repo,
        filename=model_cfg.get("checkpoint_filename", "spatialwhisperer.ckpt"),
        revision=requested_revision,
        token=token,
    )
    resolved_revision = _snapshot_revision(checkpoint, requested_revision)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    pl_model, _, _, image_processor = load_cellwhisperer_model(
        model_path=checkpoint,
        device=device,
    )
    projection_dim = int(getattr(pl_model.model, "projection_dim", -1))
    identity = {
        "baseline": BASELINE_NAME,
        "spatialwhisperer_repository": git_identity(repository),
        "huggingface_repo": hf_repo,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "checkpoint": file_identity(checkpoint, checksum=True),
        "uni2_weights": file_identity(uni_weights),
        "geneformer_weights": path_identity(geneformer_dir),
        "terra_multi_integration": [
            file_identity(Path(__file__), checksum=True),
            file_identity(Path(__file__).with_name("artifacts.py"), checksum=True),
            file_identity(
                Path(__file__).resolve().parents[1] / "lora" / "data.py",
                checksum=True,
            ),
            file_identity(
                Path(__file__).resolve().parents[1]
                / "evaluation"
                / "phenocell_data.py",
                checksum=True,
            ),
        ],
        "projection_dim": projection_dim,
        "normalized": True,
        "requested_precision": model_cfg.get("precision", "bf16"),
        "effective_precision": (
            model_cfg.get("precision", "bf16") if device.type == "cuda" else "fp32"
        ),
    }
    return SpatialWhispererEncoder(
        pl_model,
        image_processor,
        device,
        expected_dim=cfg["baseline"]["embedding_dim"],
        precision=model_cfg.get("precision", "bf16"),
        identity=identity,
    )


def prepare_model(cfg, root, device=None):
    """Load once, run a shape/norm smoke test, and save resolved provenance."""
    import torch

    encoder = load_encoder(cfg, device)
    image = Image.new("RGB", (224, 224), "white")
    features = encoder.encode(torch.stack([encoder.transform(image)]))
    payload = {
        "model_signature": encoder.model_signature,
        "model": encoder.identity,
        "smoke_embedding_shape": list(features.shape),
        "smoke_embedding_norm": float(np.linalg.norm(features[0])),
    }
    write_json(Path(root) / "model_provenance.json", payload)
    print(
        f"SpatialWhisperer ready: {features.shape[1]}-D, "
        f"revision={encoder.identity['resolved_revision']}, "
        f"signature={encoder.model_signature[:12]}"
    )
    return payload


def _hest_dataset_cfg(cfg):
    hest = cfg["hest"]
    return {
        "data_root": hest["data_root"],
        "manifest_path": hest["manifest_path"],
        "splits_path": hest["splits_path"],
        "sampling_seed": int(hest.get("sampling_seed", 0)),
        "terra_key": hest.get("terra_key", "cell_emb"),
        "max_open_wsis": int(hest.get("max_open_wsis", 64)),
        "crop": hest["crop"],
        "eval": {"label_test_fast_path": hest["label_test_fast_path"]},
    }


def _dataset_cell_ids(dataset):
    samples = getattr(dataset, "_samples", None)
    if samples is None:
        raise TypeError("unexpected HEST dataset contract: no ordered sample cells")
    return np.concatenate([_sample["cell_ids"] for _sample in samples]).astype(str)


def _infer_hest(dataset, encoder, extraction_cfg, storage_dir):
    import torch

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(extraction_cfg.get("batch_size", 64)),
        shuffle=False,
        num_workers=int(extraction_cfg.get("num_workers", 8)),
        pin_memory=encoder.device.type == "cuda",
    )
    features = np.lib.format.open_memmap(
        Path(storage_dir) / "embeddings.npy",
        mode="w+",
        dtype=np.float16,
        shape=(len(dataset), encoder.expected_dim),
    )
    cell_parts = []
    offset = 0
    for batch in loader:
        values = encoder.encode(batch["image"])
        stop = offset + len(values)
        features[offset:stop] = values
        cell_parts.append(_as_batch_strings(batch["cell_id"]))
        offset = stop
    if offset != len(dataset):
        raise ValueError(f"loader yielded {offset} rows for {len(dataset)} cells")
    features.flush()
    return np.concatenate(cell_parts), features


def _as_batch_strings(values):
    return np.asarray(values).astype(str)


def extract_hest(
    cfg,
    encoder,
    root,
    *,
    samples=None,
    max_cells_per_sample=None,
    overwrite=False,
):
    """Export one or all five fast-HEST samples from native physical crops."""
    from terra_multi.lora.data import build_label_test_dataset

    hest_cfg = cfg["hest"]
    expected = label_test_fast_samples(hest_cfg)
    if samples is None:
        wanted = expected
    else:
        requested = [str(sample) for sample in samples]
        unknown = sorted(set(requested) - set(expected))
        if unknown:
            raise ValueError(f"not label_test_fast samples: {unknown}")
        wanted = [sample for sample in expected if sample in set(requested)]
    if max_cells_per_sample is not None and int(max_cells_per_sample) < 1:
        raise ValueError("max_cells_per_sample must be positive")

    output_dir = Path(root) / "hest_label_test_fast"
    rows = []
    for sample_id in wanted:
        dataset = build_label_test_dataset(
            _hest_dataset_cfg(cfg),
            encoder.transform,
            max_cells_per_sample=max_cells_per_sample,
            sample_ids=[sample_id],
        )
        expected_ids = _dataset_cell_ids(dataset)
        settings = {
            "sample_id": sample_id,
            "context_size": int(hest_cfg["crop"]["target_size"]),
            "target_pixel_size_um": float(
                hest_cfg["crop"]["target_pixel_size_um"]
            ),
            "normalization": "ImageNet mean/std from SpatialWhisperer processor",
            "normalized_embedding": True,
            "embedding_dim": encoder.expected_dim,
            "max_cells_per_sample": max_cells_per_sample,
        }
        signature = feature_signature(
            HEST_DATASET_NAME,
            encoder.identity,
            hest_source_identity(hest_cfg, sample_id),
            settings,
        )
        destination = hest_feature_path(output_dir, sample_id)
        try:
            if not overwrite and feature_file_is_current(
                destination,
                signature,
                expected_cells=len(expected_ids),
                expected_dim=encoder.expected_dim,
                expected_cell_ids=expected_ids,
            ):
                summary = summarize_feature_file(
                    destination, expected_dim=encoder.expected_dim
                )
                print(f"[HEST/{sample_id}] current ({summary.n_cells:,} cells)")
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with TemporaryDirectory(dir=destination.parent) as temporary:
                    cell_ids, embeddings = _infer_hest(
                        dataset,
                        encoder,
                        hest_cfg["extraction"],
                        temporary,
                    )
                    if not np.array_equal(cell_ids, expected_ids):
                        raise ValueError(f"{sample_id}: DataLoader changed cell row order")
                    write_feature_file(
                        destination,
                        cell_ids,
                        embeddings,
                        signature,
                        expected_dim=encoder.expected_dim,
                        metadata={
                            "baseline_name": np.asarray(BASELINE_NAME),
                            "dataset_name": np.asarray(HEST_DATASET_NAME),
                            "artifact_sample_id": np.asarray(sample_id),
                            "model_signature": np.asarray(encoder.model_signature),
                            "provenance_json": np.asarray(
                                stable_json(
                                    {
                                        "model": encoder.identity,
                                        "source": hest_source_identity(
                                            hest_cfg, sample_id
                                        ),
                                        "settings": settings,
                                    }
                                )
                            ),
                        },
                    )
                summary = summarize_feature_file(
                    destination, expected_dim=encoder.expected_dim
                )
                print(
                    f"[HEST/{sample_id}] wrote {summary.n_cells:,} cells -> "
                    f"{destination}"
                )
            rows.append(summary)
        finally:
            dataset.close()

    if samples is None and max_cells_per_sample is None:
        return finalize_hest_manifest(cfg, output_dir)
    partial = [
        {
            "sample_id": row.sample_id,
            "n_cells": row.n_cells,
            "n_features": row.n_features,
            "path": row.path,
            "signature": row.signature,
            "model_signature": row.model_signature,
        }
        for row in rows
    ]
    suffix = "-".join(wanted)
    write_table(output_dir / "features" / f"manifest.partial.{suffix}.csv", partial)
    return pd.DataFrame(partial)


def _phenocell_source_identity(phenocell_cfg):
    data = phenocell_cfg["data"]
    return {
        "lmdb": path_identity(data["lmdb"]),
        "metadata": file_identity(data["metadata"]),
    }


def _phenocell_embeddings(image, locations, encoder, extraction_cfg, crop_cfg):
    import torch

    batch_size = int(extraction_cfg.get("batch_size", 128))
    source_size = int(crop_cfg["source_size"])
    target_size = int(crop_cfg["target_size"])
    parts = []
    for start in range(0, len(locations), batch_size):
        rows = locations.iloc[start : start + batch_size]
        images = torch.stack(
            [
                encoder.transform(
                    Image.fromarray(
                        centered_crop(image, row.x, row.y, source_size)
                    ).resize(
                        (target_size, target_size), Image.Resampling.BILINEAR
                    )
                )
                for row in rows.itertuples()
            ]
        )
        parts.append(encoder.encode(images))
    if not parts:
        raise ValueError("cannot embed an empty PhenoCell sample")
    return np.concatenate(parts)


def extract_phenocell(
    cfg,
    encoder,
    root,
    *,
    samples=None,
    max_cells_per_sample=None,
    overwrite=False,
):
    """Export PhenoCell using the established 96-source-pixel field of view."""
    phenocell_cfg = cfg["phenocell"]
    data_cfg = phenocell_cfg["data"]
    metadata = pd.read_csv(data_cfg["metadata"])
    metadata["row_id"] = np.arange(len(metadata), dtype=np.int64)
    expected_samples = metadata["sample_id"].astype(str).drop_duplicates().tolist()
    if samples is None:
        wanted = expected_samples
    else:
        requested = [str(sample) for sample in samples]
        unknown = sorted(set(requested) - set(expected_samples))
        if unknown:
            raise ValueError(f"unknown PhenoCell samples: {unknown}")
        wanted = [sample for sample in expected_samples if sample in set(requested)]
    if max_cells_per_sample is not None and int(max_cells_per_sample) < 1:
        raise ValueError("max_cells_per_sample must be positive")

    output_dir = Path(root) / "phenocell"
    image_shape = tuple(data_cfg.get("image_shape", [1440, 1920]))
    tile_size = int(data_cfg.get("tile_size", 224))
    source_identity = _phenocell_source_identity(phenocell_cfg)
    observed_samples = set()
    processed_samples = set()
    summaries = []
    wanted_set = set(wanted)
    for sample_id, entries in iter_lmdb_samples(data_cfg["lmdb"]):
        observed_samples.add(sample_id)
        if sample_id not in wanted_set:
            continue
        sample_metadata = metadata.loc[
            metadata["sample_id"].astype(str) == sample_id
        ].copy()
        if max_cells_per_sample is not None:
            sample_metadata = sample_metadata.iloc[: int(max_cells_per_sample)].copy()
        expected_ids = phenocell_cell_ids(sample_metadata)
        settings = {
            "sample_id": sample_id,
            "image_shape": list(image_shape),
            "tile_size": tile_size,
            "value_scale": float(data_cfg.get("value_scale", 250)),
            "source_crop_size": int(phenocell_cfg["crop"]["source_size"]),
            "model_input_size": int(phenocell_cfg["crop"]["target_size"]),
            "resize": "PIL bilinear",
            "normalization": "ImageNet mean/std from SpatialWhisperer processor",
            "normalized_embedding": True,
            "embedding_dim": encoder.expected_dim,
            "max_cells_per_sample": max_cells_per_sample,
        }
        signature = feature_signature(
            PHENOCELL_DATASET_NAME,
            encoder.identity,
            source_identity,
            settings,
        )
        destination = phenocell_feature_path(output_dir, sample_id)
        required_metadata = (
            "row_id",
            "sample_id",
            "tile_id",
            "cell_id",
            "label",
            "class_id",
            "split",
            "x",
            "y",
        )
        if not overwrite and feature_file_is_current(
            destination,
            signature,
            expected_cells=len(sample_metadata),
            expected_dim=encoder.expected_dim,
            expected_cell_ids=expected_ids,
            required_metadata=required_metadata,
        ):
            summary = summarize_feature_file(
                destination, expected_dim=encoder.expected_dim
            )
            print(f"[PhenoCell/{sample_id}] current ({summary.n_cells:,} cells)")
        else:
            locations = locate_cells(sample_metadata, entries, image_shape, tile_size)
            image = to_uint8_rgb(
                reconstruct_image(
                    {index: entry["image"] for index, entry in entries.items()},
                    image_shape,
                    tile_size,
                ),
                data_cfg.get("value_scale", 250),
            )
            embeddings = _phenocell_embeddings(
                image,
                locations,
                encoder,
                phenocell_cfg["extraction"],
                phenocell_cfg["crop"],
            )
            write_feature_file(
                destination,
                phenocell_cell_ids(locations),
                embeddings,
                signature,
                expected_dim=encoder.expected_dim,
                metadata={
                    "baseline_name": np.asarray(BASELINE_NAME),
                    "dataset_name": np.asarray(PHENOCELL_DATASET_NAME),
                    "artifact_sample_id": np.asarray(sample_id),
                    "model_signature": np.asarray(encoder.model_signature),
                    "provenance_json": np.asarray(
                        stable_json(
                            {
                                "model": encoder.identity,
                                "source": source_identity,
                                "settings": settings,
                            }
                        )
                    ),
                    "row_id": locations["row_id"].to_numpy(np.int64),
                    "sample_id": locations["sample_id"].to_numpy(dtype=str),
                    "tile_id": locations["tile_id"].to_numpy(dtype=str),
                    "cell_id": locations["cell_id"].to_numpy(np.int32),
                    "label": locations["label"].to_numpy(dtype=str),
                    "class_id": locations["class_id"].to_numpy(np.int16),
                    "split": locations["split_id"].to_numpy(dtype=str),
                    "x": locations["x"].to_numpy(np.float32),
                    "y": locations["y"].to_numpy(np.float32),
                },
            )
            summary = summarize_feature_file(
                destination, expected_dim=encoder.expected_dim
            )
            print(
                f"[PhenoCell/{sample_id}] wrote {summary.n_cells:,} cells -> "
                f"{destination}"
            )
        summaries.append(summary)
        processed_samples.add(sample_id)
        if samples is not None and processed_samples == wanted_set:
            break

    missing_wanted = sorted(wanted_set - processed_samples)
    if missing_wanted:
        raise ValueError(f"requested PhenoCell samples are absent from LMDB: {missing_wanted}")

    observed_expected = set(expected_samples)
    if samples is None and observed_samples != observed_expected:
        missing = sorted(observed_expected - observed_samples)
        extra = sorted(observed_samples - observed_expected)
        raise ValueError(
            f"PhenoCell LMDB/metadata mismatch: missing={missing}, extra={extra}"
        )
    if samples is None and max_cells_per_sample is None:
        return finalize_phenocell_manifest(cfg, output_dir)
    partial = [
        {
            "sample_id": row.sample_id,
            "n_cells": row.n_cells,
            "n_features": row.n_features,
            "path": row.path,
            "signature": row.signature,
            "model_signature": row.model_signature,
        }
        for row in summaries
    ]
    suffix = "-".join(str(sample).removesuffix(".hdf") for sample in wanted)
    write_table(output_dir / "features" / f"manifest.partial.{suffix}.csv", partial)
    return pd.DataFrame(partial)


def check_inputs(cfg):
    """Validate paths, fixed cohorts, and split contracts without loading a model."""
    samples = label_test_fast_samples(cfg["hest"])
    for path in (
        cfg["hest"]["manifest_path"],
        cfg["hest"]["splits_path"],
        cfg["hest"]["label_test_fast_path"],
        cfg["phenocell"]["data"]["metadata"],
        cfg["phenocell"]["data"]["split"],
        cfg["phenocell"]["data"]["lmdb"],
    ):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    for sample_id in samples:
        expected_hest_cell_ids(cfg["hest"], sample_id)

    phenocell = pd.read_csv(cfg["phenocell"]["data"]["metadata"])
    observed = phenocell[["sample_id", "split_id"]].rename(
        columns={"split_id": "split"}
    )
    validate_official_split(observed, cfg["phenocell"]["data"]["split"])
    repository = git_identity(cfg["model"]["repository"])
    print(f"SpatialWhisperer repository: {repository['commit']}")
    print(f"Fast HEST: {len(samples)} samples ({', '.join(samples)})")
    print(
        f"PhenoCell: {len(phenocell):,} cells across "
        f"{phenocell['sample_id'].nunique()} samples"
    )
    print("Embedding contract: normalized 512-D projection; PCA disabled")
    print(
        "WARNING: the released checkpoint was trained on HEST-1K. Treat fast "
        "HEST as pretraining-exposed unless exact training IDs prove otherwise; "
        "PhenoCell is the clean OOD result."
    )


def _parse_samples(value):
    if not value:
        return None
    samples = [item.strip() for item in value.split(",") if item.strip()]
    if not samples or len(samples) != len(set(samples)):
        raise ValueError("--samples must be a nonempty, unique comma-separated list")
    return samples


def _add_common(subparser, *, output=True):
    subparser.add_argument("--config", required=True)
    if output:
        subparser.add_argument("--output-dir")


def _add_extraction(subparser):
    _add_common(subparser)
    subparser.add_argument("--device")
    subparser.add_argument("--overwrite", action="store_true")
    subparser.add_argument("--samples", help="comma-separated sample IDs")
    subparser.add_argument("--max-cells-per-sample", type=int)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="validate paths and fixed cohorts")
    _add_common(check, output=False)
    prepare = commands.add_parser("prepare", help="download/load model and smoke test")
    _add_common(prepare)
    prepare.add_argument("--device")
    hest = commands.add_parser("extract-hest", help="export fast-HEST embeddings")
    _add_extraction(hest)
    phenocell = commands.add_parser(
        "extract-phenocell", help="export PhenoCell embeddings"
    )
    _add_extraction(phenocell)
    probe = commands.add_parser("probe", help="probe completed feature artifacts")
    _add_common(probe)
    probe.add_argument(
        "--dataset", choices=("all", "hest", "phenocell"), default="all"
    )
    run = commands.add_parser("run", help="extract both datasets, then probe")
    _add_common(run)
    run.add_argument("--device")
    run.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if (
        args.command in {"extract-hest", "extract-phenocell"}
        and args.max_cells_per_sample is not None
        and args.output_dir is None
    ):
        parser.error(
            "--max-cells-per-sample requires an explicit --output-dir so a "
            "smoke test cannot replace full benchmark artifacts"
        )
    cfg = load_config(args.config)
    if args.command == "check":
        check_inputs(cfg)
        return
    root = output_root(cfg, args.output_dir)
    if args.command == "probe":
        probe_all(cfg, root, args.dataset)
        return
    if args.command == "prepare":
        prepare_model(cfg, root, args.device)
        return

    encoder = load_encoder(cfg, args.device)
    try:
        if args.command == "extract-hest":
            extract_hest(
                cfg,
                encoder,
                root,
                samples=_parse_samples(args.samples),
                max_cells_per_sample=args.max_cells_per_sample,
                overwrite=args.overwrite,
            )
        elif args.command == "extract-phenocell":
            extract_phenocell(
                cfg,
                encoder,
                root,
                samples=_parse_samples(args.samples),
                max_cells_per_sample=args.max_cells_per_sample,
                overwrite=args.overwrite,
            )
        elif args.command == "run":
            extract_hest(cfg, encoder, root, overwrite=args.overwrite)
            extract_phenocell(cfg, encoder, root, overwrite=args.overwrite)
            del encoder
            encoder = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            probe_all(cfg, root, "all")
        else:
            parser.error(f"unknown command {args.command}")
    finally:
        del encoder


if __name__ == "__main__":
    main()
