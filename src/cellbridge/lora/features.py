"""LoRA feature inference and shared artifact adapters."""

from pathlib import Path

import numpy as np

from cellbridge.evaluation.artifacts import (
    artifact_row_count,
    file_identity,
    stable_signature,
    write_array_artifact,
)

FEATURE_ARMS = ("uni_original", "lora_adapted")
PCA_ARMS = {"uni_original", "lora_adapted"}
FORMAT_VERSION = "cellbridge.lora.features/1"


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


def feature_file_is_current(
    path,
    signature,
    required_keys=None,
    *,
    required_metadata=(),
    expected_cells=None,
    expected_dims=None,
):
    keys = tuple(required_keys or FEATURE_ARMS)
    dimensions = expected_dims or {}
    return artifact_row_count(
        path,
        format_version=FORMAT_VERSION,
        signature=signature,
        array_dims={name: dimensions.get(name) for name in keys},
        expected_cells=expected_cells,
        required_metadata=required_metadata,
    ) is not None


def write_feature_file(path, cell_ids, features, signature, metadata=None):
    write_array_artifact(
        path,
        format_version=FORMAT_VERSION,
        signature=signature,
        row_ids=cell_ids,
        arrays=features,
        metadata=metadata,
    )


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
        return {
            "lora_adapted": model.encode_images(images, use_lora=True),
            "uni_original": model.encode_base(images),
        }


def infer_loader(model, loader, device, storage_dir=None):
    """Collect LoRA features from a HESTCellDataset-compatible loader."""
    n_cells = len(loader.dataset)
    feature_arrays, cell_parts, sample_parts = {}, [], []
    offset = 0
    for batch in loader:
        if not {"image", "cell_id"}.issubset(batch):
            raise KeyError("evaluation batches require image and cell_id")
        outputs = infer_tensor(model, batch["image"], device)
        for name, tensor in outputs.items():
            values = tensor.float().cpu().numpy()
            if name not in feature_arrays:
                shape = (n_cells, values.shape[1])
                feature_arrays[name] = (
                    np.empty(shape, dtype=np.float32)
                    if storage_dir is None
                    else np.lib.format.open_memmap(
                        Path(storage_dir) / f"{name}.npy",
                        mode="w+",
                        dtype=np.float16,
                        shape=shape,
                    )
                )
            feature_arrays[name][offset : offset + len(values)] = values
        cells = np.asarray(batch["cell_id"]).astype(str)
        cell_parts.append(cells)
        if "sample_id" in batch:
            sample_parts.append(np.asarray(batch["sample_id"]).astype(str))
        offset += len(cells)
    if not cell_parts:
        raise ValueError("cannot export an empty dataset")
    if offset != n_cells:
        raise ValueError(f"loader yielded {offset} rows for a {n_cells}-cell dataset")
    for values in feature_arrays.values():
        if hasattr(values, "flush"):
            values.flush()
    sample_ids = np.concatenate(sample_parts) if sample_parts else None
    return np.concatenate(cell_parts), feature_arrays, sample_ids


def checkpoint_path(cfg, override=None):
    if override:
        return Path(override)
    if cfg.get("checkpoint"):
        return Path(cfg["checkpoint"])
    return Path(cfg["checkpoint_dir"]) / "best.pt"


def checkpoint_identity(cfg, path):
    weights_path = cfg.get("model", {}).get("weights_path")
    return {
        "checkpoint": file_identity(path, checksum=True),
        "encoder": "uni_v2",
        "encoder_weights": (
            file_identity(weights_path, checksum=True)
            if weights_path
            else "trident-default"
        ),
    }


def load_model(path, device, weights_path=None):
    from cellbridge.lora.model import load_checkpoint

    loaded = load_checkpoint(path, device=device, weights_path=weights_path)
    model = loaded[0] if isinstance(loaded, tuple) else loaded
    return model.to(device).eval()
