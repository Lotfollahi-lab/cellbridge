"""Export LoRA feature arms for the fixed HEST evaluation cohort."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from terra_multi.evaluation.artifacts import write_table
from terra_multi.lora.features import (
    FEATURE_ARMS,
    feature_file_is_current,
    feature_signature,
    infer_loader,
    write_feature_file,
)
from terra_multi.lora.hest import dataset, evaluation_samples, source_identity


def export_hest(
    cfg,
    model,
    device,
    output_dir,
    checkpoint_identity,
    overwrite=False,
    samples=None,
    cohort="label_test_fast",
):
    """Export all requested arms for one configured HEST cohort."""
    import torch

    output_dir = Path(output_dir)
    settings = cfg["eval"]
    rows = []
    expected_dims = {
        "uni_original": model.image_dim,
        "lora_adapted": model.image_dim,
    }
    wanted = evaluation_samples(cfg, cohort)
    if samples is not None:
        requested = {str(sample_id) for sample_id in samples}
        unknown = sorted(requested - set(wanted))
        if unknown:
            raise ValueError(f"not {cohort} samples: {unknown}")
        wanted = [sample_id for sample_id in wanted if sample_id in requested]

    for sample_id in wanted:
        source = source_identity(cfg, sample_id)
        with np.load(source["terra"]["path"], allow_pickle=True) as terra:
            expected_cells = len(terra["cell_ids"])
        signature = feature_signature(
            f"hest/{cohort}",
            checkpoint_identity,
            source,
            {
                "sample_id": sample_id,
                "encoder": "uni_v2",
                "crop": cfg.get("crop", {}),
            },
        )
        destination = output_dir / "features" / f"{sample_id}.npz"
        if not overwrite and feature_file_is_current(
            destination,
            signature,
            FEATURE_ARMS,
            expected_cells=expected_cells,
            expected_dims=expected_dims,
        ):
            with np.load(destination, allow_pickle=False) as data:
                count = len(data["cell_ids"])
            print(f"[HEST/{sample_id}] current ({count:,} cells)")
        else:
            sample_dataset = dataset(cfg, sample_id, model.eval_transforms)
            loader = torch.utils.data.DataLoader(
                sample_dataset,
                batch_size=int(settings.get("batch_size", 128)),
                shuffle=False,
                num_workers=int(settings.get("num_workers", 8)),
                pin_memory=torch.device(device).type == "cuda",
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                with TemporaryDirectory(dir=destination.parent) as temporary:
                    cell_ids, features, sample_ids = infer_loader(
                        model, loader, device, temporary
                    )
                    if sample_ids is not None and set(sample_ids) != {sample_id}:
                        raise ValueError(f"{sample_id}: dataset returned another sample")
                    write_feature_file(destination, cell_ids, features, signature)
            finally:
                sample_dataset.close()
            count = len(cell_ids)
            print(f"[HEST/{sample_id}] wrote {count:,} cells -> {destination}")
        rows.append(
            {
                "sample_id": sample_id,
                "cohort": cohort,
                "n_cells": count,
                "path": str(destination.resolve()),
                "signature": signature,
            }
        )
    write_manifest_if_complete(cfg, cohort, output_dir)
    return pd.DataFrame(rows)


def write_manifest_if_complete(cfg, cohort, output_dir) -> bool:
    """Write the feature manifest once every cohort sample is on disk.

    A sharded export only knows its own subset, so each shard checks whether the
    others have finished and the last one writes the complete manifest. Probing
    requires the manifest to list the whole cohort in order.
    """

    features_dir = Path(output_dir) / "features"
    rows = []
    for sample_id in evaluation_samples(cfg, cohort):
        path = features_dir / f"{sample_id}.npz"
        if not path.exists():
            return False
        with np.load(path, allow_pickle=False) as data:
            rows.append(
                {
                    "sample_id": sample_id,
                    "cohort": cohort,
                    "n_cells": len(data["cell_ids"]),
                    "path": str(path.resolve()),
                    "signature": str(data["signature"]),
                }
            )
    write_table(features_dir / "manifest.csv", rows)
    return True
