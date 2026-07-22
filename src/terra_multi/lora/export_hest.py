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
    required_arms = FEATURE_ARMS if model.mode == "contrastive_gex" else FEATURE_ARMS[:-1]
    expected_dims = {
        "uni_original": model.image_dim,
        "frozen_head_projected": model.terra_dim,
        "lora_adapted": model.image_dim,
        "lora_projected": model.terra_dim,
        "gex_penultimate": model.gex_hidden_dim,
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
            required_arms,
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
    if samples is None:
        write_table(output_dir / "features" / "manifest.csv", rows)
    return pd.DataFrame(rows)
