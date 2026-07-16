"""Train spatial JEPA to predict held-out TERRA cell latents."""

from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from terra_multi.distill.train import init_wandb
from terra_multi.jepa.data import (
    SpatialJEPADataset,
    WindowSpec,
    collate_windows,
    fit_target_statistics,
)
from terra_multi.jepa.model import (
    CHECKPOINT_FORMAT,
    SpatialJEPA,
    TargetNormalizer,
    jepa_loss,
)


def load_config(path) -> dict:
    """Load defaults and validate one spatial-block JEPA configuration."""

    with open(path) as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict) or cfg.get("jepa_version") != "spatial_block_v1":
        raise ValueError("JEPA training requires jepa_version: spatial_block_v1")
    for name in (
        "data_root",
        "manifest_path",
        "splits_path",
        "output_dir",
        "checkpoint_dir",
    ):
        if not cfg.get(name):
            raise ValueError(f"JEPA config must define {name}")
    cfg.setdefault("uni_key", "uni_emb")
    cfg.setdefault("terra_key", "cell_emb")

    data = cfg.setdefault("data", {})
    if not isinstance(data, dict):
        raise ValueError("data must be a mapping")
    if "max_cells_per_section" in data:
        raise ValueError(
            "data.max_cells_per_section was renamed to "
            "data.max_anchor_cells_per_section"
        )
    defaults = {
        "max_anchor_cells_per_section": 100_000,
        "max_open_sections": 64,
        "sampling_seed": 0,
        "target_cells": 8,
        "target_sampling": "anchor_plus_nearest",
        "context_cells": 64,
        "context_sampling": "uniform_within_radius",
        "context_radius_um": 300.0,
        "guard_radius_um": 80.0,
        "coordinate_normalization": "fixed_radius",
        "windows_per_section": 256,
        "max_sampling_attempts": 64,
    }
    for name, value in defaults.items():
        data.setdefault(name, value)
    if data["coordinate_normalization"] != "fixed_radius":
        raise ValueError("data.coordinate_normalization must be 'fixed_radius'")
    if data["context_sampling"] != "uniform_within_radius":
        raise ValueError("data.context_sampling must be 'uniform_within_radius'")
    if data["target_sampling"] != "anchor_plus_nearest":
        raise ValueError("data.target_sampling must be 'anchor_plus_nearest'")
    if int(data["max_anchor_cells_per_section"]) < 1:
        raise ValueError("data.max_anchor_cells_per_section must be positive")
    if int(data["max_open_sections"]) < 1:
        raise ValueError("data.max_open_sections must be positive")
    if int(data["sampling_seed"]) < 0:
        raise ValueError("data.sampling_seed must be non-negative")
    if int(data["windows_per_section"]) < 1:
        raise ValueError("data.windows_per_section must be positive")
    window_spec(cfg)

    model = cfg.setdefault("model", {})
    if not isinstance(model, dict):
        raise ValueError("model must be a mapping")
    model_defaults = {
        "image_dim": 1536,
        "target_dim": 384,
        "model_dim": 384,
        "num_heads": 8,
        "context_depth": 4,
        "predictor_depth": 2,
        "dropout": 0.1,
    }
    for name, value in model_defaults.items():
        model.setdefault(name, value)
    # Constructing the model performs the remaining dimensional validation.
    SpatialJEPA(**model)

    training = cfg.setdefault("training", {})
    if not isinstance(training, dict):
        raise ValueError("training must be a mapping")
    training_defaults = {
        "batch_size": 64,
        "max_epochs": 100,
        "patience": 10,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "precision": "bf16",
        "num_workers": 0,
        "seed": 0,
    }
    for name, value in training_defaults.items():
        training.setdefault(name, value)
    for name in ("batch_size", "max_epochs", "patience"):
        if int(training[name]) < 1:
            raise ValueError(f"training.{name} must be positive")
    if int(training["num_workers"]) < 0:
        raise ValueError("training.num_workers must be non-negative")
    if int(training["seed"]) < 0:
        raise ValueError("training.seed must be non-negative")
    if float(training["learning_rate"]) <= 0 or float(training["weight_decay"]) < 0:
        raise ValueError("training learning rate and weight decay are invalid")
    if training["precision"] not in {"bf16", "fp16", "fp32"}:
        raise ValueError("training.precision must be bf16, fp16, or fp32")
    wandb = cfg.setdefault("wandb", {})
    if not isinstance(wandb, dict):
        raise ValueError("wandb must be a mapping")
    wandb.setdefault("enabled", False)
    return cfg


def window_spec(cfg: dict) -> WindowSpec:
    settings = cfg["data"]
    return WindowSpec(
        target_cells=int(settings["target_cells"]),
        context_cells=int(settings["context_cells"]),
        context_radius_um=float(settings["context_radius_um"]),
        target_guard_um=float(settings["guard_radius_um"]),
        max_attempts=int(settings["max_sampling_attempts"]),
    )


def build_dataset(cfg: dict, split: str) -> SpatialJEPADataset:
    """Build a section-local deterministic window dataset."""

    settings = cfg["data"]
    return SpatialJEPADataset(
        cfg["data_root"],
        cfg["manifest_path"],
        cfg["splits_path"],
        split,
        int(settings["windows_per_section"]),
        window_spec(cfg),
        seed=int(settings["sampling_seed"]),
        image_key=cfg["uni_key"],
        terra_key=cfg["terra_key"],
        max_anchor_cells_per_section=int(settings["max_anchor_cells_per_section"]),
        max_open_sections=int(settings["max_open_sections"]),
    )


def fit_target_normalizer(sections) -> TargetNormalizer:
    """Fit target statistics from the capped training sections only."""

    statistics = fit_target_statistics(sections)
    return TargetNormalizer(
        torch.from_numpy(statistics.mean),
        torch.from_numpy(statistics.std),
    )


def make_loader(dataset, cfg: dict) -> DataLoader:
    """Create a loader that preserves the dataset's section-local visit order."""

    workers = int(cfg["training"]["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        collate_fn=collate_windows,
    )


def _amp_settings(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return False, torch.float32
    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 was requested but this GPU does not support it")
        return True, torch.bfloat16
    return True, torch.float16


def _batch_metrics(predictions, target_terra, normalizer):
    if predictions.shape != target_terra.shape:
        raise ValueError("predictions and targets must have matching shapes")
    if predictions.ndim != 3 or predictions.shape[1] != 1:
        raise ValueError(
            "target-centered batches must have shape [targets, 1, features]"
        )

    targets = normalizer.normalize(target_terra)
    cosine = F.cosine_similarity(predictions, targets, dim=-1).mean()
    predicted_rows = F.normalize(predictions[:, 0], dim=-1)
    target_rows = F.normalize(targets[:, 0], dim=-1)
    similarities = predicted_rows @ target_rows.T
    expected_rows = torch.arange(len(similarities), device=similarities.device)
    retrieval = (similarities.argmax(dim=1) == expected_rows).float().mean()
    return cosine, retrieval


def run_epoch(
    model,
    normalizer,
    loader,
    device,
    *,
    amp_enabled,
    amp_dtype,
    optimizer=None,
    scaler=None,
):
    """Run one train or validation epoch and return target-weighted metrics."""

    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "centered_cosine": 0.0, "retrieval_top1": 0.0}
    total_targets = 0
    for batch in loader:
        context = batch["context_image"].to(device, non_blocking=True)
        context_coords = batch["context_coords"].to(device, non_blocking=True)
        target_coords = batch["target_coords"].to(device, non_blocking=True)
        target_terra = batch["target_terra"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with (
            torch.set_grad_enabled(training),
            torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ),
        ):
            predictions = model(context, context_coords, target_coords)
            loss = jepa_loss(predictions, target_terra, normalizer)
        if training:
            if scaler is None:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        with torch.no_grad():
            cosine, retrieval = _batch_metrics(
                predictions.float(), target_terra, normalizer
            )
        n_targets = len(context)
        totals["loss"] += float(loss.detach()) * n_targets
        totals["centered_cosine"] += float(cosine) * n_targets
        totals["retrieval_top1"] += float(retrieval) * n_targets
        total_targets += n_targets
    if total_targets == 0:
        raise ValueError("cannot run a JEPA epoch without targets")
    return {name: value / total_targets for name, value in totals.items()}


def _checkpoint_payload(model, normalizer, cfg, epoch, metrics):
    return {
        "format_version": CHECKPOINT_FORMAT,
        "model_config": model.model_config,
        "model_state": model.state_dict(),
        "target_normalizer": normalizer.state_dict(),
        "config": cfg,
        "epoch": int(epoch),
        "val_loss": float(metrics["loss"]),
        "val_metrics": dict(metrics),
    }


def _save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def fit(model, normalizer, train_dataset, val_dataset, cfg, device, checkpoint, run):
    if normalizer.target_dim != model.target_dim:
        raise ValueError("target normalizer and model dimensions do not match")
    training = cfg["training"]
    train_loader = make_loader(train_dataset, cfg)
    val_loader = make_loader(val_dataset, cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    amp_enabled, amp_dtype = _amp_settings(device, training["precision"])
    scaler = (
        torch.amp.GradScaler("cuda")
        if amp_enabled and amp_dtype == torch.float16
        else None
    )

    rows = []
    best_loss = float("inf")
    best_epoch = stale_epochs = 0
    for epoch in range(1, int(training["max_epochs"]) + 1):
        train_dataset.set_epoch(epoch)
        val_dataset.set_epoch(0)
        train_metrics = run_epoch(
            model,
            normalizer,
            train_loader,
            device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            optimizer=optimizer,
            scaler=scaler,
        )
        val_metrics = run_epoch(
            model,
            normalizer,
            val_loader,
            device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        print(
            f"epoch {epoch:3d} | train {train_metrics['loss']:.4f} "
            f"cos {train_metrics['centered_cosine']:.4f} | "
            f"val {val_metrics['loss']:.4f} "
            f"cos {val_metrics['centered_cosine']:.4f}"
        )
        row = {
            "epoch": epoch,
            **{f"train_{name}": value for name, value in train_metrics.items()},
            **{f"val_{name}": value for name, value in val_metrics.items()},
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(
            Path(cfg["output_dir"]) / "epoch_metrics.csv", index=False
        )
        if run is not None:
            run.log(
                {
                    "epoch": epoch,
                    **{f"train/{name}": value for name, value in train_metrics.items()},
                    **{f"val/{name}": value for name, value in val_metrics.items()},
                }
            )

        if (
            math.isfinite(val_metrics["loss"])
            and val_metrics["loss"] < best_loss - 1e-6
        ):
            best_loss = val_metrics["loss"]
            best_epoch = epoch
            stale_epochs = 0
            _save_checkpoint(
                checkpoint,
                _checkpoint_payload(model, normalizer, cfg, epoch, val_metrics),
            )
        else:
            stale_epochs += 1
            if stale_epochs >= int(training["patience"]):
                print(f"early stopping at epoch {epoch} (best {best_epoch})")
                break
    if best_epoch == 0:
        raise RuntimeError("training produced no finite validation checkpoint")
    return best_epoch, best_loss, rows


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    seed = int(cfg["training"]["seed"])
    _set_seed(seed)
    print("=== config ===")
    print(yaml.safe_dump(cfg, sort_keys=False).rstrip())

    output_dir = Path(cfg["output_dir"])
    checkpoint_dir = Path(cfg["checkpoint_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "resolved_config.yaml", "w") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    train_dataset = build_dataset(cfg, "distill_train")
    val_dataset = build_dataset(cfg, "distill_val")
    normalizer = fit_target_normalizer(train_dataset)
    torch.save(normalizer.state_dict(), output_dir / "target_normalizer.pt")
    print(
        f"sections: train={len(train_dataset.sample_ids)} "
        f"val={len(val_dataset.sample_ids)} | "
        f"windows: train={len(train_dataset):,} val={len(val_dataset):,}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpatialJEPA(**cfg["model"]).to(device)
    normalizer = normalizer.to(device)
    run = init_wandb(cfg)
    checkpoint = checkpoint_dir / "best.pt"
    try:
        best_epoch, best_loss, _ = fit(
            model,
            normalizer,
            train_dataset,
            val_dataset,
            cfg,
            device,
            checkpoint,
            run,
        )
    finally:
        if run is not None:
            run.finish()
    print(
        f"best checkpoint: epoch {best_epoch}, val loss {best_loss:.6f} -> {checkpoint}"
    )


if __name__ == "__main__":
    main()
