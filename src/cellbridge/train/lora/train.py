"""Train UNI2-h LoRA adapters from H&E cell crops."""

import argparse
import json
import os
import random
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from cellbridge.config import load_config as load_yaml_config
from cellbridge.train.lora.data import WithinSampleBatchSampler, build_datasets
from cellbridge.train.lora.model import (
    build_uni2_distiller,
    contrastive_loss,
    make_checkpoint,
)


def load_config(path):
    cfg = load_yaml_config(path)
    if cfg.get("distill_version") != "lora":
        raise ValueError("LoRA training requires distill_version: lora")

    required = {"data_root", "splits_path", "output_dir", "checkpoint_dir"}
    missing = sorted(required - cfg.keys())
    if missing:
        raise ValueError(f"LoRA config is missing {missing}")

    cfg.setdefault("sampling_seed", 0)
    cfg.setdefault("max_cells_per_sample", 5_000)
    cfg.setdefault("max_val_cells_per_sample", 5_000)
    cfg.setdefault("max_open_wsis", 64)
    cfg.setdefault("terra_key", "cell_emb")

    crop = cfg.setdefault("crop", {})
    crop.setdefault("target_size", 224)
    crop.setdefault("target_pixel_size_um", 0.5)

    model = cfg.setdefault("model", {})
    model.setdefault("trainable_blocks", "last_2")
    model.setdefault("lora_rank", 8)
    model.setdefault("lora_alpha", 16)
    model.setdefault("lora_dropout", 0.05)
    model.setdefault("image_dim", 1536)
    model.setdefault("terra_dim", 384)

    objective = cfg.setdefault("objective", {})
    objective.setdefault("temperature", 0.07)
    objective.setdefault("bidirectional", True)
    if float(objective["temperature"]) <= 0:
        raise ValueError("objective.temperature must be positive")
    if not isinstance(objective["bidirectional"], bool):
        raise TypeError("objective.bidirectional must be true or false")

    training = cfg.setdefault("training", {})
    defaults = {
        "batch_size": 16,
        "batches_per_sample_visit": 32,
        "max_steps": 20_000,
        "eval_every": 1_000,
        "patience": 5,
        "lr_lora": 1e-4,
        "lr_heads": 3e-4,
        "weight_decay": 1e-4,
        "precision": "bf16",
        "num_workers": 8,
        "seed": 0,
        "log_every": 20,
    }
    for key, value in defaults.items():
        training.setdefault(key, value)

    if int(training["batch_size"]) < 2:
        raise ValueError("training.batch_size must be at least 2 for InfoNCE")
    positive = ("batches_per_sample_visit", "max_steps", "eval_every", "patience")
    if any(int(training[key]) < 1 for key in positive):
        raise ValueError(f"training values must be positive: {positive}")
    if int(training["log_every"]) < 1:
        raise ValueError("training.log_every must be positive")
    if int(training["num_workers"]) < 0:
        raise ValueError("training.num_workers must be non-negative")
    if training["precision"] not in {"bf16", "fp16", "fp32"}:
        raise ValueError("training.precision must be bf16, fp16, or fp32")

    cfg.setdefault("wandb", {}).setdefault("enabled", False)
    return cfg


def init_wandb(cfg):
    settings = cfg["wandb"]
    if not settings["enabled"]:
        return None
    try:
        import wandb
    except ImportError:
        print("[wandb] not installed; logging disabled")
        return None
    return wandb.init(
        project=settings.get("project"),
        group=settings.get("group"),
        name=settings.get("name", cfg.get("run_name")),
        mode=settings.get("mode", "online"),
        config=cfg,
    )


def model_config(cfg):
    settings = dict(cfg["model"])
    return settings, settings.pop("weights_path", None)


def make_optimizer(model, cfg):
    lora_parameters = list(model.lora_parameters())
    head_parameters = list(model.head_parameters())
    grouped = {id(parameter) for parameter in lora_parameters + head_parameters}
    trainable = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    if grouped != trainable:
        raise ValueError("optimizer groups do not exactly cover trainable parameters")

    training = cfg["training"]
    groups = []
    if lora_parameters:
        groups.append(
            {
                "params": lora_parameters,
                "lr": float(training["lr_lora"]),
                "name": "lora",
            }
        )
    groups.append(
        {
            "params": head_parameters,
            "lr": float(training["lr_heads"]),
            "name": "heads",
        }
    )
    optimizer = torch.optim.AdamW(
        groups,
        weight_decay=float(training["weight_decay"]),
    )
    counts = {
        "lora": sum(parameter.numel() for parameter in lora_parameters),
        "heads": sum(parameter.numel() for parameter in head_parameters),
    }
    return optimizer, counts


def make_loader(dataset, cfg, training):
    settings = cfg["training"]
    workers = int(settings["num_workers"])
    sampler = WithinSampleBatchSampler(
        dataset,
        int(settings["batch_size"]),
        seed=int(settings["seed"]),
        shuffle=training,
        batches_per_sample_visit=int(settings["batches_per_sample_visit"]),
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(int(settings["seed"])),
    )


def _amp_settings(device, precision):
    if precision == "fp32" or device.type != "cuda":
        return False, torch.float32
    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 was requested but this GPU does not support it")
        return True, torch.bfloat16
    return True, torch.float16


def _loss(model, batch, device, objective, amp_enabled, amp_dtype):
    images = batch["image"].to(device, non_blocking=True)
    terra = batch["terra"].to(device, non_blocking=True)
    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=amp_enabled,
    ):
        outputs = model(images)
        terra_features = (
            model.project_terra(terra)
            if model.alignment_dim != model.terra_dim
            else terra
        )
        losses = contrastive_loss(outputs["projected"], terra_features, **objective)
    return losses, len(images)


@torch.no_grad()
def evaluate(model, loader, device, objective, amp_enabled, amp_dtype):
    model.eval()
    totals = {}
    cell_count = 0
    for batch in loader:
        metrics, batch_size = _loss(
            model,
            batch,
            device,
            objective,
            amp_enabled,
            amp_dtype,
        )
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + float(value) * batch_size
        cell_count += batch_size
    if cell_count == 0:
        raise ValueError("validation has no contrastive batches")
    return {name: value / cell_count for name, value in totals.items()}


def atomic_torch_save(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def train_loop(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    cfg,
    checkpoint_path,
    run,
):
    training = cfg["training"]
    amp_enabled, amp_dtype = _amp_settings(device, training["precision"])
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled and amp_dtype == torch.float16
    )
    max_steps = int(training["max_steps"])
    eval_every = int(training["eval_every"])
    log_every = int(training["log_every"])
    patience = int(training["patience"])

    train_rows, val_rows = [], []
    window, window_steps = {}, 0
    best_loss, best_step, stale_evaluations = float("inf"), 0, 0
    step = epoch = 0

    while step < max_steps and stale_evaluations < patience:
        sampler = train_loader.batch_sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        for batch in train_loader:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses, _ = _loss(
                model,
                batch,
                device,
                cfg["objective"],
                amp_enabled,
                amp_dtype,
            )
            scaler.scale(losses["loss_total"]).backward()
            scaler.step(optimizer)
            scaler.update()
            step += 1
            window_steps += 1
            for name, value in losses.items():
                window[name] = window.get(name, 0.0) + float(value.detach())

            if step == 1 or step % log_every == 0:
                learning_rate = next(
                    (
                        group["lr"]
                        for group in optimizer.param_groups
                        if group.get("name") == "lora"
                    ),
                    optimizer.param_groups[0]["lr"],
                )
                train_metrics = {
                    "step": step,
                    **{name: value / window_steps for name, value in window.items()},
                    "lr": learning_rate,
                }
                window, window_steps = {}, 0
                train_rows.append(train_metrics)
                printable = " ".join(
                    f"{name}={value:.4f}"
                    for name, value in train_metrics.items()
                    if name.startswith("loss_")
                )
                print(f"step {step:6d} | {printable}")
                if run is not None:
                    run.log(
                        {
                            "step": step,
                            "lr": learning_rate,
                            **{
                                f"train/{name}": value
                                for name, value in train_metrics.items()
                                if name.startswith("loss_")
                            },
                        }
                    )

            if step % eval_every == 0 or step == max_steps:
                val_metrics = evaluate(
                    model,
                    val_loader,
                    device,
                    cfg["objective"],
                    amp_enabled,
                    amp_dtype,
                )
                val_rows.append({"step": step, **val_metrics})
                val_loss = float(val_metrics["loss_total"])
                print(f"validation {step:6d} | loss_total={val_loss:.4f}")
                if run is not None:
                    run.log(
                        {
                            "step": step,
                            **{
                                f"val/{name}": value
                                for name, value in val_metrics.items()
                            },
                        }
                    )

                if np.isfinite(val_loss) and val_loss < best_loss - 1e-6:
                    best_loss, best_step, stale_evaluations = val_loss, step, 0
                    atomic_torch_save(
                        make_checkpoint(
                            model,
                            step=step,
                            val_loss=val_loss,
                            config=cfg,
                        ),
                        checkpoint_path,
                    )
                else:
                    stale_evaluations += 1
                    if stale_evaluations >= patience:
                        print(f"early stopping at step {step}")
                        break
            if step >= max_steps:
                break
        epoch += 1

    if best_step == 0:
        raise RuntimeError("training produced no finite validation checkpoint")
    return train_rows, val_rows, best_step, best_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    seed = int(cfg["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    output_dir = Path(cfg["output_dir"])
    checkpoint_dir = Path(cfg["checkpoint_dir"])
    checkpoint_path = checkpoint_dir / "best.pt"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.unlink(missing_ok=True)

    settings, weights_path = model_config(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_uni2_distiller(weights_path=weights_path, **settings).to(device)
    train_dataset, val_dataset = build_datasets(cfg, model.eval_transforms)
    train_loader = make_loader(train_dataset, cfg, training=True)
    val_loader = make_loader(val_dataset, cfg, training=False)

    with open(output_dir / "resolved_config.yaml", "w") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    optimizer, parameter_counts = make_optimizer(model, cfg)
    print(
        f"cells: train={len(train_dataset):,} val={len(val_dataset):,} | "
        f"trainable LoRA={parameter_counts['lora']:,} "
        f"heads={parameter_counts['heads']:,}"
    )

    run = init_wandb(cfg)
    train_rows, val_rows, best_step, best_loss = train_loop(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        cfg,
        checkpoint_path,
        run,
    )
    pd.DataFrame(train_rows).to_csv(output_dir / "train_metrics.csv", index=False)
    pd.DataFrame(val_rows).to_csv(output_dir / "val_metrics.csv", index=False)
    summary = {
        "best_step": best_step,
        "best_val_loss": best_loss,
        "train_cells": len(train_dataset),
        "val_cells": len(val_dataset),
        "trainable_parameters": parameter_counts,
    }
    with open(output_dir / "train_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    if run is not None:
        run.summary.update(summary)
        run.finish()


if __name__ == "__main__":
    main()
