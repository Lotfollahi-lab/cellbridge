"""Train UNI2-h LoRA adapters directly from H&E cell crops."""

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

from terra_multi.config import load_config as load_yaml_config
from terra_multi.distill.train import init_wandb
from terra_multi.hierarchical.config import configure_hierarchy
from terra_multi.hierarchical.objective import compute_hierarchy_objective
from terra_multi.lora.data import WithinSampleBatchSampler, build_datasets
from terra_multi.lora.model import (
    build_uni2_distiller,
    compute_loss,
    make_checkpoint,
)
from terra_multi.lora.monitoring import CheckpointSelector, Criterion, MetricWindow
from terra_multi.lora.val_probe import build_validation_probe


def load_config(path):
    cfg = load_yaml_config(path)
    if cfg.get("distill_version") != "lora":
        raise ValueError("LoRA training requires distill_version: lora")

    cfg.setdefault("sampling_seed", 0)
    cfg.setdefault("max_cells_per_sample", 5_000)
    cfg.setdefault("terra_key", "cell_emb")
    cfg.setdefault("crop", {})
    cfg["crop"].setdefault("target_size", 224)
    cfg["crop"].setdefault("target_pixel_size_um", 0.5)

    model = cfg.setdefault("model", {})
    model.setdefault("trainable_blocks", "last_2")
    model.setdefault("lora_rank", 8)
    model.setdefault("lora_alpha", 16)
    model.setdefault("lora_dropout", 0.05)
    model.setdefault("image_dim", 1536)
    model.setdefault("terra_dim", 384)
    model.setdefault("gex_hidden_dim", 1024)
    model.setdefault("gex_dropout", 0.1)

    objective = cfg.setdefault("objective", {})
    objective.setdefault("mode", "contrastive")
    objective.setdefault("lambda_contrastive", 1.0)
    objective.setdefault("lambda_gex", 1.0)
    objective.setdefault("temperature", 0.07)
    objective.setdefault("gex_loss", "smooth_l1")
    objective.setdefault("targets", "exact")
    objective.setdefault("teacher_temperature", 0.1)
    objective.setdefault("exact_match_boost", 0.05)
    objective.setdefault("bidirectional", True)
    if objective["mode"] not in {"contrastive", "contrastive_gex"}:
        raise ValueError("objective.mode must be 'contrastive' or 'contrastive_gex'")
    if objective["targets"] not in {"exact", "terra_similarity"}:
        raise ValueError("objective.targets must be 'exact' or 'terra_similarity'")
    if float(objective["temperature"]) <= 0:
        raise ValueError("objective.temperature must be positive")
    if float(objective["teacher_temperature"]) <= 0:
        raise ValueError("objective.teacher_temperature must be positive")
    if float(objective["exact_match_boost"]) <= 0:
        raise ValueError("objective.exact_match_boost must be positive")
    if not isinstance(objective["bidirectional"], bool):
        raise ValueError("objective.bidirectional must be true or false")
    if objective["gex_loss"] != "smooth_l1":
        raise ValueError("objective.gex_loss must be 'smooth_l1'")
    if min(float(objective["lambda_contrastive"]), float(objective["lambda_gex"])) < 0:
        raise ValueError("objective loss weights must be non-negative")

    gex = cfg.setdefault("gex", {})
    gex.setdefault("num_selected_genes", 100)
    gex.setdefault("max_cells_for_selection", 200_000)
    if objective["mode"] == "contrastive_gex" and int(gex["num_selected_genes"]) < 1:
        raise ValueError("contrastive_gex requires gex.num_selected_genes > 0")

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
    training.setdefault("selection", None)
    if int(training["batch_size"]) < 2:
        raise ValueError("training.batch_size must be at least 2 for InfoNCE")
    if int(training["batches_per_sample_visit"]) < 1:
        raise ValueError("training.batches_per_sample_visit must be positive")
    if int(training["max_steps"]) < 1 or int(training["eval_every"]) < 1:
        raise ValueError("training.max_steps and training.eval_every must be positive")
    if int(training["patience"]) < 1 or int(training["log_every"]) < 1:
        raise ValueError("training.patience and training.log_every must be positive")
    if int(training["num_workers"]) < 0:
        raise ValueError("training.num_workers must be non-negative")
    if training["precision"] not in {"bf16", "fp16", "fp32"}:
        raise ValueError("training.precision must be bf16, fp16, or fp32")

    cfg.setdefault("wandb", {}).setdefault("enabled", False)
    return configure_hierarchy(cfg)


def selection_criteria(cfg, checkpoint_dir) -> list[Criterion]:
    """Resolve which validation metrics get their own best checkpoint.

    Hierarchy runs track two: the unweighted paired loss, which keeps ``best.pt``
    comparable across arms and with earlier runs, and the labelled validation
    probe, which keeps improving after the paired loss has plateaued.
    """

    training = cfg["training"]
    configured = training.get("selection")
    if configured:
        return [
            Criterion(entry["metric"], entry["mode"], Path(checkpoint_dir) / entry["filename"])
            for entry in configured
        ]
    primary = Criterion(
        training.get("selection_metric", "loss_total"),
        "min",
        Path(checkpoint_dir) / "best.pt",
    )
    if not cfg.get("hierarchy", {}).get("enabled", False):
        return [primary]
    return [
        primary,
        Criterion(
            "breast_val/probe_macro_f1", "max", Path(checkpoint_dir) / "best_probe.pt"
        ),
    ]


def model_config(cfg, num_genes):
    settings = dict(cfg["model"])
    weights_path = settings.pop("weights_path", None)
    settings["mode"] = cfg["objective"]["mode"]
    settings["num_genes"] = int(num_genes)
    return settings, weights_path


def make_optimizer(model, cfg):
    lora = list(model.lora_parameters())
    heads = list(model.head_parameters())
    if not heads:
        raise ValueError("no trainable heads found")
    grouped = {id(parameter) for parameter in [*lora, *heads]}
    trainable = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    if grouped != trainable:
        raise ValueError("optimizer groups do not exactly cover trainable parameters")

    training = cfg["training"]
    groups = []
    if lora:
        groups.append(
            {"params": lora, "lr": float(training["lr_lora"]), "name": "lora"}
        )
    groups.append({"params": heads, "lr": float(training["lr_heads"]), "name": "heads"})
    optimizer = torch.optim.AdamW(groups, weight_decay=float(training["weight_decay"]))
    return optimizer, {
        "lora": sum(parameter.numel() for parameter in lora),
        "heads": sum(parameter.numel() for parameter in heads),
    }


def make_loader(dataset, cfg, training):
    batch_size = int(cfg["training"]["batch_size"])
    workers = int(cfg["training"]["num_workers"])
    generator = torch.Generator().manual_seed(int(cfg["training"]["seed"]))
    loader_kwargs = {
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
        "generator": generator,
    }
    sampler = WithinSampleBatchSampler(
        dataset,
        batch_size,
        seed=int(cfg["training"]["seed"]),
        shuffle=training,
        batches_per_sample_visit=int(cfg["training"]["batches_per_sample_visit"]),
    )
    return DataLoader(dataset, batch_sampler=sampler, **loader_kwargs)


def _amp_settings(device, precision):
    if precision == "fp32" or device.type != "cuda":
        return False, torch.float32
    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 was requested but this GPU does not support it")
        return True, torch.bfloat16
    return True, torch.float16


def _to_device(batch, device):
    images = batch["image"].to(device, non_blocking=True)
    terra = batch["terra"].to(device, non_blocking=True)
    gex = batch.get("gex")
    if gex is not None:
        gex = gex.to(device, non_blocking=True)
    gex_mask = batch.get("gex_mask")
    if gex_mask is not None:
        gex_mask = gex_mask.to(device, non_blocking=True)
    return images, terra, gex, gex_mask


def _loss(model, batch, device, cfg, amp_enabled, amp_dtype):
    images, terra, gex, gex_mask = _to_device(batch, device)
    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=amp_enabled,
    ):
        outputs = model(images)
        if cfg.get("hierarchy", {}).get("enabled", False):
            losses = compute_hierarchy_objective(
                outputs,
                terra,
                batch,
                cfg["objective"],
                cfg["hierarchy"],
                gex_target=gex,
                gex_mask=gex_mask,
            )
        else:
            losses = compute_loss(
                outputs,
                terra,
                gex,
                gex_mask,
                **cfg["objective"],
            )
    return losses, len(images), outputs


@torch.inference_mode()
def evaluate(model, loader, device, cfg, amp_enabled, amp_dtype, probe=None):
    model.eval()
    totals = {}
    n_cells = 0
    embeddings, cell_ids = [], []
    for batch in loader:
        metrics, batch_size, outputs = _loss(
            model, batch, device, cfg, amp_enabled, amp_dtype
        )
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + float(value) * batch_size
        n_cells += batch_size
        if probe is not None:
            embeddings.append(outputs["projected"].float().cpu().numpy())
            cell_ids.extend(batch["cell_id"])
    if n_cells == 0:
        raise ValueError("validation has no contrastive batches")
    averaged = {name: value / n_cells for name, value in totals.items()}
    if probe is not None:
        averaged.update(probe.score(cell_ids, np.concatenate(embeddings)))
    return averaged


def evaluate_validation(model, loaders, device, cfg, amp_enabled, amp_dtype, probes=None):
    """Evaluate one legacy loader or separately named validation cohorts."""

    if not isinstance(loaders, dict):
        return evaluate(model, loaders, device, cfg, amp_enabled, amp_dtype)
    probes = probes or {}
    metrics = {}
    for cohort, loader in loaders.items():
        cohort_metrics = evaluate(
            model, loader, device, cfg, amp_enabled, amp_dtype, probes.get(cohort)
        )
        metrics.update(
            {f"{cohort}/{name}": value for name, value in cohort_metrics.items()}
        )
    return metrics


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
    model, train_loader, val_loader, optimizer, device, cfg, selector, run, probes=None
):
    training = cfg["training"]
    amp_enabled, amp_dtype = _amp_settings(device, training["precision"])
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled and amp_dtype == torch.float16
    )
    max_steps = int(training["max_steps"])
    eval_every = int(training["eval_every"])
    log_every = int(training["log_every"])
    window = MetricWindow(
        gate_key="hierarchy_annotated_batch",
        gated_prefixes=("hierarchy_", "negative_"),
    )
    train_rows, val_rows = [], []
    step = epoch = 0
    stop = False

    while step < max_steps and not stop:
        sampler = train_loader.batch_sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        for batch in train_loader:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses, _, _ = _loss(model, batch, device, cfg, amp_enabled, amp_dtype)
            scaler.scale(losses["loss_total"]).backward()
            scaler.step(optimizer)
            scaler.update()
            step += 1
            window.add({name: float(value.detach()) for name, value in losses.items()})

            if step % log_every == 0 or step == 1:
                learning_rate = next(
                    (
                        group["lr"]
                        for group in optimizer.param_groups
                        if group.get("name") == "lora"
                    ),
                    optimizer.param_groups[0]["lr"],
                )
                train_metrics = {"step": step, **window.mean(), "lr": learning_rate}
                window.reset()
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
                                if name.startswith(("loss_", "hierarchy_", "negative_"))
                            },
                        }
                    )

            should_evaluate = step % eval_every == 0 or step == max_steps
            if should_evaluate:
                val_metrics = evaluate_validation(
                    model, val_loader, device, cfg, amp_enabled, amp_dtype, probes
                )
                val_rows.append({"step": step, **val_metrics})
                improved = selector.update(val_metrics, step)
                reported = " ".join(
                    f"{criterion.name}={val_metrics[criterion.name]:.4f}"
                    for criterion in selector.criteria
                )
                print(f"validation {step:6d} | {reported}")
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
                for criterion in improved:
                    atomic_torch_save(
                        make_checkpoint(
                            model,
                            step=step,
                            val_loss=float(val_metrics[criterion.name]),
                            config=cfg,
                        ),
                        criterion.path,
                    )
                if selector.should_stop:
                    print(f"early stopping at step {step}")
                    stop = True
                    break
            if step >= max_steps:
                break
        epoch += 1

    missing = [
        criterion.name
        for criterion in selector.criteria
        if not Path(criterion.path).exists()
    ]
    if missing:
        raise RuntimeError(f"training produced no checkpoint for {missing}")
    return train_rows, val_rows


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

    out_dir = Path(cfg["output_dir"])
    checkpoint_dir = Path(cfg["checkpoint_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for criterion in selection_criteria(cfg, checkpoint_dir):
        criterion.path.unlink(missing_ok=True)

    num_genes = (
        int(cfg["gex"]["num_selected_genes"])
        if cfg["objective"]["mode"] == "contrastive_gex"
        else 0
    )
    settings, weights_path = model_config(cfg, num_genes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_uni2_distiller(weights_path=weights_path, **settings).to(device)
    transform = model.eval_transforms

    if cfg["hierarchy"]["enabled"]:
        from terra_multi.hierarchical.training_data import (
            build_hierarchy_data,
            make_hierarchy_loaders,
        )

        from terra_multi.hierarchical.prototypes import configure_target_shrinkage

        hierarchy_data = build_hierarchy_data(
            cfg, transform, out_dir / "selected_genes.json"
        )
        shrinkage = configure_target_shrinkage(cfg, hierarchy_data)
        train_dataset = hierarchy_data.train
        train_loader, val_loader = make_hierarchy_loaders(hierarchy_data, cfg)
        selected_genes = hierarchy_data.genes
        cohort_cells = {
            "base_train": len(hierarchy_data.base_train),
            "breast_train": len(hierarchy_data.breast_train),
            "base_val": len(hierarchy_data.base_val),
            "breast_val": len(hierarchy_data.breast_val),
        }
        train_cells = cohort_cells["base_train"] + cohort_cells["breast_train"]
        val_cells = cohort_cells["base_val"] + cohort_cells["breast_val"]
        probes = {"breast_val": build_validation_probe(cfg, hierarchy_data.breast_val)}
        pd.read_csv(cfg["hierarchy"]["split_manifest"]).to_csv(
            out_dir / "split_manifest.csv", index=False
        )
    else:
        train_dataset, val_dataset, selected_genes = build_datasets(
            cfg,
            transform,
            out_dir / "selected_genes.json",
        )
        train_loader = make_loader(train_dataset, cfg, training=True)
        val_loader = make_loader(val_dataset, cfg, training=False)
        train_cells, val_cells = len(train_dataset), len(val_dataset)
        cohort_cells = {}
        probes = shrinkage = None
    if len(selected_genes or []) != num_genes:
        raise ValueError(
            f"expected {num_genes} selected genes, got {len(selected_genes or [])}"
        )
    if selected_genes:
        cfg["gex"]["selected_genes"] = list(selected_genes)
    with open(out_dir / "resolved_config.yaml", "w") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    optimizer, parameter_counts = make_optimizer(model, cfg)
    print(
        f"cells: train={train_cells:,} val={val_cells:,} | "
        f"trainable LoRA={parameter_counts['lora']:,} heads={parameter_counts['heads']:,}"
    )

    run = init_wandb(cfg)
    selector = CheckpointSelector(
        selection_criteria(cfg, checkpoint_dir), int(cfg["training"]["patience"])
    )
    train_rows, val_rows = train_loop(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        cfg,
        selector,
        run,
        probes,
    )
    pd.DataFrame(train_rows).to_csv(out_dir / "train_metrics.csv", index=False)
    pd.DataFrame(val_rows).to_csv(out_dir / "val_metrics.csv", index=False)
    selection = selector.summary()
    primary = selection[selector.primary.name]
    summary = {
        "best_step": primary["best_step"],
        "best_val_loss": primary["best_value"],
        "validation_selection_metric": selector.primary.name,
        "selection": selection,
        "train_cells": train_cells,
        "val_cells": val_cells,
        "trainable_parameters": parameter_counts,
    }
    if cohort_cells:
        summary["cohort_cells"] = cohort_cells
    if shrinkage:
        summary["target_shrinkage"] = shrinkage
    with open(out_dir / "train_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    if run is not None:
        run.summary.update(summary)
        run.finish()


if __name__ == "__main__":
    main()
