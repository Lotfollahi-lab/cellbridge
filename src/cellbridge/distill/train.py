"""Train a UNI2-to-TERRA embedding projector."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from cellbridge.cohorts import split_groups
from cellbridge.config import load_config as load_yaml_config
from cellbridge.distill.data import CellEmbeddingDataset, build_split_cache
from cellbridge.distill.inference import (
    embedding_variance,
    evaluate_label_test,
    evaluate_validation,
    nmse_from_cosine_loss,
)
from cellbridge.distill.make_splits import (
    label_test_split_name,
    select_label_test_samples,
)
from cellbridge.distill.model import (
    TerraProjector,
    cosine_distill_loss,
    covariance_loss,
    variance_loss,
)


def load_config(path):
    cfg = load_yaml_config(path)
    defaults = {
        "uni_dim": 1536,
        "terra_dim": 384,
        "hidden_dim": 1024,
        "dropout": 0.1,
        "lambda_cos": 1.0,
        "lambda_var": 0.1,
        "lambda_cov": 0.01,
        "var_gamma": 1.0,
        "num_workers": 8,
        "seed": 0,
    }
    for key, value in defaults.items():
        cfg.setdefault(key, value)
    cfg.setdefault("wandb", {}).setdefault("enabled", False)
    return cfg


def load_splits(path):
    return split_groups(path)


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
        name=settings.get("name"),
        mode=settings.get("mode", "online"),
        config=cfg,
    )


def _distillation_losses(model, uni, teacher, cfg):
    normed, raw = model(uni, return_raw=True)
    losses = {
        "loss_cos": cosine_distill_loss(normed, teacher),
        "loss_var": variance_loss(raw, cfg["var_gamma"]),
        "loss_cov": covariance_loss(raw),
    }
    losses["loss_total"] = (
        cfg["lambda_cos"] * losses["loss_cos"]
        + cfg["lambda_var"] * losses["loss_var"]
        + cfg["lambda_cov"] * losses["loss_cov"]
    )
    return losses


def run_epoch(model, loader, device, cfg, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = {name: 0.0 for name in ("loss_total", "loss_cos", "loss_var", "loss_cov")}
    total_cells = 0
    for uni, teacher in loader:
        uni = uni.to(device, non_blocking=True)
        teacher = teacher.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            losses = _distillation_losses(model, uni, teacher, cfg)
        if training:
            optimizer.zero_grad(set_to_none=True)
            losses["loss_total"].backward()
            optimizer.step()
        for name, loss in losses.items():
            totals[name] += loss.item() * len(uni)
        total_cells += len(uni)
    if total_cells == 0:
        raise ValueError("cannot run an epoch with no cells")
    metrics = {name: value / total_cells for name, value in totals.items()}
    metrics["cosine_similarity"] = 1.0 - metrics["loss_cos"]
    return metrics


def make_loader(cache, cfg, training):
    return DataLoader(
        CellEmbeddingDataset(cache.uni, cache.terra),
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
        shuffle=training,
        drop_last=training,
    )


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    teacher_variance,
    cfg,
    checkpoint,
    run,
):
    best_loss = float("inf")
    best_epoch = patience = 0
    train_rows, val_rows = [], []

    for epoch in range(1, cfg["max_epochs"] + 1):
        train_metrics = run_epoch(model, train_loader, device, cfg, optimizer)
        val_metrics = run_epoch(model, val_loader, device, cfg)
        train_metrics["normalized_mse"] = nmse_from_cosine_loss(
            train_metrics["loss_cos"], teacher_variance["train"]
        )
        val_metrics["normalized_mse"] = nmse_from_cosine_loss(
            val_metrics["loss_cos"], teacher_variance["val"]
        )
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch:3d} | train total {train_metrics['loss_total']:.4f} "
            f"cos {train_metrics['cosine_similarity']:.4f} | "
            f"val total {val_metrics['loss_total']:.4f} "
            f"cos {val_metrics['cosine_similarity']:.4f}"
        )
        train_rows.append({"epoch": epoch, **train_metrics, "lr": lr})
        val_rows.append({"epoch": epoch, **val_metrics})
        if run is not None:
            run.log(
                {
                    "epoch": epoch,
                    "lr": lr,
                    **{f"train/{name}": value for name, value in train_metrics.items()},
                    **{f"val/{name}": value for name, value in val_metrics.items()},
                }
            )

        val_loss = val_metrics["loss_total"]
        if val_loss < best_loss - 1e-6:
            best_loss, best_epoch, patience = val_loss, epoch, 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                checkpoint,
            )
        else:
            patience += 1
            if patience >= cfg["patience"]:
                print(f"early stopping at epoch {epoch} (best {best_epoch})")
                break

    return best_epoch, best_loss, train_rows, val_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit-samples", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    print("=== config ===")
    print(yaml.safe_dump(cfg, sort_keys=False).rstrip())

    out_dir = Path(cfg["output_dir"])
    checkpoint_dir = Path(cfg["checkpoint_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "resolved_config.yaml", "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    split_path = Path(cfg["splits_path"])
    if not split_path.exists():
        from cellbridge.distill.make_splits import write_split_table

        write_split_table(cfg)
    splits = load_splits(split_path)
    splits["label_test"] = select_label_test_samples(cfg, splits["label_test"])
    if args.limit_samples is not None:
        splits = {
            name: samples[: args.limit_samples] for name, samples in splits.items()
        }
    print(
        "samples: "
        + " ".join(f"{name}={len(samples)}" for name, samples in splits.items())
    )

    tag = f"seed{cfg['sampling_seed']}_cap{cfg['max_cells_per_sample']}"
    if args.limit_samples is not None:
        tag += f"_lim{args.limit_samples}"
    cache_dir = Path(cfg["cache_dir"])
    train_cache = build_split_cache(
        cfg["data_root"],
        splits["distill_train"],
        cfg["max_cells_per_sample"],
        cfg["sampling_seed"],
        cfg["uni_key"],
        cfg["terra_key"],
        cache_path=cache_dir / f"distill_train_{tag}.npz",
    )
    val_cache = build_split_cache(
        cfg["data_root"],
        splits["distill_val"],
        cfg["max_cells_per_sample"],
        cfg["sampling_seed"],
        cfg["uni_key"],
        cfg["terra_key"],
        cache_path=cache_dir / f"distill_val_{tag}.npz",
    )
    print(f"cells: train={len(train_cache):,} val={len(val_cache):,}")
    teacher_variance = {
        "train": embedding_variance(train_cache.terra),
        "val": embedding_variance(val_cache.terra),
    }

    for name, cache in (("train", train_cache), ("val", val_cache)):
        pd.DataFrame({"sample_id": cache.sample_ids, "cell_id": cache.cell_ids}).to_csv(
            out_dir / f"selected_cells_{name}_{tag}.csv", index=False
        )

    train_loader = make_loader(train_cache, cfg, training=True)
    val_loader = make_loader(val_cache, cfg, training=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TerraProjector(
        cfg["uni_dim"],
        cfg["terra_dim"],
        cfg["hidden_dim"],
        cfg["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    run = init_wandb(cfg)
    checkpoint = checkpoint_dir / "best.pt"
    best_epoch, best_loss, train_rows, val_rows = fit(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        teacher_variance,
        cfg,
        checkpoint,
        run,
    )

    pd.DataFrame(train_rows).to_csv(out_dir / "train_metrics.csv", index=False)
    pd.DataFrame(val_rows).to_csv(out_dir / "val_metrics.csv", index=False)
    print(f"best val loss {best_loss:.4f} @ epoch {best_epoch} -> {checkpoint}")

    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])
    val_metrics = evaluate_validation(
        model, val_cache, device, out_dir, cfg["batch_size"]
    )
    label_test_name = label_test_split_name(cfg)
    test_metrics = evaluate_label_test(
        model,
        cfg,
        splits["label_test"],
        device,
        out_dir,
        cfg["batch_size"],
        split_name=label_test_name,
    )
    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "distill_val": val_metrics,
        label_test_name: test_metrics,
    }
    print("=== evaluation ===")
    print(json.dumps(summary, indent=2))
    with open(out_dir / "distill_eval.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    if run is not None:
        run.summary.update(
            {
                "best_epoch": best_epoch,
                "best_val_loss": best_loss,
                "distill_val/mean_cosine": val_metrics["mean_cosine"],
                "distill_val/normalized_mse": val_metrics["normalized_mse"],
                f"{label_test_name}/mean_cosine": test_metrics["mean_cosine"],
                f"{label_test_name}/normalized_mse": test_metrics["normalized_mse"],
            }
        )
        run.finish()


if __name__ == "__main__":
    main()
