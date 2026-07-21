"""Train a UNI2 student to classify TERRA molecular prototypes."""

import argparse
import gc
import json
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn import __version__ as sklearn_version
from torch.utils.data import DataLoader

from terra_multi.config import load_config as load_yaml_config
from terra_multi.distill.data import (
    build_split_cache,
    load_sample_embeddings,
    load_uni_embeddings,
)
from terra_multi.distill.make_splits import (
    label_test_split_name,
    select_label_test_samples,
)
from terra_multi.distill.prototype import (
    PrototypeDataset,
    PrototypeStudent,
    assign_prototypes,
    fit_prototypes,
    inverse_sqrt_class_weights,
    predict_student_features,
)
from terra_multi.distill.train import init_wandb, load_splits


def load_config(path):
    cfg = load_yaml_config(path)
    defaults = {
        "uni_dim": 1536,
        "hidden_dim": 1024,
        "feat_dim": 512,
        "num_prototypes": 256,
        "dropout": 0.1,
        "kmeans_seed": 0,
        "max_cells_for_kmeans": 500_000,
        "kmeans_batch_size": 8192,
        "kmeans_max_iter": 100,
        "kmeans_n_init": 3,
        "assignment_batch_size": 65_536,
        "use_class_weights": False,
        "class_weight_clip": [0.25, 4.0],
        "num_workers": 8,
        "seed": 0,
    }
    for key, value in defaults.items():
        cfg.setdefault(key, value)
    cfg.setdefault("wandb", {}).setdefault("enabled", False)
    return cfg


def build_cache(cfg, samples, split_name, limit_samples=None):
    tag = f"seed{cfg['sampling_seed']}_cap{cfg['max_cells_per_sample']}"
    if limit_samples is not None:
        tag += f"_lim{limit_samples}"
    return build_split_cache(
        cfg["data_root"],
        samples,
        cfg["max_cells_per_sample"],
        cfg["sampling_seed"],
        cfg["uni_key"],
        cfg["terra_key"],
        cache_path=Path(cfg["cache_dir"]) / f"{split_name}_{tag}.npz",
    )


def save_assignments(path, sample_ids, cell_ids, prototype_ids):
    np.savez(
        path,
        sample_id=np.asarray(sample_ids),
        cell_id=np.asarray(cell_ids),
        prototype_id=np.asarray(prototype_ids, dtype=np.int32),
    )


def assign_label_samples(model, cfg, samples):
    sample_parts, cell_parts, assignment_parts = [], [], []
    for sample_id in samples:
        _, terra, cell_ids = load_sample_embeddings(
            cfg["data_root"], sample_id, cfg["uni_key"], cfg["terra_key"]
        )
        prototype_ids = assign_prototypes(model, terra, cfg["assignment_batch_size"])
        sample_parts.append(np.full(len(cell_ids), sample_id))
        cell_parts.append(cell_ids)
        assignment_parts.append(prototype_ids)
    return (
        np.concatenate(sample_parts),
        np.concatenate(cell_parts),
        np.concatenate(assignment_parts),
    )


def make_loader(uni, prototype_ids, cfg, training):
    return DataLoader(
        PrototypeDataset(uni, prototype_ids),
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
        shuffle=training,
    )


def cross_entropy_denominator(criterion, target):
    valid = target != criterion.ignore_index
    if criterion.weight is None:
        return valid.sum().item()
    return criterion.weight[target[valid]].sum().item()


def run_epoch(model, loader, criterion, device, optimizer=None, top5=False):
    training = optimizer is not None
    model.train(training)
    loss_sum = loss_denominator = correct1 = correct5 = n_cells = 0
    for uni, target in loader:
        uni = uni.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            logits = model(uni)
            loss = criterion(logits, target)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_denominator = cross_entropy_denominator(criterion, target)
        loss_sum += loss.item() * batch_denominator
        loss_denominator += batch_denominator
        correct1 += (logits.argmax(dim=1) == target).sum().item()
        if top5:
            k = min(5, logits.shape[1])
            guesses = logits.topk(k, dim=1).indices
            correct5 += (guesses == target[:, None]).any(dim=1).sum().item()
        n_cells += len(uni)
    if n_cells == 0:
        raise ValueError("cannot run an epoch with no cells")

    metrics = {
        "loss_ce": loss_sum / loss_denominator,
        "top1_acc": correct1 / n_cells,
    }
    if top5:
        metrics["top5_acc"] = correct5 / n_cells
    return metrics


def fit_student(
    model,
    train_loader,
    val_loader,
    train_criterion,
    val_criterion,
    optimizer,
    device,
    cfg,
    checkpoint,
    run,
):
    best_loss = float("inf")
    best_epoch = stale_epochs = 0
    train_rows, val_rows = [], []
    for epoch in range(1, cfg["max_epochs"] + 1):
        train_metrics = run_epoch(
            model, train_loader, train_criterion, device, optimizer=optimizer
        )
        val_metrics = run_epoch(model, val_loader, val_criterion, device, top5=True)
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch:3d} | train ce {train_metrics['loss_ce']:.4f} "
            f"top1 {train_metrics['top1_acc']:.4f} | "
            f"val ce {val_metrics['loss_ce']:.4f} "
            f"top1 {val_metrics['top1_acc']:.4f} "
            f"top5 {val_metrics['top5_acc']:.4f}"
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

        val_loss = val_metrics["loss_ce"]
        if val_loss < best_loss - 1e-6:
            best_loss, best_epoch, stale_epochs = val_loss, epoch, 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "val_loss_ce": val_loss,
                },
                checkpoint,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= cfg["patience"]:
                print(f"early stopping at epoch {epoch} (best {best_epoch})")
                break
    return best_epoch, best_loss, train_rows, val_rows


def save_features(path, sample_ids, cell_ids, outputs):
    np.savez(
        path,
        sample_id=np.asarray(sample_ids),
        cell_id=np.asarray(cell_ids),
        **outputs,
    )


def export_label_features(
    model, cfg, samples, device, out_dir, split_name, cell_ids_by_sample=None
):
    all_samples, all_cells = [], []
    output_parts = {
        "features_penultimate": [],
        "prototype_probs": [],
        "prototype_logits": [],
    }
    sidecar_dir = out_dir / f"{split_name}_features"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    for sample_id in samples:
        uni, cell_ids = load_uni_embeddings(cfg["data_root"], sample_id, cfg["uni_key"])
        if cell_ids_by_sample is not None:
            expected_ids = np.asarray(cell_ids_by_sample[sample_id]).astype(str)
            positions = {cell_id: row for row, cell_id in enumerate(cell_ids)}
            missing = [cell_id for cell_id in expected_ids if cell_id not in positions]
            if missing:
                raise ValueError(f"{sample_id}: {len(missing)} assigned cells lack UNI")
            rows = np.fromiter(
                (positions[cell_id] for cell_id in expected_ids),
                dtype=int,
                count=len(expected_ids),
            )
            uni, cell_ids = uni[rows], expected_ids
        outputs = predict_student_features(
            model, uni, device, cfg.get("inference_batch_size", cfg["batch_size"])
        )
        np.savez(
            sidecar_dir / f"{sample_id}_prototype.npz",
            cell_ids=cell_ids,
            **outputs,
        )
        all_samples.append(np.full(len(cell_ids), sample_id))
        all_cells.append(cell_ids)
        for name, values in outputs.items():
            output_parts[name].append(values)

    outputs = {name: np.concatenate(parts) for name, parts in output_parts.items()}
    save_features(
        out_dir / f"features_{split_name}.npz",
        np.concatenate(all_samples),
        np.concatenate(all_cells),
        outputs,
    )


def save_prototype_config(cfg, path, n_fit_cells, n_train_cells):
    prototype_cfg = {
        "algorithm": "MiniBatchKMeans",
        "fit_split": "distill_train",
        "normalization": "l2",
        "num_prototypes": cfg["num_prototypes"],
        "seed": cfg["kmeans_seed"],
        "max_cells_for_kmeans": cfg["max_cells_for_kmeans"],
        "n_fit_cells": int(n_fit_cells),
        "n_train_cells": int(n_train_cells),
        "batch_size": cfg["kmeans_batch_size"],
        "max_iter": cfg["kmeans_max_iter"],
        "n_init": cfg["kmeans_n_init"],
        "sklearn_version": sklearn_version,
    }
    with open(path, "w") as fh:
        yaml.safe_dump(prototype_cfg, fh, sort_keys=False)


def save_kmeans(model, path):
    path = Path(path)
    tmp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp.joblib")
    try:
        joblib.dump(model, tmp_path, compress=3)
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit-samples", type=int)
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    np.random.seed(cfg["seed"])
    out_dir = Path(cfg["output_dir"])
    checkpoint_dir = Path(cfg["checkpoint_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "resolved_config.yaml", "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    print(yaml.safe_dump(cfg, sort_keys=False).rstrip())

    splits = load_splits(cfg["splits_path"])
    splits["label_test"] = select_label_test_samples(cfg, splits["label_test"])
    if args.limit_samples is not None:
        splits = {
            name: samples[: args.limit_samples] for name, samples in splits.items()
        }
    train_cache = build_cache(
        cfg, splits["distill_train"], "distill_train", args.limit_samples
    )
    val_cache = build_cache(
        cfg, splits["distill_val"], "distill_val", args.limit_samples
    )
    print(f"cells: train={len(train_cache):,} val={len(val_cache):,}")

    kmeans, fit_rows = fit_prototypes(
        train_cache.terra,
        cfg["num_prototypes"],
        cfg["max_cells_for_kmeans"],
        cfg["kmeans_seed"],
        cfg["kmeans_batch_size"],
        cfg["kmeans_max_iter"],
        cfg["kmeans_n_init"],
    )
    kmeans_path = out_dir / (
        f"kmeans_k{cfg['num_prototypes']}_seed{cfg['kmeans_seed']}.joblib"
    )
    save_kmeans(kmeans, kmeans_path)
    save_prototype_config(
        cfg, out_dir / "prototype_config.yaml", len(fit_rows), len(train_cache)
    )

    train_ids = assign_prototypes(
        kmeans, train_cache.terra, cfg["assignment_batch_size"]
    )
    val_ids = assign_prototypes(kmeans, val_cache.terra, cfg["assignment_batch_size"])
    label_samples, label_cells, label_ids = assign_label_samples(
        kmeans, cfg, splits["label_test"]
    )
    save_assignments(
        out_dir / "kmeans_assignments_train.npz",
        train_cache.sample_ids,
        train_cache.cell_ids,
        train_ids,
    )
    save_assignments(
        out_dir / "kmeans_assignments_val.npz",
        val_cache.sample_ids,
        val_cache.cell_ids,
        val_ids,
    )
    label_split_name = label_test_split_name(cfg)
    save_assignments(
        out_dir / f"kmeans_assignments_{label_split_name}.npz",
        label_samples,
        label_cells,
        label_ids,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PrototypeStudent(
        cfg["uni_dim"],
        cfg["hidden_dim"],
        cfg["feat_dim"],
        cfg["num_prototypes"],
        cfg["dropout"],
    ).to(device)
    class_weights = None
    if cfg["use_class_weights"]:
        class_weights = inverse_sqrt_class_weights(
            train_ids,
            cfg["num_prototypes"],
            tuple(cfg["class_weight_clip"]),
        ).to(device)
    train_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    val_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    train_loader = make_loader(train_cache.uni, train_ids, cfg, training=True)
    val_loader = make_loader(val_cache.uni, val_ids, cfg, training=False)
    run = init_wandb(cfg)
    checkpoint = checkpoint_dir / "best.pt"
    best_epoch, best_loss, train_rows, val_rows = fit_student(
        model,
        train_loader,
        val_loader,
        train_criterion,
        val_criterion,
        optimizer,
        device,
        cfg,
        checkpoint,
        run,
    )
    pd.DataFrame(train_rows).to_csv(out_dir / "train_metrics.csv", index=False)
    pd.DataFrame(val_rows).to_csv(out_dir / "val_metrics.csv", index=False)

    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])
    val_outputs = predict_student_features(
        model,
        val_cache.uni,
        device,
        cfg.get("inference_batch_size", cfg["batch_size"]),
    )
    save_features(
        out_dir / "features_distill_val.npz",
        val_cache.sample_ids,
        val_cache.cell_ids,
        val_outputs,
    )
    label_cells_by_sample = {
        sample_id: label_cells[label_samples == sample_id]
        for sample_id in splits["label_test"]
    }
    del (
        train_loader,
        val_loader,
        train_cache,
        val_cache,
        train_ids,
        val_ids,
        val_outputs,
    )
    gc.collect()
    export_label_features(
        model,
        cfg,
        splits["label_test"],
        device,
        out_dir,
        label_split_name,
        label_cells_by_sample,
    )

    summary = {"best_epoch": best_epoch, "best_val_loss_ce": best_loss}
    with open(out_dir / "prototype_eval.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    if run is not None:
        run.summary.update(summary)
        run.finish()


if __name__ == "__main__":
    main()
