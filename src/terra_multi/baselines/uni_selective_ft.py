"""Run the one-seed supervised selective-UNI baseline."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Sampler

from terra_multi.baselines.uni_selective_ft_data import (
    CachedOodCropDataset,
    GroupedBatchSampler,
    HestClassificationDataset,
    prepare_ood_crop_cache,
)
from terra_multi.baselines.uni_selective_ft_model import (
    build_classifier,
    make_checkpoint,
)
from terra_multi.config import load_config
from terra_multi.lora.data import split_samples

BASELINE_NAME = "uni_selective_ft"


def validate_config(cfg):
    required = {"baseline", "model", "training", "seed", "output_dir", "hest", "phenocell", "pannuke"}
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"config is missing {sorted(missing)}")
    if cfg["baseline"]["name"] != BASELINE_NAME:
        raise ValueError(f"baseline.name must be {BASELINE_NAME!r}")
    if int(cfg["seed"]) != 0:
        raise ValueError("this comparison uses Cellbridge's single seed: 0")
    if cfg["model"].get("encoder", "uni_v2") != "uni_v2":
        raise ValueError("selective fine-tuning must start from original UNI2-h")
    if int(cfg["model"].get("trainable_blocks", 1)) != 1:
        raise ValueError("the registered baseline selectively unfreezes exactly the last block")
    if int(cfg["hest"]["crop"]["target_size"]) != 224:
        raise ValueError("HEST UNI input must be 224x224")
    if float(cfg["hest"]["crop"]["target_pixel_size_um"]) != 0.5:
        raise ValueError("HEST must use the 0.5 um/pixel Cellbridge field of view")
    for name in ("phenocell", "pannuke"):
        if int(cfg[name]["crop"]["source_size"]) != 96:
            raise ValueError(f"{name} must preserve the existing 96px source crop")
        if int(cfg[name]["crop"]["target_size"]) != 224:
            raise ValueError(f"{name} UNI input must be 224x224")
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class IndexBatchSampler(Sampler[list[int]]):
    """Deterministic batches for a single HEST section."""

    def __init__(self, indices, batch_size, *, seed=0, shuffle=True):
        self.indices = np.asarray(indices, dtype=int)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        if self.batch_size < 1 or not len(self.indices):
            raise ValueError("batch_size and indices must be nonempty")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = self.indices
        if self.shuffle:
            indices = np.random.default_rng([self.seed, self.epoch]).permutation(indices)
        for start in range(0, len(indices), self.batch_size):
            yield indices[start : start + self.batch_size].tolist()


def _loader(dataset, indices, settings, *, seed, shuffle):
    batch_size = int(settings["batch_size"])
    if isinstance(dataset, CachedOodCropDataset):
        sampler = GroupedBatchSampler(
            dataset, indices, batch_size, seed=seed, shuffle=shuffle
        )
    else:
        sampler = IndexBatchSampler(
            indices, batch_size, seed=seed, shuffle=shuffle
        )
    workers = int(settings.get("num_workers", 8))
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    return loader, sampler


def _class_weights(targets, indices, num_classes, device):
    counts = np.bincount(np.asarray(targets)[indices], minlength=num_classes)
    weights = np.zeros(num_classes, dtype=np.float32)
    present = counts > 0
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    return torch.from_numpy(weights).to(device)


def _device(cfg):
    requested = cfg["training"].get("device", "cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def _build(cfg, num_classes, device):
    set_seed(int(cfg["seed"]))
    weights_path = cfg["model"].get("weights_path")
    model = build_classifier(
        num_classes,
        weights_path=weights_path,
        image_dim=int(cfg["model"].get("image_dim", 1536)),
        trainable_blocks=int(cfg["model"].get("trainable_blocks", 1)),
    )
    return model.to(device)


def _optimizer(model, settings):
    return torch.optim.AdamW(
        [
            {
                "params": list(model.backbone_parameters()),
                "lr": float(settings["backbone_lr"]),
            },
            {
                "params": list(model.head.parameters()),
                "lr": float(settings["head_lr"]),
            },
        ],
        weight_decay=float(settings.get("weight_decay", 0.05)),
    )


def _autocast(model, device):
    return torch.autocast(
        device_type=device.type,
        dtype=model.precision,
        enabled=device.type == "cuda",
    )


def train_epoch(model, loader, optimizer, class_weights, device, grad_clip):
    model.train()
    total_loss, total = 0.0, 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(model, device):
            logits = model(images)
            loss = F.cross_entropy(logits, targets, weight=class_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()
        total_loss += float(loss.detach()) * len(targets)
        total += len(targets)
    return total_loss / max(total, 1)


def _row_ids(batch):
    values = batch["row_id"]
    if torch.is_tensor(values):
        return values.cpu().numpy()
    return np.asarray(values)


def predict(model, loader, device):
    model.eval()
    targets, probabilities, row_ids = [], [], []
    total_loss, total = 0.0, 0
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            with _autocast(model, device):
                logits = model(images)
                loss = F.cross_entropy(logits, target)
            targets.append(target.cpu().numpy())
            probabilities.append(logits.float().softmax(dim=1).cpu().numpy())
            row_ids.append(_row_ids(batch))
            total_loss += float(loss) * len(target)
            total += len(target)
    return {
        "loss": total_loss / max(total, 1),
        "target": np.concatenate(targets),
        "probability": np.concatenate(probabilities),
        "row_id": np.concatenate(row_ids),
    }


def _macro_auroc(target, probability, num_classes):
    try:
        if num_classes == 2:
            return float(roc_auc_score(target, probability[:, 1]))
        return float(
            roc_auc_score(
                target,
                probability,
                multi_class="ovr",
                average="macro",
                labels=np.arange(num_classes),
            )
        )
    except ValueError:
        return float("nan")


def score_prediction(prediction, classes):
    target = prediction["target"]
    predicted = prediction["probability"].argmax(axis=1)
    labels = np.arange(len(classes))
    return {
        "loss": float(prediction["loss"]),
        "accuracy": float(accuracy_score(target, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(target, predicted)),
        "macro_f1": float(
            f1_score(target, predicted, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(target, predicted, labels=labels, average="weighted", zero_division=0)
        ),
        "macro_auroc": _macro_auroc(target, prediction["probability"], len(classes)),
    }


def select_epoch(model, dataset, train_indices, val_indices, cfg, device):
    settings = cfg["training"]
    train_loader, train_sampler = _loader(
        dataset, train_indices, settings, seed=cfg["seed"], shuffle=True
    )
    val_loader, _ = _loader(
        dataset, val_indices, settings, seed=cfg["seed"], shuffle=False
    )
    optimizer = _optimizer(model, settings)
    weights = _class_weights(dataset.targets, train_indices, len(dataset.classes), device)
    history, best_epoch, best_score, stale = [], 0, -np.inf, 0
    for epoch in range(int(settings["max_epochs"])):
        train_sampler.set_epoch(epoch)
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            weights,
            device,
            settings.get("grad_clip", 1.0),
        )
        validation = score_prediction(predict(model, val_loader, device), dataset.classes)
        history.append(
            {"stage": "selection", "epoch": epoch + 1, "train_loss": train_loss, **{f"val_{key}": value for key, value in validation.items()}}
        )
        print(
            f"[selection epoch {epoch + 1}] train loss={train_loss:.4f}, "
            f"val macro-F1={validation['macro_f1']:.4f}"
        )
        if validation["macro_f1"] > best_score:
            best_epoch, best_score, stale = epoch, validation["macro_f1"], 0
        else:
            stale += 1
        if stale >= int(settings.get("patience", 2)):
            break
    return best_epoch, best_score, history


def refit(model, dataset, train_indices, epochs, cfg, device):
    settings = cfg["training"]
    loader, sampler = _loader(
        dataset, train_indices, settings, seed=cfg["seed"], shuffle=True
    )
    optimizer = _optimizer(model, settings)
    weights = _class_weights(dataset.targets, train_indices, len(dataset.classes), device)
    history = []
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        loss = train_epoch(
            model,
            loader,
            optimizer,
            weights,
            device,
            settings.get("grad_clip", 1.0),
        )
        history.append({"stage": "refit", "epoch": epoch + 1, "train_loss": loss})
        print(f"[refit epoch {epoch + 1}/{epochs}] train loss={loss:.4f}")
    return history


def _atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_result(output_dir, model, dataset, prediction, metrics, history, cfg, best_epoch):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predicted = prediction["probability"].argmax(axis=1)
    classes = np.asarray(dataset.classes).astype(str)
    np.savez(
        output_dir / "predictions.npz",
        row_id=prediction["row_id"],
        y_true=classes[prediction["target"]],
        y_pred=classes[predicted],
        y_proba=prediction["probability"],
        classes=classes,
    )
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    per_class = f1_score(
        prediction["target"],
        predicted,
        labels=np.arange(len(classes)),
        average=None,
        zero_division=0,
    )
    pd.DataFrame({"class": classes, "f1": per_class}).to_csv(
        output_dir / "per_class_f1.csv", index=False
    )
    with open(output_dir / "resolved_config.yaml", "w") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    _atomic_torch_save(
        make_checkpoint(
            model,
            classes=classes,
            best_epoch=best_epoch,
            seed=cfg["seed"],
            config=cfg,
        ),
        output_dir / "model.pt",
    )


def run_training(dataset, output_dir, cfg, *, selection_train, val, refit_train, test, identity):
    device = _device(cfg)
    model = _build(cfg, len(dataset.classes), device)
    transform = model.eval_transforms
    if transform is None:
        raise ValueError("Trident UNI did not expose eval_transforms")
    dataset.transform = transform
    best_epoch, best_score, history = select_epoch(
        model, dataset, selection_train, val, cfg, device
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = _build(cfg, len(dataset.classes), device)
    dataset.transform = model.eval_transforms
    refit_history = refit(model, dataset, refit_train, best_epoch + 1, cfg, device)
    test_loader, _ = _loader(
        dataset, test, cfg["training"], seed=cfg["seed"], shuffle=False
    )
    prediction = predict(model, test_loader, device)
    scores = score_prediction(prediction, dataset.classes)
    metrics = {
        "baseline": BASELINE_NAME,
        "initialization": "original_uni2_h",
        "unfrozen": "last_transformer_block+final_norm+head",
        "seed": int(cfg["seed"]),
        **identity,
        "n_train_selection": len(selection_train),
        "n_val": len(val),
        "n_train_refit": len(refit_train),
        "n_test": len(test),
        "n_classes": len(dataset.classes),
        "best_epoch": best_epoch + 1,
        "val_macro_f1": best_score,
        **{f"test_{key}": value for key, value in scores.items()},
    }
    _write_result(
        output_dir,
        model,
        dataset,
        prediction,
        metrics,
        history + refit_history,
        cfg,
        best_epoch + 1,
    )
    print(
        f"[{identity}] test macro-F1={scores['macro_f1']:.4f}, "
        f"balanced accuracy={scores['balanced_accuracy']:.4f}"
    )
    return metrics


def train_hest(cfg, sample_id):
    dataset = HestClassificationDataset(
        cfg["hest"], sample_id, seed=cfg["seed"]
    )
    output = Path(cfg["output_dir"]) / "hest" / str(sample_id)
    return run_training(
        dataset,
        output,
        cfg,
        selection_train=dataset.selection_train_indices,
        val=dataset.val_indices,
        refit_train=dataset.train_indices,
        test=dataset.test_indices,
        identity={"dataset": "hest", "sample_id": str(sample_id), "task": "all"},
    )


def train_ood(cfg, dataset_name, task):
    dataset_cfg = cfg[dataset_name]
    allowed = dataset_cfg.get("tasks", ["all"])
    if task not in allowed:
        raise ValueError(f"{dataset_name}: task {task!r} is not configured in {allowed}")
    dataset = CachedOodCropDataset(dataset_cfg, task)
    output = Path(cfg["output_dir"]) / dataset_name / task
    return run_training(
        dataset,
        output,
        cfg,
        selection_train=dataset.train_indices,
        val=dataset.val_indices,
        refit_train=np.concatenate([dataset.train_indices, dataset.val_indices]),
        test=dataset.test_indices,
        identity={"dataset": dataset_name, "sample_id": None, "task": task},
    )


def check(cfg):
    samples = split_samples(
        cfg["hest"]["splits_path"],
        "label_test_fast",
        cfg["hest"]["label_test_fast_path"],
    )
    if len(samples) != int(cfg["hest"].get("label_test_fast_n", 5)):
        raise ValueError(f"unexpected fast-HEST cohort: {samples}")
    print(f"seed: {cfg['seed']} (Cellbridge)")
    print("initialization: original UNI2-h")
    print("trainable: last transformer block + final norm + task head")
    print(f"fast HEST: {', '.join(samples)}")
    for name in ("phenocell", "pannuke"):
        metadata = pd.read_csv(cfg[name]["data"]["metadata"])
        labels = sorted(metadata["label"].astype(str).unique())
        print(f"{name}: {len(metadata):,} cells, {len(labels)} source classes, tasks={cfg[name]['tasks']}")


def aggregate(cfg):
    paths = sorted(Path(cfg["output_dir"]).glob("**/metrics.csv"))
    rows = []
    for path in paths:
        table = pd.read_csv(path)
        if len(table) == 1 and table.iloc[0].get("baseline") == BASELINE_NAME:
            rows.append(table.iloc[0].to_dict())
    if not rows:
        raise ValueError("no completed selective-UNI metrics were found")
    output = Path(cfg["output_dir"]) / "summary.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"wrote {len(rows)} result rows -> {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check")
    prep = commands.add_parser("prepare-ood")
    prep.add_argument("--dataset", choices=("phenocell", "pannuke"), required=True)
    prep.add_argument("--overwrite", action="store_true")
    prep.add_argument("--limit-samples", type=int)
    hest = commands.add_parser("train-hest")
    hest.add_argument("--sample", required=True)
    ood = commands.add_parser("train-ood")
    ood.add_argument("--dataset", choices=("phenocell", "pannuke"), required=True)
    ood.add_argument("--task", choices=("all", "immune"), default="all")
    commands.add_parser("aggregate")
    args = parser.parse_args(argv)
    cfg = validate_config(load_config(args.config))

    if args.command == "check":
        check(cfg)
    elif args.command == "prepare-ood":
        prepare_ood_crop_cache(
            args.dataset,
            cfg[args.dataset],
            overwrite=args.overwrite,
            limit_samples=args.limit_samples,
        )
    elif args.command == "train-hest":
        train_hest(cfg, args.sample)
    elif args.command == "train-ood":
        train_ood(cfg, args.dataset, args.task)
    else:
        aggregate(cfg)


if __name__ == "__main__":
    main()
