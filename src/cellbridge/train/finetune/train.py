"""Train the selective UNI2-h baseline on an OOD classification dataset."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import ConcatDataset, DataLoader

from cellbridge.config import load_config
from cellbridge.evaluation import (
    ClassificationResult,
    evaluate_predictions,
    save_result as save_classification_result,
)
from cellbridge.inference.embed import sources
from cellbridge.train.finetune.data import iter_datasets, prepare_task
from cellbridge.train.finetune.model import (
    UNFREEZING_POLICY,
    build_classifier,
    make_checkpoint,
)

METHOD = "uni_selective_finetune"


def validate_config(config: dict) -> dict:
    """Validate the small, explicit fine-tuning configuration."""
    required = {"profile", "training", "output_dir"}
    if not required <= set(config):
        raise ValueError(f"config is missing {sorted(required - set(config))}")

    training = config["training"]
    required_training = {
        "batch_size",
        "max_epochs",
        "patience",
        "backbone_lr",
        "head_lr",
        "weight_decay",
        "num_workers",
        "seed",
    }
    if not required_training <= set(training):
        raise ValueError(
            "training config is missing " f"{sorted(required_training - set(training))}"
        )
    if int(training["batch_size"]) < 1:
        raise ValueError("training.batch_size must be positive")
    if int(training["max_epochs"]) < 1 or int(training["patience"]) < 1:
        raise ValueError("training.max_epochs and patience must be positive")
    if min(float(training["backbone_lr"]), float(training["head_lr"])) <= 0:
        raise ValueError("training learning rates must be positive")
    if float(training["weight_decay"]) < 0:
        raise ValueError("training.weight_decay must be non-negative")
    if int(training["num_workers"]) < 0:
        raise ValueError("training.num_workers must be non-negative")

    weights_path = config.get("weights_path")
    if weights_path and not Path(weights_path).exists():
        raise FileNotFoundError(f"weights_path does not exist: {weights_path}")
    return config


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(requested: str | None) -> torch.device:
    device = torch.device(requested or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _build_model(config, num_classes, device):
    _set_seed(int(config["training"]["seed"]))
    model = build_classifier(
        num_classes,
        weights_path=config.get("weights_path"),
    ).to(device)
    if model.transform is None:
        raise ValueError("Trident UNI did not expose an evaluation transform")
    return model


def _optimizer(model, training):
    backbone = list(model.backbone_parameters())
    head = list(model.head.parameters())
    grouped = {id(parameter) for parameter in [*backbone, *head]}
    trainable = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    if not backbone or grouped != trainable:
        raise ValueError("optimizer groups do not match trainable parameters")
    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": float(training["backbone_lr"])},
            {"params": head, "lr": float(training["head_lr"])},
        ],
        weight_decay=float(training["weight_decay"]),
    )


def _loader(task, splits, transform, training, device, *, shuffle, seed):
    datasets = tuple(iter_datasets(task, splits, transform=transform))
    if not datasets:
        raise ValueError(f"no datasets found for splits {sorted(splits)}")
    num_workers = int(training["num_workers"])
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        ConcatDataset(datasets),
        batch_size=int(training["batch_size"]),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def _class_weights(task, splits, device):
    rows = task.metadata.loc[task.metadata["split"].isin(splits), "label"]
    targets = rows.map(task.class_to_index).to_numpy(dtype=int)
    counts = np.bincount(targets, minlength=len(task.classes))
    if np.any(counts == 0):
        raise ValueError("training data does not contain every class")
    weights = len(targets) / (len(task.classes) * counts)
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def _autocast(model, device):
    dtype = model.precision
    enabled = device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def _train_epoch(
    model,
    task,
    splits,
    optimizer,
    class_weights,
    training,
    device,
    epoch,
    *,
    loader=None,
):
    model.train()
    total_loss = 0.0
    total_weight = 0.0
    total = 0
    if loader is None:
        loader = _loader(
            task,
            splits,
            model.transform,
            training,
            device,
            shuffle=True,
            seed=int(training["seed"]) + int(epoch),
        )
    for batch in loader:
        images = batch["image"].to(device, non_blocking=device.type == "cuda")
        targets = batch["target"].to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        with _autocast(model, device):
            losses = F.cross_entropy(
                model(images),
                targets,
                weight=class_weights,
                reduction="none",
            )
            batch_weight = class_weights[targets].sum()
            loss = losses.sum() / batch_weight
        loss.backward()
        optimizer.step()
        total_loss += float(losses.detach().sum())
        total_weight += float(batch_weight.detach())
        total += len(targets)
    expected = int(task.metadata["split"].isin(splits).sum())
    if total != expected:
        raise ValueError(f"trained on {total} cells; expected {expected}")
    return total_loss / total_weight


def _fit(model, task, splits, epochs, config, device):
    training = config["training"]
    optimizer = _optimizer(model, training)
    class_weights = _class_weights(task, splits, device)
    loader = _loader(
        task,
        splits,
        model.transform,
        training,
        device,
        shuffle=True,
        seed=int(training["seed"]),
    )
    for epoch in range(1, int(epochs) + 1):
        yield (
            epoch,
            _train_epoch(
                model,
                task,
                splits,
                optimizer,
                class_weights,
                training,
                device,
                epoch,
                loader=loader,
            ),
        )


@torch.no_grad()
def _predict(model, task, split, config, device, *, loader=None):
    model.eval()
    targets = []
    probabilities = []
    row_ids = []
    total = 0
    if loader is None:
        loader = _loader(
            task,
            (split,),
            model.transform,
            config["training"],
            device,
            shuffle=False,
            seed=int(config["training"]["seed"]),
        )
    for batch in loader:
        images = batch["image"].to(device, non_blocking=device.type == "cuda")
        target = batch["target"].to(device, non_blocking=device.type == "cuda")
        with _autocast(model, device):
            logits = model(images)
        targets.append(target.cpu().numpy())
        probabilities.append(logits.float().softmax(dim=1).cpu().numpy())
        row_ids.extend(map(str, batch["row_id"]))
        total += len(target)
    expected = int((task.metadata["split"] == split).sum())
    if total != expected:
        raise ValueError(f"predicted {total} {split} cells; expected {expected}")
    return {
        "target": np.concatenate(targets),
        "probability": np.concatenate(probabilities),
        "row_id": np.asarray(row_ids, dtype=str),
    }


def _evaluate_prediction(prediction, classes, metrics=None) -> ClassificationResult:
    classes = np.asarray(classes, dtype=str)
    return evaluate_predictions(
        ids=prediction["row_id"],
        y_true=classes[prediction["target"]],
        probabilities=prediction["probability"],
        classes=classes,
        metrics=metrics,
    )


def _select_epochs(model, task, config, device):
    training = config["training"]
    best_epoch = None
    best_score = -np.inf
    stale = 0
    history = []
    validation_loader = _loader(
        task,
        {"val"},
        model.transform,
        training,
        device,
        shuffle=False,
        seed=int(training["seed"]),
    )
    for epoch, train_loss in _fit(
        model,
        task,
        {"train"},
        training["max_epochs"],
        config,
        device,
    ):
        validation = _evaluate_prediction(
            _predict(
                model,
                task,
                "val",
                config,
                device,
                loader=validation_loader,
            ),
            task.classes,
        ).metrics
        history.append(
            {
                "stage": "selection",
                "epoch": epoch,
                "train_loss": train_loss,
                **{f"val_{name}": value for name, value in validation.items()},
            }
        )
        print(
            f"[selection {epoch}] train_loss={train_loss:.4f} "
            f"val_macro_f1={validation['macro_f1']:.4f}"
        )
        if validation["macro_f1"] > best_score:
            best_epoch = epoch
            best_score = validation["macro_f1"]
            stale = 0
        else:
            stale += 1
        if stale >= int(training["patience"]):
            break
    return best_epoch, best_score, history


def _refit(model, task, epochs, config, device):
    history = []
    for epoch, loss in _fit(
        model,
        task,
        {"train", "val"},
        epochs,
        config,
        device,
    ):
        history.append({"stage": "refit", "epoch": epoch, "train_loss": loss})
        print(f"[refit {epoch}/{epochs}] train_loss={loss:.4f}")
    return history


def _save_result(
    output_dir,
    model,
    result,
    history,
    config,
    dataset_config,
    epochs,
):
    output_dir = Path(output_dir)
    save_classification_result(output_dir, result)
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    with open(output_dir / "resolved_config.yaml", "w") as handle:
        yaml.safe_dump(
            {"finetune": config, "dataset": dataset_config},
            handle,
            sort_keys=False,
        )
    torch.save(
        make_checkpoint(
            model,
            classes=result.classes,
            epochs=epochs,
            seed=int(config["training"]["seed"]),
        ),
        output_dir / "model.pt",
    )


def run(config, dataset_config, task, task_name, device=None):
    """Select an epoch count, refit, and score the untouched test split."""
    device = _resolve_device(device)
    selection_model = _build_model(config, len(task.classes), device)
    epochs, validation_score, history = _select_epochs(
        selection_model,
        task,
        config,
        device,
    )
    del selection_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = _build_model(config, len(task.classes), device)
    history.extend(_refit(model, task, epochs, config, device))
    counts = task.counts
    result = _evaluate_prediction(
        _predict(model, task, "test", config, device),
        task.classes,
        metrics={
            "dataset": str(dataset_config["dataset"]),
            "method": METHOD,
            "task": task_name,
            "initialization": "original_uni2_h",
            "unfreezing": UNFREEZING_POLICY,
            "seed": int(config["training"]["seed"]),
            "n_train": counts["train"],
            "n_validation": counts["val"],
            "n_reference": counts["train"] + counts["val"],
            "n_test": counts["test"],
            "n_classes": len(task.classes),
            "selected_epochs": epochs,
            "selection_macro_f1": validation_score,
        },
    )
    output_dir = Path(config["output_dir"]) / str(dataset_config["dataset"]) / task_name
    _save_result(
        output_dir,
        model,
        result,
        history,
        config,
        dataset_config,
        epochs,
    )
    print(
        f"[{dataset_config['dataset']}/{task_name}] "
        f"test_macro_f1={result.metrics['macro_f1']:.4f} -> {output_dir}"
    )
    return result.metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "train"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--task", choices=("all", "immune"), default="all")
    parser.add_argument("--device")
    args = parser.parse_args(argv)

    config = validate_config(load_config(args.config))
    dataset_config = load_config(args.dataset_config)
    if not dataset_config.get("dataset") or not isinstance(
        dataset_config.get("data"), dict
    ):
        raise ValueError("dataset config requires dataset and data")
    configured_tasks = dataset_config.get("probe", {}).get("tasks", ["all"])
    if args.task not in configured_tasks:
        raise ValueError(
            f"{dataset_config['dataset']} does not configure task {args.task!r}"
        )

    profile = load_config(config["profile"])
    source = sources.load(dataset_config["data"])
    metadata = source.check(dataset_config["data"])
    available_row_ids = None
    if dataset_config["data"].get("skip_missing_cells", False):
        available_row_ids = source.available_row_ids(dataset_config["data"])
        unavailable = metadata.loc[~metadata["row_id"].isin(available_row_ids)]
        if len(unavailable):
            counts = (
                unavailable["split"]
                .replace({"valid": "val"})
                .value_counts()
                .sort_index()
            )
            details = ", ".join(
                f"{split_name}={count:,}" for split_name, count in counts.items()
            )
            print(
                f"{dataset_config['dataset']}: excluding {len(unavailable):,} "
                f"cells missing from LMDB masks ({details})"
            )
    task = prepare_task(
        dataset_config["data"],
        profile,
        task=args.task,
        metadata=metadata,
        source=source,
        available_row_ids=available_row_ids,
    )
    counts = task.counts
    print(
        f"{dataset_config['dataset']}/{args.task}: "
        f"train={counts['train']:,}, val={counts['val']:,}, "
        f"test={counts['test']:,}, classes={len(task.classes)}"
    )
    print(f"trainable: {UNFREEZING_POLICY}")
    if args.command == "train":
        run(config, dataset_config, task, args.task, args.device)


if __name__ == "__main__":
    main()
