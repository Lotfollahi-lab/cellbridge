"""Generation and leakage-safe phenotype evaluation for the flow pilot."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from cellbridge.evaluation.classification import evaluate_classification
from cellbridge.evaluation.linear_probe import reduce_features
from cellbridge.flow.path import sample_endpoint
from cellbridge.flow.training import amp_settings


@torch.no_grad()
def generate_embeddings(
    model,
    prompts,
    normalizer,
    dataset,
    settings,
    device,
    *,
    seed: int,
) -> np.ndarray:
    """Generate one paired-noise TERRA embedding per dataset anchor."""

    model.eval()
    if prompts is not None:
        prompts.eval()
    loader = DataLoader(
        dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    generator = torch.Generator(device=device).manual_seed(int(seed))
    amp_enabled, amp_dtype = amp_settings(device, settings["precision"])
    generated = []
    for batch in loader:
        condition = batch["condition"].to(device, non_blocking=True)
        coordinates = batch["condition_coordinates"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            normalized = sample_endpoint(
                model,
                condition,
                coordinates,
                model.target_dim,
                steps=int(settings["steps"]),
                generator=generator,
                prompts=prompts,
            )
        generated.append(normalizer.denormalize(normalized.float()).cpu().numpy())
    return np.concatenate(generated).astype(np.float32, copy=False)


def latent_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    predicted = torch.as_tensor(predicted, dtype=torch.float32)
    target = torch.as_tensor(target, dtype=torch.float32)
    if predicted.shape != target.shape:
        raise ValueError("predicted and target embeddings must have matching shapes")
    return {
        "latent_cosine": float(F.cosine_similarity(predicted, target, dim=-1).mean()),
        "latent_mse": float(F.mse_loss(predicted, target)),
    }


def _classification_arm(
    name,
    train_features,
    test_features,
    train_labels,
    test_labels,
    train_ids,
    test_ids,
    settings,
):
    features = np.concatenate((train_features, test_features)).astype(np.float32)
    labels = np.concatenate((train_labels, test_labels))
    row_ids = np.concatenate((train_ids, test_ids))
    train = np.arange(len(train_features))
    test = np.arange(len(train_features), len(features))
    if name == "direct_he":
        features, explained = reduce_features(
            features,
            train,
            int(settings["uni_pca_components"]),
            int(settings["seed"]),
        )
    else:
        explained = None
    result = evaluate_classification(
        features,
        labels,
        train,
        test,
        "linear",
        settings,
        row_ids=row_ids,
    )
    metrics = {"arm": name, **result.metrics, "pca_explained_variance": explained}
    classes = result.predictions["classes"]
    scores = f1_score(
        result.predictions["y_true"],
        result.predictions["y_pred"],
        labels=classes,
        average=None,
        zero_division=0,
    )
    per_class = [
        {"arm": name, "class": str(label), "f1": float(score)}
        for label, score in zip(classes, scores, strict=True)
    ]
    return metrics, per_class


def evaluate_phenotypes(
    train_dataset,
    test_dataset,
    base_train,
    base_test,
    prompt_train,
    prompt_test,
    settings,
) -> tuple[list[dict], list[dict]]:
    """Compare real TERRA, direct H&E, base flow, and prompted flow."""

    arms = {
        "real_terra": (train_dataset.anchor_terra, test_dataset.anchor_terra),
        "direct_he": (train_dataset.anchor_image, test_dataset.anchor_image),
        "flow": (base_train, base_test),
        "flow_softrepa": (prompt_train, prompt_test),
    }
    metrics, per_class = [], []
    for name, (train_features, test_features) in arms.items():
        row, scores = _classification_arm(
            name,
            train_features,
            test_features,
            train_dataset.anchor_labels,
            test_dataset.anchor_labels,
            train_dataset.anchor_cell_ids,
            test_dataset.anchor_cell_ids,
            settings,
        )
        metrics.append(row)
        per_class.extend(scores)
    return metrics, per_class


def scale_up_gate(base_diagnostic, prompt_diagnostic, phenotype_metrics) -> dict:
    by_arm = {row["arm"]: row for row in phenotype_metrics}
    compatibility_improved = (
        prompt_diagnostic["relative_error_reduction"]
        > base_diagnostic["relative_error_reduction"]
        and prompt_diagnostic["retrieval_top1"] > base_diagnostic["retrieval_top1"]
    )
    phenotype_improved = (
        by_arm["flow_softrepa"]["macro_f1"] > by_arm["flow"]["macro_f1"]
        and by_arm["flow_softrepa"]["balanced_accuracy"]
        > by_arm["flow"]["balanced_accuracy"]
    )
    return {
        "compatibility_improved": bool(compatibility_improved),
        "phenotype_improved": bool(phenotype_improved),
        "scale_up": bool(compatibility_improved and phenotype_improved),
    }
