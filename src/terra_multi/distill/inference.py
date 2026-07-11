"""Inference and metrics for trained projectors."""

from pathlib import Path

import numpy as np
import torch

from terra_multi.distill.data import load_sample_embeddings


@torch.inference_mode()
def predict_embeddings(model, uni, device, batch_size=8192):
    model.eval()
    outputs = []
    for start in range(0, len(uni), batch_size):
        batch = np.asarray(uni[start : start + batch_size], dtype=np.float32)
        outputs.append(model(torch.from_numpy(batch).to(device)).cpu().numpy())
    if outputs:
        return np.concatenate(outputs).astype(np.float32)
    return np.empty((0, model.out_dim), dtype=np.float32)


def _unit_norm(values):
    return values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-8)


def cosine_and_nmse(student, teacher):
    if student.shape != teacher.shape or len(student) == 0:
        raise ValueError(
            f"expected non-empty matching arrays, got {student.shape} and {teacher.shape}"
        )
    student = _unit_norm(student)
    teacher = _unit_norm(teacher)
    cosine = np.sum(student * teacher, axis=1)
    mse = np.mean(np.sum((student - teacher) ** 2, axis=1))
    variance = np.mean(
        np.sum((teacher - teacher.mean(axis=0, keepdims=True)) ** 2, axis=1)
    )
    return {
        "mean_cosine": float(cosine.mean()),
        "median_cosine": float(np.median(cosine)),
        "normalized_mse": float(mse / variance) if variance > 0 else float("nan"),
        "n_cells": int(len(cosine)),
    }


def evaluate_validation(model, cache, device, out_dir, batch_size):
    predicted = predict_embeddings(model, cache.uni, device, batch_size)
    metrics = cosine_and_nmse(predicted, np.asarray(cache.terra, dtype=np.float32))
    np.savez(
        Path(out_dir) / "pred_embeddings_distill_val.npz",
        cell_ids=cache.cell_ids,
        sample_id=cache.sample_ids,
        pred_emb=predicted,
    )
    return metrics


def evaluate_label_test(model, cfg, samples, device, out_dir, batch_size):
    if not samples:
        raise ValueError("label_test split is empty")

    sample_dir = Path(out_dir) / "label_test_distilled"
    sample_dir.mkdir(parents=True, exist_ok=True)
    all_ids, all_samples, all_predicted, all_cosine = [], [], [], []
    squared_error = teacher_squared = 0.0
    teacher_sum = np.zeros(cfg["terra_dim"], dtype=np.float64)
    n_cells = 0

    for sample_id in samples:
        uni, terra, cell_ids = load_sample_embeddings(
            cfg["data_root"], sample_id, cfg["uni_key"], cfg["terra_key"]
        )
        predicted = predict_embeddings(model, uni, device, batch_size)
        np.savez(
            sample_dir / f"{sample_id}_terra.npz",
            cell_ids=cell_ids,
            cell_emb=predicted,
        )

        teacher = _unit_norm(terra)
        all_cosine.append(np.sum(predicted * teacher, axis=1))
        squared_error += float(np.sum((predicted - teacher) ** 2))
        teacher_squared += float(np.sum(teacher**2))
        teacher_sum += teacher.sum(axis=0)
        n_cells += len(teacher)
        all_ids.append(cell_ids)
        all_samples.append(np.full(len(cell_ids), sample_id))
        all_predicted.append(predicted)

    np.savez(
        Path(out_dir) / "pred_embeddings_label_test.npz",
        cell_ids=np.concatenate(all_ids),
        sample_id=np.concatenate(all_samples),
        pred_emb=np.concatenate(all_predicted),
    )
    cosine = np.concatenate(all_cosine)
    variance = teacher_squared - float(teacher_sum @ teacher_sum) / n_cells
    return {
        "mean_cosine": float(cosine.mean()),
        "median_cosine": float(np.median(cosine)),
        "normalized_mse": squared_error / variance,
        "n_cells": int(n_cells),
        "n_samples": len(samples),
    }
