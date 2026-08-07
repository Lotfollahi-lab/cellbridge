"""Optimization and held-out diagnostics for the TERRA cell flow."""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cellbridge.flow.compatibility import (
    endpoint_error_matrix,
    false_negative_mask,
    nearest_terra_indices,
    softrepa_loss,
    summarize_error_matrices,
)
from cellbridge.flow.path import linear_path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def amp_settings(device: torch.device, precision: str) -> tuple[bool, torch.dtype]:
    if precision not in {"fp32", "bf16"}:
        raise ValueError("precision must be fp32 or bf16")
    if device.type != "cuda" or precision == "fp32":
        return False, torch.float32
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 was requested but is not supported")
    return True, torch.bfloat16


def _loader(dataset, batch_size, *, shuffle, seed, drop_last=False) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _cuda_generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(int(seed))


def train_base(model, normalizer, dataset, settings, device) -> list[dict]:
    """Fit the ordinary conditional flow using STFlow's clean-endpoint MSE."""

    seed = int(settings["seed"])
    set_seed(seed)
    model.train()
    normalizer.to(device)
    loader = _loader(dataset, settings["batch_size"], shuffle=True, seed=seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    amp_enabled, amp_dtype = amp_settings(device, settings["precision"])
    path_generator = _cuda_generator(device, seed + 101)
    records = []
    for epoch in range(1, int(settings["epochs"]) + 1):
        total_loss = 0.0
        total_cells = 0
        for raw_batch in loader:
            batch = _to_device(raw_batch, device)
            target = normalizer.normalize(batch["target"])
            state, time, _ = linear_path(target, generator=path_generator)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                prediction = model(
                    state,
                    time,
                    batch["condition"],
                    batch["condition_coordinates"],
                )
                loss = F.mse_loss(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(target)
            total_cells += len(target)
        record = {"epoch": epoch, "loss": total_loss / total_cells}
        records.append(record)
        print(f"base epoch {epoch}: loss={record['loss']:.6f}", flush=True)
    return records


@torch.no_grad()
def diagnose(model, normalizer, dataset, settings, device, *, prompts=None) -> dict:
    """Measure matched/shuffled native endpoint error on held-out cells."""

    model.eval()
    if prompts is not None:
        prompts.eval()
    seed = int(settings["seed"])
    loader = _loader(
        dataset,
        settings["batch_size"],
        shuffle=True,
        seed=seed,
        drop_last=True,
    )
    generator = _cuda_generator(device, seed + 211)
    amp_enabled, amp_dtype = amp_settings(device, settings["precision"])
    matrices = []
    for batch_number, raw_batch in enumerate(loader):
        if batch_number >= int(settings["max_batches"]):
            break
        batch = _to_device(raw_batch, device)
        target = normalizer.normalize(batch["target"])
        state, time, _ = linear_path(target, generator=generator)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            errors = endpoint_error_matrix(
                model,
                state,
                target,
                time,
                batch["condition"],
                batch["condition_coordinates"],
                prompts=prompts,
                candidate_chunk=int(settings["candidate_chunk"]),
            )
        matrices.append(errors.float().cpu().numpy())
    return summarize_error_matrices(matrices)


def train_prompts(model, prompts, normalizer, dataset, settings, device) -> list[dict]:
    """Freeze the base flow and optimize only SoftREPA conditioning tokens."""

    seed = int(settings["seed"])
    set_seed(seed)
    model.requires_grad_(False).eval()
    prompts.train()
    optimizer = torch.optim.AdamW(
        prompts.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    loader = _loader(dataset, settings["batch_size"], shuffle=True, seed=seed, drop_last=True)
    iterator = iter(loader)
    generator = _cuda_generator(device, seed + 307)
    amp_enabled, amp_dtype = amp_settings(device, settings["precision"])
    exclusion = settings["false_negative_exclusion"]
    nearest = nearest_terra_indices(
        dataset.anchor_terra,
        int(exclusion["nearest_terra"]),
        device,
        chunk_size=int(exclusion.get("chunk_size", 512)),
    ) if exclusion.get("enabled", True) else None
    records, running = [], 0.0
    log_every = int(settings.get("log_every", 50))
    for step in range(1, int(settings["max_steps"]) + 1):
        try:
            raw_batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            raw_batch = next(iterator)
        batch = _to_device(raw_batch, device)
        target = normalizer.normalize(batch["target"])
        state, time, _ = linear_path(target, generator=generator)
        valid = None
        if exclusion.get("enabled", True):
            valid = false_negative_mask(
                batch["anchor_position"],
                batch["center_coordinates"],
                nearest,
                spatial_radius_um=float(exclusion["spatial_radius_um"]),
            )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            errors = endpoint_error_matrix(
                model,
                state,
                target,
                time,
                batch["condition"],
                batch["condition_coordinates"],
                prompts=prompts,
                candidate_chunk=int(settings["candidate_chunk"]),
            )
            loss = softrepa_loss(
                errors.float(),
                scale=float(settings["scale"]),
                temperature=float(settings["temperature"]),
                valid_negatives=valid,
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prompts.parameters(), 1.0)
        optimizer.step()
        running += float(loss.detach())
        if step % log_every == 0 or step == int(settings["max_steps"]):
            record = {"step": step, "loss": running / (step % log_every or log_every)}
            records.append(record)
            running = 0.0
            print(f"prompts step {step}: loss={record['loss']:.6f}", flush=True)
    return records


def diagnostic_passes(metrics: dict, gate: dict) -> bool:
    return (
        metrics["relative_error_reduction"] >= float(gate["min_error_reduction"])
        and metrics["retrieval_top1"]
        >= float(gate["min_chance_multiple"]) * metrics["chance_top1"]
    )
