"""Versioned checkpoints for the TERRA flow and SoftREPA prompts."""

from __future__ import annotations

import torch

from cellbridge.flow.model import CellFlow, SoftPromptTokens
from cellbridge.jepa.model import TargetNormalizer

BASE_FORMAT = "cellbridge.terra_cell_flow/1"
PROMPT_FORMAT = "cellbridge.terra_cell_flow_prompts/1"


def save_base(path, model, normalizer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": BASE_FORMAT,
            "model_config": model.config,
            "model_state": model.state_dict(),
            "normalizer_state": normalizer.state_dict(),
        },
        path,
    )


def load_base(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != BASE_FORMAT:
        raise ValueError(f"{path}: unsupported checkpoint format")
    model = CellFlow(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    state = checkpoint["normalizer_state"]
    normalizer = TargetNormalizer(state["mean"], state["scale"])
    return model.to(device), normalizer.to(device)


def save_prompts(path, prompts: SoftPromptTokens) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": PROMPT_FORMAT,
            "prompt_config": {
                "depth": prompts.depth,
                "token_count": prompts.token_count,
                "model_dim": prompts.model_dim,
                "time_dependent": prompts.time_tokens is not None,
                "time_bins": prompts.time_bins,
            },
            "prompt_state": prompts.state_dict(),
        },
        path,
    )


def load_prompts(path, device) -> SoftPromptTokens:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != PROMPT_FORMAT:
        raise ValueError(f"{path}: unsupported prompt checkpoint format")
    prompts = SoftPromptTokens(**checkpoint["prompt_config"])
    prompts.load_state_dict(checkpoint["prompt_state"])
    return prompts.to(device)
