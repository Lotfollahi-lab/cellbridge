"""Strict image-encoder adapter for the released PAST checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from cellbridge.evaluation.artifacts import (
    file_identity,
    git_identity,
    stable_signature,
    write_json,
)

INPUT_SIZE = 224
EMBEDDING_DIM = 1024
IMAGE_PREFIX = "image_encoder.model."


def _image_state_dict(checkpoint):
    state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise ValueError("PAST checkpoint has no model state_dict")
    image_state = {}
    for raw_key, value in state.items():
        key = str(raw_key).removeprefix("module.")
        if not key.startswith(IMAGE_PREFIX):
            continue
        key = key.removeprefix(IMAGE_PREFIX)
        if key in image_state:
            raise ValueError(f"duplicate PAST image key: {key}")
        image_state[key] = value
    if not image_state:
        raise ValueError(f"PAST checkpoint has no {IMAGE_PREFIX!r} weights")
    return image_state


def _load_checkpoint(torch, path):
    # The released training checkpoint stores argparse.Namespace under ``args``.
    with torch.serialization.safe_globals([argparse.Namespace]):
        return torch.load(path, map_location="cpu", weights_only=True)


class PastEncoder:
    """Expose PAST's raw, fine-tuned UNI image-tower representation."""

    def __init__(self, model, device, identity):
        import torch

        self.model = model
        self.device = torch.device(device)
        self.expected_dim = EMBEDDING_DIM
        self.identity = identity
        self.model_signature = stable_signature(identity)
        self._mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
        self._std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)

    def transform(self, image):
        import torch

        image = Image.fromarray(np.asarray(image)).convert("RGB")
        if image.size != (INPUT_SIZE, INPUT_SIZE):
            image = image.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
        values = np.asarray(image, dtype=np.float32) / 255.0
        values = (values - self._mean) / self._std
        values = np.ascontiguousarray(values.transpose(2, 0, 1))
        return torch.from_numpy(values)

    def encode(self, images):
        import torch

        expected = (3, INPUT_SIZE, INPUT_SIZE)
        if images.ndim != 4 or tuple(images.shape[1:]) != expected:
            raise ValueError(
                f"PAST input must be (B,3,224,224), got {tuple(images.shape)}"
            )
        with torch.inference_mode():
            features = self.model(images.to(self.device, non_blocking=True)).float()
        if features.ndim != 2 or features.shape[1] != self.expected_dim:
            raise ValueError(
                f"expected (B,{self.expected_dim}) features, got {tuple(features.shape)}"
            )
        if not torch.isfinite(features).all():
            raise ValueError("PAST returned non-finite features")
        return features.cpu().numpy()


def load_encoder(cfg, device=None):
    import timm
    import torch

    model_cfg = cfg["model"]
    checkpoint_path = Path(model_cfg["checkpoint"]).resolve()
    checkpoint_identity = file_identity(checkpoint_path, checksum=True)
    if checkpoint_identity["size"] != int(model_cfg["checkpoint_size"]):
        raise ValueError("PAST checkpoint size does not match release metadata")
    if checkpoint_identity["sha256"] != model_cfg["checkpoint_sha256"]:
        raise ValueError(
            "PAST checkpoint SHA-256 does not match the Hugging Face release"
        )

    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=INPUT_SIZE,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True,
    )
    model.load_state_dict(
        _image_state_dict(_load_checkpoint(torch, checkpoint_path)), strict=True
    )
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device).eval()

    integration = [Path(path) for path in model_cfg["integration_paths"]]
    identity = {
        "baseline": cfg["baseline"]["name"],
        "past_repository": git_identity(model_cfg["repository"]),
        "huggingface_repo": model_cfg["hf_repo"],
        "huggingface_revision": model_cfg["revision"],
        "checkpoint": checkpoint_identity,
        "architecture": "vit_large_patch16_224",
        "feature": "image_encoder.model output",
        "embedding_dim": EMBEDDING_DIM,
        "normalized": False,
        "input_size": INPUT_SIZE,
        "integration": [file_identity(path, checksum=True) for path in integration],
    }
    return PastEncoder(model, device, identity)


def prepare_model(cfg, output_dir, device=None):

    encoder = load_encoder(cfg, device)
    batch = encoder.transform(Image.new("RGB", (INPUT_SIZE, INPUT_SIZE))).unsqueeze(0)
    features = encoder.encode(batch)
    payload = {
        "model_signature": encoder.model_signature,
        "model": encoder.identity,
        "smoke_embedding_shape": list(features.shape),
        "smoke_embedding_norm": float(np.linalg.norm(features[0])),
    }
    write_json(Path(output_dir) / "model_provenance.json", payload)
    print(
        f"PAST ready: {features.shape[1]}-D, signature={encoder.model_signature[:12]}"
    )
    return payload
