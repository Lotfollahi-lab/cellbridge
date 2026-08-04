"""Strict adapter for the released VOLTA ovarian momentum encoder."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image

from terra_multi.evaluation.artifacts import (
    file_identity,
    git_identity,
    stable_signature,
    write_json,
)

INPUT_SIZE = 32
EMBEDDING_DIM = 512


def _load_backbone_module(repository: Path):
    path = repository / "backbones" / "resnet" / "model.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("volta_released_resnet", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import VOLTA backbone from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _teacher_state_dict(checkpoint):
    if "state_dict" not in checkpoint:
        raise ValueError("VOLTA checkpoint has no state_dict")
    clean = {}
    for key, value in checkpoint["state_dict"].items():
        key = str(key).removeprefix("module.")
        if key in clean:
            raise ValueError(f"duplicate checkpoint key after prefix removal: {key}")
        clean[key] = value
    prefix = "encoder_k."
    teacher = {
        key.removeprefix(prefix): value
        for key, value in clean.items()
        if key.startswith(prefix)
    }
    if not teacher:
        raise ValueError("VOLTA checkpoint has no momentum encoder weights")
    return teacher


class VoltaEncoder:
    def __init__(self, model, device, mean, std, identity):
        import torch

        self.model = model
        self.device = torch.device(device)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
        self.expected_dim = EMBEDDING_DIM
        self.identity = identity
        self.model_signature = stable_signature(identity)

    def transform(self, image):
        import cv2
        import torch

        image = Image.fromarray(np.asarray(image)).convert("RGB")
        values = cv2.resize(
            np.asarray(image),
            (INPUT_SIZE, INPUT_SIZE),
            interpolation=cv2.INTER_CUBIC,
        ).astype(np.float32)
        values = (values / 255.0 - self.mean) / self.std
        return torch.from_numpy(np.ascontiguousarray(values.transpose(2, 0, 1)))

    def encode(self, images):
        import torch

        if images.ndim != 4 or tuple(images.shape[1:]) != (3, INPUT_SIZE, INPUT_SIZE):
            raise ValueError(
                f"VOLTA input must be (B,3,32,32), got {tuple(images.shape)}"
            )
        with torch.inference_mode():
            features = self.model(images.to(self.device, non_blocking=True)).float()
        if features.ndim != 2 or features.shape[1] != self.expected_dim:
            raise ValueError(
                f"expected (B,{self.expected_dim}) features, got {tuple(features.shape)}"
            )
        if not torch.isfinite(features).all():
            raise ValueError("VOLTA returned non-finite features")
        return features.cpu().numpy()


def load_encoder(cfg, device=None) -> VoltaEncoder:
    import torch

    model_cfg = cfg["model"]
    repository = Path(model_cfg["repository"]).resolve()
    checkpoint_path = Path(model_cfg["checkpoint"]).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if str(checkpoint.get("arch")) != "preact_resnet18":
        raise ValueError(f"unexpected VOLTA architecture: {checkpoint.get('arch')!r}")

    module = _load_backbone_module(repository)
    model = module.preact_resnet18(num_classes=64)
    model.fc = torch.nn.Identity()
    model.load_state_dict(_teacher_state_dict(checkpoint), strict=True)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device).eval()

    integration = tuple(Path(path) for path in model_cfg["integration_paths"])
    checkpoint_identity = file_identity(checkpoint_path, checksum=True)
    if checkpoint_identity["sha256"] != model_cfg["checkpoint_sha256"]:
        raise ValueError("VOLTA checkpoint SHA-256 does not match release metadata")
    identity = {
        "baseline": cfg["baseline"]["name"],
        "checkpoint_domain": model_cfg["checkpoint_domain"],
        "volta_repository": git_identity(repository),
        "checkpoint": checkpoint_identity,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "architecture": str(checkpoint["arch"]),
        "embedding_dim": EMBEDDING_DIM,
        "normalized": False,
        "input_size": INPUT_SIZE,
        "normalization": model_cfg["normalization"],
        "integration": [file_identity(path, checksum=True) for path in integration],
    }
    normalization = model_cfg["normalization"]
    return VoltaEncoder(
        model, device, normalization["mean"], normalization["std"], identity
    )


def prepare_model(cfg, output_dir, device=None):
    encoder = load_encoder(cfg, device)
    example = Path(cfg["model"]["repository"]) / "examples" / "example1.png"
    image = Image.open(example).convert("RGB")
    batch = encoder.transform(image).unsqueeze(0)
    features = encoder.encode(batch)
    payload = {
        "model_signature": encoder.model_signature,
        "model": encoder.identity,
        "smoke_embedding_shape": list(features.shape),
        "smoke_embedding_norm": float(np.linalg.norm(features[0])),
    }
    write_json(Path(output_dir) / "model_provenance.json", payload)
    print(
        f"VOLTA ready: {features.shape[1]}-D, signature={encoder.model_signature[:12]}"
    )
    return payload
