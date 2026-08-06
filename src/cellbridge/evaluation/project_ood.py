"""Project cached out-of-distribution UNI embeddings with one distillation checkpoint."""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from cellbridge.config import load_config

from cellbridge.distill.inference import predict_embeddings
from cellbridge.distill.model import TerraProjector
from cellbridge.distill.prototype import PrototypeStudent, predict_student_features

# v1 prototype students expose two usable representations (cf. the label_test arms).
REPRESENTATIONS = {
    "penultimate": "features_penultimate",
    "probs": "prototype_probs",
}


def load_projector(path, device):
    """Load a v0 regression projector or a v1 prototype student."""
    checkpoint = torch.load(path, map_location=device)
    cfg = checkpoint["config"]
    if cfg.get("distill_version") == "v1" or "num_prototypes" in cfg:
        model = PrototypeStudent(
            cfg.get("uni_dim", 1536),
            cfg.get("hidden_dim", 1024),
            cfg.get("feat_dim", 512),
            cfg.get("num_prototypes", 256),
            cfg.get("dropout", 0.1),
        ).to(device)
    else:
        model = TerraProjector(
            cfg.get("uni_dim", 1536),
            cfg.get("terra_dim", 384),
            cfg.get("hidden_dim", 1024),
            cfg.get("dropout", 0.1),
        ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model.eval(), cfg


def project(model, uni, device, batch_size, representation):
    """Features the downstream probe consumes, for either student type."""
    if isinstance(model, PrototypeStudent):
        if representation not in REPRESENTATIONS:
            raise ValueError(
                f"projection.representation must be one of {sorted(REPRESENTATIONS)}, "
                f"got {representation!r}"
            )
        outputs = predict_student_features(model, uni, device, batch_size)
        return outputs[REPRESENTATIONS[representation]]
    return predict_embeddings(model, uni, device, batch_size)


def output_path(projection_dir, source_path):
    return projection_dir / "samples" / Path(source_path).name


def write_projection(path, row_id, sample_id, embeddings):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(
        temporary,
        row_id=row_id,
        sample_id=sample_id,
        distilled_emb=embeddings.astype(np.float16),
    )
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)

    source_dir = Path(cfg["embedding"]["output_dir"])
    source_manifest = pd.read_csv(source_dir / "manifest.csv")
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    projection_dir = output_dir / "embeddings" / "distilled"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    projector, projector_cfg = load_projector(cfg["projection"]["checkpoint"], device)
    batch_size = int(cfg["projection"].get("batch_size", 8192))
    representation = cfg["projection"].get("representation", "penultimate")

    rows = []
    for source_path in source_manifest["path"]:
        destination = output_path(projection_dir, source_path)
        if destination.exists() and not args.overwrite:
            with np.load(destination, allow_pickle=False) as cached:
                count = len(cached["row_id"])
            print(f"[{Path(source_path).stem}] already projected ({count:,} cells)")
        else:
            with np.load(source_path, allow_pickle=False) as source:
                row_id = source["row_id"]
                sample_id = source["sample_id"]
                uni = source["uni_emb"]
                if uni.shape[1] != projector_cfg.get("uni_dim", 1536):
                    raise ValueError(
                        f"{source_path}: UNI dimension {uni.shape[1]} does not match "
                        f"projector dimension {projector_cfg.get('uni_dim', 1536)}"
                    )
                projected = project(projector, uni, device, batch_size, representation)
            write_projection(destination, row_id, sample_id, projected)
            count = len(row_id)
            print(
                f"[{Path(source_path).stem}] projected {count:,} cells -> {destination}"
            )
        rows.append(
            {
                "sample_id": source_manifest.loc[
                    source_manifest["path"] == source_path, "sample_id"
                ].iloc[0],
                "n_cells": count,
                "path": str(destination),
            }
        )

    projection_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(projection_dir / "manifest.csv", index=False)
    with open(output_dir / "resolved_config.yaml", "w") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    print(f"projected {sum(row['n_cells'] for row in rows):,} cells")


if __name__ == "__main__":
    main()
