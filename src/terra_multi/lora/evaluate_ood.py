"""Export and probe LoRA feature arms on one OOD dataset (PhenoCell, PanNuke)."""

import argparse
from pathlib import Path

import pandas as pd

from terra_multi.config import load_config
from terra_multi.lora.export_phenocell import export_phenocell
from terra_multi.lora.features import checkpoint_identity, checkpoint_path, load_model
from terra_multi.lora.probe_phenocell import probe_phenocell


def merge_shard_manifests(features_dir, num_shards):
    """Join parallel export slices into the manifest the probe reads."""
    shards = sorted(features_dir.glob("manifest.shard*.csv"))
    if len(shards) != num_shards:
        raise ValueError(f"expected {num_shards} shard manifests, found {len(shards)}")
    merged = pd.concat([pd.read_csv(path) for path in shards], ignore_index=True)
    merged = merged.sort_values("sample_id", ignore_index=True)
    merged.to_csv(features_dir / "manifest.csv", index=False)
    return merged


def _arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="LoRA training config")
    parser.add_argument("--ood-config", required=True, help="OOD dataset config")
    parser.add_argument("--dataset", default="phenocell")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device")
    parser.add_argument("--shard", type=int, help="export this slice only")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.shard is not None and not 0 <= args.shard < args.num_shards:
        parser.error("--shard must be in [0, --num-shards)")
    return args


def main(argv=None):
    args = _arguments(argv)
    cfg = load_config(args.config)
    ood_cfg = load_config(args.ood_config)
    output_dir = Path(ood_cfg["output_dir"])

    if args.probe_only:
        if args.num_shards > 1:
            merge_shard_manifests(output_dir / "features", args.num_shards)
        probe_phenocell(ood_cfg, output_dir)
        return

    import torch

    checkpoint = checkpoint_path(cfg, args.checkpoint)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint, device, cfg.get("model", {}).get("weights_path"))
    export_phenocell(
        ood_cfg,
        model,
        device,
        output_dir,
        checkpoint_identity(cfg, checkpoint),
        args.overwrite,
        dataset=args.dataset,
        shard=args.shard,
        num_shards=args.num_shards,
    )


if __name__ == "__main__":
    main()
