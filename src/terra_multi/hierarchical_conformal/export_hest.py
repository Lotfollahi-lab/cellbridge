"""Export frozen feature arms for the annotated HEST breast-training cohort."""

from __future__ import annotations

import argparse
from pathlib import Path

from terra_multi.config import load_config
from terra_multi.lora.export_hest import export_hest
from terra_multi.lora.features import checkpoint_identity, checkpoint_path, load_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--samples", help="comma-separated subset")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    checkpoint = checkpoint_path(cfg, args.checkpoint)
    identity = checkpoint_identity(cfg, checkpoint)
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint, device, cfg.get("model", {}).get("weights_path"))
    output = Path(
        args.output_dir
        or Path(cfg["output_dir"]) / "evaluation" / "hest_breast_train"
    )
    samples = args.samples.split(",") if args.samples else None
    export_hest(
        cfg,
        model,
        device,
        output,
        identity,
        overwrite=args.overwrite,
        samples=samples,
        cohort="breast_train",
    )


if __name__ == "__main__":
    main()
