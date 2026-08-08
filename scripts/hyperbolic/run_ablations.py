#!/usr/bin/env python3
"""Materialize and optionally run the complete geometry ablation matrix."""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

import yaml

VARIANTS = (
    ("euclidean_d64", "euclidean", 64, False),
    ("hyperbolic_d64", "hyperbolic", 64, False),
    ("hyperbolic_cones_d32", "hyperbolic", 32, True),
    ("hyperbolic_cones_d64", "hyperbolic", 64, True),
    ("hyperbolic_cones_d128", "hyperbolic", 128, True),
    ("hyperbolic_cones_d384", "hyperbolic", 384, True),
)
VARIANT_NAMES = tuple(item[0] for item in VARIANTS)


def _arguments():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=root / "configs" / "hyperbolic" / "cellbridge.yaml",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=root / "configs" / "hyperbolic" / "ablations",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/hyperbolic/ablations"),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("checkpoints/hyperbolic/ablations"),
    )
    parser.add_argument(
        "--wandb-group",
        default="cellbridge_hyperbolic_ablations",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=VARIANT_NAMES,
        metavar="VARIANT",
        help="materialize only the named variants",
    )
    parser.add_argument(
        "--action",
        choices=("write", "submit", "local"),
        default="write",
        help="write configs only, submit LSF jobs, or train/evaluate sequentially",
    )
    return parser.parse_args(), root


def materialize(
    base, config_dir, output_root, checkpoint_root, wandb_group, only=None
):
    configs = []
    selected = set(only or VARIANT_NAMES)
    for name, mode, dimension, cones in VARIANTS:
        if name not in selected:
            continue
        cfg = copy.deepcopy(base)
        cfg["run_name"] = f"cellbridge_{name}"
        cfg["geometry"]["mode"] = mode
        cfg["geometry"]["dim"] = dimension
        cfg["hierarchy"]["cones"]["enabled"] = cones
        cfg["output_dir"] = str(output_root / name)
        cfg["checkpoint_dir"] = str(checkpoint_root / name)
        cfg["wandb"]["name"] = f"cellbridge_{name}"
        cfg["wandb"]["group"] = wandb_group
        path = config_dir / f"{name}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        configs.append(path)
    return configs


def main():
    args, root = _arguments()
    base = yaml.safe_load(args.base_config.read_text())
    configs = materialize(
        base,
        args.config_dir,
        args.output_root,
        args.checkpoint_root,
        args.wandb_group,
        args.only,
    )
    for path in configs:
        print(path)
        if args.action == "submit":
            subprocess.run(
                [str(root / "scripts" / "lora" / "submit_experiment.sh"), str(path)],
                check=True,
            )
        elif args.action == "local":
            subprocess.run(
                [sys.executable, "-m", "cellbridge.lora.train", "--config", str(path)],
                cwd=root,
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "cellbridge.lora.evaluate",
                    "--config",
                    str(path),
                ],
                cwd=root,
                check=True,
            )


if __name__ == "__main__":
    main()
