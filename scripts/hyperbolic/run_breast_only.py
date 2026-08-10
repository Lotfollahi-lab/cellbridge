#!/usr/bin/env python3
"""Materialize and run the matched breast-only geometry screen."""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

import yaml

VARIANTS = (
    ("euclidean_d64", "euclidean", False),
    ("hyperbolic_d64", "hyperbolic", False),
    ("hyperbolic_cones_d64", "hyperbolic", True),
)
VARIANT_NAMES = tuple(name for name, _, _ in VARIANTS)


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
        default=root / "configs" / "hyperbolic" / "breast_only",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/hyperbolic/breast_only_screen"),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("checkpoints/hyperbolic/breast_only_screen"),
    )
    parser.add_argument(
        "--wandb-group",
        default="cellbridge_hyperbolic_breast_only_screen",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=VARIANT_NAMES,
        metavar="VARIANT",
    )
    parser.add_argument(
        "--action",
        choices=("write", "submit", "local"),
        default="write",
    )
    return parser.parse_args(), root


def materialize(base, args):
    selected = set(args.only or VARIANT_NAMES)
    configs = []
    for name, mode, cones in VARIANTS:
        if name not in selected:
            continue
        cfg = copy.deepcopy(base)
        run_name = f"cellbridge_breast_only_{name}"
        cfg["run_name"] = run_name
        cfg["geometry"]["mode"] = mode
        cfg["geometry"]["dim"] = 64
        cfg["hierarchy"]["breast_only"] = True
        cfg["hierarchy"]["cones"]["enabled"] = cones
        cfg["eval"]["hest_cohorts"] = ["breast_test"]
        cfg["eval"]["skip_phenocell"] = True
        cfg["eval"]["hierarchy_metric_config"] = (
            "${CELLBRIDGE_ROOT}/configs/exploration/"
            "terra_phenotype_structure_breast_test.yaml"
        )
        cfg["output_dir"] = str(args.output_root / name)
        cfg["checkpoint_dir"] = str(args.checkpoint_root / name)
        cfg["wandb"]["name"] = run_name
        cfg["wandb"]["group"] = args.wandb_group
        path = args.config_dir / f"{name}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        configs.append(path)
    return configs


def main():
    args, root = _arguments()
    configs = materialize(yaml.safe_load(args.base_config.read_text()), args)
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
