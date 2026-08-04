"""Extract and probe released VOLTA ovarian cell embeddings."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import pandas as pd

from terra_multi.baselines.evaluation import (
    expected_hest_cell_ids,
    hest_manifest_row,
    label_test_fast_samples,
)
from terra_multi.baselines.volta_hest import extract_hest
from terra_multi.baselines.volta_model import EMBEDDING_DIM, load_encoder, prepare_model
from terra_multi.baselines.volta_ood_extract import extract_ood
from terra_multi.baselines.volta_probe import probe_all
from terra_multi.config import load_config as load_yaml_config
from terra_multi.evaluation.artifacts import git_identity
from terra_multi.evaluation.ood import validate_official_split

BASELINE_NAME = "volta_ovarian"
DATASETS = ("hest", "phenocell", "pannuke")


def _integration_paths():
    return [
        Path(__file__).with_name("volta_model.py"),
        Path(__file__).with_name("volta_crops.py"),
    ]


def load_config(path):
    cfg = load_yaml_config(path)
    cfg["model"]["integration_paths"] = [str(path) for path in _integration_paths()]
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    required = {"baseline", "model", "crop", "output_dir", *DATASETS}
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"config is missing sections {sorted(missing)}")
    baseline = cfg["baseline"]
    if baseline.get("name") != BASELINE_NAME:
        raise ValueError(f"baseline.name must be {BASELINE_NAME!r}")
    if int(baseline.get("embedding_dim", 0)) != EMBEDDING_DIM:
        raise ValueError("released VOLTA preactivation ResNet18 features are 512-D")
    if baseline.get("normalized") is not False:
        raise ValueError("released VOLTA features are not L2-normalized")
    if baseline.get("save_dtype") != "float16":
        raise ValueError("VOLTA artifacts must use float16 storage")

    model = cfg["model"]
    repository = Path(model["repository"])
    checkpoint = Path(model["checkpoint"])
    for path in (repository / "backbones/resnet/model.py", checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if model.get("checkpoint_domain") != "ovarian":
        raise ValueError("this integration requires the ovarian checkpoint")
    if int(model.get("checkpoint_size", 0)) != checkpoint.stat().st_size:
        raise ValueError("VOLTA checkpoint size does not match its release metadata")
    crop = cfg["crop"]
    if float(crop.get("scale", 0)) != 2.0 or crop.get("padding") != "white":
        raise ValueError("VOLTA crops require scale: 2.0 and padding: white")
    for name in DATASETS:
        probe = cfg[name]["probe"]
        if probe.get("pca", False):
            raise ValueError(f"{name} must use pca: false")
        if probe.get("methods") != ["linear", "knn"]:
            raise ValueError(f"{name} must evaluate linear and knn")
    if cfg["phenocell"]["probe"].get("tasks") != ["all", "immune"]:
        raise ValueError("PhenoCell tasks must be all and immune")
    if cfg["pannuke"]["probe"].get("tasks") != ["all"]:
        raise ValueError("PanNuke task must be all")


def output_root(cfg, override=None):
    return Path(override or cfg["output_dir"])


def check_inputs(cfg):
    hest = cfg["hest"]
    samples = label_test_fast_samples(hest)
    paths = [hest["manifest_path"], hest["splits_path"], hest["label_test_fast_path"]]
    for sample_id in samples:
        expected_hest_cell_ids(hest, sample_id)
        paths.extend(
            [
                hest_manifest_row(hest, sample_id)[key]
                for key in ("wsi", "xenium_cell_seg")
            ]
        )
    for name in ("phenocell", "pannuke"):
        data = cfg[name]["data"]
        paths.extend(data[key] for key in ("lmdb", "metadata", "split"))
        metadata = pd.read_csv(data["metadata"])
        observed = metadata[["sample_id", "split_id"]].rename(
            columns={"split_id": "split"}
        )
        validate_official_split(
            observed,
            data["split"],
            require_full_coverage=not data.get("allow_unlabeled_samples", False),
        )
        if name == "pannuke" and metadata["label"].nunique() != 5:
            raise ValueError("PanNuke metadata must contain five classes")
        print(
            f"{name}: {len(metadata):,} cells across "
            f"{metadata['sample_id'].nunique():,} samples"
        )
    for path in paths:
        if not Path(path).exists():
            raise FileNotFoundError(path)
    repository = git_identity(cfg["model"]["repository"])
    print(f"VOLTA repository: {repository['commit']}")
    print(f"Fast HEST: {len(samples)} samples ({', '.join(samples)})")
    print(
        "Embedding contract: raw 512-D ovarian momentum-encoder features; PCA disabled"
    )


def _parse_samples(value):
    if not value:
        return None
    samples = [item.strip() for item in value.split(",") if item.strip()]
    if not samples or len(samples) != len(set(samples)):
        raise ValueError("--samples must be a unique comma-separated list")
    return samples


def _add_common(parser, output=True):
    parser.add_argument("--config", required=True)
    if output:
        parser.add_argument("--output-dir")


def _add_extraction(parser):
    _add_common(parser)
    parser.add_argument("--device")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--samples")
    parser.add_argument("--max-cells-per-sample", type=int)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    _add_common(commands.add_parser("check"), output=False)
    prepare = commands.add_parser("prepare")
    _add_common(prepare)
    prepare.add_argument("--device")
    for name in DATASETS:
        _add_extraction(commands.add_parser(f"extract-{name}"))
    probe = commands.add_parser("probe")
    _add_common(probe)
    probe.add_argument("--dataset", choices=("all", *DATASETS), default="all")
    probe.add_argument("--samples")
    run = commands.add_parser("run")
    _add_common(run)
    run.add_argument("--device")
    run.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if (
        args.command.startswith("extract-")
        and args.max_cells_per_sample
        and not args.output_dir
    ):
        parser.error("--max-cells-per-sample requires --output-dir")
    cfg = load_config(args.config)
    if args.command == "check":
        check_inputs(cfg)
        return
    root = output_root(cfg, args.output_dir)
    if args.command == "prepare":
        prepare_model(cfg, root, args.device)
        return
    if args.command == "probe":
        samples = _parse_samples(args.samples)
        if samples is not None and args.dataset != "hest":
            parser.error("--samples is only valid with --dataset hest")
        probe_all(cfg, root, args.dataset, samples=samples)
        return

    encoder = load_encoder(cfg, args.device)
    try:
        if args.command == "extract-hest":
            extract_hest(
                cfg,
                encoder,
                root,
                samples=_parse_samples(args.samples),
                max_cells_per_sample=args.max_cells_per_sample,
                overwrite=args.overwrite,
            )
        elif args.command.startswith("extract-"):
            name = args.command.removeprefix("extract-")
            extract_ood(
                cfg,
                encoder,
                root,
                name,
                samples=_parse_samples(args.samples),
                max_cells_per_sample=args.max_cells_per_sample,
                overwrite=args.overwrite,
            )
        else:
            extract_hest(cfg, encoder, root, overwrite=args.overwrite)
            extract_ood(cfg, encoder, root, "phenocell", overwrite=args.overwrite)
            extract_ood(cfg, encoder, root, "pannuke", overwrite=args.overwrite)
            del encoder
            encoder = None
            gc.collect()
            probe_all(cfg, root, "all")
    finally:
        del encoder


if __name__ == "__main__":
    main()
