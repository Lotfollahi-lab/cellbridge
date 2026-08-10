"""Extract and probe released PAST image features."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import pandas as pd

from cellbridge.baselines.evaluation import (
    expected_hest_cell_ids,
    hest_manifest_row,
    label_test_fast_samples,
)
from cellbridge.baselines.past_hest import extract_hest
from cellbridge.baselines.past_model import EMBEDDING_DIM, load_encoder, prepare_model
from cellbridge.baselines.past_ood import extract_ood
from cellbridge.baselines.past_probe import probe_all
from cellbridge.config import load_config as load_yaml_config
from cellbridge.evaluation.artifacts import git_identity
from cellbridge.evaluation.ood import validate_official_split

BASELINE_NAME = "past"
DATASETS = ("hest", "phenocell", "pannuke")


def _integration_paths():
    directory = Path(__file__).parent
    names = ("past.py", "past_model.py", "past_crops.py", "past_hest.py", "past_ood.py")
    return [directory / name for name in names]


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
        raise ValueError("released PAST image-tower features are 1024-D")
    if baseline.get("normalized") is not False:
        raise ValueError("released PAST image-tower features are not L2-normalized")
    if baseline.get("save_dtype") != "float16":
        raise ValueError("PAST artifacts must use float16 storage")

    model = cfg["model"]
    repository = Path(model["repository"])
    checkpoint = Path(model["checkpoint"])
    for path in (repository / "modules.py", repository / "README.md", checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if int(model.get("checkpoint_size", 0)) != checkpoint.stat().st_size:
        raise ValueError("PAST checkpoint size does not match release metadata")
    if not model.get("checkpoint_sha256"):
        raise ValueError("model.checkpoint_sha256 is required")
    if not all(Path(path).is_file() for path in model["integration_paths"]):
        raise FileNotFoundError("a PAST integration source file is missing")

    crop = cfg["crop"]
    expected_crop = {"geometry": "cell_center", "source_size": 224, "padding": "black"}
    if any(crop.get(key) != value for key, value in expected_crop.items()):
        raise ValueError(f"PAST crop must be {expected_crop}")
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
        paths.append(hest_manifest_row(hest, sample_id)["wsi"])
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
        print(
            f"{name}: {len(metadata):,} cells across "
            f"{metadata['sample_id'].nunique():,} samples"
        )
    for path in paths:
        if not Path(path).exists():
            raise FileNotFoundError(path)
    repository = git_identity(cfg["model"]["repository"])
    print(f"PAST repository: {repository['commit']}")
    print(f"Fast HEST: {len(samples)} samples ({', '.join(samples)})")
    print("Embedding contract: raw 1024-D PAST image-tower features; PCA disabled")


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
