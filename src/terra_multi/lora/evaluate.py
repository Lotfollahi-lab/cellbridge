"""Export and probe LoRA features on HEST and PhenoCell."""

import argparse
from pathlib import Path

import pandas as pd

from terra_multi.config import load_config
from terra_multi.evaluation.artifacts import file_identity
from terra_multi.evaluation.classification import resolve_methods
from terra_multi.lora.export_hest import export_hest
from terra_multi.lora.export_phenocell import export_phenocell, phenocell_cell_ids
from terra_multi.lora.features import (
    FEATURE_ARMS,
    FORMAT_VERSION,
    PCA_ARMS,
    checkpoint_path,
    feature_file_is_current,
    feature_signature,
    infer_loader,
    infer_tensor,
    load_model,
    write_feature_file,
)
from terra_multi.lora.hest import label_test_fast_samples
from terra_multi.lora.probe_hest import evaluate_hest_sample, probe_hest
from terra_multi.lora.probe_phenocell import (
    evaluate_phenocell_arm,
    load_phenocell_arm,
    probe_phenocell,
)
from terra_multi.lora.summary import write_evaluation_summary

# Backward-compatible name used by existing scripts/tests.
_checkpoint_path = checkpoint_path


def load_phenocell_config(cfg):
    path = cfg.get("phenocell_config")
    if not path:
        raise ValueError("LoRA config requires top-level phenocell_config")
    return load_config(path)


def _arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-hest", action="store_true")
    parser.add_argument("--skip-phenocell", action="store_true")
    parser.add_argument("--skip-probes", action="store_true")
    parser.add_argument("--samples", help="comma-separated HEST samples to export")
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="probe already-exported features without loading a model",
    )
    parser.add_argument("--probe-samples", help="comma-separated HEST samples to probe")
    args = parser.parse_args(argv)
    samples = args.samples.split(",") if args.samples else None
    probe_samples = args.probe_samples.split(",") if args.probe_samples else None
    if samples is not None and not args.skip_probes:
        parser.error("--samples exports a subset, so it requires --skip-probes")
    if (args.probe_only or probe_samples is not None) and samples is not None:
        parser.error("--samples exports; --probe-only probes. Use one or the other")
    return args, samples, probe_samples


def _checkpoint_identity(cfg, path):
    weights_path = cfg.get("model", {}).get("weights_path")
    return {
        "checkpoint": file_identity(path, checksum=True),
        "encoder": "uni_v2",
        "encoder_weights": (
            file_identity(weights_path, checksum=True)
            if weights_path
            else "trident-default"
        ),
    }


def _write_other_method_summaries(cfg, root, phenocell_cfg, args):
    methods = set(resolve_methods(cfg["eval"]))
    if phenocell_cfg is not None:
        methods.update(resolve_methods(phenocell_cfg.get("probe", {})))
    for method in sorted(methods - {"linear"}):
        hest_path = root / "hest_label_test_fast" / f"probe_metrics_{method}.csv"
        phenocell_path = root / "phenocell" / f"probe_summary_{method}.csv"
        hest = pd.read_csv(hest_path) if not args.skip_hest and hest_path.exists() else None
        phenocell = (
            pd.read_csv(phenocell_path)
            if not args.skip_phenocell and phenocell_path.exists()
            else None
        )
        write_evaluation_summary(cfg, root, hest, phenocell, method)


def main(argv=None):
    args, samples, probe_samples = _arguments(argv)
    cfg = load_config(args.config)
    root = Path(
        args.output_dir
        or cfg.get("eval", {}).get("output_dir", Path(cfg["output_dir"]) / "evaluation")
    )
    if args.probe_only or probe_samples is not None:
        probe_hest(cfg, root / "hest_label_test_fast", samples=probe_samples)
        return

    checkpoint = _checkpoint_path(cfg, args.checkpoint)
    identity = _checkpoint_identity(cfg, checkpoint)
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint, device, cfg.get("model", {}).get("weights_path"))
    hest_metrics = phenocell_metrics = phenocell_cfg = None
    if not args.skip_hest:
        directory = root / "hest_label_test_fast"
        export_hest(cfg, model, device, directory, identity, args.overwrite, samples)
        if not args.skip_probes:
            hest_metrics, _ = probe_hest(cfg, directory)
    if not args.skip_phenocell:
        phenocell_cfg = load_phenocell_config(cfg)
        directory = root / "phenocell"
        export_phenocell(
            phenocell_cfg, model, device, directory, identity, args.overwrite
        )
        if not args.skip_probes:
            phenocell_metrics = probe_phenocell(phenocell_cfg, directory)
    if not args.skip_probes:
        write_evaluation_summary(cfg, root, hest_metrics, phenocell_metrics)
        _write_other_method_summaries(cfg, root, phenocell_cfg, args)


if __name__ == "__main__":
    main()
