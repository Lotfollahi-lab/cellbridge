"""Export and probe LoRA features on HEST and PhenoCell."""

import argparse
from pathlib import Path

import pandas as pd

from cellbridge.config import load_config
from cellbridge.evaluation.classification import resolve_methods
from cellbridge.lora.export_hest import export_hest
from cellbridge.lora.export_phenocell import export_phenocell, phenocell_cell_ids
from cellbridge.lora.features import (
    FEATURE_ARMS,
    FORMAT_VERSION,
    PCA_ARMS,
    checkpoint_identity,
    checkpoint_path,
    feature_file_is_current,
    feature_signature,
    infer_loader,
    infer_tensor,
    load_model,
    write_feature_file,
)
from cellbridge.lora.hest import evaluation_cohorts
from cellbridge.lora.probe_hest import evaluate_hest_sample, probe_hest
from cellbridge.lora.probe_phenocell import (
    evaluate_phenocell_arm,
    load_phenocell_arm,
    probe_phenocell,
)
from cellbridge.lora.summary import write_evaluation_summary

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


def _hest_directory(root, cohort):
    return Path(root) / f"hest_{cohort}"


def _selected_hest_cohorts(cfg, selected=None):
    hierarchy_enabled = cfg.get("hierarchy", {}).get("enabled", False)
    if selected is None and not hierarchy_enabled:
        return {"label_test_fast": None}
    cohorts = evaluation_cohorts(cfg)
    if selected is None:
        return {cohort: None for cohort in cohorts}
    requested = {str(sample_id) for sample_id in selected}
    configured = {sample for samples in cohorts.values() for sample in samples}
    unknown = sorted(requested - configured)
    if unknown:
        raise ValueError(f"not configured HEST evaluation samples: {unknown}")
    return {
        cohort: [sample for sample in cohort_samples if sample in requested]
        for cohort, cohort_samples in cohorts.items()
        if requested.intersection(cohort_samples)
    }


def _run_hest_probe(cfg, directory, samples, cohort):
    if cohort == "label_test_fast":
        return probe_hest(cfg, directory, samples=samples)
    return probe_hest(cfg, directory, samples=samples, cohort=cohort)


def _write_other_method_summaries(cfg, root, phenocell_cfg, args):
    methods = set(resolve_methods(cfg["eval"]))
    if phenocell_cfg is not None:
        methods.update(resolve_methods(phenocell_cfg.get("probe", {})))
    for method in sorted(methods - {"linear"}):
        hest = {}
        if not args.skip_hest:
            for cohort in evaluation_cohorts(cfg):
                path = _hest_directory(root, cohort) / f"probe_metrics_{method}.csv"
                if path.exists():
                    hest[cohort] = pd.read_csv(path)
        phenocell_path = root / "phenocell" / f"probe_summary_{method}.csv"
        phenocell = (
            pd.read_csv(phenocell_path)
            if not args.skip_phenocell and phenocell_path.exists()
            else None
        )
        write_evaluation_summary(cfg, root, hest or None, phenocell, method)


def main(argv=None):
    args, samples, probe_samples = _arguments(argv)
    cfg = load_config(args.config)
    root = Path(
        args.output_dir
        or cfg.get("eval", {}).get("output_dir", Path(cfg["output_dir"]) / "evaluation")
    )
    if args.probe_only or probe_samples is not None:
        for cohort, selected in _selected_hest_cohorts(cfg, probe_samples).items():
            _run_hest_probe(cfg, _hest_directory(root, cohort), selected, cohort)
        return

    checkpoint = _checkpoint_path(cfg, args.checkpoint)
    identity = checkpoint_identity(cfg, checkpoint)
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint, device, cfg.get("model", {}).get("weights_path"))
    hest_metrics = {}
    phenocell_metrics = phenocell_cfg = None
    if not args.skip_hest:
        for cohort, selected in _selected_hest_cohorts(cfg, samples).items():
            directory = _hest_directory(root, cohort)
            if cohort == "label_test_fast":
                export_hest(
                    cfg, model, device, directory, identity, args.overwrite, selected
                )
            else:
                export_hest(
                    cfg,
                    model,
                    device,
                    directory,
                    identity,
                    args.overwrite,
                    selected,
                    cohort,
                )
            if not args.skip_probes:
                hest_metrics[cohort], _ = _run_hest_probe(
                    cfg, directory, None, cohort
                )
    if not args.skip_phenocell:
        phenocell_cfg = load_phenocell_config(cfg)
        directory = root / "phenocell"
        export_phenocell(
            phenocell_cfg, model, device, directory, identity, args.overwrite
        )
        if not args.skip_probes:
            phenocell_metrics = probe_phenocell(phenocell_cfg, directory)
    if not args.skip_probes:
        write_evaluation_summary(
            cfg, root, hest_metrics or None, phenocell_metrics
        )
        _write_other_method_summaries(cfg, root, phenocell_cfg, args)


if __name__ == "__main__":
    main()
