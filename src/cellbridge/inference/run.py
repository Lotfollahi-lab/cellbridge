"""Run embedding extraction and probe evaluation for one configured dataset."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from cellbridge.config import config_root, load_config
from cellbridge.inference.embed import embed, sources
from cellbridge.inference.embed import methods as embedding_methods
from cellbridge.inference.embed.artifacts import (
    load_manifest,
    load_manifest_sample,
    save_sample,
    write_manifests,
)
from cellbridge.inference.embed.methods.lora import resolve_checkpoint
from cellbridge.inference.embed.sources.metadata import cell_uids
from cellbridge.inference.conformal import bootstrap as conformal_bootstrap
from cellbridge.inference.conformal import conformal_hest as hest_conformal
from cellbridge.inference.conformal import conformal_ood as conformal_api
from cellbridge.inference.evaluate import bootstrap as bootstrap_api
from cellbridge.inference.evaluate import hest as hest_api
from cellbridge.inference.evaluate import ood
from cellbridge.inference.evaluate import probe as probe_api
from cellbridge.ontology import load_mapping, load_ontology

TASKS = ("all", "immune")


def _load_suite(path):
    path = Path(path)
    config = load_config(path)
    if not config.get("dataset"):
        raise ValueError("dataset must be named")
    configured = config.get("methods")
    if not isinstance(configured, list) or not configured:
        raise ValueError("methods must be a non-empty list")

    profiles = {}
    for method in configured:
        profile_path = path.parent / "methods" / f"{method}.yaml"
        if not profile_path.is_file():
            profile_path = config_root() / "configs/inference/methods" / f"{method}.yaml"
        profile = load_config(profile_path)
        if profile.get("method") != method:
            raise ValueError(f"{profile_path} does not describe method {method!r}")
        profiles[method] = profile
    return config, profiles


def _select(requested, configured, allowed, kind):
    selected = list(dict.fromkeys(requested or configured))
    unknown = set(selected) - set(allowed)
    if not selected or unknown:
        raise ValueError(f"invalid {kind}: {sorted(unknown)}")
    return selected


def _load_conformal(config):
    settings = config.get("conformal")
    if not isinstance(settings, dict):
        raise ValueError("conformal settings must be a mapping")
    missing = {"mapping", "ontology"} - set(settings)
    if missing:
        raise ValueError(f"conformal settings are missing {sorted(missing)}")

    mapping = load_mapping(settings["mapping"])
    ontology = load_ontology(settings["ontology"])
    if mapping.dataset != str(config["dataset"]):
        raise ValueError(
            f"mapping dataset {mapping.dataset!r} does not match "
            f"{config['dataset']!r}"
        )
    mapping.validate_ontology(ontology)
    alpha = float(settings.get("alpha", 0.1))
    if not 0 < alpha < 1:
        raise ValueError("conformal alpha must be between zero and one")
    branch_roots = settings.get("branch_roots")
    if not isinstance(branch_roots, list) or not branch_roots:
        raise ValueError("conformal branch_roots must be a non-empty list")
    branch_roots = tuple(map(str, branch_roots))
    if len(set(branch_roots)) != len(branch_roots):
        raise ValueError("conformal branch_roots must be unique")
    unknown = sorted(set(branch_roots) - set(ontology.names))
    if unknown:
        raise ValueError(f"unknown conformal branch roots: {unknown}")
    return mapping, ontology, alpha, branch_roots


def check(config, profiles, methods, probes, tasks, *, pending_checkpoint=False) -> None:
    """Validate inputs; pending_checkpoint allows a dependency to create weights."""
    data = config["data"]
    source = sources.load(data)
    metadata = source.check(data)

    for method in methods:
        profile = profiles[method]
        embedding_methods.arms(method, config.get("arms"))
        if method == "cellbridge" and not pending_checkpoint:
            settings = profile.get("load", {})
            resolve_checkpoint(**{
                key: settings[key]
                for key in ("checkpoint", "repo_id", "filename", "revision")
                if key in settings
            })
        source.make_cropper(profile, data)
        extraction = profile.get("extraction", {})
        if int(extraction.get("batch_size", 0)) < 1:
            raise ValueError(f"{method}: batch_size must be positive")
        if int(extraction.get("num_workers", 0)) < 0:
            raise ValueError(f"{method}: num_workers must be non-negative")
        np.dtype(extraction.get("storage_dtype", "float16"))
        for key in ("checkpoint", "repository", "encoder_weights"):
            value = profile.get("load", {}).get(key)
            if key == "checkpoint" and (pending_checkpoint or method == "cellbridge"):
                continue
            if value is not None and not Path(value).exists():
                raise FileNotFoundError(f"{method}: {key} does not exist: {value}")

    if data.get("format") == "hest":
        mapping_dir = data.get("mapping_dir")
        if not mapping_dir:
            raise ValueError("HEST data.mapping_dir is required")
        sample_ids = source.sample_ids(data)
        for sample_id in sample_ids:
            hest_api.load_sample_mapping(
                mapping_dir, sample_id, config["dataset"]
            )
        print(f"HEST mappings: {len(sample_ids)} reviewed samples")

        if "conformal" in config:
            settings = [
                hest_conformal.ontology_settings(config, sample_id)
                for sample_id in sample_ids
            ]
            alphas = {item[2] for item in settings}
            print(
                f"Conformal: alpha={next(iter(alphas)):g}, "
                f"{len(sample_ids)} per-sample mappings"
            )
    elif "conformal" in config:
        mapping, _, alpha, _ = _load_conformal(config)
        _, included = mapping.map_labels(metadata["label"])
        if not included.any():
            raise ValueError("dataset has no ontology-mapped cells")
        print(
            f"Conformal: alpha={alpha:g}, "
            f"{int(included.sum()):,}/{len(included):,} mapped cells"
        )

    if data.get("format") != "hest" and "bootstrap" in config["probe"]:
        test_metadata = metadata.loc[
            metadata["split"].replace({"valid": "val"}) == "test"
        ]
        groups = bootstrap_api.resolve_groups(
            test_metadata,
            test_metadata["row_id"].to_numpy(),
            config["probe"]["bootstrap"],
        )
        print(f"Bootstrap: {len(np.unique(groups))} independent test groups")

    print(
        f"{config['dataset']}: {len(metadata):,} cells, "
        f"{metadata['sample_id'].nunique()} labeled samples"
    )
    print(f"Methods: {', '.join(methods)}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Probes: {', '.join(probes)}")
    print(f"Output: {config['output_dir']}")


def _extract_method(config, profile, method, device) -> None:
    load_settings = dict(profile.get("load", {}))
    load_settings["device"] = device
    embedder = embedding_methods.load(method, **load_settings)
    data = config["data"]
    source = sources.load(data)
    cropper = source.make_cropper(profile, data)
    extraction = profile["extraction"]
    output_dir = Path(config["output_dir"]) / "embeddings"

    arms = embedding_methods.arms(method, config.get("arms"))
    records = []
    for sample in source.iter_samples(
        data,
        transform=embedder.transform,
        cropper=cropper,
    ):
        result = embed(
            sample.dataset,
            embedder,
            batch_size=int(extraction["batch_size"]),
            num_workers=int(extraction.get("num_workers", 0)),
        )
        result = replace(
            result, embeddings={arm: result.embeddings[arm] for arm in arms}
        )
        records.extend(
            save_sample(
                output_dir,
                method,
                sample.sample_id,
                sample.row_ids,
                result,
                storage_dtype=extraction.get("storage_dtype", "float16"),
                arms=arms,
            )
        )
        print(f"[{method}] {sample.sample_id}: {len(sample.dataset):,} cells")
    write_manifests(output_dir, method, records, arms=arms)


def extract(config, profiles, methods, device=None) -> None:
    """Extract every selected method and write per-arm manifests."""
    for method in methods:
        _extract_method(config, profiles[method], method, device)


def _align_metadata(config, loaded):
    data = config["data"]
    source = sources.load(data)
    allow_unlabeled = bool(data.get("allow_unlabeled_samples", False))
    metadata = source.load_metadata(data)
    indexed = metadata.set_index("row_id", drop=False)
    missing = set(loaded.row_ids) - set(indexed.index)
    if missing:
        raise ValueError(
            f"embedding artifacts contain unknown row IDs: {sorted(missing)[:5]}"
        )
    aligned = indexed.loc[loaded.row_ids].reset_index(drop=True)
    if not np.array_equal(cell_uids(aligned), loaded.cell_ids):
        raise ValueError("embedding cell IDs do not match dataset metadata")
    if not np.array_equal(aligned["sample_id"].astype(str), loaded.sample_ids):
        raise ValueError("embedding sample IDs do not match dataset metadata")
    ood.validate_split(
        aligned,
        data["split"],
        require_all_samples=not allow_unlabeled,
    )
    return aligned


def _save_hest_metrics(output, rows):
    output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(output / "metrics.csv", index=False)

    summary = {
        name: table[name].iloc[0]
        for name in ("dataset", "method", "arm", "task", "probe")
    }
    summary["n_samples"] = len(table)
    for name in (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_auroc",
    ):
        summary[f"{name}_mean"] = table[name].mean()
        summary[f"{name}_std"] = table[name].std()
    pd.DataFrame([summary]).to_csv(output / "summary.csv", index=False)


def _evaluate_hest(config, methods, probes, tasks) -> None:
    if tasks != ["all"]:
        raise ValueError("HEST supports only the all-cell-type task")

    data = config["data"]
    source = sources.load(data)
    sample_ids = source.sample_ids(data)
    probe_config = config["probe"]
    embedding_dir = Path(config["output_dir"]) / "embeddings"
    evaluation_dir = Path(config["output_dir"]) / "evaluation"
    seed = int(probe_config.get("seed", 0))

    for method in methods:
        for arm in embedding_methods.arms(method, config.get("arms")):
            manifest = embedding_dir / arm / "manifest.csv"
            metrics = {probe: [] for probe in probes}
            for sample_id in sample_ids:
                loaded = load_manifest_sample(manifest, sample_id)
                metadata = hest_api.align_metadata(
                    source, data, loaded, sample_id
                )
                mapping = hest_api.load_sample_mapping(
                    data["mapping_dir"], sample_id, config["dataset"]
                )
                labels, coordinates = source.evaluation_paths(data, sample_id)
                task = hest_api.prepare_task(
                    loaded.embeddings,
                    metadata["cell_id"].to_numpy(),
                    labels,
                    coordinates,
                    id_column=probe_config.get("label_id_column", "cell_id"),
                    label_column=probe_config.get("label_column", "group"),
                    min_class_count=int(probe_config.get("min_class_count", 25)),
                    test_fraction=float(probe_config.get("test_fraction", 0.2)),
                    split_axis=probe_config.get("split_axis", "y"),
                    arm=arm,
                    mapping=mapping,
                    pca_components=int(probe_config.get("pca_components", 256)),
                    seed=seed,
                )
                for probe in probes:
                    settings = dict(probe_config.get(probe, {}))
                    settings.setdefault("seed", seed)
                    result = probe_api.evaluate(task, probe, settings)
                    result = replace(
                        result,
                        metrics={
                            "dataset": config["dataset"],
                            "sample_id": sample_id,
                            "method": method,
                            "arm": arm,
                            "task": "all",
                            **result.metrics,
                        },
                    )
                    output = evaluation_dir / arm / "all" / probe / sample_id
                    probe_api.save_result(output, result)
                    metrics[probe].append(result.metrics)
                    print(
                        f"[{method}/{arm}/{sample_id}/{probe}] "
                        f"macro_f1={result.metrics['macro_f1']:.4f}"
                    )
            for probe, rows in metrics.items():
                output = evaluation_dir / arm / "all" / probe
                _save_hest_metrics(output, rows)


def evaluate(config, methods, probes, tasks) -> None:
    """Evaluate every arm of the selected extracted methods."""
    if config["data"].get("format") == "hest":
        _evaluate_hest(config, methods, probes, tasks)
        return

    probe_config = config["probe"]
    embedding_dir = Path(config["output_dir"]) / "embeddings"
    evaluation_dir = Path(config["output_dir"]) / "evaluation"
    seed = int(probe_config.get("seed", 0))
    bootstrap_config = probe_config.get("bootstrap")

    for method in methods:
        for arm in embedding_methods.arms(method, config.get("arms")):
            manifest = embedding_dir / arm / "manifest.csv"
            loaded = load_manifest(manifest)
            metadata = _align_metadata(config, loaded)
            for task_name in tasks:
                task = ood.prepare_task(
                    loaded.embeddings,
                    metadata,
                    task=task_name,
                    arm=arm,
                    pca_components=int(probe_config.get("pca_components", 256)),
                    seed=seed,
                )
                groups = None
                if bootstrap_config is not None:
                    groups = bootstrap_api.resolve_groups(
                        metadata,
                        np.asarray(task.ids)[task.test],
                        bootstrap_config,
                    )
                for probe in probes:
                    settings = dict(probe_config.get(probe, {}))
                    settings.setdefault("seed", seed)
                    if bootstrap_config is not None:
                        settings.setdefault(
                            "min_test_support",
                            bootstrap_config.get("min_test_support"),
                        )
                    result = probe_api.evaluate(task, probe, settings)
                    result = replace(
                        result,
                        metrics={
                            "dataset": config["dataset"],
                            "method": method,
                            "arm": arm,
                            "task": task_name,
                            **result.metrics,
                        },
                    )
                    output = evaluation_dir / arm / task_name / probe
                    probe_api.save_result(output, result)
                    uncertainty = None
                    if bootstrap_config is not None:
                        uncertainty, replicates = bootstrap_api.summarize(
                            result, groups, bootstrap_config
                        )
                        bootstrap_api.save_result(output, uncertainty, replicates)
                    suffix = ""
                    if uncertainty is not None:
                        macro_se = uncertainty.loc[
                            uncertainty["metric"] == "macro_f1", "bootstrap_se"
                        ].iloc[0]
                        suffix = f", bootstrap_se={macro_se:.4f}"
                    print(
                        f"[{method}/{arm}/{task_name}/{probe}] "
                        f"macro_f1={result.metrics['macro_f1']:.4f}{suffix}"
                    )


def evaluate_conformal(config, methods, probes, tasks) -> None:
    """Run split and graph conformal evaluation for every extracted arm."""
    if config["data"].get("format") == "hest":
        hest_conformal.evaluate(config, methods, probes, tasks)
        return

    mapping, ontology, alpha, branch_roots = _load_conformal(config)
    probe_config = config["probe"]
    embedding_dir = Path(config["output_dir"]) / "embeddings"
    evaluation_dir = Path(config["output_dir"]) / "evaluation"
    output_subdir = config["conformal"].get("output_subdir", "conformal")
    seed = int(probe_config.get("seed", 0))
    pca_components = int(probe_config.get("pca_components", 256))
    bootstrap_config = probe_config.get("bootstrap")

    for method in methods:
        for arm in embedding_methods.arms(method, config.get("arms")):
            manifest = embedding_dir / arm / "manifest.csv"
            loaded = load_manifest(manifest)
            metadata = _align_metadata(config, loaded)
            for task_name in tasks:
                task = conformal_api.prepare_task(
                    loaded.embeddings,
                    metadata,
                    mapping,
                    task=task_name,
                    arm=arm,
                    pca_components=pca_components,
                    min_class_count_per_split=int(
                        config["conformal"].get("min_class_count_per_split", 20)
                    ),
                    seed=seed,
                )
                for probe in probes:
                    settings = dict(probe_config.get(probe, {}))
                    settings.setdefault("seed", seed)
                    result = conformal_api.evaluate(
                        task,
                        probe,
                        settings,
                        mapping,
                        ontology,
                        branch_roots=branch_roots,
                        alpha=alpha,
                    )
                    context = {
                        "dataset": config["dataset"],
                        "method": method,
                        "arm": arm,
                        "task": task_name,
                        "probe": probe,
                    }
                    result = replace(
                        result,
                        metrics=tuple({**context, **row} for row in result.metrics),
                        per_class_coverage=tuple(
                            {**context, **row} for row in result.per_class_coverage
                        ),
                        calibration=tuple(
                            {**context, **row} for row in result.calibration
                        ),
                    )
                    output = (
                        evaluation_dir
                        / arm
                        / task_name
                        / probe
                        / output_subdir
                    )
                    conformal_api.save_result(output, result)
                    if bootstrap_config is not None:
                        uncertainty, replicates = conformal_bootstrap.summarize(
                            conformal_bootstrap.prediction_sets(result),
                            result.y_true,
                            result.classes,
                            bootstrap_api.resolve_groups(
                                metadata, result.ids, bootstrap_config
                            ),
                            bootstrap_config,
                            target_coverage=1.0 - float(alpha),
                        )
                        bootstrap_api.save_result(output, uncertainty, replicates)
                    graph_metrics = result.metrics[1]
                    print(
                        f"[{method}/{arm}/{task_name}/{probe}/{output_subdir}] "
                        f"coverage={graph_metrics['marginal_coverage']:.4f}, "
                        f"size={graph_metrics['mean_prediction_set_size']:.2f}"
                    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("check", "extract", "evaluate", "conformal", "run"),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", action="append", choices=embedding_methods.METHODS)
    parser.add_argument("--probe", action="append", choices=probe_api.PROBES)
    parser.add_argument("--task", action="append", choices=TASKS)
    parser.add_argument(
        "--sample-id",
        action="append",
        help="Restrict a HEST command to one or more fixed-cohort samples",
    )
    parser.add_argument(
        "--arm", action="append", help="Restrict stored and evaluated representations"
    )
    parser.add_argument("--device")
    parser.add_argument(
        "--checkpoint",
        help="Override the CellBridge checkpoint without changing a suite",
    )
    parser.add_argument("--output-dir", help="Write this run to a separate directory")
    args = parser.parse_args()

    config, profiles = _load_suite(args.config)
    if args.sample_id:
        if config["data"].get("format") != "hest":
            raise ValueError("--sample-id is supported only for HEST")
        config["data"]["sample_ids"] = args.sample_id
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if not config.get("output_dir"):
        raise ValueError("--output-dir is required")
    if args.checkpoint:
        if "cellbridge" not in profiles:
            raise ValueError(
                "--checkpoint requires cellbridge in the configured methods"
            )
        profiles["cellbridge"].setdefault("load", {})["checkpoint"] = args.checkpoint
    methods = _select(
        args.method, config["methods"], embedding_methods.METHODS, "methods"
    )
    if args.command in ("extract", "run") and "cellbridge" in methods:
        settings = profiles["cellbridge"].get("load", {})
        if not (settings.get("checkpoint") or settings.get("repo_id")):
            raise ValueError("cellbridge extraction requires a checkpoint or repo_id")
    if args.arm:
        config["arms"] = args.arm
    for method in methods:
        embedding_methods.arms(method, config.get("arms"))
    probe_config = config["probe"]
    probes = _select(args.probe, probe_config["methods"], probe_api.PROBES, "probes")
    tasks = _select(args.task, probe_config["tasks"], TASKS, "tasks")

    if args.command == "check":
        check(config, profiles, methods, probes, tasks)
    elif args.command == "extract":
        extract(config, profiles, methods, args.device)
    elif args.command == "evaluate":
        evaluate(config, methods, probes, tasks)
    elif args.command == "conformal":
        evaluate_conformal(config, methods, probes, tasks)
    else:
        check(config, profiles, methods, probes, tasks)
        extract(config, profiles, methods, args.device)
        evaluate(config, methods, probes, tasks)
        if "conformal" in config:
            evaluate_conformal(config, methods, probes, tasks)


if __name__ == "__main__":
    main()
