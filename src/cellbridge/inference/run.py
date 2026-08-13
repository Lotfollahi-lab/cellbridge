"""Run embedding extraction and probe evaluation for one configured dataset."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from cellbridge.config import load_config
from cellbridge.inference.embed import embed, sources
from cellbridge.inference.embed import methods as embedding_methods
from cellbridge.inference.embed.artifacts import (
    load_manifest,
    save_sample,
    write_manifests,
)
from cellbridge.inference.embed.sources.common import cell_uids
from cellbridge.inference.evaluate import ood
from cellbridge.inference.evaluate import probe as probe_api

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


def check(config, profiles, methods, probes, tasks) -> None:
    """Validate suite paths and settings without loading a model."""
    data = config["data"]
    source = sources.load(data)
    metadata = source.check(data)

    for method in methods:
        profile = profiles[method]
        embedding_methods.arms(method)
        source.make_cropper(profile, data)
        extraction = profile.get("extraction", {})
        if int(extraction.get("batch_size", 0)) < 1:
            raise ValueError(f"{method}: batch_size must be positive")
        if int(extraction.get("num_workers", 0)) < 0:
            raise ValueError(f"{method}: num_workers must be non-negative")
        np.dtype(extraction.get("storage_dtype", "float16"))
        for key in ("checkpoint", "repository", "encoder_weights"):
            value = profile.get("load", {}).get(key)
            if value is not None and not Path(value).exists():
                raise FileNotFoundError(f"{method}: {key} does not exist: {value}")

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
        records.extend(
            save_sample(
                output_dir,
                method,
                sample.sample_id,
                sample.row_ids,
                result,
                storage_dtype=extraction.get("storage_dtype", "float16"),
            )
        )
        print(f"[{method}] {sample.sample_id}: {len(sample.dataset):,} cells")
    write_manifests(output_dir, method, records)


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


def evaluate(config, methods, probes, tasks) -> None:
    """Evaluate every arm of the selected extracted methods."""
    probe_config = config["probe"]
    embedding_dir = Path(config["output_dir"]) / "embeddings"
    evaluation_dir = Path(config["output_dir"]) / "evaluation"
    seed = int(probe_config.get("seed", 0))

    for method in methods:
        for arm in embedding_methods.arms(method):
            manifest = embedding_dir / method / arm / "manifest.csv"
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
                for probe in probes:
                    settings = dict(probe_config.get(probe, {}))
                    settings.setdefault("seed", seed)
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
                    output = evaluation_dir / method / arm / task_name / probe
                    probe_api.save_result(output, result)
                    print(
                        f"[{method}/{arm}/{task_name}/{probe}] "
                        f"macro_f1={result.metrics['macro_f1']:.4f}"
                    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "extract", "evaluate", "run"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", action="append", choices=embedding_methods.METHODS)
    parser.add_argument("--probe", action="append", choices=probe_api.PROBES)
    parser.add_argument("--task", action="append", choices=TASKS)
    parser.add_argument("--device")
    args = parser.parse_args()

    config, profiles = _load_suite(args.config)
    methods = _select(
        args.method, config["methods"], embedding_methods.METHODS, "methods"
    )
    probe_config = config["probe"]
    probes = _select(args.probe, probe_config["methods"], probe_api.PROBES, "probes")
    tasks = _select(args.task, probe_config["tasks"], TASKS, "tasks")

    if args.command == "check":
        check(config, profiles, methods, probes, tasks)
    elif args.command == "extract":
        extract(config, profiles, methods, args.device)
    elif args.command == "evaluate":
        evaluate(config, methods, probes, tasks)
    else:
        check(config, profiles, methods, probes, tasks)
        extract(config, profiles, methods, args.device)
        evaluate(config, methods, probes, tasks)


if __name__ == "__main__":
    main()
